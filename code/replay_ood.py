"""Latent OOD signal for the replay-finetuned checkpoint (off-large, 2026-09-12).

The earlier OOD before/after numbers in OOD_NOTES.md were measured on the
*forgetting* checkpoints. This re-measures them for ``runs/rot250L_replay.model``
-- the fix -- and answers the question the energy gate cannot: does the replay
fine-tune still pull the rotaxane into distribution, or did the replay head buy
back the benchmark chemistry by giving up the latent gain?

Pool-matching rule change
-------------------------
The old rule was "match the pool in encoder, size AND composition", where
composition meant CHNOF-only: the forgetting checkpoints' z_table had dropped
P/S/Cl/Br/I, so their pools silently lost those OFF23 frames and the stock
control had to be CHNOF-filtered to compensate. The replay checkpoint keeps all
10 OFF23 elements, so its pool drops nothing and the control is a plain
full-element size-matched subsample of the existing stock pool. Reusing the old
``off23_pool_largeL_CHNOF_sub.npz`` here would be an invalid comparison.

Stages (each cached; a reap costs at most one frame):

    python replay_ood.py --stage pool      # encode 2500 OFF23 frames through the ft model
    python replay_ood.py --stage match     # size-matched stock control by row subsample
    python replay_ood.py --stage score     # rotaxane / dethreading / benchmarks
    python replay_ood.py --stage report
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("replay_ood")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_REPO = Path(__file__).resolve().parent.parent
_DATA = _REPO / "data"

FT_MODEL = _REPO / "runs" / "rot250L_replay.model"
FT_POOL = _DATA / "off23_pool_ft_replay.npz"          # encoded through the fix
STOCK_POOL = _DATA / "off23_pool_large.npz"           # existing, expensive, reused
MATCH_POOL = _DATA / "off23_pool_large_sub_replay.npz"  # full-element size match
SCORES = _DATA / "replay_ood_scores.json"

ROT250 = _DATA / "rot1_sampled_250.xyz"
DETHREAD = _DATA / "dethread1_sampled_10_fixed.xyz"


def _ft_calc():
    import torch
    from mace.calculators import MACECalculator

    _orig = torch.jit.load                      # CUDA-saved buffers -> CPU
    torch.jit.load = lambda *a, **kw: _orig(
        *a, **{**kw, "map_location": kw.get("map_location", "cpu")})
    return MACECalculator(model_paths=str(FT_MODEL), device="cpu",
                          default_dtype="float64")


def stage_pool(n_frames: int = 2500):
    """Encode the OFF23 test-split sample through the replay checkpoint."""
    import mace_calc as mc
    from activation_ood import ReferencePool

    if FT_POOL.exists():
        p = ReferencePool.load(FT_POOL)
        log.info("ft pool already built: %d atoms", p.atom_vecs.shape[0])
        return p
    ft = _ft_calc()
    orig = mc.get_calculator
    mc.get_calculator = lambda **kw: ft          # build runs the ft encoder
    try:
        p = ReferencePool.build(n_frames=n_frames, model="off-large", out=FT_POOL)
    finally:
        mc.get_calculator = orig
    log.info("ft pool built: %d atoms, elements %s", p.atom_vecs.shape[0],
             sorted(set(p.atom_elements.tolist())))
    return p


def stage_match():
    """Size-matched stock control: full-element row subsample of the stock pool."""
    from activation_ood import ReferencePool

    if MATCH_POOL.exists():
        p = ReferencePool.load(MATCH_POOL)
        log.info("matched stock pool already built: %d atoms", p.atom_vecs.shape[0])
        return p
    ft = ReferencePool.load(FT_POOL)
    big = ReferencePool.load(STOCK_POOL)
    n_ft, n_big = ft.atom_vecs.shape[0], big.atom_vecs.shape[0]
    log.info("stock pool %d atoms; ft pool %d atoms", n_big, n_ft)
    if n_big < n_ft:
        raise SystemExit(f"stock pool ({n_big}) smaller than ft pool ({n_ft})")
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(n_big, size=n_ft, replace=False))
    vecs = big.atom_vecs[sel]
    els = np.asarray([big.atom_elements[i] for i in sel], dtype=object)
    p = ReferencePool(vecs, els, ReferencePool._self_stats(vecs, els),
                      big.pair_vecs, big.pair_types)
    p.save(MATCH_POOL)
    log.info("matched stock pool: %d atoms, elements %s", p.atom_vecs.shape[0],
             sorted(set(p.atom_elements.tolist())))
    return p


def _mean_ood(atoms, calc, pool):
    from activation_ood import atom_ood_scores

    atoms.calc = calc
    d = atom_ood_scores(atoms, pool)["distances"]
    d = d[~np.isnan(d)]
    atoms.calc = None
    return float(d.mean()) if d.size else float("nan")


def stage_score():
    """Score rotaxane training frames, held-out dethreading frames, benchmarks."""
    import mace_calc as mc
    from activation_ood import ReferencePool
    from trust_frames import read_frames

    ft_pool, st_pool = ReferencePool.load(FT_POOL), ReferencePool.load(MATCH_POOL)
    ft = _ft_calc()
    stock = mc.get_calculator(model="off-large", dtype="float64")

    sets = {"rot250": read_frames(ROT250), "dethread10": read_frames(DETHREAD)}
    import forgetting as fg
    sets["benchmarks"] = [row[3] for row in fg.systems()]     # the complex/dimer

    res = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    for name, frames in sets.items():
        res.setdefault(name, {"before": [], "after": []})
        done = len(res[name]["before"])
        for k, at in enumerate(frames):
            if k < done:
                continue
            res[name]["before"].append(_mean_ood(at, stock, st_pool))
            res[name]["after"].append(_mean_ood(at, ft, ft_pool))
            SCORES.write_text(json.dumps(res, indent=1))
            if k % 10 == 0:
                log.info("%s %3d/%d  before %.3f  after %.3f", name, k, len(frames),
                         res[name]["before"][-1], res[name]["after"][-1])
    return res


def stage_report():
    res = json.loads(SCORES.read_text())
    print("\n=== Latent OOD, replay checkpoint vs stock (off-large, matched pools) ===")
    print("  pools: ft = encoded through runs/rot250L_replay.model; stock = "
          "full-element\n  size-matched subsample of data/off23_pool_large.npz "
          "(new rule: all 10 elements).")
    print(f"\n  {'set':12s} {'n':>4s} {'before':>8s} {'after':>8s} {'delta':>8s}"
          f"  {'before range':>16s} {'after range':>16s}")
    for name, d in res.items():
        b, a = np.array(d["before"]), np.array(d["after"])
        if not b.size:
            continue
        print(f"  {name:12s} {b.size:4d} {b.mean():8.3f} {a.mean():8.3f} "
              f"{a.mean() - b.mean():+8.3f}  "
              f"[{b.min():.3f}, {b.max():.3f}]".ljust(64)
              + f"[{a.min():.3f}, {a.max():.3f}]")
    print("\n  trust.py flags VERIFY above mean OOD 0.25.")
    print("  Old (forgetting) checkpoint for comparison: rot250 -0.021, "
          "dethread10 -0.028,\n  benchmarks S66 -0.034 / S30L -0.028 -- the same "
          "shift on chemistry it had\n  destroyed, which is why the energy gate "
          "(forgetting.py) is the real test.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=["pool", "match", "score", "report"])
    p.add_argument("--n-frames", type=int, default=2500)
    a = p.parse_args(argv)
    {"pool": lambda: stage_pool(a.n_frames), "match": stage_match,
     "score": stage_score, "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()

"""Catastrophic-forgetting check: do the GFN2-finetuned checkpoints still get the
original benchmark complexes right?

A fine-tune sees only rotaxane frames labeled with GFN2-xTB. That pulls rotaxanes
into distribution (OOD_NOTES.md), but without a replay head it also *destroys*
what the base model knew -- which is what this script exists to catch, and what
it caught on the first attempt (see OOD_NOTES.md, footnote). The original grounded OOD test
sets are exactly the right probe, because they come with reference interaction
energies:

* **S66** -- 66 non-covalent dimers, CCSD(T)/CBS reference (``BIND`` in Psi4's
  S66.py). All CHNO, all inside the finetuned z_table.
* **S30L-CI** -- host-guest complexes, computed wB97X-D3/def2-QZVP reference
  (Table S1). Only the 16 CHNOF-only systems are usable: the finetuned models
  dropped Cl from the z_table, so the 7 in-scope Cl systems cannot be run at all
  (that element loss is itself a reported result).

Interaction energy ``E_int = E(AB) - E(A) - E(B)`` is the right observable: the
composition is identical on both sides, so the refit ``E0s=average`` atomic
references (which shift the finetuned total energy by ~10^5 eV) cancel exactly.

The control that makes the result interpretable is **GFN2-xTB itself**. The
fine-tune's target is GFN2, so the finetuned model *should* move toward GFN2's
accuracy on these benchmarks, not toward the base model's. Degradation down to
the GFN2 line = successful transfer of the new level of theory; degradation far
*past* GFN2 = catastrophic forgetting.

Stages (each cached to ``data/forget_*.json``, so a rerun skips finished work):

    python forgetting.py --stage gfn2      # torch-free (tblite/torch libomp clash)
    python forgetting.py --stage mace      # stock + finetuned, both sizes
    python forgetting.py --stage ood       # latent OOD of the benchmarks, before/after
    python forgetting.py --stage report    # tables
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

log = logging.getLogger("forgetting")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_REPO = Path(__file__).resolve().parent.parent
_DATA = _REPO / "data"
_KCAL = 23.0605487415

# Finetuned checkpoints (trained on Colab GPUs -> CUDA-saved, float32).
FT_MODELS = {
    "ft-medium": _REPO / "runs" / "rot250M_replay.model",
    "ft-large": _REPO / "runs" / "rot250L_replay.model",
}
STOCK_MODELS = ["off-medium", "off-large"]
# off-large latent pools, like-for-like (both CHNOF, same atom count).
POOL_STOCK_L = _DATA / "off23_pool_largeL_CHNOF_sub.npz"
POOL_FT_L = _DATA / "off23_pool_ftL.npz"

MACE_JSON = _DATA / "forget_mace.json"
GFN2_JSON = _DATA / "forget_gfn2.json"
OOD_JSON = _DATA / "forget_ood.json"


# ---------------------------------------------------------------------------
# Benchmark systems: (set, idx, label, complex, partA, partB, reference kcal/mol)
# ---------------------------------------------------------------------------
def systems():
    """S66 dimers + the CHNOF-only S30L complexes, with reference E_int."""
    import ood_datasets as od

    keep = {"H", "C", "N", "O", "F"}
    out = []
    bind = od.s66_bind_ref()
    for idx, lab, dimer, mA, mB in od.fetch_s66():
        out.append(("S66", idx, lab, dimer, mA, mB, bind.get(idx, float("nan"))))
    for n, chg, host, guest, cplx in od.load_s30l():
        syms = set(cplx.get_chemical_symbols()) | set(host.get_chemical_symbols()) \
            | set(guest.get_chemical_symbols())
        if not syms.issubset(keep):
            log.info("S30L %d: %s outside the finetuned z_table; skipped",
                     n, sorted(syms - keep))
            continue
        out.append(("S30L", n, f"S30L-{n}", cplx, host, guest,
                    od.s30l_computed_ref(n, "wB97XD3")))
    return out


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------------------
# Stage: GFN2-xTB (the fine-tune's target level) -- torch must NOT be imported
# ---------------------------------------------------------------------------
def stage_gfn2():
    from tblite.ase import TBLite

    res = _load(GFN2_JSON)
    for tag, idx, lab, cplx, a, b, ref in systems():
        key = f"{tag}:{idx}"
        if key in res:
            continue
        try:
            es = []
            for at in (cplx, a, b):
                at.calc = TBLite(method="GFN2-xTB", verbosity=0)
                es.append(float(at.get_potential_energy()))
                at.calc = None
            res[key] = {"e_int": (es[0] - es[1] - es[2]) * _KCAL, "ref": ref}
            log.info("%-9s GFN2 E_int %8.2f   ref %8.2f", key,
                     res[key]["e_int"], ref)
        except Exception as e:                       # SCF failure on a big host
            res[key] = {"e_int": None, "ref": ref, "error": str(e)[:200]}
            log.warning("%-9s GFN2 failed: %s", key, str(e)[:120])
        _save(GFN2_JSON, res)
    return res


# ---------------------------------------------------------------------------
# Stage: MACE energies + forces, stock and finetuned
# ---------------------------------------------------------------------------
def _device() -> str:
    """"cuda" when a GPU is present (Modal), else "cpu" (the laptop)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ft_calculator(path):
    """Load a GPU-trained (CUDA-saved, float32) checkpoint, CPU or GPU."""
    import torch
    from mace.calculators import MACECalculator

    dev = _device()
    if dev == "cpu":                             # e3nn TorchScript buffers are CUDA
        _orig = torch.jit.load
        torch.jit.load = lambda *a, **kw: _orig(
            *a, **{**kw, "map_location": kw.get("map_location", "cpu")})
    return MACECalculator(model_paths=str(path), device=dev,
                          default_dtype="float64")


def stage_mace(models=None):
    import mace_calc as mc

    models = models or (STOCK_MODELS + list(FT_MODELS))
    res = _load(MACE_JSON)
    sysl = systems()
    for name in models:
        calc = (_ft_calculator(FT_MODELS[name]) if name in FT_MODELS
                else mc.get_calculator(model=name, dtype="float64",
                                       device=_device()))
        for tag, idx, lab, cplx, a, b, ref in sysl:
            key = f"{name}|{tag}:{idx}"
            if key in res:
                continue
            es, fmax = [], []
            for at in (cplx, a, b):
                at.calc = calc
                es.append(float(at.get_potential_energy()))
                fmax.append(float(np.abs(at.get_forces()).max()))
                at.calc = None
            res[key] = {"e_int": (es[0] - es[1] - es[2]) * _KCAL, "ref": ref,
                        "e_cplx": es[0], "fmax_cplx": fmax[0],
                        "n": len(cplx)}
            log.info("%-22s E_int %9.2f  ref %8.2f  err %8.2f  |F|max %6.3f",
                     key, res[key]["e_int"], ref, res[key]["e_int"] - ref, fmax[0])
            _save(MACE_JSON, res)
    return res


# ---------------------------------------------------------------------------
# Stage: latent OOD of the benchmarks, stock vs finetuned (off-large only)
# ---------------------------------------------------------------------------
def stage_ood():
    import mace_calc as mc
    from activation_ood import ReferencePool, atom_ood_scores

    stock = mc.get_calculator(model="off-large", dtype="float64")
    ft = _ft_calculator(FT_MODELS["ft-large"])
    pools = {"before": ReferencePool.load(POOL_STOCK_L),
             "after": ReferencePool.load(POOL_FT_L)}
    res = _load(OOD_JSON)
    for tag, idx, lab, cplx, a, b, ref in systems():
        key = f"{tag}:{idx}"
        if key in res:
            continue
        row = {}
        for stage, calc in (("before", stock), ("after", ft)):
            cplx.calc = calc
            d = atom_ood_scores(cplx, pools[stage])["distances"]
            d = d[~np.isnan(d)]
            row[stage] = float(d.mean()) if d.size else float("nan")
            row[stage + "_max"] = float(d.max()) if d.size else float("nan")
        cplx.calc = None
        res[key] = row
        log.info("%-9s OOD before %.3f  after %.3f  (%+.3f)", key,
                 row["before"], row["after"], row["after"] - row["before"])
        _save(OOD_JSON, res)
    return res



# ---------------------------------------------------------------------------
# Control: does the finetuned checkpoint still reproduce GFN2 on the chemistry
# it was TRAINED on? (rules out "the checkpoint is broken / mis-loaded")
# ---------------------------------------------------------------------------
CONTROL_JSON = _DATA / "forget_control.json"


def stage_control(n_frames: int = 25, names=("off-large", "ft-large")):
    """Relative energies over GFN2-labeled rotaxane frames: ft vs the GFN2 labels.

    ``data/rot1_gfn2.xyz`` carries GFN2 ``REF_energy`` per frame -- the same
    molecule and level of theory the fine-tune targeted. E0s differ, so the
    comparison is on *relative* energies (each series minus its own mean). If
    the finetuned model tracks GFN2 here but fails S66/S30L, the checkpoint
    loads and runs correctly and the benchmark failure is forgetting.
    """
    from ase.io import read
    import mace_calc as mc

    frames = read(_DATA / "rot1_gfn2.xyz", ":")
    idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
    frames = [frames[i] for i in idx]
    ref = np.array([f.info["REF_energy"] for f in frames])

    out = _load(CONTROL_JSON)
    for name in names:
        if name in out:
            continue
        calc = (_ft_calculator(FT_MODELS[name]) if name in FT_MODELS
                else mc.get_calculator(model=name, dtype="float64",
                                       device=_device()))
        es = []
        for k, at in enumerate(frames):
            at.calc = calc
            es.append(float(at.get_potential_energy()))
            at.calc = None
            if k % 5 == 0:
                log.info("control %s frame %d/%d", name, k, len(frames))
        out[name] = es
        _save(CONTROL_JSON, out)

    print("\n=== Control: GFN2-labeled rotaxane frames (relative energies, kcal/mol) ===")
    r = (ref - ref.mean()) * _KCAL
    for name, es in out.items():
        v = (np.array(es) - np.mean(es)) * _KCAL
        mae = np.abs(v - r).mean()
        corr = np.corrcoef(v, r)[0, 1]
        print(f"  {name:10s}  MAE vs GFN2 {mae:7.2f}   r {corr:6.3f}   "
              f"spread {v.max() - v.min():7.2f} (GFN2 {r.max() - r.min():.2f})")
    print("  -> the fine-tune DID learn GFN2 on its own distribution if ft beats stock here.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _errs(rows, tag):
    v = np.array([r["e_int"] - r["ref"] for k, r in rows.items()
                  if k.split("|")[-1].startswith(tag + ":")
                  and r.get("e_int") is not None])
    return v


def stage_report():
    mace, gfn2, ood = _load(MACE_JSON), _load(GFN2_JSON), _load(OOD_JSON)
    present = []                       # every model scored into the JSON, in order
    for k in mace:
        m = k.split("|")[0]
        if m not in present:
            present.append(m)
    known = [m for m in STOCK_MODELS + list(FT_MODELS) if m in present]
    names = known + [m for m in present if m not in known]

    print("\n=== Interaction-energy error vs benchmark reference (kcal/mol) ===")
    print("  S66: CCSD(T)/CBS.  S30L: wB97X-D3/def2-QZVP (16 CHNOF systems).")
    print(f"\n  {'model':12s} | {'S66 MAE':>8s} {'med':>6s} {'max':>7s} "
          f"{'MSE':>7s} | {'S30L MAE':>9s} {'med':>6s} {'max':>7s} {'MSE':>8s}")
    print("  " + "-" * 78)
    for name in names + (["GFN2-xTB"] if gfn2 else []):
        rows = ({k: v for k, v in mace.items() if k.startswith(name + "|")}
                if name != "GFN2-xTB" else gfn2)
        line = f"  {name:12s} |"
        for tag in ("S66", "S30L"):
            e = _errs(rows, tag)
            if not e.size:
                line += f" {'--':>8s}" * 4 + " |"
                continue
            line += (f" {np.abs(e).mean():8.2f} {np.median(np.abs(e)):6.2f} "
                     f"{np.abs(e).max():7.2f} {e.mean():7.2f} |")
        print(line)
    print("\n  MSE = mean signed error (negative = over-binding).")

    pairs = [("medium", "off-medium", "ft-medium"), ("large", "off-large", "ft-large")]
    pairs += [(m, "off-medium" if "medium" in m else "off-large", m)
              for m in names if m not in ("ft-medium", "ft-large") + tuple(STOCK_MODELS)]
    for size, stock, ft in pairs:
        if not any(k.startswith(ft + "|") for k in mace):
            continue
        print(f"\n=== off-{size}: stock -> finetuned, per system (kcal/mol) ===")
        print(f"  {'system':10s} {'n':>4s} {'ref':>9s} {'stock':>9s} {'ft':>9s} "
              f"{'GFN2':>9s} | {'err stock':>9s} {'err ft':>8s} {'|F|max ft':>9s}")
        rows = []
        for k, v in mace.items():
            m, key = k.split("|")
            if m != ft:
                continue
            s = mace.get(f"{stock}|{key}")
            g = gfn2.get(key, {})
            rows.append((key, v, s, g))
        rows.sort(key=lambda r: -abs(r[1]["e_int"] - r[1]["ref"]))
        for key, v, s, g in rows[:12]:
            ge = g.get("e_int")
            print(f"  {key:10s} {v['n']:4d} {v['ref']:9.2f} "
                  f"{(s['e_int'] if s else float('nan')):9.2f} {v['e_int']:9.2f} "
                  f"{(ge if ge is not None else float('nan')):9.2f} | "
                  f"{(s['e_int'] - s['ref'] if s else float('nan')):9.2f} "
                  f"{v['e_int'] - v['ref']:8.2f} {v['fmax_cplx']:9.3f}")
        print("  (12 worst finetuned systems by |error|)")

        f_s = np.array([v["fmax_cplx"] for k, v in mace.items()
                        if k.startswith(stock + "|")])
        f_f = np.array([v["fmax_cplx"] for k, v in mace.items()
                        if k.startswith(ft + "|")])
        print(f"\n  |F|max on the benchmark geometries (true force ~0, eV/A): "
              f"stock median {np.median(f_s):.3f} / max {f_s.max():.3f}   "
              f"ft median {np.median(f_f):.3f} / max {f_f.max():.3f}")

    if ood:
        print("\n=== Latent OOD of the benchmarks, off-large (like-for-like pools) ===")
        for tag in ("S66", "S30L"):
            b = np.array([v["before"] for k, v in ood.items() if k.startswith(tag + ":")])
            a = np.array([v["after"] for k, v in ood.items() if k.startswith(tag + ":")])
            if not b.size:
                continue
            print(f"  {tag:5s} n={b.size:3d}  before {b.mean():.3f} "
                  f"[{b.min():.3f}, {b.max():.3f}]   after {a.mean():.3f} "
                  f"[{a.min():.3f}, {a.max():.3f}]   delta {a.mean() - b.mean():+.3f}")
        print("  (trust.py flags VERIFY above mean OOD 0.25)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=["gfn2", "mace", "ood", "control", "report"])
    p.add_argument("--models", nargs="*", default=None,
                   help="subset of models for --stage mace")
    p.add_argument("--control-models", nargs="*", default=None,
                   help="models for --stage control (default off-large ft-large)")
    p.add_argument("--ft-model", action="append", default=[], metavar="NAME=PATH",
                   help="register an extra finetuned checkpoint to score, e.g. "
                        "--ft-model ft-replay=../runs/rot250L_replay.model")
    a = p.parse_args(argv)
    for spec in a.ft_model:
        name, _, path = spec.partition("=")
        if not path:
            p.error(f"--ft-model wants NAME=PATH, got {spec!r}")
        FT_MODELS[name] = Path(path)
    {"gfn2": lambda: stage_gfn2(), "mace": lambda: stage_mace(a.models),
     "ood": stage_ood,
     "control": lambda: stage_control(
         names=tuple(a.control_models) if a.control_models
         else ("off-large", "ft-large")),
     "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()

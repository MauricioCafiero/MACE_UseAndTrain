"""Latent OOD for the replay-finetuned checkpoint, on a Modal GPU.

Builds BOTH reference pools in one container -- stock off-large and the replay
checkpoint -- from the same 2500 OFF23 test frames, same seed, same precision.
The only difference between them is the encoder, so the before/after comparison
is matched by construction: no CHNOF filtering, no row subsampling, no
size-matching fudge. (The old ft pools needed that because their z_table had
lost P/S/Cl/Br/I and silently dropped those frames; the replay checkpoint keeps
all 10 OFF23 elements, so both pools see identical frames.)

Then scores three sets before/after: the 250 rotaxane training frames, the 10
held-out dethreading frames, and the 82 S66/S30L benchmark complexes.

    modal run code/modal_ood.py                # L4
    modal run code/modal_ood.py --gpu A10G --n-frames 2500
"""
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
_CODE, _DATA = _REPO / "code", _REPO / "data"
_CACHE = Path.home() / ".cache" / "mace"
_FOUNDATIONS = {"off-medium": _CACHE / "MACE-OFF23_medium.model",
                "off-large": _CACHE / "MACE-OFF23_large.model"}
# finetuned checkpoints, by the foundation they were built from
_CKPTS = {"off-medium": "rot250M_replay.model", "off-large": "rot250L_replay.model"}
_BASE = {"off-medium": "off-medium", "off-large": "off-large"}
# Ship the OFF23 test tarball rather than letting the container fetch it: the
# Cambridge repository has been flaky, and a mid-run failure costs GPU minutes.
_OFF23_TGZ = _DATA / "off23_test" / "test_large_neut_no_bad_clean.tar.gz"

app = modal.App("mace-replay-ood")
vol = modal.Volume.from_name("mace-ft", create_if_missing=True)

_MODULES = ["activation_ood.py", "mace_calc.py", "ood_datasets.py",
            "forgetting.py", "trust_frames.py", "replay_ood.py",
            "inspect_activations.py", "trust.py"]
_INPUTS = ["rot1_sampled_250.xyz", "dethread1_sampled_10_fixed.xyz"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("mace-torch==0.3.16", "ase==3.29.0", "e3nn==0.4.4",
                 "numpy==2.5.1", "torch==2.13.0")
    .add_local_file(_FOUNDATIONS["off-medium"],
                    "/root/foundation/MACE-OFF23_medium.model")
    .add_local_file(_FOUNDATIONS["off-large"],
                    "/root/foundation/MACE-OFF23_large.model")
    .add_local_file(_OFF23_TGZ, f"/root/data/off23_test/{_OFF23_TGZ.name}")

    .add_local_dir(_DATA / "s66", "/root/data/s66")
    .add_local_dir(_DATA / "s30l", "/root/data/s30l")
)
# Ship whichever finetuned checkpoints exist locally: the medium one does not
# exist until its training run has finished, and an absent file is a build error.
for _ck in _CKPTS.values():
    if (_REPO / "runs" / _ck).exists():
        image = image.add_local_file(_REPO / "runs" / _ck, f"/root/runs/{_ck}")

for _m in _MODULES:
    image = image.add_local_file(_CODE / _m, f"/root/code/{_m}")
for _f in _INPUTS:
    image = image.add_local_file(_DATA / _f, f"/root/data/{_f}")


@app.function(image=image, gpu="L4", timeout=3 * 3600, memory=32768,
              volumes={"/vol": vol})
def run(n_frames: int = 2500, model: str = "off-large") -> str:
    import json
    import sys

    sys.path.insert(0, "/root/code")
    import numpy as np
    import torch

    import mace_calc as mc
    from activation_ood import ReferencePool, atom_ood_scores
    from mace.calculators import MACECalculator
    from trust_frames import read_frames

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)

    ckpt = f"/root/runs/{_CKPTS[model]}"
    base = _BASE[model]
    print("stock:", base, "| finetuned:", ckpt, flush=True)
    stock = mc.get_calculator(model=base, dtype="float64", device=dev)
    ft = MACECalculator(model_paths=ckpt, device=dev, default_dtype="float64")

    # --- both pools, same frames, differing only in encoder -----------------
    pools, orig = {}, mc.get_calculator
    for tag, calc in (("stock", stock), ("ft", ft)):
        size = "medium" if model == "off-medium" else "large"
        out = Path(f"/vol/pools/off23_pool_{size}_{tag}_replayrun.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            pools[tag] = ReferencePool.load(out)
        else:
            mc.get_calculator = lambda _c=calc, **kw: _c
            pools[tag] = ReferencePool.build(n_frames=n_frames, model=base,
                                             out=out)
            mc.get_calculator = orig
            vol.commit()
        print(f"{tag} pool: {pools[tag].atom_vecs.shape[0]} atoms, elements "
              f"{sorted(set(pools[tag].atom_elements.tolist()))}", flush=True)

    n_s, n_f = pools["stock"].atom_vecs.shape[0], pools["ft"].atom_vecs.shape[0]
    print(f"pool atom counts: stock {n_s}, ft {n_f}, "
          f"{'EXACT MATCH' if n_s == n_f else 'MISMATCH -> size-match needed'}",
          flush=True)

    # --- score the three sets ----------------------------------------------
    import forgetting as fg
    sets = {"rot250": read_frames("/root/data/rot1_sampled_250.xyz"),
            "dethread10": read_frames("/root/data/dethread1_sampled_10_fixed.xyz"),
            "benchmarks": [r[3] for r in fg.systems()]}

    def mean_ood(atoms, calc, pool):
        atoms.calc = calc
        d = atom_ood_scores(atoms, pool)["distances"]
        d = d[~np.isnan(d)]
        atoms.calc = None
        return float(d.mean()) if d.size else float("nan")

    res = {"pool_atoms": {"stock": n_s, "ft": n_f}}
    for name, frames in sets.items():
        b = [mean_ood(a, stock, pools["stock"]) for a in frames]
        a_ = [mean_ood(a, ft, pools["ft"]) for a in frames]
        res[name] = {"before": b, "after": a_}
        print(f"{name}: before {np.mean(b):.3f}  after {np.mean(a_):.3f}  "
              f"delta {np.mean(a_) - np.mean(b):+.3f}", flush=True)

    Path(f"/vol/replay_ood_scores_{model}.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return json.dumps(res)


@app.local_entrypoint()
def main(gpu: str = "L4", n_frames: int = 2500, pull_pools: bool = True,
         model: str = "off-large"):
    if model not in _CKPTS:
        raise SystemExit(f"model must be one of {sorted(_CKPTS)}")
    opts = {"gpu": gpu} if gpu != "L4" else {}
    out = run.with_options(**opts).remote(n_frames=n_frames, model=model)
    dest = _DATA / ("replay_ood_scores.json" if model == "off-large"
                    else f"replay_ood_scores_{model}.json")
    dest.write_text(out)
    print(f"\nscores -> {dest}")

    if not pull_pools:
        print("pools left on the volume (--no-pull-pools)")
        return
    # Pull the pools down: each embeds 2500 GPU-encoded frames, so they are the
    # expensive artifact here -- the scores JSON is cheap to regenerate from
    # them, they are not cheap to regenerate from anything.
    size = "medium" if model == "off-medium" else "large"
    for tag in ("stock", "ft"):
        rel = f"pools/off23_pool_{size}_{tag}_replayrun.npz"
        local = _DATA / Path(rel).name
        if local.exists():
            print(f"already local, skipping: {local.name}")
            continue
        print(f"downloading {rel} ...", flush=True)
        # Stream-writing these ~1-2 GB pools with vol.read_file produced silently
        # truncated files (2 of 4 attempts), so shell out to the CLI, which has
        # been reliable, and verify the result actually opens before keeping it.
        import subprocess
        import zipfile
        subprocess.run(["modal", "volume", "get", "mace-ft", rel, str(local)],
                       check=True)
        try:
            with zipfile.ZipFile(local) as z:
                bad = z.testzip()
            if bad:
                raise zipfile.BadZipFile(f"corrupt member {bad}")
        except Exception as e:
            local.unlink(missing_ok=True)
            raise SystemExit(f"download of {rel} is corrupt ({e}); rerun to retry")
        print(f"  -> {local} ({local.stat().st_size / 1e9:.2f} GB, verified)")

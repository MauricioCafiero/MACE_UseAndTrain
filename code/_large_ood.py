"""One-off: build an off-large reference pool and re-run the reliability separation
with off-large's own latent space. Answers: with off-large, which systems are still
OOD, and is the remaining energy failure (Cl4 25/26) flagged?"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
# hold off macOS idle sleep for the whole run (dies with this process)
try:
    import subprocess
    subprocess.Popen(["caffeinate", "-w", str(os.getpid())],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass
import numpy as np
from pathlib import Path
from activation_ood import ReferencePool
from ood_datasets import (fetch_s66, s66_bind_ref, load_s30l, s30l_computed_ref,
                          _KCAL, _mace_energy_eV, score_atoms)

POOL = Path(__file__).resolve().parent.parent / "data" / "off23_pool_large.npz"
MODEL = "off-large"
REL = 10.0

# 1. Build the off-large pool (skip if cached).
if not POOL.exists():
    print(f"building off-large pool -> {POOL} (2500 frames)...", flush=True)
    ReferencePool.build(n_frames=2500, model=MODEL, out=POOL)
    print("pool built.", flush=True)
else:
    print(f"using cached off-large pool {POOL}", flush=True)

pool = ReferencePool.load(POOL)


def _e(atoms):
    atoms.calc = None
    return _mace_energy_eV(atoms, model=MODEL, dtype="float64")


def _lat(atoms):
    atoms.calc = None
    return score_atoms(atoms, pool, model=MODEL, dtype="float64")


rows = []  # (set, idx, formula, err, latMean, latMax)
print("scoring S66 (66 positives)...", flush=True)
bind = s66_bind_ref()
for idx, lab, dimer, mA, mB in fetch_s66():
    err = (_e(dimer) - _e(mA) - _e(mB)) * _KCAL - bind.get(idx, float("nan"))
    sc = _lat(dimer)
    rows.append(("S66", idx, dimer.get_chemical_formula(), err, sc["mean"], sc["max"]))
print("scoring S30L (23)...", flush=True)
for n, ch, host, guest, cplx in load_s30l():
    err = (_e(cplx) - _e(host) - _e(guest)) * _KCAL - s30l_computed_ref(n, "wB97XD3")
    sc = _lat(cplx)
    rows.append(("S30L", n, cplx.get_chemical_formula(), err, sc["mean"], sc["max"]))

s66r = [r for r in rows if r[0] == "S66"]
s30 = [r for r in rows if r[0] == "S30L"]
s30_bad = [r for r in s30 if abs(r[3]) >= REL]
print(f"\n=== off-large latent OOD ===")
print(f"  S66 (positives): latent mean [{min(r[4] for r in s66r):.3f}, "
      f"{np.median([r[4] for r in s66r]):.3f}, {max(r[4] for r in s66r):.3f}] "
      f"max [{min(r[5] for r in s66r):.3f}, {np.median([r[5] for r in s66r]):.3f}, "
      f"{max(r[5] for r in s66r):.3f}]")
print(f"  S30L energy failures (|err|>={REL:.0f}): "
      + (", ".join(f"{r[1]}(err{r[3]:+.1f},mean{r[4]:.3f},max{r[5]:.3f})"
                   for r in s30_bad) if s30_bad else "NONE"))
print(f"\n  {'set':5s} {'#':>3} {'formula':16s} {'err':>8s} {'latMean':>8s} "
      f"{'latMax':>7s} {'label':>6s}")
for s, i, f, e, lm, lmx in sorted(rows, key=lambda r: r[5]):
    lab = "OK" if abs(e) < REL else "FAIL"
    print(f"  {s:5s} {i:3d} {f:16s} {e:8.2f} {lm:8.3f} {lmx:7.3f} {lab:>6s}")
"""Where does the validation energy error live? Per-frame breakdown.

Evaluates a (CPU) model on the validation file and groups the per-atom energy
error by |charge|, multiplicity, frame size, and the rarest element in the
frame (counted as frames-containing-it in the training file). Writes the
per-frame table to longrange/eval/<label>/frame_errors.tsv.

    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/frame_errors.py \
        runs/scratch/omol_omol25_wide20k_r6/omol_omol25_wide20k_r6.cpu.model r6 [data_prefix]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

REPO = Path(__file__).resolve().parents[1]
model_path, label = sys.argv[1], sys.argv[2]
data = sys.argv[3] if len(sys.argv) > 3 else "omol25_wide20k"
train = read(REPO / "data" / f"{data}_train.xyz", index=":", format="extxyz")
valid = read(REPO / "data" / f"{data}_valid.xyz", index=":", format="extxyz")
el_frames = Counter(s for at in train for s in set(at.get_chemical_symbols()))

from mace.calculators import MACECalculator
calc = MACECalculator(model_paths=[model_path], device="cpu", default_dtype="float64")

rows = []
for i, at in enumerate(valid):
    ref = at.info["REF_energy"]
    q, m = at.info["total_charge"], at.info["total_spin"]
    a = at.copy()
    a.info["charge"], a.info["spin"] = q, m
    a.calc = calc
    err = (a.get_potential_energy() - ref) / len(at) * 1000      # meV/atom
    rare = min(set(at.get_chemical_symbols()), key=lambda s: el_frames.get(s, 0))
    rows.append((i, err, int(q), int(m), len(at), rare, el_frames.get(rare, 0),
                 at.get_chemical_formula()))

err = np.array([r[1] for r in rows])
print(f"{label}: valid RMSE_E {np.sqrt((err**2).mean()):.1f}  MAE_E {np.abs(err).mean():.1f}  "
      f"median |err| {np.median(np.abs(err)):.1f} meV/atom  (n={len(err)})")
o = np.argsort(-np.abs(err))
share = (err[o] ** 2).cumsum() / (err ** 2).sum()
for k in (10, 20, 50, 100, 200):
    print(f"  worst {k:3d} frames carry {100 * share[k - 1]:5.1f}% of the squared error")


def group(name, keyfn):
    g = {}
    for r in rows:
        g.setdefault(keyfn(r), []).append(r[1])
    print(f"\n  by {name}:  n | RMSE_E | MAE_E")
    for k in sorted(g):
        e = np.array(g[k])
        print(f"   {str(k):>14} {len(e):5d} | {np.sqrt((e**2).mean()):6.1f} | {np.abs(e).mean():6.1f}")


group("|charge|", lambda r: min(abs(r[2]), 4))
group("multiplicity", lambda r: min(r[3], 5))
group("rarest-element train frames", lambda r: ("<10" if r[6] < 10 else "10-99" if r[6] < 100
                                                 else "100-999" if r[6] < 1000 else ">=1000"))
group("atoms", lambda r: (r[4] // 20) * 20)

print("\n  worst 15 frames: err meV/atom | q | M | atoms | rarest el (train frames) | formula")
for i in o[:15]:
    r = rows[i]
    print(f"   {r[1]:+8.0f} | {r[2]:+d} | {r[3]} | {r[4]:3d} | {r[5]:>2} ({r[6]}) | {r[7]}")

out = REPO / "longrange" / "eval" / label
out.mkdir(parents=True, exist_ok=True)
with open(out / "frame_errors.tsv", "w") as fh:
    fh.write("idx\terr_meV_per_atom\tq\tM\tnatoms\trarest_el\trarest_train_frames\tformula\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")

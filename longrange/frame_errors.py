"""Where does the validation energy error live? Per-frame breakdown.

Evaluates a (CPU) model on the validation file and groups the per-atom energy
error by |charge|, multiplicity, frame size, and the rarest element in the
frame (counted as frames-containing-it in the training file). Also reports each frame's force RMSE and its most-isolated-fragment gap (same
definition as measure_fragment_gaps.py: covalent-radius bonding graph; the
largest over fragments of the min distance to the rest), grouped by gap.
Writes the per-frame table to longrange/eval/<label>/frame_errors.tsv.

PolarMACE models (``--polar``): needs graph_longrange, i.e. cafquant's
.venv/bin/python; applies the same non-periodic k-grid shrink as the Modal
trainer (energies unchanged, memory bounded).

    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/frame_errors.py \
        runs/scratch/omol_omol25_wide20k_r6/omol_omol25_wide20k_r6.cpu.model r6 [data_prefix] [--polar]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[1]
POLAR = "--polar" in sys.argv
argv = [a for a in sys.argv if a != "--polar"]
model_path, label = argv[1], argv[2]
data = argv[3] if len(argv) > 3 else "omol25_wide20k"
train = read(REPO / "data" / f"{data}_train.xyz", index=":", format="extxyz")
valid = read(REPO / "data" / f"{data}_valid.xyz", index=":", format="extxyz")
el_frames = Counter(s for at in train for s in set(at.get_chemical_symbols()))

if POLAR:
    import mace.modules.extensions as ext       # all frames pbc F F F: realspace path
    _kv = ext.compute_k_vectors_flat
    ext.compute_k_vectors_flat = lambda cut, cell, rcell: _kv(1e-6, cell, rcell)
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths=[model_path], device="cpu", default_dtype="float64",
                      **({"charges_key": "charge"} if POLAR else {}))


def frag_gap(at):
    p = at.positions
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    r = covalent_radii[at.numbers]
    n, lab = connected_components(d < 1.2 * (r[:, None] + r[None]), directed=False)
    return 0.0 if n == 1 else max(d[np.ix_(lab == a, lab != a)].min() for a in range(n))


rows = []
for i, at in enumerate(valid):
    ref = at.info["REF_energy"]
    q, m = at.info["total_charge"], at.info["total_spin"]
    a = at.copy()
    a.info["charge"], a.info["spin"] = q, m
    a.calc = calc
    err = (a.get_potential_energy() - ref) / len(at) * 1000      # meV/atom
    ferr = np.sqrt(((a.get_forces() - at.arrays["REF_forces"]) ** 2).mean()) * 1000   # meV/A
    rare = min(set(at.get_chemical_symbols()), key=lambda s: el_frames.get(s, 0))
    rows.append((i, err, int(q), int(m), len(at), rare, el_frames.get(rare, 0),
                 at.get_chemical_formula(), ferr, frag_gap(at)))

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
GAPS = [(0, "1 fragment"), (3, "gap <3"), (4.5, "3-4.5"), (6, "4.5-6"), (8, "6-8"), (1e9, ">8")]
gapbin = lambda g: "1 fragment" if g == 0 else next(n for c, n in GAPS[1:] if g < c)
group("isolated-fragment gap (A)", lambda r: gapbin(r[9]))
gf = {}
for r in rows:
    gf.setdefault(gapbin(r[9]), []).append(r[8])
print("\n  force RMSE by isolated-fragment gap:  n | RMSE_F (meV/A) | median")
for _, n in GAPS:
    if n in gf:
        f = np.array(gf[n])
        print(f"   {n:>14} {len(f):5d} | {np.sqrt((f**2).mean()):8.1f} | {np.median(f):8.1f}")
fo = np.argsort(-np.array([r[8] for r in rows]))
print("\n  worst 15 frames by force: F err meV/A | E err meV/atom | gap A | q | M | atoms | formula")
for i in fo[:15]:
    r = rows[i]
    print(f"   {r[8]:9.0f} | {r[1]:+8.0f} | {r[9]:5.2f} | {r[2]:+d} | {r[3]} | {r[4]:3d} | {r[7]}")

print("\n  worst 15 frames: err meV/atom | q | M | atoms | rarest el (train frames) | formula")
for i in o[:15]:
    r = rows[i]
    print(f"   {r[1]:+8.0f} | {r[2]:+d} | {r[3]} | {r[4]:3d} | {r[5]:>2} ({r[6]}) | {r[7]}")

out = REPO / "longrange" / "eval" / label
out.mkdir(parents=True, exist_ok=True)
with open(out / "frame_errors.tsv", "w") as fh:
    fh.write("idx\terr_meV_per_atom\tq\tM\tnatoms\trarest_el\trarest_train_frames\tformula"
             "\tF_rmse_meV_per_A\tfrag_gap_A\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")

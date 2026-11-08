"""Top-N frames by most-isolated-fragment gap from the OMol25 scratch pool,
plus per-frame edge counts at 6 and 12 A for cost scaling, and a sense-check
of the selection (per-element coverage, charge/spin mix, frame sizes).

Original wide20k (all 83 elements, any charge):
    .venv/bin/python longrange/make_wide20k.py --out omol25_wide20k
SPICE-element, |q| <= 2, all metal frames forced in, from the full pool:
    .venv/bin/python longrange/make_wide20k.py --pool omol25_full60 --spice --max-abs-q 2 \
        --force-elements Li,Na,K,Mg,Ca --out omol25_spice20k
"""
import argparse
from collections import Counter

import numpy as np
from ase.io import read, write
from ase.data import covalent_radii
from scipy.sparse.csgraph import connected_components

SPICE = {"H", "Li", "B", "C", "N", "O", "F", "Na", "Mg", "Si", "P", "S", "Cl",
         "K", "Ca", "Br", "I"}          # SPICE 2 element set (17)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="omol25_wide20k")
ap.add_argument("--pool", default="omol25_scratch", help="data/<pool>_{train,valid}.xyz to select from")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--spice", action="store_true", help="keep only frames whose elements are all in SPICE 2")
ap.add_argument("--max-abs-q", type=int, default=None)
ap.add_argument("--force-elements", default="", help="comma list: frames containing any of these are always kept, rest filled by gap")
a = ap.parse_args()

frames = (read(f"data/{a.pool}_train.xyz", index=":", format="extxyz")
          + read(f"data/{a.pool}_valid.xyz", index=":", format="extxyz"))
qall = Counter(int(at.info["total_charge"]) for at in frames)
print("pool charge counts |q|>=3:", {q: c for q, c in sorted(qall.items()) if abs(q) >= 3})
keep = np.ones(len(frames), bool)
if a.spice:
    keep &= np.array([set(at.get_chemical_symbols()) <= SPICE for at in frames])
if a.max_abs_q is not None:
    keep &= np.array([abs(at.info["total_charge"]) <= a.max_abs_q for at in frames])
print(f"pool {len(frames)} -> {keep.sum()} after filters (spice={a.spice}, max|q|={a.max_abs_q})")

gap, nat, e6, e12 = [], [], [], []
for at in frames:
    d = at.get_all_distances(); np.fill_diagonal(d, np.inf)
    r = covalent_radii[at.numbers]
    n, lab = connected_components(d < 1.2 * (r[:, None] + r[None]), directed=False)
    gap.append(0.0 if n == 1 else max(d[np.ix_(lab == a_, lab != a_)].min() for a_ in range(n)))
    nat.append(len(at)); e6.append((d < 6).sum()); e12.append((d < 12).sum())
gap, nat, e6, e12 = map(np.array, (gap, nat, e6, e12))
force = {e for e in a.force_elements.split(",") if e}
forced = np.array([bool(force & set(at.get_chemical_symbols())) for at in frames]) & keep
print(f"forced-in frames (contain any of {sorted(force)}): {forced.sum()}")
rank = np.where(keep, gap, -np.inf) + np.where(forced, 1e6, 0.0)
order = np.argsort(-rank, kind="stable")
top = order[:min(a.n, int(keep.sum()))]
top = top[np.argsort(-gap[top], kind="stable")]     # interleave forced frames into the valid stride
print("selected %d: gap min/median/max: %.2f / %.2f / %.2f A" % (
    len(top), gap[top].min(), np.median(gap[top]), gap[top].max()))
print("selected: gap>4.5: %d  >6: %d  >8: %d  >12: %d  single-fragment: %d" % (
    tuple(int((gap[top] > c).sum()) for c in (4.5, 6, 8, 12)) + (int((gap[top] == 0).sum()),)))
print("mean atoms: selected %.1f" % nat[top].mean())
vi = top[::10]; ti = np.setdiff1d(top, vi)
print("split: train %d valid %d" % (len(ti), len(vi)))
# cost: calibrate to the measured wide20k epochs (train+valid edges), L4
ref = read("data/omol25_wide20k_train.xyz", index=":", format="extxyz") + \
      read("data/omol25_wide20k_valid.xyz", index=":", format="extxyz")
def _edges(ats, c):
    tot = 0
    for at in ats:
        d = at.get_all_distances(); np.fill_diagonal(d, np.inf); tot += (d < c).sum()
    return tot
for c, mins in ((6, 7.6), (12, 11.5)):
    new = e6[ti].sum() + e6[vi].sum() if c == 6 else e12[ti].sum() + e12[vi].sum()
    f = new / _edges(ref, c)
    print(f"r{c}: edges vs wide20k x{f:.3f} -> ~{mins * f:.1f} min/epoch on L4")

# ---- sense check ---------------------------------------------------------
tr = [frames[i] for i in ti]
fr_el = Counter(s for at in tr for s in set(at.get_chemical_symbols()))
at_el = Counter(s for at in tr for s in at.get_chemical_symbols())
print("\nper-element coverage in TRAIN (frames containing it | atoms):")
for s, c in sorted(fr_el.items(), key=lambda x: x[1]):
    print(f"  {s:2s} {c:6d} | {at_el[s]:7d}" + ("   <-- under 200 frames" if c < 200 else ""))
print("charge mix (train):", dict(sorted(Counter(int(at.info['total_charge']) for at in tr).items())))
print("multiplicity mix (train):", dict(sorted(Counter(int(at.info['total_spin']) for at in tr).items())))
sz = np.array([len(at) for at in tr])
print("frame size (train): <10 atoms %d, 10-19 %d, 20-39 %d, >=40 %d" % (
    (sz < 10).sum(), ((sz >= 10) & (sz < 20)).sum(), ((sz >= 20) & (sz < 40)).sum(), (sz >= 40).sum()))
small = [at for at in tr if len(at) < 10]
if small:
    print("  smallest-frame examples:", ", ".join(
        f"{at.get_chemical_formula()}(q{int(at.info['total_charge']):+d},M{int(at.info['total_spin'])})" for at in small[:12]))

write(f"data/{a.out}_train.xyz", tr, format="extxyz")
write(f"data/{a.out}_valid.xyz", [frames[i] for i in sorted(vi)], format="extxyz")
print(f"written data/{a.out}_{{train,valid}}.xyz")

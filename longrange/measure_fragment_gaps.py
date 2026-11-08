import sys, numpy as np
from ase.io import iread
from ase.data import covalent_radii
from scipy.sparse.csgraph import connected_components
ext, nfrag, gap, q = [], [], [], []
for f in sys.argv[1:]:
    for at in iread(f, index=":", format="extxyz"):
        p = at.positions; d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        ext.append(d.max())
        r = covalent_radii[at.numbers]; bond = d < 1.2 * (r[:, None] + r[None])
        n, lab = connected_components(bond, directed=False)
        nfrag.append(n); q.append(at.info.get("total_charge", 0))
        if n > 1:
            g = max(d[np.ix_(lab == a, lab != a)].min() for a in range(n))
            gap.append(g)  # smallest distance from any fragment to the rest
        else:
            gap.append(np.nan)
ext, nfrag, gap, q = map(np.array, (ext, nfrag, gap, q))
N = len(ext); print("frames", N)
print("extent (max pair dist) pct 50/90/99/max:", np.round(np.percentile(ext, [50, 90, 99, 100]), 1))
for c in (6, 12, 18):
    print(f"  extent > {c} A: {np.mean(ext > c)*100:.1f}%")
m = nfrag > 1
print(f"multi-fragment frames: {m.sum()} ({m.mean()*100:.1f}%)")
gm = gap[m]
print("  most-isolated-fragment gap pct 50/90/99/max:", np.round(np.percentile(gm, [50, 90, 99, 100]), 2))
for c in (3, 4.5, 6, 8, 12):
    print(f"  gap > {c} A: {(gm > c).sum()}")
print("  multi-fragment & |q|>0:", int((m & (q != 0)).sum()))

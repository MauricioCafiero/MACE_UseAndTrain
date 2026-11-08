"""Even-stride subset of the OMol25 scratch set, forcing in every frame whose
most-isolated fragment sits > 6 A from the rest (the beyond-cutoff signal)."""
import numpy as np
from ase.io import read, write
from ase.data import covalent_radii
from scipy.sparse.csgraph import connected_components

def iso_gap(at):
    if len(at) < 2: return 0.0
    d = at.get_all_distances(); r = covalent_radii[at.numbers]
    n, lab = connected_components(d < 1.2 * (r[:, None] + r[None]), directed=False)
    return 0.0 if n == 1 else max(d[np.ix_(lab == a, lab != a)].min() for a in range(n))

for split, n in (("train", 10000), ("valid", 1000)):
    frames = read(f"data/omol25_scratch_{split}.xyz", index=":", format="extxyz")
    far = {i for i, at in enumerate(frames) if iso_gap(at) > 6.0}
    even = set(np.linspace(0, len(frames) - 1, n - len(far)).astype(int))
    keep = sorted(even | far)
    write(f"data/omol25_scratch10k_{split}.xyz", [frames[i] for i in keep], format="extxyz")
    print(split, "kept", len(keep), "far(>6A)", len(far))

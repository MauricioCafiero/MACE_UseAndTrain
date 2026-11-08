"""Rigid distance scans between neutral, mixed and charged fragment pairs.

Question: how much does a longer message-passing cutoff (r_max 6 vs 12 A)
help long-range interactions of different physical types -- ion-ion (r^-1),
ion-dipole (r^-2), dipole-dipole (r^-3), induction (r^-4), dispersion (r^-6)?

Fragments are built once from SMILES (RDKit ETKDG + MMFF) and kept rigid.
Each is oriented so a chosen "front" vector points along +x (A) or -x (B),
then B is translated along +x until the minimum A..B gap hits each target
(placement copied from cafquant/dft_check/charge_sweep.py place_pair).
Writes, under longrange/scan/:
  frames/<stem>.xyz       every monomer and pair geometry
  orca/<stem>/<stem>.inp  ORCA single points (wB97M-V/def2-TZVPD RIJCOSX def2/J)
  jobs.txt                run order for run_scan_seq.sh (smallest first)
  meta.json               pairs, charges, realised d_min / d_com per point

    .venv/bin/python longrange/scan/make_scans.py
"""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
GAPS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 16.0, 20.0]

# name -> (SMILES, charge). Single ions are handled as bare atoms.
FRAG = {"methane": ("C", 0), "benzene": ("c1ccccc1", 0), "water": ("O", 0),
        "acetonitrile": ("CC#N", 0), "Na+": ("[Na+]", 1), "Cl-": ("[Cl-]", -1),
        "ammonium": ("[NH4+]", 1), "acetate": ("CC(=O)[O-]", -1)}


def front(name, pos, sym):
    """Unit vector that should face the partner (None = any orientation)."""
    if name == "water":            # O -> H-bisector (dipole direction)
        o = pos[sym.index("O")]; h = pos[[i for i, s in enumerate(sym) if s == "H"]].mean(0)
        return h - o
    if name == "acetonitrile":     # CH3 C -> N
        c = pos[[i for i, s in enumerate(sym) if s == "C"]]
        n = pos[sym.index("N")]
        far = c[np.argmax(np.linalg.norm(c - n, axis=1))]
        return n - far
    if name == "benzene":          # ring normal
        c = pos[[i for i, s in enumerate(sym) if s == "C"]] - pos[[i for i, s in enumerate(sym) if s == "C"]].mean(0)
        return np.linalg.svd(c)[2][-1]
    if name == "acetate":          # methyl C -> carboxyl C (carboxylate end)
        ci = [i for i, s in enumerate(sym) if s == "C"]
        oi = [i for i, s in enumerate(sym) if s == "O"]
        cc = min(ci, key=lambda i: np.linalg.norm(pos[oi] - pos[i], axis=1).min())
        cm = [i for i in ci if i != cc][0]
        return pos[cc] - pos[cm]
    if name == "ammonium":         # one N-H bond
        n = pos[sym.index("N")]; h = pos[sym.index("H")]
        return h - n
    return None


# (label, type, A, B, A front points +x?, B front points -x?) -- B is placed at +x of A
PAIRS = [
    ("CH4-CH4",         "neutral nonpolar (disp, r^-6)",       "methane", "methane", None, None),
    ("bz-bz_stack",     "neutral nonpolar (disp+quad)",        "benzene", "benzene", True, True),
    ("H2O-H2O_aligned", "neutral dipolar (dip-dip, r^-3)",     "water", "water", True, False),
    ("MeCN-MeCN_h2t",   "neutral dipolar (dip-dip, r^-3)",     "acetonitrile", "acetonitrile", True, False),
    ("Na+-H2O",         "ion-dipole (r^-2)",                   "Na+", "water", None, False),
    ("Cl--H2O",         "ion-dipole (r^-2)",                   "Cl-", "water", None, True),
    ("Na+-CH4",         "ion-induced dipole (r^-4)",           "Na+", "methane", None, None),
    ("Na+-bz",          "cation-pi + induction",               "Na+", "benzene", None, True),
    ("NH4+-OAc-",       "ion-ion attractive (r^-1)",           "ammonium", "acetate", True, True),
    ("NH4+-Cl-",        "ion-ion attractive (r^-1)",           "ammonium", "Cl-", True, None),
    ("NH4+-NH4+",       "ion-ion repulsive (r^-1)",            "ammonium", "ammonium", True, True),
]
# Orientation notes (front vector after rotation):
#  water: O->H bisector. A "+x" + B "+x" (False = front along +x for B too) gives
#  head-to-tail aligned dipoles (A's H face B's O). Na+-H2O: B front along +x -> O faces Na+.
#  Cl--H2O: B front along -x -> H's face Cl-. MeCN: both CH3->N along +x -> A's N faces B's CH3.
#  acetate front along -x -> carboxylate faces A. ammonium front +x -> one N-H points at B.


def build(name):
    smi, q = FRAG[name]
    if smi.startswith("[") and smi.count("[") == 1 and smi[1:3].rstrip("+-]") in ("Na", "Cl"):
        return Atoms(smi[1:3].rstrip("+-]"), positions=[[0, 0, 0]]), q
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    pos = m.GetConformer().GetPositions()
    return Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=pos - pos.mean(0)), q


def rotate_to(pos, v, target):
    """Rotate positions so unit(v) -> unit(target) (Rodrigues)."""
    a = v / np.linalg.norm(v); b = target / np.linalg.norm(target)
    c = np.cross(a, b); s = np.linalg.norm(c); d = float(a @ b)
    if s < 1e-8:
        return pos if d > 0 else pos @ np.diag([-1, -1, 1]) if abs(a[2]) < 0.9 else pos @ np.diag([-1, 1, -1])
    k = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]]) / s
    R = np.eye(3) + s * k + (1 - d) * k @ k
    return pos @ R.T


def min_gap(a, b):
    return float(np.linalg.norm(a[:, None] - b[None], axis=-1).min())


def place_pair(a, b0, gap_target):
    """From cafquant charge_sweep.py: rigid +x translation of b to hit gap."""
    acom, bcom0 = a.mean(0), b0.mean(0)
    shifts = np.arange(-40.0, 80.0, 0.02)
    gaps = np.array([min_gap(a, b0 + (acom + [s, 0, 0] - bcom0)) for s in shifts])
    i = int(gaps.argmin())
    s = shifts[i] if gap_target <= gaps[i] else float(np.interp(gap_target, gaps[i:], shifts[i:]))
    b = b0 + (acom + [s, 0, 0] - bcom0)
    return b, min_gap(a, b), float(np.linalg.norm(a.mean(0) - b.mean(0)))


def orca_inp(stem, q):
    return ("! wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3\n"
            "%maxcore 2000\n%pal nprocs 4 end\n"
            f"*xyzfile {q} 1 {stem}.xyz\n")


def main():
    (HERE / "frames").mkdir(exist_ok=True); (HERE / "orca").mkdir(exist_ok=True)
    jobs, meta = {}, {"gaps": GAPS, "pairs": []}

    def add(stem, atoms, q):
        write(HERE / "frames" / f"{stem}.xyz", atoms, format="xyz")
        d = HERE / "orca" / stem; d.mkdir(exist_ok=True)
        write(d / f"{stem}.xyz", atoms, format="xyz")
        (d / f"{stem}.inp").write_text(orca_inp(stem, q))
        jobs[stem] = len(atoms)

    frags = {}
    for name in FRAG:
        at, q = build(name)
        frags[name] = (at, q)
        add(f"mono_{name.replace('+', 'p').replace('-', 'm')}", at, q)

    for label, kind, A, B, fa, fb in PAIRS:
        a_at, qa = frags[A]; b_at, qb = frags[B]
        sa, sb = a_at.get_chemical_symbols(), b_at.get_chemical_symbols()
        pa, pb = a_at.get_positions(), b_at.get_positions()
        va, vb = front(A, pa, sa), front(B, pb, sb)
        if fa is not None and va is not None:
            pa = rotate_to(pa, va, np.array([1.0, 0, 0]) if fa else np.array([-1.0, 0, 0]))
        if fb is not None and vb is not None:
            pb = rotate_to(pb, vb, np.array([-1.0, 0, 0]) if fb else np.array([1.0, 0, 0]))
        stem_l = label.replace("+", "p").replace("-", "m").replace("__", "_")
        pts = []
        for g in GAPS:
            b_new, dmin, dcom = place_pair(pa, pb, g)
            pair = Atoms(sa + sb, positions=np.vstack([pa, b_new]))
            stem = f"pair_{stem_l}_{g:04.1f}"
            add(stem, pair, qa + qb)
            pts.append({"stem": stem, "target_gap": g, "d_min": dmin, "d_com": dcom})
        meta["pairs"].append({"label": label, "type": kind, "A": A, "B": B, "qA": qa, "qB": qb,
                              "monoA": f"mono_{A.replace('+', 'p').replace('-', 'm')}",
                              "monoB": f"mono_{B.replace('+', 'p').replace('-', 'm')}",
                              "points": pts})
        print(f"{label:16s} {kind:32s} q={qa+qb:+d}  d_min {pts[0]['d_min']:.2f}..{pts[-1]['d_min']:.2f}"
              f"  d_com {pts[0]['d_com']:.2f}..{pts[-1]['d_com']:.2f}")

    order = sorted(jobs, key=lambda s: (jobs[s], s))
    (HERE / "jobs.txt").write_text("\n".join(order) + "\n")
    (HERE / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(jobs)} ORCA jobs (largest {max(jobs.values())} atoms) -> {HERE/'jobs.txt'}")


if __name__ == "__main__":
    main()

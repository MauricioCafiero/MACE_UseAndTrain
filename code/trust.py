"""SMILES or XYZ -> MACE single point -> energy + TRUST/VERIFY verdict.

    python trust.py "CC(=O)Nc1ccc(cc1)O"          # SMILES (RDKit 3D embed)
    python trust.py molecule.xyz                  # existing geometry
    python trust.py "CCO" --relax                 # MACE-relax geometry first
    python trust.py "CCO" --model off-medium      # faster model (pool must match)

Prints the MACE energy, the latent OOD signal (how far each atom's internal
representation sits from a reference pool built from the model's own training
distribution), and a verdict:

  TRUST   mean OOD <= 0.25 -- the molecule is in-distribution. On comparable
          in-distribution systems (S66) MACE-OFF matches CCSD(T)/CBS to
          0.22-0.26 kcal/mol MAE (max 0.9), so the energy is reliable.
  VERIFY  mean OOD  > 0.25 -- outside the calibration range. This does NOT
          mean the energy is wrong (some reliable systems score 0.3+); it
          means the error is no longer bounded by the benchmark, so
          cross-check with another model size or a higher level of theory.

Why a threshold at 0.25: S66 (error < 1 kcal/mol) spans 0.04-0.24; every
observed energy catastrophe (bare fullerene: 73-85 kcal/mol off-medium) sits
above 0.28. See OOD_NOTES.md for the full study.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np

import mace_calc as mc
from activation_ood import ReferencePool, atom_ood_scores

_REPO = Path(__file__).resolve().parent.parent

# Pool must match the model (latent spaces are not comparable across sizes).
POOLS = {
    "off-medium": _REPO / "data" / "off23_pool.npz",
    "off-large": _REPO / "data" / "off23_pool_large.npz",
}
MEAN_TRUST = 0.25


def verdict(mean_ood: float, model: str) -> str:
    other = "off-medium" if model == "off-large" else "off-large"
    lines = []
    if mean_ood <= MEAN_TRUST:
        lines.append("VERDICT: TRUST")
        lines.append(
            f"  mean OOD {mean_ood:.2f} is inside the in-distribution range "
            f"(S66: 0.04-0.24), where MACE-OFF matches CCSD(T)/CBS to\n"
            f"  ~0.25 kcal/mol MAE (max 0.9). The energy is reliable."
        )
    else:
        lines.append("VERDICT: VERIFY")
        lines.append(
            f"  mean OOD {mean_ood:.2f} is above the in-distribution range. "
            f"This does NOT mean the energy is wrong -- some reliable\n"
            f"  systems score equally high -- but its error is no longer "
            f"bounded by the benchmark. Cross-check with another model size\n"
            f"  (--model {other}) or a higher level of theory. Worst "
            f"atoms above show where the model is least at home."
        )
    return "\n".join(lines)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("molecule", help="SMILES string or path to an XYZ file")
    p.add_argument("--model", default="off-large", choices=sorted(POOLS),
                   help="MACE-OFF23 size (default: off-large; best MAE)")
    p.add_argument("--relax", action="store_true",
                   help="MACE-relax the geometry before the single point")
    args = p.parse_args(argv)

    # --- input: SMILES or XYZ ------------------------------------------------
    if args.molecule.lower().endswith(".xyz"):
        atoms = mc.from_xyz(args.molecule)
        label = args.molecule
    else:
        atoms = mc.smiles_to_atoms(args.molecule)
        label = args.molecule
        print("note: geometry from RDKit/MMFF (not MACE-relaxed); "
              "use --relax for a MACE-optimized geometry")

    # --- single point --------------------------------------------------------
    if args.relax:
        out = mc.optimize(atoms, model=args.model, fmax=0.01, logfile=None)
        print(f"relaxed: converged={out['converged']} fmax={out['fmax']:.3f} eV/A")
    out = mc.singlepoint(atoms, model=args.model)
    e = out["energy"]
    fmax = float(np.linalg.norm(out["forces"], axis=1).max())

    # --- signal: latent distance to the training-distribution pool -----------
    pool_path = POOLS[args.model]
    if not pool_path.exists():
        sys.exit(f"reference pool {pool_path.name} missing -- build it with "
                 f"activation_ood.ReferencePool.build(model='{args.model}')")
    pool = ReferencePool.load(pool_path)
    ood = atom_ood_scores(atoms, pool, model=args.model)

    d, els, flags = ood["distances"], ood["elements"], ood["flags"]
    mean_ood, max_ood = float(np.nanmean(d)), float(np.nanmax(d))

    # --- report --------------------------------------------------------------
    print()
    print(f"{atoms.get_chemical_formula()}  ({label})")
    print(f"model: {args.model}   energy: {e:.4f} eV   max|F|: {fmax:.3f} eV/A")
    print(f"OOD signal (latent distance to training-distribution pool):")
    print(f"  molecule mean: {mean_ood:.3f}   worst atom: {max_ood:.3f} "
          f"({els[int(np.nanargmax(d))]} #{int(np.nanargmax(d))})   "
          f"atoms over their element's p99: {int(flags.sum())}/{len(d)}")
    if ood["used_global"]:
        print(f"  elements absent from the pool (strong OOD): "
              f"{sorted(set(ood['used_global']))}")
    order = np.argsort(-np.where(np.isnan(d), -np.inf, d))[:5]
    print("  5 most-unusual atoms:")
    for i in order:
        star = " *" if flags[i] else ""
        print(f"    #{int(i):4d} {els[i]:>2s}  dist={d[i]:.3f}{star}")
    print()
    print(verdict(mean_ood, args.model))


if __name__ == "__main__":
    main()
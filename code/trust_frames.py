"""Trust/verify verdict for every frame of a multi-frame XYZ (e.g. a trajectory).

    python trust_frames.py ../data/rot1_sampled_10.xyz
    python trust_frames.py path.xyz --model off-medium

Same signal and threshold as trust.py (latent distance to the model's
training-distribution pool; mean <= 0.25 -> TRUST), applied frame by frame.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

import mace_calc as mc
from activation_ood import ReferencePool, atom_ood_scores
from trust import POOLS, verdict

_REPO = Path(__file__).resolve().parent.parent


def read_frames(path: str) -> list[Atoms]:
    """Plain multi-frame XYZ reader; MDTraj-style names (O1x) -> element (O)."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        n = int(lines[i].split()[0])
        syms, pos = [], []
        for line in lines[i + 2:i + 2 + n]:
            f = line.split()
            syms.append(re.match(r"[A-Za-z]+", f[0]).group(0).capitalize())
            pos.append([float(x) for x in f[1:4]])
        frames.append(Atoms(syms, positions=pos))
        i += 2 + n
    return frames


def analyze(atoms: Atoms, pool: ReferencePool, model: str, label: str) -> dict:
    out = mc.singlepoint(atoms, model=model)
    ood = atom_ood_scores(atoms, pool, model=model)
    d, els, flags = ood["distances"], ood["elements"], ood["flags"]
    mean_ood, max_ood = float(np.nanmean(d)), float(np.nanmax(d))
    j = int(np.nanargmax(d))

    print(f"\n--- frame {label}: {atoms.get_chemical_formula()} ---")
    print(f"model: {model}   energy: {out['energy']:.4f} eV   "
          f"max|F|: {float(np.linalg.norm(out['forces'], axis=1).max()):.3f} eV/A")
    print(f"  mean OOD: {mean_ood:.3f}   worst atom: {max_ood:.3f} "
          f"({els[j]} #{j})   atoms over element p99: {int(flags.sum())}/{len(d)}")
    if ood["used_global"]:
        print(f"  elements absent from the pool (strong OOD): "
              f"{sorted(set(ood['used_global']))}")
    order = np.argsort(-np.where(np.isnan(d), -np.inf, d))[:5]
    for i in order:
        star = " *" if flags[i] else ""
        print(f"    #{int(i):4d} {els[i]:>2s}  dist={d[i]:.3f}{star}")
    print(verdict(mean_ood, model))
    return {"frame": label, "energy": out["energy"], "mean_ood": mean_ood,
            "max_ood": max_ood, "n_flagged": int(flags.sum()),
            "verdict": "TRUST" if mean_ood <= 0.25 else "VERIFY"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xyz")
    p.add_argument("--model", default="off-large", choices=sorted(POOLS))
    args = p.parse_args()

    frames = read_frames(args.xyz)
    print(f"{len(frames)} frames, {len(frames[0])} atoms each "
          f"({frames[0].get_chemical_formula()})")

    pool_path = POOLS[args.model]
    if not pool_path.exists():
        sys.exit(f"reference pool {pool_path.name} missing")
    pool = ReferencePool.load(pool_path)

    results = []
    shared_calc = None
    for k, atoms in enumerate(frames):
        if shared_calc is None:
            mc.attach(atoms, model=args.model)
            shared_calc = atoms.calc      # one model load for all frames
        else:
            atoms.calc = shared_calc
        results.append(analyze(atoms, pool, args.model, str(k)))

    print("\n=== summary ===")
    print(f"{'frame':>5s}  {'energy (eV)':>12s}  {'mean OOD':>9s}  "
          f"{'worst':>6s}  {'p99 flags':>9s}  verdict")
    for r in results:
        print(f"{r['frame']:>5s}  {r['energy']:12.4f}  {r['mean_ood']:9.3f}  "
              f"{r['max_ood']:6.3f}  {r['n_flagged']:5d}/144  {r['verdict']}")


if __name__ == "__main__":
    main()
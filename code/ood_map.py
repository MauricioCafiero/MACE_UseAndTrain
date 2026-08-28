"""Per-atom OOD map: render an XYZ molecule or trajectory with each atom colored
by how unusual its MACE latent representation is, highlighting the unusual
chemistry in an image.

    python ood_map.py ../data/rot1_sampled_10.xyz            # all frames
    python ood_map.py mol.xyz --frame 3                      # one frame only
    python ood_map.py mol.xyz --model off-medium             # pool must match

Companion to trust.py (molecule-level verdict) and activation_viz.py (2D RDKit
coloring, needs a SMILES). This one works directly on geometries: per-atom
latent cosine distance to the training-distribution pool, drawn on the 3D
structure. The color scale is FIXED (0-0.5 cosine distance) so colors mean the
same thing in every frame and every molecule; the 0.25 trust threshold from
trust.py is marked on the colorbar.

Caveat carried over from the study: this localizes UNUSUAL chemistry, not
unreliable energies -- novel-but-reliable atoms flag identically to
novel-and-wrong ones. The verdict still comes from the molecule mean (trust.py).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mace_calc as mc
from activation_ood import ReferencePool, atom_ood_scores
from ase.data import chemical_symbols, covalent_radii
from ase.neighborlist import natural_cutoffs, neighbor_list
from trust import POOLS
from trust_frames import read_frames

CMAP = "plasma"
VMAX = 0.5      # fixed colour ceiling (cosine distance); S66 in-dist. max is 0.24
TRUST = 0.25


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotation matrix aligning centered P onto centered Q (for the grid view)."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _bond_pairs(atoms) -> np.ndarray:
    """Covalent bonds as an (n, 2) index array, inferred from covalent radii."""
    i, j, d = neighbor_list("ijd", atoms, natural_cutoffs(atoms, mult=1.1))
    keep = i < j                      # neighbor_list emits both directions
    i, j, d = i[keep], j[keep], d[keep]
    cut = (np.array(natural_cutoffs(atoms))[i]
           + np.array(natural_cutoffs(atoms))[j]) * 1.15
    return np.column_stack([i[j < cut], j[j < cut]])


def _draw_panel(ax, atoms, dist, els, radii_scale=45.0, view=None):
    """One 3D panel: neutral-gray ball-and-stick, with a colored halo on every
    atom whose latent OOD distance exceeds the in-distribution ceiling (0.25).
    Halo color = OOD distance on the fixed 0-0.5 plasma scale; atoms inside
    the distribution get no halo, so unusual chemistry is the only thing that
    draws the eye. ``view`` = (elev, azim); None keeps matplotlib's default
    camera (elev 30, azim -60) on the raw XYZ orientation."""
    pos = atoms.positions - atoms.positions.mean(axis=0)
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    # marker/line sizes are absolute points, but the molecule is scaled to its
    # axes box -- rescale by axes width so a big panel and a small grid cell
    # render the same picture, proportionally (32 ~ a 3.2-in grid panel).
    fig = ax.figure
    w_in = ax.get_position().width * fig.get_size_inches()[0]
    f = max(w_in, 0.1) / 3.2
    radii_scale = radii_scale * f

    for i, j in _bond_pairs(atoms):
        ax.plot(*zip(pos[i], pos[j]), color="0.82", lw=1.0,
                solid_capstyle="round", zorder=1)

    z = atoms.get_atomic_numbers()
    size = (covalent_radii[z] * radii_scale) ** 2
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], facecolors="0.80",
               s=size, depthshade=False, edgecolors="0.35",
               linewidths=0.4, zorder=2)

    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(0.0, VMAX)
    hi = (dist > TRUST) & ~np.isnan(dist)
    if hi.any():
        cols = plt.colormaps[CMAP](norm(np.clip(dist[hi], 0.0, VMAX)))
        ax.scatter(pos[hi, 0], pos[hi, 1], pos[hi, 2], facecolors="none",
                   edgecolors=cols, s=size[hi] * 3.2, depthshade=False,
                   linewidths=2.2 * f, zorder=3)

    ax.set_axis_off()
    ax.set_box_aspect([max(np.ptp(pos[:, k]), 1.0) for k in range(3)])


def _ood_sm():
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(0.0, VMAX),
                               cmap=plt.colormaps[CMAP])
    sm.set_array([])
    return sm


def _colorbar(fig, sm, ax, label="latent OOD distance (cosine)"):
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, fontsize=9)
    cb.ax.axhline(TRUST, color="0.2", lw=1.2, ls="--")
    cb.ax.text(1.3, TRUST, "halos start\nhere", fontsize=7, va="center",
               transform=cb.ax.get_yaxis_transform())
    cb.ax.tick_params(labelsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xyz")
    p.add_argument("--model", default="off-large", choices=sorted(POOLS))
    p.add_argument("--frame", default=None,
                   help="single frame index (default: all frames)")
    p.add_argument("--out-dir", default="viz")
    p.add_argument("--view", default=None,
                   help="camera as 'ELEV,AZIM' (default: matplotlib's 30,-60 "
                        "on the raw XYZ orientation)")
    args = p.parse_args()
    view = tuple(float(v) for v in args.view.split(",")) if args.view else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.xyz)
    if args.frame is not None:
        frames = [frames[int(args.frame)]]
    print(f"{len(frames)} frame(s), {len(frames[0])} atoms "
          f"({frames[0].get_chemical_formula()})")

    pool = ReferencePool.load(POOLS[args.model])
    per_frame = []          # (k, energy, mean_ood, verdict, atoms, d, els, flags)

    # --- scoring cache: geometry+model+pool fully determine the numbers, so
    #     hash them; redraws (colour/alpha/radii/threshold) then skip MACE ---
    pool_stat = POOLS[args.model].stat().st_mtime_ns
    key = hashlib.sha1(
        Path(args.xyz).read_bytes()
        + f"|{args.model}|{POOLS[args.model].name}|{pool_stat}".encode()
    ).hexdigest()
    cache_file = out_dir / f".ood_cache_{args.model}.npz"
    cached = None
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        keys = z["keys"].tolist()
        if key in keys:
            i = keys.index(key)
            cached = (z[f"e{i}"], z[f"d{i}"])
            print(f"(cache hit {key[:12]}: skipping MACE scoring)")

    if cached is None:
        shared_calc = None
        energies, dists = [], []
        for k, atoms in enumerate(frames):
            if shared_calc is None:
                mc.attach(atoms, model=args.model)
                shared_calc = atoms.calc
            else:
                atoms.calc = shared_calc
            out = mc.singlepoint(atoms, model=args.model)
            ood = atom_ood_scores(atoms, pool, model=args.model)
            energies.append(float(out["energy"]))
            dists.append(ood["distances"])
        cached = (np.array(energies), np.array(dists))
        z = dict(np.load(cache_file, allow_pickle=True)) if cache_file.exists() else {}
        keys = [str(k) for k in z.pop("keys", [])]
        i = len(keys)
        z["keys"] = keys + [key]
        z[f"e{i}"], z[f"d{i}"] = cached[0], cached[1]
        np.savez(cache_file, **z)

    for k, atoms in enumerate(frames):
        e, d = float(cached[0][k]), cached[1][k]
        els = [chemical_symbols[int(zz)] for zz in atoms.get_atomic_numbers()]
        flags = (d > TRUST) & ~np.isnan(d)        # "unusual" atoms (the halos)
        mean_ood = float(np.nanmean(d))
        verdict = "TRUST" if mean_ood <= TRUST else "VERIFY"
        per_frame.append((k, e, mean_ood, verdict, atoms, d, els, flags))

    # --- one full image per frame -------------------------------------------
    for k, e, mean_ood, verdict, atoms, d, els, flags in per_frame:
        fig = plt.figure(figsize=(7.6, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        sc = _draw_panel(ax, atoms, d, els, view=view)
        order = np.argsort(-np.where(np.isnan(d), -np.inf, d))[:5]
        lst = "   ".join(f"#{i} {els[i]} {d[i]:.2f}" for i in order)
        ax.text2D(0.01, 0.99,
                  f"unusual: {int(flags.sum())}/{len(d)}   most-unusual:  {lst}",
                  transform=ax.transAxes, fontsize=8, va="top", ha="left",
                  color="0.15")
        ax.set_title(f"frame {k}   E = {e:.4f} eV   mean OOD = {mean_ood:.3f}   "
                     f"[{verdict}]\n{atoms.get_chemical_formula()}", fontsize=10,
                     pad=16)
        _colorbar(fig, _ood_sm(), ax)
        out_png = out_dir / f"frame{k:02d}_ood.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_png}")

    # --- summary grid: all frames aligned to frame 0, one shared colorbar ----
    if len(per_frame) > 1:
        ref = per_frame[0][4].positions - per_frame[0][4].positions.mean(axis=0)
        n = len(per_frame)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        fig = plt.figure(figsize=(3.4 * ncols, 3.6 * nrows))
        for r, (k, e, mean_ood, verdict, atoms, d, els, flags) in enumerate(per_frame):
            pos = atoms.positions - atoms.positions.mean(axis=0)
            if r > 0:
                pos = pos @ _kabsch(pos, ref)      # rotate onto frame 0
            atoms_tmp = atoms.copy()
            atoms_tmp.positions = pos
            ax = fig.add_subplot(nrows, ncols, r + 1, projection="3d")
            _draw_panel(ax, atoms_tmp, d, els, view=view)
            ax.set_title(f"f{k}  mean {mean_ood:.3f} [{verdict}]  "
                         f"{int(flags.sum())} unusual", fontsize=9)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(_ood_sm(), cax=cax)
        cb.set_label("latent OOD distance (cosine)", fontsize=9)
        cb.ax.axhline(TRUST, color="0.2", lw=1.2, ls="--")
        cb.ax.tick_params(labelsize=8)
        grid_png = out_dir / "summary_grid.png"
        fig.savefig(grid_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {grid_png}")

    print("\n=== summary ===")
    print(f"{'frame':>5s}  {'energy (eV)':>12s}  {'mean OOD':>9s}  "
          f"{'unusual':>8s}  verdict")
    for k, e, mean_ood, verdict, atoms, d, els, flags in per_frame:
        print(f"{k:5d}  {e:12.4f}  {mean_ood:9.3f}  "
              f"{int(flags.sum()):4d}/{len(d):<3d}  {verdict}")


if __name__ == "__main__":
    main()
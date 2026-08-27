"""Visualize MACE layer activations and OOD scores on an RDKit molecule image.

Maps per-atom and per-bond quantities onto a 2D depiction of the molecule so you
can *see* which parts of the structure correspond to which neurons/layers firing
(raw activation magnitude) and which parts are out-of-distribution for MACE-OFF23
(OOD score from :mod:`activation_ood`).

Atom mapping is exact: ``mace_calc.smiles_to_atoms`` builds ASE atoms in the same
order as RDKit's heavy atoms (AddHs appends Hs at the end, preserving heavy-atom
indices), so heavy-atom ASE index ``i`` == RDKit atom index ``i``. Bonds are
mapped back to MACE's per-edge radial-MLP output via :func:`inspect_activations
.mace_edge_index`.

Produces two side-by-side images per molecule: left colored by **raw activation
magnitude** (``interactions.1.linear`` per-atom norm, ``interactions.0.conv_tp_
weights`` per-bond norm), right colored by **OOD score** (atom / bond latent
distance to the OFF23 reference pool). A combined PNG and the two individuals
are written.

Example
-------
>>> from activation_viz import draw_comparison
>>> import mace_calc as mc
>>> from activation_ood import ReferencePool
>>> pool = ReferencePool.load_or_build()
>>> a = mc.smiles_to_atoms("CC(=O)Nc1ccc(cc1)O")
>>> draw_comparison(a, "CC(=O)Nc1ccc(cc1)O", pool, out_name="paracetamol")
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch import). ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import logging
from pathlib import Path

import numpy as np

import mace_calc as mc  # noqa: E402
from inspect_activations import capture_activations, mace_edge_index  # noqa: E402
from activation_ood import (  # noqa: E402
    ATOM_LAYER, PAIR_LAYER, _POOL_PATH, ReferencePool, atom_ood_scores,
    pair_ood_scores,
)

log = logging.getLogger("activation_viz")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Scalar extraction (raw activation magnitude + OOD)
# ---------------------------------------------------------------------------
def _ensure_calc(atoms, model="off-medium", dtype="float64"):
    if not (hasattr(atoms, "calc") and atoms.calc is not None and hasattr(atoms.calc, "models")):
        mc.attach(atoms, model=model, dtype=dtype)


def raw_atom_norms(atoms, model="off-medium", dtype="float64",
                   layer=ATOM_LAYER) -> np.ndarray:
    """Per-atom L2 norm of the ``layer`` activation (how hard each atom fires)."""
    _ensure_calc(atoms, model, dtype)
    acts = capture_activations(atoms, layers=[layer], model=model, dtype=dtype)
    v = acts[layer].numpy()
    return np.linalg.norm(v, axis=-1)


def raw_bond_norms(atoms, model="off-medium", dtype="float64",
                  pair_layer=PAIR_LAYER) -> dict:
    """Per-covalent-bond L2 norm of the radial-MLP activation -> {frozenset(i,j): norm}."""
    from rdkit import Chem

    _ensure_calc(atoms, model, dtype)
    # bond norms need a SMILES -> heavy mol; derive from atoms.info if present
    smi = atoms.info.get("smiles")
    if smi is None:
        raise ValueError("atoms has no 'smiles' info; pass smi to draw_comparison instead.")
    acts = capture_activations(atoms, layers=[pair_layer], model=model, dtype=dtype)
    pv = acts[pair_layer].numpy()
    ei = mace_edge_index(atoms, model=model, dtype=dtype)
    edge_map = {frozenset((int(ei[0, e]), int(ei[1, e]))): e for e in range(ei.shape[1])}
    mol = Chem.MolFromSmiles(smi)
    out = {}
    for b in mol.GetBonds():
        key = frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        e = edge_map.get(key)
        if e is not None:
            out[key] = float(np.linalg.norm(pv[e]))
    return out


def _ood_atom_scalar(atoms, pool, model="off-medium", dtype="float64") -> np.ndarray:
    return atom_ood_scores(atoms, pool, model=model, dtype=dtype)["distances"]


def _ood_bond_scalar(atoms, smi, pool, model="off-medium", dtype="float64") -> dict:
    p = pair_ood_scores(atoms, smi, pool, model=model, dtype=dtype)
    return {frozenset(b): float(d) for b, d in zip(p["bonds"], p["distances"])}


# ---------------------------------------------------------------------------
# Fixed colour-scale calibration (so images are comparable across molecules)
# ---------------------------------------------------------------------------
# Both molecule images are drawn on a fixed [0, 1] scale so a colour means the
# same thing on every molecule -- OOD is a cosine distance (bounded 0..1), and
# activation magnitude is shown as a fraction of a calibrated ceiling (the p99
# of raw activation norms over the OFF23 pool), so "1.0" = "fires as hard as the
# typical training-distribution maximum".
_ACT_SCALE_PATH = _POOL_PATH.parent / "act_scale.npz"


def calibrate_activation_scale(n_frames=80, max_atoms=100, model="off-medium",
                               dtype="float64", atom_layer=ATOM_LAYER,
                               pair_layer=PAIR_LAYER, out=_ACT_SCALE_PATH,
                               seed=0) -> dict:
    """Sample ``n_frames`` from the OFF23 test split and store the p99 of raw
    per-atom and per-pair activation norms as fixed colour-scale ceilings.

    One-time; cached to ``data/act_scale.npz``. Cheap (norms only, no vectors
    kept). Returns ``{"act_atom_vmax", "act_bond_vmax"}``.
    """
    import random
    from activation_ood import _iter_frames, download_off23_test_data

    xyz_dir = download_off23_test_data()
    calc = mc.get_calculator(model=model, dtype=dtype)
    rng = random.Random(seed)

    reservoir, seen = [], 0
    for frame in _iter_frames(xyz_dir):
        seen += 1
        if len(frame) > max_atoms:
            continue
        if len(reservoir) < n_frames:
            reservoir.append(frame)
        else:
            j = rng.randint(0, seen - 1)
            if j < n_frames:
                reservoir[j] = frame

    anorms, bnorms = [], []
    for frame in reservoir:
        frame.calc = calc
        try:
            acts = capture_activations(frame, layers=[atom_layer, pair_layer],
                                       model=model, dtype=dtype)
        except Exception as exc:
            log.warning("scale: a frame failed (%s); skipping", exc)
            continue
        anorms.append(np.linalg.norm(acts[atom_layer].numpy(), axis=1))
        bnorms.append(np.linalg.norm(acts[pair_layer].numpy(), axis=1))

    anorms = np.concatenate(anorms) if anorms else np.array([1.0])
    bnorms = np.concatenate(bnorms) if bnorms else np.array([1.0])
    avmax = float(np.percentile(anorms, 99)) or 1.0
    bvmax = float(np.percentile(bnorms, 99)) or 1.0

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, act_atom_vmax=avmax, act_bond_vmax=bvmax,
             n_frames=len(reservoir), n_atoms=len(anorms), n_bonds=len(bnorms))
    log.info("calibrated activation scale: atom p99=%.3f, bond p99=%.3f "
             "(%d atoms / %d pairs from %d frames)", avmax, bvmax,
             len(anorms), len(bnorms), len(reservoir))
    return {"act_atom_vmax": avmax, "act_bond_vmax": bvmax}


def get_activation_scale(model="off-medium", dtype="float64") -> dict:
    """Load the cached activation colour-scale, or calibrate it once on demand."""
    if _ACT_SCALE_PATH.exists():
        d = np.load(_ACT_SCALE_PATH, allow_pickle=True)
        return {"act_atom_vmax": float(d["act_atom_vmax"]),
                "act_bond_vmax": float(d["act_bond_vmax"])}
    return calibrate_activation_scale(model=model, dtype=dtype)


# ---------------------------------------------------------------------------
# RDKit drawing
# ---------------------------------------------------------------------------
def _robust_limits(values, lo=2, hi=98):
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def _draw_mol_png(smi, atom_values, bond_values, title, cmap_name="plasma",
                  size=(520, 420), vmin=None, vmax=None):
    """Render one colored molecule -> PNG bytes."""
    import matplotlib
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
    AllChem.Compute2DCoords(mol)

    # accept either a dict {atom_id: value} or a full-ASE ndarray (indexed by id)
    if isinstance(atom_values, np.ndarray):
        atom_values = {i: float(atom_values[i]) for i in range(len(atom_values))}

    atom_ids = [a.GetIdx() for a in mol.GetAtoms()]
    avals = np.array([atom_values.get(i, np.nan) for i in atom_ids], dtype=float)
    if vmin is None or vmax is None:
        vmin2, vmax2 = _robust_limits(avals)
        vmin = vmin if vmin is not None else vmin2
        vmax = vmax if vmax is not None else vmax2
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    highlight_atom_colors = {}
    highlight_atom_radii = {}
    for i in atom_ids:
        v = atom_values.get(i)
        if v is None or np.isnan(v):
            continue
        highlight_atom_colors[i] = tuple(cmap(norm(v))[:3])
        highlight_atom_radii[i] = 0.18 + 0.30 * float(min(max((v - vmin) / (vmax - vmin), 0), 1))

    highlight_bond_ids, highlight_bond_colors = [], {}
    for b in mol.GetBonds():
        key = frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        v = bond_values.get(key)
        if v is None or np.isnan(v):
            continue
        highlight_bond_ids.append(b.GetIdx())
        highlight_bond_colors[b.GetIdx()] = tuple(cmap(norm(v))[:3])

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(
        mol, highlightAtoms=atom_ids,
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii=highlight_atom_radii,
        highlightBonds=highlight_bond_ids,
        highlightBondColors=highlight_bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText(), (vmin, vmax)


def _annotate_with_colorbar(png_bytes, vmin, vmax, cmap_name, label, out_png, title):
    """Place the RDKit PNG on a matplotlib figure with a colorbar; save PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    img = imread(io.BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=matplotlib.colormaps[cmap_name])
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def draw_molecule(atoms, smi, atom_values, bond_values, out_png, title="",
                  cmap="plasma", vmin=None, vmax=None):
    """Color the molecule by per-atom (array over ASE atoms) & per-bond (dict)
    scalars and save a PNG with a colorbar."""
    # atom_values may be a full-ASE array (incl. H); index by heavy-atom id == ASE id
    if isinstance(atom_values, np.ndarray):
        atom_values = {i: float(atom_values[i]) for i in range(len(atom_values))}
    png, (vmin, vmax) = _draw_mol_png(smi, atom_values, bond_values, title or out_png,
                                      cmap_name=cmap, vmin=vmin, vmax=vmax)
    _annotate_with_colorbar(png, vmin, vmax, cmap, title or "value", out_png, title)
    log.info("wrote %s", out_png)
    return out_png


def draw_comparison(atoms, smi, pool: ReferencePool, out_name=None,
                    out_dir=".", model="off-medium", dtype="float64",
                    cmap="plasma"):
    """Render the activation image + the OOD image (side by side) for a molecule.

    Writes ``<name>_activation.png``, ``<name>_ood.png`` and a combined
    ``<name>_comparison.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    atoms.info["smiles"] = smi  # for raw_bond_norms
    _ensure_calc(atoms, model, dtype)

    # --- raw activation magnitudes, put on a fixed [0,1] "firing intensity"
    #     scale (fraction of the calibrated OFF23 p99) so the colourbar is
    #     comparable across molecules and atoms/bonds share one scale. ---
    scale = get_activation_scale(model=model, dtype=dtype)
    a_atom = np.clip(raw_atom_norms(atoms, model=model, dtype=dtype)
                     / scale["act_atom_vmax"], 0.0, 1.0)
    a_bond = {k: float(np.clip(v / scale["act_bond_vmax"], 0.0, 1.0))
              for k, v in raw_bond_norms(atoms, model=model, dtype=dtype).items()}
    # --- OOD scores: cosine distance is already bounded [0,1] ---
    o_atom = np.clip(_ood_atom_scalar(atoms, pool, model=model, dtype=dtype), 0.0, 1.0)
    o_bond = {k: float(np.clip(v, 0.0, 1.0))
              for k, v in _ood_bond_scalar(atoms, smi, pool, model=model, dtype=dtype).items()}

    name = out_name or atoms.get_chemical_formula()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # both images share the SAME fixed [0,1] scale across every molecule
    a_png, _ = _draw_mol_png(
        smi, a_atom, a_bond, "activation (firing intensity)",
        cmap_name=cmap, vmin=0.0, vmax=1.0)
    o_png, _ = _draw_mol_png(
        smi, o_atom, o_bond, "OOD score (cosine distance)",
        cmap_name=cmap, vmin=0.0, vmax=1.0)

    # save individuals with colorbars (fixed 0..1)
    _annotate_with_colorbar(a_png, 0.0, 1.0, cmap, "firing intensity",
                            out_dir / f"{name}_activation.png", "activation (fixed scale)")
    _annotate_with_colorbar(o_png, 0.0, 1.0, cmap, "OOD distance",
                            out_dir / f"{name}_ood.png", "OOD score (fixed scale)")

    # combined side-by-side with per-panel colorbars
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
    axL.imshow(imread(io.BytesIO(a_png))); axL.set_title("activation (firing intensity)", fontsize=12); axL.axis("off")
    axR.imshow(imread(io.BytesIO(o_png))); axR.set_title("OOD score (cosine distance)", fontsize=12); axR.axis("off")
    cb1 = fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(0.0, 1.0),
                      matplotlib.colormaps[cmap]), ax=axL, fraction=0.046, pad=0.04)
    cb1.set_label("firing intensity (frac. of calib. p99)")
    cb2 = fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(0.0, 1.0),
                      matplotlib.colormaps[cmap]), ax=axR, fraction=0.046, pad=0.04)
    cb2.set_label("OOD distance (cosine)")
    fig.suptitle(f"{name}  ({smi})", fontsize=13)
    fig.tight_layout()
    combined = out_dir / f"{name}_comparison.png"
    fig.savefig(combined, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", combined)
    print(f"images: {out_dir}/{name}_activation.png, {out_dir}/{name}_ood.png, {combined}")
    return {"activation": str(out_dir / f"{name}_activation.png"),
            "ood": str(out_dir / f"{name}_ood.png"),
            "comparison": str(combined),
            "atom_ood": o_atom, "bond_ood": o_bond,
            "atom_activation": a_atom, "bond_activation": a_bond}


# ---------------------------------------------------------------------------
# Network-firing diagram: per-layer neuron x (atom|bond) heatmaps
# ---------------------------------------------------------------------------
# The flat, neuron-like layers we lay out as rows of the diagram. Order matters:
# this is the forward direction (embedding -> interactions -> readout), so the
# stack reads top-to-bottom like the network itself.
DIAGRAM_ATOM_LAYERS = [
    "node_embedding.linear",
    "interactions.0.linear",
    "interactions.1.linear",
    "readouts.1.linear_1",
    "readouts.1.non_linearity",
]


def _bin_columns(M, n_bins):
    """Average columns of a (R, D) matrix down to ~n_bins for readable heatmaps.

    Returns ``(matrix, n_cols_shown)``. Narrow layers (D <= 2*n_bins) are left at
    full resolution.
    """
    D = M.shape[1]
    if D <= n_bins * 2:
        return M, D
    parts = np.array_split(M, n_bins, axis=1)
    return np.stack([p.mean(axis=1) for p in parts], axis=1), n_bins


def _bond_rows(atoms, smi, pair_acts, model="off-medium", dtype="float64"):
    """Covalent-bond activations -> (labels, matrix) for the radial-MLP panel.

    Each RDKit heavy-atom bond is mapped to its MACE edge (via mace_edge_index)
    and its per-pair activation vector pulled out. ``labels`` are element-pair
    strings (e.g. ``"C-O"``) so the rows are chemically readable.
    """
    from ase.data import chemical_symbols
    from rdkit import Chem

    ei = mace_edge_index(atoms, model=model, dtype=dtype)
    edge_map = {frozenset((int(ei[0, e]), int(ei[1, e]))): e for e in range(ei.shape[1])}
    pv = pair_acts.numpy()
    z = atoms.get_atomic_numbers()
    mol = Chem.MolFromSmiles(smi)
    labels, rows = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        e = edge_map.get(frozenset((i, j)))
        if e is None:
            continue
        pair = "-".join(sorted((chemical_symbols[int(z[i])], chemical_symbols[int(z[j])])))
        labels.append(f"{pair} {i}-{j}")
        rows.append(pv[e])
    if not rows:
        return [], np.empty((0, 0))
    return labels, np.stack(rows)


def draw_neuron_firing(atoms, smi, out_png, layers=DIAGRAM_ATOM_LAYERS,
                      pair_layer=PAIR_LAYER, neuron_bins=128, cmap="magma",
                      model="off-medium", dtype="float64", title=None):
    """Draw a per-layer neuron-firing diagram for one molecule.

    Produces a vertical stack of heatmaps -- one per layer -- with **rows = atoms**
    (or covalent **bonds** for the radial MLP) and **columns = neurons** (binned
    when a layer is wide), colored by |activation|. This is the complement to the
    RDKit molecule coloring: instead of asking "which *atoms* fire," it shows
    "which *neurons* fire, and for which atoms," across the network depth.

    A single forward pass captures every layer at once. Each panel is normalized
    independently (layers operate at very different scales), so read a panel's
    pattern, not its absolute brightness across panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from ase.data import chemical_symbols

    atoms.info["smiles"] = smi
    _ensure_calc(atoms, model, dtype)
    want = list(layers) + ([pair_layer] if (pair_layer and smi) else [])
    acts = capture_activations(atoms, layers=want, model=model, dtype=dtype)

    z = atoms.get_atomic_numbers()
    atom_labels = [chemical_symbols[int(zz)] for zz in z]

    panels = []  # (label, matrix, row_labels, ncols, D)
    for L in layers:
        if L not in acts:
            log.warning("no activation captured for %s; skipping", L)
            continue
        M = np.abs(acts[L].numpy())          # (N, D)
        Mb, ncols = _bin_columns(M, neuron_bins)
        panels.append((L, Mb, atom_labels, ncols, M.shape[1]))

    # pair panel needs a SMILES to map covalent bonds -> skip for complexes (smi=None)
    if pair_layer and smi and pair_layer in acts:
        blabels, bM = _bond_rows(atoms, smi, acts[pair_layer], model=model, dtype=dtype)
        if bM.size:
            Mb, ncols = _bin_columns(bM, neuron_bins)
            panels.append((pair_layer, Mb, blabels, ncols, bM.shape[1]))

    if not panels:
        raise ValueError("no layers captured; check layer names vs the model.")

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n + 1.0), squeeze=False)
    cmap_obj = matplotlib.colormaps[cmap]
    for ax, (L, M, rlabels, ncols, D) in zip(axes[:, 0], panels):
        vmax = float(np.percentile(M, 99)) if M.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap_obj,
                       norm=Normalize(0.0, vmax))
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(rlabels, fontsize=max(4, min(9, 120 // max(1, M.shape[0]))))
        ax.set_ylabel(f"{M.shape[0]} rows", fontsize=8)
        binned = f", binned {D}->{ncols}" if ncols != D else ""
        ax.set_title(f"{L}   [{M.shape[0]} x {D}{binned}]", fontsize=9, loc="left")
        ax.set_xlabel("neuron (channel)" if ncols == D else "neuron bin", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
        cb.set_label("|act|", fontsize=7)
        cb.ax.tick_params(labelsize=7)

    head = title or f"{atoms.get_chemical_formula()}  ({smi})"
    fig.suptitle(f"{head}  --  E = {atoms.get_potential_energy():.4f} eV",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_png)
    print(f"diagram: {out_png}")
    return str(out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_atoms(arg: str):
    from ase.build import molecule as _molecule
    try:
        return _molecule(arg)
    except (KeyError, ValueError):
        return mc.smiles_to_atoms(arg)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Draw a molecule colored by MACE activation magnitude and "
                    "OOD score (side by side).",
    )
    p.add_argument("molecule", help="SMILES (or ASE molecule name)")
    p.add_argument("--out", default=None, help="output name (default: formula)")
    p.add_argument("--out-dir", default=".", help="output directory")
    p.add_argument("--model", default="off-medium")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--cmap", default="plasma")
    p.add_argument("--build-pool", action="store_true",
                   help="(re)build the OOD reference pool first")
    p.add_argument("--diagram", action="store_true",
                   help="draw a per-layer neuron-firing diagram instead of the "
                        "RDKit molecule coloring (no pool needed)")
    p.add_argument("--neuron-bins", type=int, default=128,
                   help="with --diagram, bin wide layers to this many columns")
    p.add_argument("--recalibrate-scale", action="store_true",
                   help="(re)compute the fixed activation colour-scale ceiling "
                        "from the OFF23 pool (data/act_scale.npz)")
    args = p.parse_args(argv)

    if args.recalibrate_scale:
        calibrate_activation_scale(model=args.model, dtype=args.dtype)

    atoms = _build_atoms(args.molecule)
    smi = args.molecule if not _looks_like_ase_name(args.molecule) else None

    if args.diagram:
        if smi is None:
            p.error("--diagram needs a SMILES (to map the bond panel); pass one.")
        name = args.out or atoms.get_chemical_formula()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        draw_neuron_firing(
            atoms, smi, out_dir / f"{name}_firing.png",
            neuron_bins=args.neuron_bins, cmap=args.cmap,
            model=args.model, dtype=args.dtype,
        )
        return

    if smi is None:
        p.error("OOD/bond coloring needs a SMILES; pass a SMILES string.")

    if args.build_pool or not _POOL_PATH.exists():
        pool = ReferencePool.build(model=args.model, dtype=args.dtype)
    else:
        pool = ReferencePool.load(_POOL_PATH)

    draw_comparison(atoms, smi, pool, out_name=args.out, out_dir=args.out_dir,
                    model=args.model, dtype=args.dtype, cmap=args.cmap)


def _looks_like_ase_name(arg: str) -> bool:
    from ase.build import molecule as _m
    try:
        _m(arg)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
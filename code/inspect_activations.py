"""Inspect MACE model activations during a single-point calculation.

MACE foundation models are ordinary ``torch.nn.Module`` instances, so any
layer's output can be captured with PyTorch forward hooks. This module wraps
that plumbing so you can ask "which layers/neurons fire for this molecule"
without re-deriving the hook bookkeeping each time.

The torch network lives at ``atoms.calc.models[0]`` (a ``ScaleShiftMACE``) once
a MACE calculator is attached via :mod:`mace_calc`.

Note
----
MACE is equivariant (built on e3nn): most layers output irreducible tensors
(angular-momentum channels), not flat neuron vectors. The flat, neuron-like
activations are the invariant Linears/MLPs -- the node embedding, the per-atom
mixing ``interactions.*.linear``, the radial/bond MLP
``interactions.*.conv_tp_weights``, and the readout head. ``DEFAULT_PATTERNS``
targets exactly those; use :func:`list_layers` to discover the rest.

Example
-------
>>> import mace_calc as mc
>>> from inspect_activations import capture_activations, summarize
>>> from ase.build import molecule
>>> acts = capture_activations(molecule("H2O"), model="off-medium")
>>> summarize(acts)
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch import; see
#     mace_calc.py for rationale). mace_calc import sets these too, but we set
#     them explicitly so this file is safe to import on its own.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fnmatch
import logging

import numpy as np

import mace_calc as mc  # noqa: E402  (sets the OMP guards, loads mace/torch)

import torch  # noqa: E402

log = logging.getLogger("inspect_activations")

# Flat, invariant layers present in every OFF23/OMOL MACE model. These are the
# ones whose output is a plain ``(N, channels)`` tensor (per-atom) or
# ``(N_pairs, channels)`` (radial MLP) -- i.e. genuine "neuron activations".
DEFAULT_PATTERNS: list[str] = [
    "node_embedding.linear",
    "interactions.*.linear",
    "interactions.*.conv_tp_weights",   # radial / bond MLP (per-pair)
    "readouts.*.linear_1",
    "readouts.*.non_linearity",
    "readouts.*.linear_2",
]


def _has_mace_calc(atoms) -> bool:
    return atoms.calc is not None and hasattr(atoms.calc, "models")


def get_model(atoms):
    """Return the underlying torch ``nn.Module`` (``ScaleShiftMACE``) backing ``atoms``.

    The calculator must already be attached (see :func:`capture_activations`,
    which attaches one if missing).
    """
    if not _has_mace_calc(atoms):
        raise ValueError(
            "atoms has no MACE calculator; pass it through capture_activations "
            "(which attaches one) or call mace_calc.attach first."
        )
    model = atoms.calc.models[0]
    # torch.compile (used when compile_mode is set) wraps the module; unwrap so
    # named_modules() reaches the real submodules.
    return getattr(model, "_orig_mod", model)


def list_layers(atoms, max_depth: int = 2, params_only: bool = False) -> None:
    """Print the named module hierarchy of the model backing ``atoms``.

    Useful to discover exact layer names before passing them to
    :func:`capture_activations`. ``max_depth`` limits the dotted-name depth
    shown; ``params_only`` hides submodules with no own parameters.
    """
    if not _has_mace_calc(atoms):
        mc.attach(atoms, model="auto", dtype="float64")
    net = get_model(atoms)
    for name, mod in net.named_modules():
        if not name or name.count(".") > max_depth:
            continue
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        if params_only and n == 0:
            continue
        print(f"{name:46s} {type(mod).__name__:22s} params={n}")


def _resolve_names(model, layers) -> list[str]:
    """Expand fnmatch patterns / exact names against the model's module names."""
    all_names = sorted(n for n, _ in model.named_modules())
    out: list[str] = []
    for pat in layers:
        hits = fnmatch.filter(all_names, pat)
        if hits:
            out.extend(hits)
        elif pat in all_names:
            out.append(pat)
        else:
            log.warning("no layer matched %r (run list_layers() to see names)", pat)
    # dedupe, preserve order
    seen: set[str] = set()
    return [n for n in out if not (n in seen or seen.add(n))]


def capture_activations(
    atoms,
    layers=DEFAULT_PATTERNS,
    model: str = "auto",
    dtype: str = "float64",
) -> dict:
    """Run one single-point forward and return ``{layer_name: tensor}``.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure. A MACE calculator is attached if not already present.
    layers : list[str]
        Module names or fnmatch patterns (e.g. ``"interactions.*.linear"``).
        Defaults to :data:`DEFAULT_PATTERNS` (the flat invariant layers).
    model : str
        MACE alias (``"auto"``, ``"off-medium"``, ``"omol"`` ...) used only if a
        calculator must be attached.
    dtype : str
        ``"float64"`` (default, recommended) or ``"float32"``.

    Returns
    -------
    dict
        ``{layer_name: torch.Tensor}`` of detached, CPU-resident activations.
        Only the energy forward is run, so each layer appears once.
    """
    if not _has_mace_calc(atoms):
        mc.attach(atoms, model=model, dtype=dtype)
    net = get_model(atoms)
    names = set(_resolve_names(net, layers))

    acts: dict = {}
    handles = []

    def hook(name):
        def _h(_mod, _inp, out):
            acts[name] = out.detach().cpu() if torch.is_tensor(out) else out
        return _h

    for name, mod in net.named_modules():
        if name in names:
            handles.append(mod.register_forward_hook(hook(name)))
    if not handles:
        raise ValueError(
            f"no hooks registered -- none of {layers} matched. "
            f"Run list_layers(atoms) to see available names."
        )

    # A single energy forward fires every hook once. (Forces would re-trigger;
    # we skip them so each layer is captured exactly once.) ASE caches results,
    # so reset first -- otherwise a second capture call (e.g. a different layer
    # on the same atoms) returns the cached energy and the hooks never fire.
    calc = atoms.calc
    if hasattr(calc, "reset"):
        calc.reset()
    atoms.get_potential_energy()
    for h in handles:
        h.remove()
    return acts


def summarize(acts: dict) -> list[tuple]:
    """Print a shape + magnitude summary of captured activations.

    For tensors with >2 dims the trailing axes are flattened so the stats are
    per-leading-index (per atom / per pair). Returns the rows for programmatic
    use.
    """
    rows = []
    print(f"{'layer':34s} {'shape':20s} {'mean|.|':10s} {'max|.|':10s}")
    for name, t in acts.items():
        if not torch.is_tensor(t):
            print(f"{name:34s} (non-tensor output: {type(t).__name__})")
            continue
        flat = t if t.dim() <= 2 else t.reshape(t.shape[0], -1)
        mn, mx = flat.abs().mean().item(), flat.abs().max().item()
        rows.append((name, tuple(t.shape), mn, mx))
        print(f"{name:34s} {str(tuple(t.shape)):20s} {mn:10.3e} {mx:10.3e}")
    return rows


# ---------------------------------------------------------------------------
# Per-neighbor-pair breakdown of a radial-MLP (per-edge) layer
# ---------------------------------------------------------------------------
def mace_edge_index(atoms, model: str = "auto", dtype: str = "float64"):
    """Rebuild MACE's edge graph and return ``edge_index`` as a ``(2, E)`` int array.

    Uses the calculator's own ``r_max``/``z_table`` (via ``config_from_atoms`` +
    ``AtomicData.from_config``), so the edge ordering matches the per-edge layer
    outputs captured by :func:`capture_activations` exactly. Each column is a
    directed neighbor pair ``(sender, receiver)``; both directions are present and
    every atom pair within ``r_max`` (5.0 A for the organic models) is an edge --
    not just covalent bonds.

    A MACE calculator is attached if not already present.
    """
    import mace.data as mdata

    if not _has_mace_calc(atoms):
        mc.attach(atoms, model=model, dtype=dtype)
    calc = atoms.calc
    cfg = mdata.config_from_atoms(atoms, head_name=calc.head)
    graph = mdata.AtomicData.from_config(
        cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads,
    )
    return graph["edge_index"].cpu().numpy().astype(int)  # (2, E)


def neighbor_pair_breakdown(
    atoms,
    layer: str = "interactions.0.conv_tp_weights",
    model: str = "auto",
    dtype: str = "float64",
    top: Optional[int] = None,
) -> list[tuple]:
    """Break a radial-MLP layer's per-edge activations down by neighbor-pair type.

    MACE's graph connects every atom pair within ``model.r_max`` (5.0 A for the
    organic models), so an "edge" is a *neighbor pair*, not just a covalent bond
    -- it includes 1-2, 1-3, 1-4 and any close contacts. The radial MLP
    (``interactions.*.conv_tp_weights``) outputs one vector per such edge, and
    that output depends only on the pair distance (Bessel basis x cutoff fed
    through an MLP), so the two directed edges i->j and j->i carry identical
    activations.

    This rebuilds MACE's own edge graph (``config_from_atoms`` +
    ``AtomicData.from_config`` with the calculator's ``r_max``/``z_table``) so
    the edge ordering matches the captured activations exactly, then groups
    edges by the unordered element pair of their endpoints.

    Parameters
    ----------
    layer : str
        A per-edge layer name. Defaults to the first radial MLP. The second
        one is ``"interactions.1.conv_tp_weights"``.
    top : int, optional
        If given, print only the top-N pair types by edge count.

    Returns
    -------
    list[tuple]
        ``(pair_label, n_edges, n_pairs, mean_abs, max_abs, mean_norm)`` per
        pair type, sorted by edge count descending.
    """
    import collections
    from ase.data import chemical_symbols

    if not _has_mace_calc(atoms):
        mc.attach(atoms, model=model, dtype=dtype)
    calc = atoms.calc
    net = get_model(atoms)

    # resolve the single layer name, then rebuild the exact graph the model
    # sees (edge ordering matches the captured per-edge activations).
    names = _resolve_names(net, [layer])
    if len(names) != 1:
        raise ValueError(
            f"layer {layer!r} matched {names}; pass a single radial-MLP name "
            f"(e.g. 'interactions.0.conv_tp_weights')."
        )
    layer = names[0]

    edge_index = mace_edge_index(atoms, model=model, dtype=dtype)
    n_edges = edge_index.shape[1]

    acts = capture_activations(atoms, layers=[layer], model=model, dtype=dtype)
    t = acts[layer]
    if not torch.is_tensor(t) or t.shape[0] != n_edges:
        raise ValueError(
            f"edge/activation mismatch: graph has {n_edges} edges but "
            f"{layer} captured shape {tuple(t.shape)}. This can happen with "
            f"torch.compile padding; use a non-compiled model."
        )
    arr = t.numpy()  # (E, C)

    syms = [chemical_symbols[int(z)] for z in atoms.get_atomic_numbers()]
    groups: dict = collections.defaultdict(list)      # label -> [edge idx]
    pair_sets: dict = collections.defaultdict(set)   # label -> {frozenset(i,j)}
    for e in range(n_edges):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        label = tuple(sorted((syms[i], syms[j])))
        groups[label].append(e)
        pair_sets[label].add(frozenset((i, j)))

    rows = []
    for label, idxs in groups.items():
        sub = arr[idxs]
        rows.append((
            "-".join(label),
            len(idxs),
            len(pair_sets[label]),
            float(abs(sub).mean()),
            float(abs(sub).max()),
            float(np.linalg.norm(sub, axis=-1).mean()),
        ))
    rows.sort(key=lambda r: r[1], reverse=True)
    if top is not None:
        rows = rows[:top]

    r_max = float(calc.r_max)
    print(f"{atoms.get_chemical_formula()}  layer={layer}  cutoff={r_max:.1f} A  "
          f"edges={n_edges}  pairs={sum(len(s) for s in pair_sets.values())}\n")
    print(f"{'pair':10s} {'n_edges':>8s} {'n_pairs':>8s} {'mean|.|':>10s} "
          f"{'max|.|':>10s} {'mean||.||':>10s}")
    for label, ne, np_, mn, mx, nrm in rows:
        print(f"{label:10s} {ne:8d} {np_:8d} {mn:10.3e} {mx:10.3e} {nrm:10.3e}")
    return rows


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def _build_atoms(arg: str):
    """Interpret ``arg`` as an ASE molecule name, else as a SMILES string."""
    from ase.build import molecule as _molecule

    try:
        return _molecule(arg)
    except (KeyError, ValueError):
        return mc.smiles_to_atoms(arg)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Capture MACE layer activations for a single-point calculation.",
    )
    p.add_argument("molecule", nargs="?", default="H2O",
                   help="ASE molecule name (e.g. H2O, CH3OH) or a SMILES string")
    p.add_argument("--model", default="auto",
                   help="MACE alias (auto, off-medium, omol, ...)")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--list-layers", action="store_true",
                   help="print the model's module hierarchy and exit")
    p.add_argument("--layer", action="append", default=None,
                   help="layer name/pattern to capture (repeatable); "
                        "default = the flat invariant layers")
    p.add_argument("--by-pair", action="store_true",
                   help="break a radial-MLP layer down by neighbor-pair element "
                        "type (uses MACE's own 5.0 A edge graph)")
    p.add_argument("--pair-layer", default="interactions.0.conv_tp_weights",
                   help="radial-MLP layer to use with --by-pair "
                        "(default: interactions.0.conv_tp_weights; "
                        "also try interactions.1.conv_tp_weights)")
    p.add_argument("--top", type=int, default=None,
                   help="with --by-pair, print only the top-N pair types")
    args = p.parse_args(argv)

    atoms = _build_atoms(args.molecule)

    if args.list_layers:
        list_layers(atoms)
        return

    if args.by_pair:
        neighbor_pair_breakdown(atoms, layer=args.pair_layer,
                                 model=args.model, dtype=args.dtype, top=args.top)
        return

    acts = capture_activations(
        atoms,
        layers=args.layer or DEFAULT_PATTERNS,
        model=args.model,
        dtype=args.dtype,
    )
    print(f"{atoms.get_chemical_formula()}  E = {atoms.get_potential_energy():.6f} eV\n")
    summarize(acts)


if __name__ == "__main__":
    main()
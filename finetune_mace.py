"""Helpers to fine-tune a MACE foundation model on GFN2-xTB reference data.

This wraps the ``mace_run_train`` CLI that ships with mace-torch. It builds the
command line for a *foundation-model fine-tune* (the recommended way to adapt a
MACE foundation model to a new level of theory like GFN2-xTB) and optionally
computes isolated-atom reference energies (``E0s``) with GFN2 for a physically
clean energy baseline.

A typical fine-tune call looks like:

    mace_run_train \
        --foundation_model=/path/to/MACE-OFF23_medium.model \
        --train_file=gfn2_data_train.xyz --valid_file=gfn2_data_valid.xyz \
        --energy_key=REF_energy --forces_key=REF_forces \
        --E0s=average --model=MACE \
        --max_num_epochs=200 --ema --ema_decay=0.99 --scheduler=patience \
        --lr=1e-3 --weight_decay=1e-8 --forces_weight=1 --energy_weight=100 \
        --device=cpu --default_dtype=float64 \
        --name=gfn2_finetune --results_dir=runs --checkpoints_dir=runs

Use :func:`build_train_command` to construct this (passing ``foundation_model``
as a :mod:`mace_calc` alias such as ``"off-medium"`` resolves it to the cached
``.model`` path automatically) and :func:`run_finetune` to launch it. The model
artifact is written to ``<results_dir>/<name>_run-<i>.model``.
"""

from __future__ import annotations

import os

# macOS / Apple-Silicon stability guards (see mace_calc.py for rationale).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from ase import Atoms

log = logging.getLogger("finetune_mace")


# Ground-state atomic term multiplicities for the common organic elements
# (neutral atoms). Used as the default spin state when computing isolated-atom
# GFN2 energies; override or extend per element via the ``multiplicity`` arg of
# :func:`compute_e0s_gfn2`.
_ATOMIC_MULTIPLICITIES: dict[str, int] = {
    "H": 2,   # 2S
    "B": 2,   # 2P
    "C": 3,   # 3P
    "N": 4,   # 4S
    "O": 3,   # 3P
    "F": 2,   # 2P
    "Si": 3,  # 3P
    "P": 4,   # 4S
    "S": 3,   # 3P
    "Cl": 2,  # 2P
    "Br": 2,  # 2P
    "I": 2,   # 2P
}


def compute_e0s_gfn2(
    elements: Sequence[Union[int, str]],
    method: str = "GFN2-xTB",
    box_A: float = 12.0,
    multiplicity: Optional[dict[Union[int, str], int]] = None,
    **calc_kw,
) -> dict[int, float]:
    """Compute isolated-atom GFN2 energies for the given elements.

    Each atom is placed alone in a large box (non-periodic) and its GFN2
    single-point energy is recorded. Returns a mapping ``Z -> energy (eV)``
    suitable for ``mace_run_train --E0s='{...}'``.

    Spin handling: the energy of an isolated atom depends on its electronic
    ground state, so the correct multiplicity must be set. ``multiplicity`` is
    an optional ``{element: mult}`` mapping (keys may be element symbols or
    atomic numbers) that overrides or extends the built-in table of ground-state
    multiplicities for the common organic elements (H, B, C, N, O, F, Si, P, S,
    Cl, Br, I). For any element not covered, the electron-parity heuristic
    (doublet for odd Z, singlet for even Z) is used with a warning -- this is
    often wrong, so pass an explicit ``multiplicity`` entry for unusual
    elements.

    This is optional; ``--E0s=average`` (the default in :func:`build_train_command`)
    is simpler and usually fine for fine-tuning. Use this when you want a
    physically meaningful atomic-energy baseline.
    """
    from ase import Atom
    from gfn2_data import get_gfn2_calculator

    mult_overrides = multiplicity or {}

    def _mult_for(symbol: str, z: int) -> int:
        # User overrides (by symbol or Z) take precedence.
        if symbol in mult_overrides:
            return int(mult_overrides[symbol])
        if z in mult_overrides:
            return int(mult_overrides[z])
        # Built-in ground-state table for the common organic elements.
        if symbol in _ATOMIC_MULTIPLICITIES:
            return _ATOMIC_MULTIPLICITIES[symbol]
        # Last resort: parity heuristic (often wrong -- warn so it's not silent).
        fallback = 2 if z % 2 == 1 else 1
        log.warning(
            "No ground-state multiplicity known for %s (Z=%d); falling back to "
            "the parity heuristic (mult=%d), which is often wrong. Pass "
            "multiplicity={'%s': <mult>} to set it explicitly.",
            symbol, z, fallback, symbol,
        )
        return fallback

    e0s: dict[int, float] = {}
    for el in elements:
        atom = Atom(el)
        z = atom.number
        at = Atoms([atom], positions=[[0.0, 0.0, 0.0]])
        at.cell = [box_A, box_A, box_A]
        at.pbc = False
        mult = _mult_for(atom.symbol, z)
        at.calc = get_gfn2_calculator(method=method, multiplicity=mult, **calc_kw)
        e = float(at.get_potential_energy())
        e0s[z] = e
        log.info("E0[%s (Z=%d)] = %.6f eV (mult=%d)", atom.symbol, z, e, mult)
    return e0s


def elements_in_file(
    path: Union[str, Path],
    *,
    max_frames: Optional[int] = None,
    stable_frames: Optional[int] = None,
) -> list[str]:
    """Return the sorted unique element symbols present in an extxyz file.

    Frames are streamed lazily via :func:`ase.io.iread`, so memory stays low
    regardless of file size (a 4-5k-frame file is fine). By default every frame
    is read for a complete element set. To sample a large file instead, pass:

    * ``max_frames`` -- stop after reading this many frames.
    * ``stable_frames`` -- stop after this many consecutive frames add no new
      element (early stop once the element set has converged).

    Sampling trades completeness for speed: a rare element appearing only late
    in the file can be missed, so the scanned frame count is logged (and a
    warning is emitted) when sampling is active.
    """
    from ase.io import iread

    symbols: set[str] = set()
    since_new = 0
    n = 0
    stopped_early = False
    it = iread(str(path), format="extxyz")
    for a in it:
        before = len(symbols)
        symbols.update(a.get_chemical_symbols())
        n += 1
        since_new = 0 if len(symbols) > before else since_new + 1
        if (max_frames is not None and n >= max_frames) or (
            stable_frames is not None and since_new >= stable_frames
        ):
            # We hit a cap. Peek the next frame to tell a genuine early stop
            # (more frames remain) from having read the whole file. If one more
            # frame exists, fold its elements in too.
            try:
                nxt = next(it)
                stopped_early = True
                symbols.update(nxt.get_chemical_symbols())
            except StopIteration:
                pass
            break
    log.info("elements_in_file(%s): scanned %d frame(s)%s -> %s",
             path, n, " (sampled)" if stopped_early else "", sorted(symbols))
    if stopped_early:
        log.warning(
            "Element scan stopped after %d frame(s) of %s; a rare element "
            "appearing only late could be missed. Drop max_frames/stable_frames "
            "for a complete scan.", n, path,
        )
    return sorted(symbols)


def auto_e0s_from_train(
    train_file: Union[str, Path],
    valid_file: Optional[Union[str, Path]] = None,
    *,
    multiplicity: Optional[dict[Union[int, str], int]] = None,
    method: str = "GFN2-xTB",
    box_A: float = 12.0,
    max_frames: Optional[int] = None,
    stable_frames: Optional[int] = None,
    **calc_kw,
) -> dict[int, float]:
    """Auto-examine the training set and compute isolated-atom GFN2 E0s.

    Scans the training (and, if given, validation) extxyz files, collects the
    unique element symbols, and runs :func:`compute_e0s_gfn2` on exactly that
    set. Returns the ``e0s`` dict (``Z -> energy (eV)``) ready to pass to
    :func:`build_train_command` as ``e0s=`` -- or use ``e0s="auto"`` there to
    call this implicitly.

    ``multiplicity`` overrides/extends the built-in ground-state table for any
    elements not covered (see :func:`compute_e0s_gfn2`).

    ``max_frames`` / ``stable_frames`` are forwarded to :func:`elements_in_file`
    to sample very large files instead of reading them in full. The default
    streams all frames lazily, which already handles 4-5k-frame files in a few
    seconds with low memory; only reach for sampling on much larger files.
    """
    files = [train_file] + ([valid_file] if valid_file else [])
    symbols: set[str] = set()
    for f in files:
        symbols.update(elements_in_file(
            f, max_frames=max_frames, stable_frames=stable_frames))
    if not symbols:
        raise RuntimeError(f"No atoms found in {files}")
    elements = sorted(symbols)
    log.info("Elements detected across training set: %s", elements)
    return compute_e0s_gfn2(
        elements, method=method, box_A=box_A,
        multiplicity=multiplicity, **calc_kw,
    )


# alias -> filename that MACE writes into its download cache (~/.cache/mace).
# These are stable for the current mace-torch; if a future version renames them,
# the fallback glob in cached_model_path() still finds the most recent .model.
_CACHE_FILENAMES = {
    "off-small": "MACE-OFF23_small.model",
    "off-medium": "MACE-OFF23_medium.model",
    "off-large": "MACE-OFF23_large.model",
    "omol": "MACE-omol-0-extra-large-1024.model",
    "omol-extra-large": "MACE-omol-0-extra-large-1024.model",
}


def cached_model_path(alias: str = "off-medium") -> str:
    """Return the local path to a cached MACE model, downloading it if needed.

    ``alias`` is a :mod:`mace_calc` alias (e.g. ``"off-medium"``, ``"omol"``,
    ``"off-large"``). The path is what ``mace_run_train --foundation_model``
    expects to fine-tune from that checkpoint.
    """
    import mace_calc as mc

    if alias not in _CACHE_FILENAMES:
        raise ValueError(
            f"No cached-path mapping for alias {alias!r}. Pass a .model path "
            f"directly to foundation_model= instead."
        )
    # Trigger the download + cache write (no-op if already cached).
    mc.get_calculator(model=alias)

    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    cache_dir = Path(base) / "mace"
    p = cache_dir / _CACHE_FILENAMES[alias]
    if p.exists():
        return str(p)
    # Fallback: newest .model in the cache dir.
    cands = sorted(cache_dir.glob("*.model"), key=lambda x: x.stat().st_mtime)
    if cands:
        log.warning("Expected %s not found; falling back to %s", p, cands[-1])
        return str(cands[-1])
    raise RuntimeError(f"Could not locate cached model for {alias!r} in {cache_dir}.")


def build_train_command(
    train_file: Union[str, Path],
    valid_file: Union[str, Path],
    foundation_model: str = "off-medium",
    *,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: Optional[str] = None,
    e0s: Union[str, dict[int, float]] = "average",
    e0s_multiplicity: Optional[dict[Union[int, str], int]] = None,
    e0s_box_A: float = 12.0,
    model: str = "MACE",
    name: str = "gfn2_finetune",
    results_dir: Union[str, Path] = "runs",
    checkpoints_dir: Optional[Union[str, Path]] = None,
    max_num_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-8,
    energy_weight: float = 100.0,
    forces_weight: float = 1.0,
    ema: bool = True,
    ema_decay: float = 0.99,
    scheduler: str = "patience",
    patience: int = 25,
    device: str = "cpu",
    default_dtype: str = "float64",
    config_type_weights: Optional[dict] = None,
    extra: Optional[Sequence[str]] = None,
) -> list[str]:
    """Construct the ``mace_run_train`` argument list for a fine-tune.

    Returns a list of CLI tokens. Pass it to :func:`run_finetune` or
    ``subprocess.run(cmd)``.

    ``e0s`` selects the per-element reference-energy baseline:

    * ``"average"`` (default) -- MACE fits per-element E0s from the data.
    * ``"auto"`` -- scan ``train_file``/``valid_file`` for the elements present
      and compute their isolated-atom GFN2 energies via
      :func:`auto_e0s_from_train` (uses ``e0s_multiplicity`` / ``e0s_box_A``).
    * a ``{Z: energy}`` dict -- passed through verbatim.

    """
    if checkpoints_dir is None:
        checkpoints_dir = results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Resolve a mace_calc alias (e.g. "off-medium") to a cached .model path so
    # we fine-tune from an organic checkpoint. A bare path or a "medium"/"large"
    # shorthand is passed through unchanged.
    try:
        from mace_calc import MODEL_ALIASES

        if str(foundation_model) in MODEL_ALIASES:
            foundation_model = cached_model_path(str(foundation_model))
    except Exception as e:  # pragma: no cover
        log.warning("Could not resolve foundation_model alias %r: %s",
                    foundation_model, e)
    foundation_model = str(foundation_model)

    e0s_arg: str
    if e0s == "auto":
        # Scan the training/validation files for the elements present and
        # compute their isolated-atom GFN2 energies on the fly.
        e0s_dict = auto_e0s_from_train(
            train_file, valid_file,
            multiplicity=e0s_multiplicity, box_A=e0s_box_A,
        )
        e0s_arg = json.dumps({str(k): v for k, v in sorted(e0s_dict.items())})
    elif isinstance(e0s, dict):
        e0s_arg = json.dumps({str(k): v for k, v in sorted(e0s.items())})
    else:
        e0s_arg = str(e0s)

    cmd = [
        "mace_run_train",
        "--foundation_model", foundation_model,
        "--train_file", str(train_file),
        "--valid_file", str(valid_file),
        "--energy_key", energy_key,
        "--forces_key", forces_key,
        "--E0s", e0s_arg,
        "--model", model,
        "--name", name,
        "--results_dir", str(results_dir),
        "--checkpoints_dir", str(checkpoints_dir),
        "--max_num_epochs", str(max_num_epochs),
        "--scheduler", scheduler,
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--energy_weight", str(energy_weight),
        "--forces_weight", str(forces_weight),
        "--device", device,
        "--default_dtype", default_dtype,
        "--patience", str(patience),
    ]
    if stress_key is not None:
        cmd += ["--stress_key", stress_key, "--stress_weight", "1.0"]
    if ema:
        cmd += ["--ema", "--ema_decay", str(ema_decay)]
    if config_type_weights is not None:
        cmd += ["--config_type_weights", json.dumps(config_type_weights)]
    if extra:
        cmd += list(extra)
    return cmd


def run_finetune(
    train_file: Union[str, Path],
    valid_file: Union[str, Path],
    *,
    run: bool = True,
    **kw,
) -> Union[list[str], subprocess.CompletedProcess]:
    """Build and (optionally) launch the fine-tuning command.

    If ``run=False`` (or the ``MACE_DRY_RUN`` env var is set), the command list
    is returned without executing -- useful for inspection or batch submission.
    """
    cmd = build_train_command(train_file, valid_file, **kw)
    if not run or os.environ.get("MACE_DRY_RUN"):
        log.info("Dry run; command:\n  %s", " ".join(cmd))
        return cmd
    log.info("Launching: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True)


def find_latest_model(results_dir: Union[str, Path], name: str) -> Optional[Path]:
    """Return the most recent ``<name>_run-*.model`` produced by training."""
    base = Path(results_dir)
    candidates = sorted(
        base.glob(f"{name}_run-*.model"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def eval_model(
    model_path: Union[str, Path],
    atoms: Atoms,
    device: str = "cpu",
    dtype: str = "float64",
) -> dict:
    """Quick sanity check: load a fine-tuned model and run a single point."""
    from mace.calculators import MACECalculator

    atoms = atoms.copy()
    atoms.calc = MACECalculator(model_paths=str(model_path), device=device,
                                default_dtype=dtype)
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
    }
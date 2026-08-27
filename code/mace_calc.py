"""MACE foundation models as an ASE calculator front-end for small molecules.

This module wraps the public organic MACE foundation-model loaders
``mace_off`` (MACE-OFF23, 10 elements: H,C,N,O,F,P,S,Cl,Br,I, wB97M-D3, with
small/medium/large variants) and ``mace_omol`` (MACE-OMOL-0, Z = 1..83 incl.
main-group and transition metals, wB97M-VV10, extra-large only) and exposes
them through ASE for the four classic atomistic tasks:

    * single-point energy / forces / stress
    * geometry optimization
    * vibrational analysis (harmonic frequencies)
    * molecular dynamics (NVE / NVT)

Model selection defaults to ``"auto"``: organic elements use the fast
``"off-medium"`` checkpoint, and any element outside the OFF23 set (up to Bi,
Z=83) automatically escalates to ``"omol"``. See :func:`select_model`.

It is designed to run on Apple Silicon laptops. The default device is ``cpu``
because the MACE/e3nn stack still has gaps on the MPS (Metal) backend -- for the
small molecules this module targets, CPU is fast enough and fully reliable.
MPS can be requested with ``device="mps"`` but is considered experimental.

All energies are in eV, lengths in Angstrom, forces in eV/Angstrom -- ASE's
native units. MACE returns energies in eV already, so no conversion is needed.

Example
-------
>>> from mace_calc import singlepoint, optimize, run_md
>>> from ase.build import molecule
>>> atoms = molecule("H2O")
>>> e, f = singlepoint(atoms)              # auto -> off-medium (organic)
>>> optimize(atoms, fmax=0.01)             # auto
>>> run_md(atoms, T_K=300, steps=1000, trajectory="water_md.traj")
>>> # a transition-metal complex auto-escalates to omol:
>>> from ase import Atoms
>>> fe_complex = Atoms("FeN4C2", positions=[[0,0,0],[1,0,0],[-1,0,0],
...                                          [0,1,0],[0,-1,0],[0,0,1],[-1,0,1]])
>>> singlepoint(fe_complex)                # auto -> omol
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch/tblite import)
# 1. PyTorch and tblite each ship a copy of libomp; loading both aborts with
#    "OMP: Error #15" unless this is allowed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 2. The e3nn/torch multi-threaded OpenMP backend segfaults on Apple Silicon
#    (SIGSEGV during the MACE forward pass). Pinning OpenMP/MKL to a single
#    thread eliminates the crash. For the small molecules targeted here this
#    has negligible cost. Override by exporting the var before importing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ase import Atoms, units
from ase.io import Trajectory, read, write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.vibrations import Vibrations
from ase.md import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

log = logging.getLogger("mace_calc")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Aliases -> (loader_name, model_string_or_None).
# loader_name is "mace_omol" or "mace_off".
MODEL_ALIASES: dict[str, tuple[str, Optional[str]]] = {
    # MACE-OFF23 family (organic molecules, wB97M-D3, 10 elements).
    # Lighter than OMOL and available in small/medium/large -> default for fast
    # interactive small-molecule work.
    "off-small": ("mace_off", "small"),
    "off-medium": ("mace_off", "medium"),
    "off-large": ("mace_off", "large"),
    # MACE-OMOL family (organic molecules + main-group/transition metals,
    # wB97M-VV10, Z = 1..83). Ships ONLY the extra-large (1024-channel)
    # checkpoint -> ~90 s first load, ~0.4 s/call after. Used automatically when
    # the input contains elements OFF23 cannot handle.
    "omol": ("mace_omol", None),
    "omol-extra-large": ("mace_omol", None),
}

# Sensible default dtype per loader. Both use float64 by convention; pass
# dtype="float32" for long MD to go faster.
_DEFAULT_DTYPE = {
    "mace_omol": "float64",
    "mace_off": "float64",
}

# --- Element coverage (read directly from the cached model files) ------------
# OFF23 supports exactly H,C,N,O,F,P,S,Cl,Br,I:
_OFF23_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
# OMOL supports Z = 1..83 (H through Bi, incl. transition metals):
_OMOL_MAX_Z = 83


def off23_elements() -> frozenset[int]:
    """Atomic numbers supported by MACE-OFF23 (H,C,N,O,F,P,S,Cl,Br,I)."""
    return _OFF23_ELEMENTS


def omol_max_z() -> int:
    """Highest atomic number supported by MACE-OMOL (83 = Bi)."""
    return _OMOL_MAX_Z


def select_model(atoms: Atoms, base: str = "off-medium") -> str:
    """Pick a model alias based on the elements present in ``atoms``.

    * all elements in the OFF23 set -> ``base`` (default ``"off-medium"``)
    * any element outside OFF23 but within Z = 1..83 -> ``"omol"``
    * anything heavier than Bi (Z > 83) -> ``ValueError`` (no model here covers it)
    """
    zs = {int(z) for z in atoms.get_atomic_numbers()}
    if zs <= _OFF23_ELEMENTS:
        return base
    if max(zs) <= _OMOL_MAX_Z:
        return "omol"
    from ase.data import chemical_symbols

    bad = sorted(z for z in zs if z < 1 or z > _OMOL_MAX_Z)
    names = [chemical_symbols[z] for z in bad]
    raise ValueError(
        f"Elements {names} (Z={bad}) are not supported by MACE-OFF23 or "
        f"MACE-OMOL (OMOL covers Z=1..{_OMOL_MAX_Z}). No suitable model in this "
        f"package; consider adding a MACE-MP model for heavy elements."
    )


def _resolve_model_for_atoms(model: Optional[str], atoms: Atoms) -> str:
    """Resolve ``"auto"``/``None`` to a concrete alias using :func:`select_model`."""
    if model in (None, "auto"):
        return select_model(atoms)
    return model


def list_models() -> list[str]:
    """Return the available model aliases."""
    return sorted(MODEL_ALIASES)


def _resolve_alias(model: str) -> tuple[str, Optional[str]]:
    try:
        return MODEL_ALIASES[model]
    except KeyError as exc:  # pragma: no cover - helpful error
        raise ValueError(
            f"Unknown model alias {model!r}. "
            f"Available: {', '.join(list_models())}"
        ) from exc


def _pick_device(device: str) -> str:
    """Resolve 'auto' to a concrete device, preferring CPU for reliability."""
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            log.warning(
                "device='auto' selected MPS. MPS support in MACE/e3nn is "
                "incomplete; falling back to CPU. Pass device='mps' to force."
            )
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def get_calculator(
    model: str = "off-medium",
    device: str = "cpu",
    dtype: Optional[str] = None,
    dispersion: bool = False,
    **kwargs,
):
    """Build an ASE-compatible MACE calculator.

    Parameters
    ----------
    model : str
        Concrete alias from :func:`list_models` (e.g. ``"off-medium"``,
        ``"omol"``). Use ``"auto"`` only via :func:`attach` / the task functions
        (which need an ``Atoms`` object to inspect elements); passing
        ``"auto"`` here raises ``ValueError``.
    device : str
        ``"cpu"`` (default, recommended on Apple Silicon), ``"mps"``
        (experimental), ``"cuda"``, or ``"auto"`` (resolves to CPU for
        reliability).
    dtype : str, optional
        ``"float32"`` or ``"float64"``. Defaults to float64. Use float64 for
        tight geometry optimizations and vibrations; float32 for long MD.
    dispersion : bool
        Reserved (accepted for API compatibility); OFF23/OMOL do not add a
        separate D3 term here.
    **kwargs
        Passed through to the underlying loader.
    """
    if model in (None, "auto"):
        raise ValueError(
            "get_calculator() needs a concrete model alias; 'auto' requires an "
            "Atoms object -- use attach()/singlepoint()/optimize()/... instead."
        )
    from mace.calculators import mace_off, mace_omol

    loader_name, model_str = _resolve_alias(model)
    dtype = dtype or _DEFAULT_DTYPE[loader_name]
    device = _pick_device(device)

    if loader_name == "mace_omol":
        calc = mace_omol(
            model=model_str,
            device=device,
            default_dtype=dtype,
            **kwargs,
        )
    elif loader_name == "mace_off":
        calc = mace_off(
            model=model_str,
            device=device,
            default_dtype=dtype,
            **kwargs,
        )
    else:  # pragma: no cover - guarded by _resolve_alias
        raise ValueError(f"Unknown loader {loader_name!r}")

    log.info("Loaded MACE model %s (%s, %s, %s)", model, loader_name, dtype, device)
    return calc


def attach(atoms: Atoms, model: str = "auto", **calc_kw) -> Atoms:
    """Attach a MACE calculator to ``atoms`` and return ``atoms`` (in-place).

    ``model="auto"`` (the default) inspects the atoms and picks ``"off-medium"``
    for organic elements (H,C,N,O,F,P,S,Cl,Br,I) or escalates to ``"omol"`` when
    the structure contains other elements up to Bi (Z=83).
    """
    model = _resolve_model_for_atoms(model, atoms)
    atoms.calc = get_calculator(model=model, **calc_kw)
    return atoms


# ---------------------------------------------------------------------------
# Task 1: single point
# ---------------------------------------------------------------------------
def singlepoint(
    atoms: Atoms,
    model: str = "auto",
    stress: bool = False,
    **calc_kw,
) -> dict:
    """Run a single-point calculation.

    Returns a dict with ``energy`` (eV), ``forces`` (N,3; eV/Angstrom) and,
    if requested and the system is periodic, ``stress`` (eV/Angstrom^3).
    """
    if atoms.calc is None or not _is_mace_calc(atoms.calc):
        attach(atoms, model=model, **calc_kw)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    out = {"energy": energy, "forces": forces}
    if stress:
        if not atoms.pbc.any():
            log.warning("stress requested but system is non-periodic; skipping.")
        else:
            out["stress"] = np.asarray(atoms.get_stress())
    return out


def _is_mace_calc(calc) -> bool:
    name = type(calc).__name__
    return name.startswith("MACECalculator") or "MACE" in name


# ---------------------------------------------------------------------------
# Task 2: geometry optimization
# ---------------------------------------------------------------------------
_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}


def optimize(
    atoms: Atoms,
    model: str = "auto",
    fmax: float = 0.01,  # eV/Angstrom
    steps: int = 500,
    optimizer: str = "FIRE",
    trajectory: Optional[Union[str, Path]] = None,
    logfile: Optional[Union[str, Path]] = "-",
    **calc_kw,
) -> dict:
    """Relax the geometry to ``fmax`` (eV/Angstrom).

    Returns a dict with the relaxed ``energy``, final ``forces``, and whether
    the optimizer converged. The ``atoms`` object is updated in place.
    """
    if atoms.calc is None or not _is_mace_calc(atoms.calc):
        attach(atoms, model=model, **calc_kw)
    try:
        Opt = _OPTIMIZERS[optimizer]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. Choose from {list(_OPTIMIZERS)}."
        ) from exc
    dyn = Opt(atoms, trajectory=str(trajectory) if trajectory else None, logfile=logfile)
    converged = dyn.run(fmax=fmax, steps=steps)
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
        "converged": bool(converged),
        "fmax": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
    }


# ---------------------------------------------------------------------------
# Task 3: vibrations
# ---------------------------------------------------------------------------
def vibrations(
    atoms: Atoms,
    model: str = "auto",
    indices: Optional[Sequence[int]] = None,
    nfree: int = 2,
    delta: float = 0.01,  # Angstrom
    name: str = "vib",
    **calc_kw,
) -> dict:
    """Harmonic vibrational analysis by finite differences.

    Returns a dict with ``frequencies`` (cm^-1, ASE convention: imaginary are
    negative), ``energies`` (meV), and the path to the vib data. The calculator
    is attached if missing. Requires a reasonably relaxed geometry for
    meaningful real frequencies.

    Notes
    -----
    Uses float64 internally for accurate gradients. Run on a relaxed geometry
    for meaningful real frequencies; an unrelaxed structure yields imaginary
    modes. ``"off-medium"``/``"off-large"`` are good for organic molecules,
    ``"omol"`` when transition metals are present.
    """
    if atoms.calc is None or not _is_mace_calc(atoms.calc):
        # vibrations are sensitive to gradient noise -> prefer float64
        calc_kw.setdefault("dtype", "float64")
        attach(atoms, model=model, **calc_kw)
    vib = Vibrations(atoms, indices=indices, nfree=nfree, delta=delta, name=name)
    # ASE caches per-displacement results to <name>.pckl and skips recomputation
    # if present. Stale/partial files (e.g. from a killed run) make get_frequencies
    # crash with None forces, so wipe the cache before a fresh run.
    try:
        vib.clean()
    except Exception:  # pragma: no cover
        pass
    vib.run()
    freqs = np.asarray(vib.get_frequencies())  # cm^-1
    energies = np.asarray(vib.get_energies())  # eV
    try:
        vib.summary()
    except Exception:  # pragma: no cover
        pass
    return {
        "frequencies_cm1": freqs,
        "energies_meV": energies * 1000.0,
        "name": name,
        "nmodes": len(freqs),
    }


# ---------------------------------------------------------------------------
# Task 4: molecular dynamics
# ---------------------------------------------------------------------------
def run_md(
    atoms: Atoms,
    model: str = "auto",
    T_K: float = 300.0,
    timestep_fs: float = 0.5,
    steps: int = 1000,
    ensemble: str = "nvt",
    friction: float = 0.01,  # 1/fs, Langevin
    trajectory: Optional[Union[str, Path]] = None,
    traj_interval: int = 10,
    logfile: Optional[Union[str, Path]] = "-",
    loginterval: int = 100,
    seed: Optional[int] = None,
    **calc_kw,
) -> dict:
    """Run a short MD trajectory.

    Parameters
    ----------
    ensemble : str
        ``"nvt"`` (Langevin, default) or ``"nve"`` (velocity Verlet).
    trajectory : path
        If given, an ASE ``Trajectory`` is written every ``traj_interval`` steps.
    seed : int, optional
        Seed for the Maxwell-Boltzmann velocity distribution.
    """
    if atoms.calc is None or not _is_mace_calc(atoms.calc):
        attach(atoms, model=model, **calc_kw)

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=rng, force_temp=True)
    Stationary(atoms)  # remove total momentum
    ZeroRotation(atoms)  # remove angular momentum

    dt = timestep_fs * units.fs

    if ensemble.lower() == "nvt":
        dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction / units.fs,
                       logfile=logfile, loginterval=loginterval, rng=rng)
    elif ensemble.lower() == "nve":
        dyn = VelocityVerlet(atoms, dt, logfile=logfile, loginterval=loginterval)
    else:
        raise ValueError(f"Unknown ensemble {ensemble!r}; use 'nvt' or 'nve'.")

    if trajectory:
        traj = Trajectory(str(trajectory), "w", atoms)
        dyn.attach(traj.write, interval=traj_interval)

    dyn.run(steps)

    return {
        "ensemble": ensemble,
        "T_K": T_K,
        "steps": steps,
        "timestep_fs": timestep_fs,
        "final_energy": float(atoms.get_potential_energy()),
        "trajectory": str(trajectory) if trajectory else None,
    }


# ---------------------------------------------------------------------------
# Convenience: build inputs
# ---------------------------------------------------------------------------
def molecule(name: str, **kw) -> Atoms:
    """Build a small molecule from the ASE database (e.g. 'H2O', 'CH3OH')."""
    from ase.build import molecule as _molecule

    return _molecule(name, **kw)


def from_xyz(path: Union[str, Path]) -> Atoms:
    """Read a single geometry from an XYZ file."""
    return read(str(path))


def smiles_to_atoms(
    smiles: str,
    n_conformers: int = 5,
    seed: int = 42,
    forcefield: str = "mmff",
    minimize_iters: int = 500,
    prune_rms: float = 0.5,
) -> Atoms:
    """Build a 3D molecule from a SMILES string via an RDKit conformer search.

    Pipeline: parse -> add Hs -> embed ``n_conformers`` (ETKDGv3, RMS-pruned) ->
    minimize each with MMFF (falls back to UFF if MMFF is unavailable for the
    element set) -> return the lowest-energy conformer as an ASE ``Atoms``
    object (positions in Angstrom).

    Requires the optional ``rdkit`` dependency (``uv pip install rdkit``).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "smiles_to_atoms requires rdkit. Install with: "
            "uv pip install rdkit (or pip install rdkit)."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if not cids:
        raise RuntimeError(f"RDKit failed to embed any conformers for {smiles!r}")

    ff_name = forcefield.lower()
    energies: list[tuple[float, int, str]] = []  # (energy, conf_id, method)
    for cid in cids:
        method_used = ff_name
        if ff_name == "mmff":
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                # MMFF not parameterized for this element set -> use UFF
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                method_used = "uff"
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            method_used = "uff"
        ff.Minimize(maxIts=minimize_iters)
        energies.append((ff.CalcEnergy(), cid, method_used))

    best_e, best_cid, best_method = min(energies)
    log.info("smiles_to_atoms: %d conformers, lowest E=%.3f (%s) confId=%d",
             len(energies), best_e, best_method, best_cid)

    conf = mol.GetConformer(best_cid)
    from ase import Atom

    atoms = Atoms(
        [Atom(atom.GetSymbol(), conf.GetAtomPosition(i))
         for i, atom in enumerate(mol.GetAtoms())]
    )
    atoms.info["smiles"] = smiles
    atoms.info["conformer_energy"] = best_e
    atoms.info["forcefield"] = best_method
    atoms.info["n_conformers_sampled"] = len(energies)
    return atoms


def from_smiles(smiles: str, n_conformers: int = 5, seed: int = 42, **kw) -> Atoms:
    """Build a 3D molecule from SMILES (multi-conformer MMFF). See
    :func:`smiles_to_atoms`. Kept for backwards compatibility."""
    return smiles_to_atoms(smiles, n_conformers=n_conformers, seed=seed)


def smiles_to_xyz(
    smiles: str,
    path: Union[str, Path],
    n_conformers: int = 5,
    seed: int = 42,
    comment: Optional[str] = None,
    **kw,
) -> Path:
    """SMILES -> lowest-energy MMFF conformer -> XYZ file ready for MACE.

    Writes a single-geometry XYZ (ASE format) at ``path`` and returns the
    path. The geometry is the lowest-energy conformer from
    :func:`smiles_to_atoms`; pass it to ``mace_calc.optimize``/``vibrations``
    or load it elsewhere with ``mace_calc.from_xyz``.
    """
    atoms = smiles_to_atoms(smiles, n_conformers=n_conformers, seed=seed)
    c = comment or f"{atoms.get_chemical_formula()} from SMILES; ff={atoms.info.get('forcefield')}"
    write(str(path), atoms, format="xyz", comment=c)
    log.info("smiles_to_xyz: wrote %s (%s)", path, atoms.get_chemical_formula())
    return Path(path)
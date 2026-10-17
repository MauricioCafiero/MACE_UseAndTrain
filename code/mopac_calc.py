"""MOPAC semi-empirical methods (PM7, PM6-D3H4, ...) as an ASE calculator front-end.

A thin companion to ``mace_calc.py``: same four tasks (single point, geometry
optimization, vibrations, MD) driven through ASE, but with MOPAC as the
energy/force engine. Useful as a cheap QM baseline next to the MACE models.

MOPAC is the conda-forge build in its own env (``conda create -n mopac -c
conda-forge mopac``), symlinked to ``~/.local/bin/mopac``. The binary path is
resolved explicitly so it also works from shells that lack ``~/.local/bin`` on
PATH (cron, detached Popen jobs).

Units are ASE-native: eV, Angstrom, eV/Angstrom. ASE reports the MOPAC *heat of
formation* (converted to eV) as the potential energy, and forces are its
gradient -- so energies are comparable between MOPAC runs, not to MACE/DFT
absolute energies.

Each calculation writes ``<label>.mop`` / ``<label>.out`` into ``workdir``
(default ``mopac_runs/``).

Example
-------
>>> from mopac_calc import singlepoint, optimize, vibrations
>>> from ase.build import molecule
>>> atoms = molecule("H2O")
>>> optimize(atoms, method="PM7", fmax=0.01)
>>> vibrations(atoms, method="PM7")
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ase import Atoms, units
from ase.calculators.mopac import MOPAC
from ase.io import Trajectory
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE, LBFGS
from ase.vibrations import Vibrations

MOPAC_EXE = shutil.which("mopac") or str(Path.home() / ".local/bin/mopac")


def get_calculator(
    method: str = "PM7",
    charge: int = 0,
    workdir: Union[str, Path] = "mopac_runs",
    label: str = "mopac",
    extra: str = "",
) -> MOPAC:
    """Build an ASE MOPAC calculator.

    ``method`` is any MOPAC Hamiltonian (PM7, PM6, PM6-D3H4, AM1, ...).
    ``extra`` is appended to the keyword line (e.g. ``"EPS=78.4"`` for COSMO
    water, ``"UHF TRIPLET"``). ``PRECISE`` tightens SCF/gradient thresholds,
    which finite-difference vibrations and MD energy conservation need.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)
    return MOPAC(
        command=f"{MOPAC_EXE} PREFIX.mop",  # explicit command -> ASE does not append the input file
        directory=str(workdir),
        label=label,
        method=method,
        task=f"1SCF GRADIENTS PRECISE {extra}".strip(),
        charge=charge,
    )


def singlepoint(atoms: Atoms, **calc_kw) -> dict:
    """Energy (eV, heat of formation) and forces (eV/Angstrom)."""
    atoms.calc = get_calculator(**calc_kw)
    return {"energy": float(atoms.get_potential_energy()),
            "forces": np.asarray(atoms.get_forces())}


_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}


def optimize(
    atoms: Atoms,
    fmax: float = 0.01,  # eV/Angstrom
    steps: int = 500,
    optimizer: str = "BFGS",
    trajectory: Optional[Union[str, Path]] = None,
    logfile: Optional[str] = "-",
    **calc_kw,
) -> dict:
    """Relax ``atoms`` in place with an ASE optimizer on MOPAC forces."""
    atoms.calc = get_calculator(**calc_kw)
    dyn = _OPTIMIZERS[optimizer](atoms, trajectory=str(trajectory) if trajectory else None,
                                 logfile=logfile)
    converged = dyn.run(fmax=fmax, steps=steps)
    forces = atoms.get_forces()
    return {"energy": float(atoms.get_potential_energy()),
            "forces": np.asarray(forces),
            "converged": bool(converged),
            "fmax": float(np.linalg.norm(forces, axis=1).max())}


def vibrations(
    atoms: Atoms,
    delta: float = 0.01,  # Angstrom
    nfree: int = 2,
    name: str = "mopac_vib",
    **calc_kw,
) -> dict:
    """Harmonic frequencies (cm^-1, imaginary as negative) by finite differences.

    Run on a relaxed geometry. Stale displacement caches from a killed run make
    ASE crash, so the cache is wiped first (same as ``mace_calc.vibrations``).
    """
    atoms.calc = get_calculator(**calc_kw)
    vib = Vibrations(atoms, nfree=nfree, delta=delta, name=name)
    vib.clean()
    vib.run()
    vib.summary()
    freqs = np.asarray(vib.get_frequencies())
    return {"frequencies_cm1": freqs, "name": name, "nmodes": len(freqs)}


def run_md(
    atoms: Atoms,
    T_K: float = 300.0,
    timestep_fs: float = 0.5,
    steps: int = 1000,
    ensemble: str = "nvt",
    friction: float = 0.01,  # 1/fs, Langevin
    trajectory: Optional[Union[str, Path]] = None,
    traj_interval: int = 10,
    logfile: Optional[str] = "-",
    loginterval: int = 100,
    seed: Optional[int] = None,
    **calc_kw,
) -> dict:
    """Short NVT (Langevin) or NVE MD. One MOPAC process launch per step, so
    this is for small molecules / short runs."""
    atoms.calc = get_calculator(**calc_kw)
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=rng, force_temp=True)
    Stationary(atoms)
    ZeroRotation(atoms)
    dt = timestep_fs * units.fs
    if ensemble == "nvt":
        dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction / units.fs,
                       logfile=logfile, loginterval=loginterval, rng=rng)
    else:
        dyn = VelocityVerlet(atoms, dt, logfile=logfile, loginterval=loginterval)
    if trajectory:
        dyn.attach(Trajectory(str(trajectory), "w", atoms).write, interval=traj_interval)
    dyn.run(steps)
    return {"ensemble": ensemble, "T_K": T_K, "steps": steps,
            "final_energy": float(atoms.get_potential_energy())}


if __name__ == "__main__":
    # Smoke test: PM7 water -> heat of formation (expt -57.8 kcal/mol) + vibrations.
    from ase.build import molecule

    water = molecule("H2O")
    res = optimize(water, method="PM7", logfile=None)
    print(f"PM7 water: dHf = {res['energy'] / (units.kcal / units.mol):.2f} kcal/mol, "
          f"converged={res['converged']}, fmax={res['fmax']:.4f} eV/A")
    print("frequencies (cm^-1):", np.round(vibrations(water, method="PM7")["frequencies_cm1"].real[-3:], 1))

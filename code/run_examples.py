"""End-to-end demo: MACE single point / optimize / vibrations / MD + GFN2 data.

Run with the project venv activated:

    source .venv/bin/activate
    python run_examples.py

It uses small molecules and short trajectories so it finishes in a few minutes
on a CPU-only Apple Silicon laptop. Edit the knobs at the top to scale up.
"""

from __future__ import annotations

import logging
import os

# macOS / Apple-Silicon stability guards (see mace_calc.py for rationale).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
from ase.build import molecule

import mace_calc as mc
import gfn2_data as gd

# ---- knobs -----------------------------------------------------------------
# "auto" picks off-medium for organic elements (H2O/CH3OH/NH3 below) and
# escalates to omol only when needed. Set to "omol" or "off-large" explicitly
# to force a specific model.
MACE_MODEL = "auto"
MD_STEPS = 200
MD_TEMP_K = 300.0


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    section(f"1. MACE single point on H2O  (model={MACE_MODEL})")
    water = molecule("H2O")
    sp = mc.singlepoint(water, model=MACE_MODEL, dtype="float64")
    print(f"energy = {sp['energy']:.6f} eV")
    print(f"max |force| = {np.linalg.norm(sp['forces'], axis=1).max():.4f} eV/A")

    section("2. Geometry optimization of H2O")
    opt = mc.optimize(water, model=MACE_MODEL, fmax=0.01, dtype="float64",
                      trajectory="water_opt.traj")
    print(f"converged={opt['converged']}  E={opt['energy']:.6f} eV  "
          f"fmax={opt['fmax']:.4f} eV/A")

    section("3. Vibrational frequencies of relaxed H2O")
    vib = mc.vibrations(water, model=MACE_MODEL, dtype="float64", name="water_vib")
    print(f"{vib['nmodes']} modes, frequencies (cm^-1):")
    print(np.array2string(vib["frequencies_cm1"], precision=1))

    section(f"4. NVT MD of H2O @ {MD_TEMP_K} K, {MD_STEPS} steps")
    md = mc.run_md(water, model=MACE_MODEL, T_K=MD_TEMP_K, steps=MD_STEPS,
                   timestep_fs=0.5, ensemble="nvt", trajectory="water_md.traj",
                   traj_interval=10, seed=1)
    print(f"final E = {md['final_energy']:.4f} eV  (traj: {md['trajectory']})")

    section("5. GFN2-xTB reference data for fine-tuning (H2O, CH3OH, NH3)")
    mols = [molecule("H2O"), molecule("CH3OH"), molecule("NH3")]
    for m, name in zip(mols, ["water", "methanol", "ammonia"]):
        m.info["label"] = name
    stats = gd.generate_dataset(
        mols,
        outfile="gfn2_data.xyz",
        temperatures_K=[100.0, 300.0],
        md_steps=400,
        sample_every=40,
        do_normal_modes=True,
        nmodes_per_mol=4,
        nmodes_displacements=2,
        valid_fraction=0.15,
        seed=0,
    )
    print(f"frames: total={stats['n_total']} train={stats['n_train']} "
          f"valid={stats['n_valid']}")
    print(f"train file: {stats['train_file']}")
    print(f"valid file: {stats['valid_file']}")

    section("6. (dry-run) MACE fine-tune command from that data")
    import finetune_mace as ft
    cmd = ft.build_train_command(
        stats["train_file"], stats["valid_file"],
        foundation_model="off-medium",
        results_dir="runs",
    )
    print(" ".join(cmd))

    print("\nDone. To actually fine-tune, run:\n  " + " ".join(cmd))


if __name__ == "__main__":
    main()
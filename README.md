# MACE + ASE on Apple Silicon (small molecules)

Run energy, geometry optimization, vibrational analysis, and MD on small
molecules with [MACE](https://github.com/ACEsuit/mace) organic foundation
models through [ASE](https://wiki.fysik.dtu.dk/ase/), plus GFN2-xTB
reference-data generation (via [TBLite](https://github.com/tblite/tblite)) for
fine-tuning a MACE model.

Tested on an Apple A18 Pro laptop, macOS, **CPU-only**.

## Files

| File | Purpose |
|---|---|
| `mace_calc.py` | MACE as an ASE calculator: single point, optimize, vibrations, MD. Autodetects OFF23 vs OMOL from the elements. |
| `gfn2_data.py` | GFN2-xTB (TBLite/ASE): relax → MD + normal-mode sampling → extxyz with energy & forces. |
| `finetune_mace.py` | Builds/launches `mace_run_train` to fine-tune a MACE foundation model on the GFN2 data. |
| `run_examples.py` | End-to-end demo (all of the above on H₂O/CH₃OH/NH₃). |

## Environment

A `uv`-managed venv (Python 3.12 — the best-supported interpreter for the
MACE/torch/e3nn stack) lives in `.venv/`.

```bash
# clone the repo
git clone https://github.com/MauricioCafiero/MACE_UseAndTrain.git
cd MACE_UseAndTrain

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mace-torch ase "tblite[ase]" matplotlib
source .venv/bin/activate
```

Verified versions: mace-torch 0.3.16, torch 2.13.0, e3nn 0.4.4, ase 3.29.0,
tblite 0.7.0 (macOS arm64 wheel), numpy 2.5.1.

## Feasibility notes (read this first)

* **It works on Apple Silicon, on the CPU.** PyTorch 2.13 ships macOS arm64
  wheels and MACE is pure-Python, so the whole stack installs cleanly. Models
  run on the **CPU**, not the Metal GPU.
* **MPS (Apple GPU) is NOT used by default.** MACE/e3nn still has unsupported
  ops on MPS, so `device="cpu"` is the reliable path and is fast enough for
  small molecules. You can try `device="mps"`, but expect possible errors.
* **Two stability guards are required on macOS and are baked into every
  module** (set before torch/tblite import):
  * `KMP_DUPLICATE_LIB_OK=TRUE` — torch and tblite each bundle `libomp`;
    without this the process aborts with `OMP: Error #15`.
  * `OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`) — the e3nn/torch
    multi-threaded OpenMP path **segfaults** (SIGSEGV in the MACE forward pass)
    on Apple Silicon. Single-threaded execution eliminates the crash and costs
    nothing for small systems. Raise it by exporting the var *before* importing
    these modules (at your own risk).
* **Python 3.14 (the Homebrew default) is avoided** in favor of 3.12 to stay on
  the best-supported interpreter. `uv` fetches 3.12 for you.

## Models (organic only)

Two model families are wired in; the right one is chosen automatically.

| Alias | Family | Elements | Notes |
|---|---|---|---|
| `off-small` / `off-medium` *(default)* / `off-large` | MACE-OFF23 | H,C,N,O,F,P,S,Cl,Br,I | wB97M-D3, light & fast (~2 s). Best for everyday organic small molecules. ASL. |
| `omol` | MACE-OMOL-0 | Z = 1..83 (incl. main-group & transition metals) | wB97M-VV10, extra-large only (~90 s first load, ~0.4 s/call after). Auto-used when OFF23 can't cover the elements. ASL. |

```bash
python -c "import mace_calc as m; print(m.list_models())"
```

**Autodetect (`model="auto"`, the default):**
* all elements in the OFF23 set → `off-medium`
* any element outside OFF23 but within Z = 1..83 → `omol`
* heavier than Bi (Z > 83) → `ValueError` (no organic model covers it)

```python
import mace_calc as mc
print(mc.select_model(molecule("H2O")))        # -> 'off-medium'
print(mc.select_model(molecule("FeCl3")))      # -> 'omol'  (Fe not in OFF23)
```

Use `dtype="float64"` for optimizations and vibrations, `dtype="float32"` for
long MD.

## Quick usage

The normal entry point is a **SMILES string**: `smiles_to_atoms` runs an RDKit
conformer search (5 ETKDGv3 conformers, RMS-pruned, MMFF-minimized, lowest kept)
and returns an ASE `Atoms` object. (`mc.from_xyz` reads an existing XYZ;
`ase.build.molecule("H2O")` works for the few small molecules ASE ships.)

```python
import mace_calc as mc

# SMILES -> 3D geometry (RDKit, requires `uv pip install rdkit`)
atoms = mc.smiles_to_atoms("CCO", n_conformers=5, seed=1)

# 1. single point (auto -> off-medium for organic elements)
sp = mc.singlepoint(atoms, dtype="float64")
print(sp["energy"], sp["forces"])

# 2. geometry optimization
mc.optimize(atoms, fmax=0.01, trajectory="opt.traj")

# 3. vibrations (run after relaxing)
vib = mc.vibrations(atoms, name="etoh_vib")
print(vib["frequencies_cm1"])

# 4. MD (NVT Langevin or NVE)
mc.run_md(atoms, T_K=300, steps=1000, ensemble="nvt",
          trajectory="md.traj", traj_interval=10)

# 5. force a specific model
mc.singlepoint(atoms, model="omol")

# write a MACE-ready XYZ directly from SMILES:
mc.smiles_to_xyz("CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
                 "rosuvastatin.xyz", n_conformers=5, seed=1)
```

If MMFF is unavailable for an element set, `smiles_to_atoms` falls back to UFF.
The MMFF-relaxed conformer is a good starting point for a MACE optimization.

## Full pipeline: SMILES → GFN2 data → fine-tuned MACE

The end-to-end workflow for adapting a MACE foundation model to GFN2-xTB. Inputs
are SMILES strings (the usual case); each is turned into a 3D conformer with
`smiles_to_atoms`, passed to `generate_dataset` for GFN2 energy/forces
reference data, then used to fine-tune, and finally the fine-tuned model is
loaded back for ASE simulations.

### 1. SMILES → GFN2 reference data

`gfn2_data.generate_dataset` relaxes each molecule with GFN2-xTB, samples
configurations via NVT MD (multiple temperatures) and normal-mode displacements,
recomputes GFN2 energy + forces on every frame, and writes extended XYZ:

```python
import mace_calc as mc
import gfn2_data as gd

smiles = {
    "rosuvastatin": "CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
    "atorvastatin": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "ethanol":      "CCO",
}
# SMILES -> lowest-energy MMFF conformer -> Atoms (tag each with a label)
mols = []
for name, smi in smiles.items():
    a = mc.smiles_to_atoms(smi, n_conformers=5, seed=hash(name) & 0xffff)
    a.info["label"] = name
    mols.append(a)

stats = gd.generate_dataset(
    mols, outfile="gfn2_data.xyz",
    temperatures_K=[100, 300, 500], md_steps=2000, sample_every=50,
    do_normal_modes=True, valid_fraction=0.1, seed=0,
)
# -> gfn2_data_train.xyz, gfn2_data_valid.xyz
```

The frames carry **both** `REF_energy`/`REF_forces` (robust to ASE's
special-casing of `energy`/`forces`) and `energy`/`forces`. Fine-tune with the
`REF_*` keys (the default).

### 2. Fine-tune a MACE foundation model

```python
import finetune_mace as ft

# foundation_model accepts a mace_calc alias (resolved to a cached .model path)
# or a bare path / "medium" / "large" shorthand.
cmd = ft.build_train_command(
    stats["train_file"], stats["valid_file"],
    foundation_model="off-medium",
    energy_key="REF_energy", forces_key="REF_forces",
    results_dir="runs",
)
# inspect the command, then launch (CPU-slow — see notes below):
ft.run_finetune(stats["train_file"], stats["valid_file"],
                foundation_model="off-medium", results_dir="runs")
```

This wraps `mace_run_train`. The fine-tuned model is written to
`runs/gfn2_finetune_run-<i>.model`.

### 3. Use the fine-tuned model

```python
from mace.calculators import MACECalculator
import mace_calc as mc
import finetune_mace as ft

model_path = ft.find_latest_model("runs", "gfn2_finetune")  # or pass the path
atoms = mc.smiles_to_atoms("CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
                           n_conformers=5, seed=1)
atoms.calc = MACECalculator(model_paths=str(model_path), device="cpu", default_dtype="float64")
print(atoms.get_potential_energy(), atoms.get_forces())
mc.optimize(atoms)  # keeps the attached fine-tuned calc (any MACECalculator is reused)
# or a one-shot check:
ft.eval_model(model_path, atoms)
```

Notes on fine-tuning:
* `--E0s=average` (default) derives per-element reference energies from the
  data. For a cleaner baseline, compute GFN2 isolated-atom energies with
  `ft.compute_e0s_gfn2(["H","C","N","O","F","S"])` and pass that dict as `e0s=`.
  Ground-state multiplicities are built in for the common organic elements
  (H, B, C, N, O, F, Si, P, S, Cl, Br, I); pass `multiplicity={"Fe": 5}` to
  cover others (by symbol or atomic number).
* To auto-detect the elements from the data itself, pass `e0s="auto"` (or call
  `ft.auto_e0s_from_train(train_file, valid_file)`). It scans the training
  extxyz, collects the unique elements, and computes their isolated-atom GFN2
  energies in one step. Frames are streamed lazily, so 4-5k-frame files are
  fine; for much larger files, sample with `max_frames=` / `stable_frames=` on
  `auto_e0s_from_train` (a rare element appearing only late could then be
  missed, so it warns):

  ```python
  ft.run_finetune(stats["train_file"], stats["valid_file"],
                  foundation_model="off-medium", e0s="auto",
                  e0s_multiplicity={"Fe": 5},  # only needed for non-standard elements
                  results_dir="runs")
  ```
* `foundation_model="off-medium"` adapts the MACE-OFF23 medium checkpoint (an
  organic starting point). Use `"omol"` to adapt OMOL, or pass any `.model`
  path.
* Training is CPU-slow; for real runs, prefer a CUDA machine or be patient.
  The GFN2 data-generation step is fast and is the part you do here.

## Demo

```bash
source .venv/bin/activate
python run_examples.py
```

Runs single point → optimize → vibrations → MD on water (auto → off-medium),
then generates a small GFN2 dataset and prints the fine-tune command.

## Licenses

MACE code: MIT. MACE-OMOL-0 and MACE-OFF23 weights: Academic Software License
(non-commercial). TBLite: LGPL-3.0+. GFN2-xTB is free for academic use. Check
the terms before commercial use.
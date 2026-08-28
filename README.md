# MACE + ASE on Apple Silicon (small molecules)

Run energy, geometry optimization, vibrational analysis, and MD on small
molecules with [MACE](https://github.com/ACEsuit/mace) organic foundation
models through [ASE](https://wiki.fysik.dtu.dk/ase/), plus GFN2-xTB
reference-data generation (via [TBLite](https://github.com/tblite/tblite)) for
fine-tuning a MACE model — and score molecules for **out-of-distribution-ness
from the network's internals** (latent-space distance, per-atom energy
residuals, precision disagreement).

Tested on an Apple A18 Pro laptop, macOS, **CPU-only**.

**No GPU? Run it on Colab:** [`notebooks/colab_gfn2_finetune.ipynb`](notebooks/colab_gfn2_finetune.ipynb)
clones this repo, GFN2-labels the 250-frame rotaxane sample, builds the latent
reference pool, fine-tunes `off-medium` on GPU (energy + forces), and re-scores
before/after with per-atom OOD maps — no local install needed.

## Files

All Python modules live in `code/`. Run scripts from the repo root as
`python code/<script>.py` (the script's directory is auto-added to `sys.path`,
so the sibling `import mace_calc` / `import gfn2_data` imports resolve). For the
interactive `import` snippets below, `cd code` first (or put `code/` on your
path).

| File | Purpose |
|---|---|
| `code/trust.py` | **The main entry point:** SMILES/XYZ → MACE single point → energy + OOD signal → TRUST/VERIFY verdict. See `OOD_NOTES.md`. |
| `code/mace_calc.py` | MACE as an ASE calculator: single point, optimize, vibrations, MD. Autodetects OFF23 vs OMOL from the elements. |
| `code/gfn2_data.py` | GFN2-xTB (TBLite/ASE): relax → MD + normal-mode sampling → extxyz with energy & forces. |
| `code/finetune_mace.py` | Builds/launches `mace_run_train` to fine-tune a MACE foundation model on the GFN2 data. |
| `code/run_examples.py` | End-to-end demo (all of the above on H₂O/CH₃OH/NH₃). |
| `code/time_rosuvastatin.py` | Timed single-point benchmark on rosuvastatin (writes `rosuvastatin.xyz`). |
| `code/inspect_activations.py` | Capture MACE layer/neuron activations during a single-point pass (PyTorch forward hooks). |
| `code/activation_ood.py` | Score molecules for out-of-distribution-ness vs a reference pool built from the real MACE-OFF23 test split (latent-distance proxy). |
| `code/activation_viz.py` | Draw an RDKit molecule image colored by activation magnitude and OOD score (side by side), plus neuron-firing diagrams. |
| `code/ood_datasets.py` | Grounded OOD test set from computation-ready datasets (S66 non-covalent dimers + S30L-CI host-guest complexes), scored vs the OFF23 reference pool. |
| `code/energy_ood.py` | Energy-space OOD signals: per-atom interaction-energy residual (`eZ`, |z| + signed molecule-mean), latent-kNN energy consistency (`kZ`), and the layer-contribution profile (system-level). |
| `code/precision_ood.py` | Precision-disagreement signal: \|E_assoc(f32) − E_assoc(f64)\| — system-level, pool-free. |
| `code/layer_sweep.py` | Sweep candidate latent layers for the OOD signal (`data/layer_sweep_results.json`). |
| `code/trust_frames.py` | trust.py per frame: multi-frame XYZ → per-frame energy + mean OOD + verdict summary table. |
| `code/ood_map.py` | 3D per-atom OOD maps: neutral ball-and-stick with halos on unusual atoms (fixed 0–0.5 scale, cached scoring). |
| `code/gfn2_label.py` | Torch-free GFN2-xTB labeling of XYZ frames → extxyz for `mace_run_train` (multiprocess; avoids the torch/tblite libomp clash). |
| `notebooks/colab_gfn2_finetune.ipynb` | The 250-frame GFN2 fine-tune + OOD-rescore workflow, end to end, on a Colab GPU. |

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
cd code
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
python code/run_examples.py
```

Runs single point → optimize → vibrations → MD on water (auto → off-medium),
then generates a small GFN2 dataset and prints the fine-tune command.

## Inspecting model activations

`code/inspect_activations.py` captures MACE layer outputs during a single-point
forward pass via PyTorch forward hooks — useful for asking which
layers/"neurons" fire for a given molecule. MACE is equivariant (e3nn), so most
layers output angular-momentum irreps; the flat, neuron-like activations are the
invariant Linears/MLPs (node embedding, per-atom `interactions.*.linear`, the
radial/bond MLP `interactions.*.conv_tp_weights`, and the readout head), which
the default capture targets.

```bash
source .venv/bin/activate
python code/inspect_activations.py H2O                 # ASE molecule name
python code/inspect_activations.py CCO                 # SMILES (auto-detected)
python code/inspect_activations.py H2O --list-layers   # discover layer names
python code/inspect_activations.py H2O --layer 'interactions.0.skip_tp'  # a specific/equivariant layer
```

As a library:

```python
from inspect_activations import capture_activations, summarize
acts = capture_activations(atoms, model="off-medium")  # -> {layer_name: tensor}
summarize(acts)
```

The `interactions.*.conv_tp_weights` rows are indexed per atom-**pair** (the
radial MLPs — the most chemically interpretable, "which bond-neurons fire");
the `interactions.*.linear` and readout rows are per-**atom**.

## Trust check: is this MACE energy reliable?

`code/trust.py` is the one-command workflow for a new molecule: **SMILES or
XYZ → MACE single point → energy + OOD signal → TRUST/VERIFY verdict.**

```bash
source .venv/bin/activate
python code/trust.py "CC(=O)Nc1ccc(cc1)O"     # SMILES (RDKit embeds 3D)
python code/trust.py molecule.xyz             # or an existing geometry
python code/trust.py "CCO" --relax            # MACE-relax before scoring
python code/trust.py mol.xyz --model off-medium
```

The OOD signal is the **latent distance** of the molecule to a reference pool
built from MACE-OFF23's own training distribution (per-atom cosine distance of
deep-layer activations, molecule-mean over atoms). The verdict rule:

- **mean OOD ≤ 0.25 → TRUST**: in-distribution, where MACE-OFF matches
  CCSD(T)/CBS to ~0.25 kcal/mol MAE on the S66 benchmark.
- **mean OOD > 0.25 → VERIFY**: outside the calibration range. Not necessarily
  wrong — but the error is no longer bounded by any benchmark, so cross-check
  another model size or a higher level of theory.

The signal is a trust boundary, not an error predictor: errors that are
collective well-depth offsets spread over many individually-normal atoms are
invisible to any per-atom signal (details, calibration, and the full
signal-comparison study in [`OOD_NOTES.md`](OOD_NOTES.md)). Requires the
reference pool for the chosen model in `data/` (see OOD_NOTES.md for the
one-time build).

## OOD scoring & activation visualization

`code/activation_ood.py` scores a molecule's atoms and bonds for
**out-of-distribution-ness** vs a reference pool built from the real MACE-OFF23
test split (~81 MB, same SPICE/wB97M-D3 distribution the model was trained on —
source: `doi.org/10.17863/CAM.107498`). The proxy is **latent distance**: each
atom's deep-layer activation vector (`interactions.1.linear`) is compared by
cosine distance to the nearest same-element reference atom; an atom is flagged
when it exceeds that element's in-distribution p99. Per-bond OOD uses the
radial-MLP (`conv_tp_weights`, a function of bond length only), reported as a
continuous "how unusual is this bond length" score with a within-molecule flag.
This is a heuristic exploratory tool, not a calibrated detector — scores are
relative and meaningful only across molecules scored against the same pool.

```bash
source .venv/bin/activate
# first run downloads ~81 MB once + builds/caches the pool (~2-3 min):
python code/activation_ood.py "CC(=O)Nc1ccc(cc1)O"        # paracetamol
python code/activation_ood.py "IC(I)I"                    # iodoform (rare I -> OOD)
python code/activation_ood.py <SMILES> --build-pool --n-frames 250
```

`code/activation_viz.py` draws an **RDKit molecule image** with atoms/bonds
colored two ways, side by side: left = **activation magnitude** (which parts
fire hardest), right = **OOD score** (which parts are unusual for MACE). This
localizes the signal onto the structure.

Both images use a **fixed [0, 1] colour scale** so a colour means the same
thing on every molecule (no per-image auto-rescaling): OOD is a cosine distance
(0 = identical to a training atom, 1 = orthogonal), and activation is shown as
*firing intensity* — each atom/bond's raw `||layer output||` divided by the p99
of that quantity over the OFF23 pool (calibrated once into
`data/act_scale.npz`; 1.0 ≈ "fires as hard as the typical training max").

```bash
python code/activation_viz.py "CC(=O)Nc1ccc(cc1)O" --out paracetamol --out-dir figures
# -> figures/paracetamol_activation.png, paracetamol_ood.png, paracetamol_comparison.png
python code/activation_viz.py --recalibrate-scale          # recompute the fixed ceiling
```

A complementary **neuron-firing diagram** (`--diagram`) shows the network itself
rather than the molecule: a vertical stack of heatmaps, one per layer
(node embedding → interactions → readout, plus the radial-MLP bond panel), with
**rows = atoms/bonds** and **columns = neurons** (binned when wide), colored by
`|activation|` — the dual of the RDKit coloring.

```bash
python code/activation_viz.py "CC(=O)Nc1ccc(cc1)O" --diagram --out paracetamol --out-dir figures
# -> figures/paracetamol_firing.png
```

As a library:

```python
import mace_calc as mc
from activation_ood import ReferencePool, score_molecule
from activation_viz import draw_comparison

pool = ReferencePool.load_or_build()                      # cached after first build
a = mc.smiles_to_atoms("CC(=O)Nc1ccc(cc1)O")
score_molecule(a, "CC(=O)Nc1ccc(cc1)O", pool)              # prints per-atom/bond OOD
draw_comparison(a, "CC(=O)Nc1ccc(cc1)O", pool, out_name="paracetamol")

# neuron-firing diagram (no pool needed): per-layer neuron x atom/bond heatmaps
from activation_viz import draw_neuron_firing
draw_neuron_firing(a, "CC(=O)Nc1ccc(cc1)O", "figures/paracetamol_firing.png")
```

The downloaded OFF23 test data, cached pool, and activation colour-scale live in
`data/` (gitignored: ~81 MB tarball + ~80 MB pool + tiny `act_scale.npz`).

## Grounded OOD test set (S66 + S30L-CI)

`code/ood_datasets.py` scores a set of **real** out-of-distribution structures
from computation-ready datasets against the OFF23 reference pool, kept strictly
within the H,C,N,O,F,Cl element scope. This grounds the OOD detector on actual
benchmark geometries (DFT-D optimized) rather than hand-built approximations.

Sources:
- **S66** (Řezáč, Riley, Hobza, JCTC 2011, doi:10.1021/ct2002946) — 66
  non-covalent dimers parsed from the Psi4 `S66.py` module. The `--` fragment
  separator gives both monomers, so each dimer yields a paired **monomer
  (in-distribution) vs dimer (non-covalent contact)** contrast.
- **S30L-CI** (Sure & Grimme, JCTC 2015, doi:10.1021/acs.jctc.5b00296) — 30
  realistic host-guest complexes (cucurbituril, calixarene, octaacid, exohedral
  fullerene, cyclodextrin, ...), DFT-D optimized, up to ~200 atoms. The ACS
  Supporting Information ships each system pre-split into **host (A) / guest (B)
  / complex (AB)** Turbomole coord files, so the monomer-vs-complex decomposition
  comes from the dataset itself — no extraction. We keep the 23 systems whose
  host+guest+complex fall entirely in scope (drop the I/Na/S systems). The SI is
  paywalled; obtain it from ACS and place the `s30lci_test_set` folder under
  `data/s30l/`.

Each S30L complex is scored as **host / guest / complex** separately, mirroring
the S66 monomer split, so the per-component table answers whether the OOD lives
in the host's own environment, the guest's, or the inclusion geometry.

```bash
python code/ood_datasets.py                  # fetch S66 + load S30L-CI, score vs pool
python code/ood_datasets.py --no-fetch       # use cached S66.py / S30L coords only
python code/ood_datasets.py --energies       # also compare MACE association energies
                                             # to the Grimme Table S1 wB97X-D3/QZ reference
python code/ood_datasets.py --ensemble       # ensemble OOD: spread of E_assoc across
                                             # off-small/medium/large (model disagreement)
python code/ood_datasets.py --separation     # reliability test: do the NN signals
                                             # separate reliable from unreliable energetics?
python code/ood_datasets.py --trend          # model-size trend: S66 + S30L accuracy for
                                             # off-small/medium/large
python code/ood_datasets.py --forces         # force-residual signal: MACE |F| on the
                                             # optimized geometries (true F~0)
python code/ood_datasets.py --diagram        # neuron-firing diagram of the most-OOD
                                             # S30L complex -> figures/ood_S30L-NN_firing.png
```

## Energy-space OOD signals

`code/energy_ood.py` looks at the network's *output* decomposition instead of
its latent inputs. MACE decomposes `E = Σ(E0_i + f_i)`; `f_i` (the per-atom
interaction energy) is the output of the model's `scale_shift` block, captured
with a forward hook, and decomposes further by readout depth as
`f_i = scale·(r_shallow_i + r_deep_i)`.

* **`eZ` — per-atom energy residual.** `|f_i − μ_el| / σ_el` against
  per-element reference stats from a 2500-frame OFF23 reference
  (`data/energy_pool_<model>.npz`, ~78k atoms, built in ~6 min).
* **`ezSign` — signed molecule-mean z.** Coherent per-atom shifts (what a
  collective binding error looks like) survive averaging, unlike |z|; the
  molecule-mean of `f_i` is tight in-distribution.
* **`kZ` — latent-kNN energy consistency.** Predict each atom's `f_i` from
  the k=8 nearest same-element reference atoms *in latent space* and score
  `|f_i − f_kNN| / σ_resid` (calibrated leave-one-out on the reference).
  Asks "does the model assign this atom the energy its latent neighbours
  usually get?" — the data-side analogue of DeepMD's model deviation for a
  single checkpoint.
* **`profDeep` / `fracDeep` — layer-contribution profile (system-level).**
  z-scores of the molecule-mean deep-readout contribution and of the deep
  fraction `mean(r_deep)/mean(f)` — a scale-free composition descriptor.

```bash
python code/energy_ood.py "CC(=O)Nc1ccc(cc1)O"     # score one molecule
python code/energy_ood.py "CC(=O)Nc1ccc(cc1)O" --build-ref   # (re)build the reference
python code/energy_ood.py --separation             # S66/S30L reliability verdict
```

## Precision-disagreement OOD signal

`code/precision_ood.py` computes a **system-level** signal, immune to per-atom
dilution: run the whole energy twice, at float64 and float32, and compare —
`|E_assoc(f32) − E_assoc(f64)|`. The model's prediction is a large cancellation
(totals ~10³ eV, quantities of interest ~10⁻¹ eV), so reduced precision
perturbs every intermediate sum; the output disagreement measures how much the
input's cancellation structure *amplifies* rounding — a proxy for how far the
network extrapolates. Two forwards, no pool, fully deployable. (float32 is the
lowest workable dtype: MACE's graph build uses `lu_cpu`, which torch does not
implement for bfloat16/float16 on CPU.)

```bash
python code/precision_ood.py "CC(=O)Nc1ccc(cc1)O"          # one molecule
python code/precision_ood.py --separation                  # S66/S30L verdict
```

## Licenses

MACE code: MIT. MACE-OMOL-0 and MACE-OFF23 weights: Academic Software License
(non-commercial). TBLite: LGPL-3.0+. GFN2-xTB is free for academic use. Check
the terms before commercial use.
# Training the long-range cutoff models on Modal

How the from-scratch, charge/spin-aware MACE models `omol_spice20k_r6` and `omol_spice20k_r12` were built and trained (2026-11-07). They are identical apart from `r_max`. Every command below is the one actually run, taken from the logs.

Ground rules that apply throughout:

- Ask before spending on Modal, and give a cost estimate. Never stop, cancel or kill a running job without asking, even for budget.
- Don't edit a shell script while it is running; zsh reads scripts incrementally.
- Caffeinate anything long. Launch multi-hour jobs detached (see step 3). The launcher's `caffeinate -w $$` dies with the launcher and the laptop then sleeps at once, so also keep a project-long `caffeinate -ims` running that is tied to no process. Start it detached: `~/miniforge3/bin/python -c "import subprocess; p=subprocess.Popen(['caffeinate','-ims'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True); open('longrange/caffeinate.pid','w').write(str(p.pid))"`. Stop it by hand with `kill $(cat longrange/caffeinate.pid)`.
- Results stay in `longrange/`, not a scratchpad.

## Why from scratch

A fine-tune can't change `r_max`. `mace_run_train` overwrites `args.r_max` with the foundation model's value (`run_train.py`), and `finetuning_utils.py` asserts that they are equal. So the 12 Å model can't be made from a 6 Å checkpoint. `build_train_command(foundation_model=None)` in `code/finetune_mace.py` gives a from-scratch command with no `--foundation_model` flag.

## 1. Data

**Pool.** The OMol25 validation parquet, `data/omol25_val/co_0.parquet` (400k rows), filtered to frames with 60 atoms or fewer and stride 1, giving 102,044 frames. The builder lives in cafquant and needs pyarrow, so it runs in cafquant's venv:

```sh
cd /Users/cafierom/python_mac/mace
caffeinate -i /Users/cafierom/python_mac/cafquant/.venv/bin/python -I \
  /Users/cafierom/python_mac/cafquant/dft_check/legacy_reference/modal/make_replay_omol25.py \
  --parquet data/omol25_val/co_0.parquet --n 400000 --max-atoms 60 \
  --out data/omol25_full60 > data/omol25_full60_build.log 2>&1
```

**Selection.** This step selects 20k frames:

- SPICE-2 elements only (17: H Li B C N O F Na Mg Si P S Cl K Ca Br I);
- |q| ≤ 2;
- every frame containing a metal (Li, Na, K, Mg, Ca) forced in;
- the rest chosen by widest gap to the most isolated fragment.

Every 10th frame goes to validation. The script prints per-element coverage and the charge, multiplicity and size mixes; read them before launching.

```sh
.venv/bin/python longrange/make_wide20k.py --pool omol25_full60 --spice --max-abs-q 2 \
    --force-elements Li,Na,K,Mg,Ca --out omol25_spice20k
```

Output: `data/omol25_spice20k_{train,valid}.xyz`. Gzipped copies are in git (`*.xyz.gz`); gunzip them to use.

Round 1 (`omol25_wide20k`: all 83 elements, any charge, `average` E0s) gave a validation RMSE_E of about 240 meV/atom. The error was dominated by tiny, highly charged clusters of rare elements. Don't repeat it.

## 2. Atom reference energies (E0s)

Isolated neutral atoms are computed with ORCA on the laptop at the OMol25 level: ωB97M-V/def2-TZVPD, ground-state multiplicity. The runner is copied from `moltui/surface_gen_scripts/run_opts_seq.sh`.

```sh
.venv/bin/python longrange/atom_e0s.py make
caffeinate -i bash longrange/atom_e0s/run_atoms_seq.sh
.venv/bin/python longrange/atom_e0s.py collect     # -> longrange/atom_e0s/e0s_wb97mv_tzvpd.json
```

The JSON has Hartree and eV dicts; the launcher uses the `ev` dict, keyed by Z. ORCA 6.1.1 is at `~/opt/orca-6.1.1/orca`, with `~/opt/openmpi-4.1.8/bin` on PATH. ORCA's input must come from `/dev/null` in any loop, because its mpirun swallows stdin. Delete ORCA scratch files (`.gbw`, `.densities`, `.tmp` and so on) after a run; keep only `.inp`, `.xyz` and `.out`.

## 3. Launch on Modal

- **Entrypoint:** `code/modal_finetune.py::scratch`, which runs the `train_scratch` function on an L4 GPU.
- **Launcher:** `code/run_modal_scratch.sh`. It's tracked in git (force-added, since `.git/info/exclude` ignores the other `code/run_*.sh` launchers); its full text is reproduced below.

Usage:

```
zsh code/run_modal_scratch.sh <r_max> <epochs> <timeout_h> [name] [data] [extra flags]
```

The two runs, launched fully detached from the Claude session (the harness otherwise reaps long jobs):

```sh
cd /Users/cafierom/python_mac/mace
~/miniforge3/bin/python -c "import subprocess; subprocess.Popen(['zsh','code/run_modal_scratch.sh','6','15','2.5','omol_spice20k_r6','omol25_spice20k','--lr','1e-3','--energy-weight','10','--forces-weight','10','--e0s-json','/Users/cafierom/python_mac/mace/longrange/atom_e0s/e0s_wb97mv_tzvpd.json'], stdout=open('code/modal_spice20k_r6.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd='/Users/cafierom/python_mac/mace')"
~/miniforge3/bin/python -c "import subprocess; subprocess.Popen(['zsh','code/run_modal_scratch.sh','12','15','3.5','omol_spice20k_r12','omol25_spice20k','--lr','1e-3','--energy-weight','10','--forces-weight','10','--e0s-json','/Users/cafierom/python_mac/mace/longrange/atom_e0s/e0s_wb97mv_tzvpd.json'], stdout=open('code/modal_spice20k_r12.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd='/Users/cafierom/python_mac/mace')"
```

Check with `ps -ax -o pid,ppid,sess,command | grep run_modal_scratch`: the launcher should show PPID 1.

Settings:

| Setting | Value |
|---|---|
| Model | ScaleShiftMACE, 128 channels, max_L 1, 2 interactions, correlation 3, max_ell 3 |
| r_max | 6.0 or 12.0 Å |
| Embeddings | total_charge categorical (201 classes, offset 100, dim 64); total_spin categorical (101 classes, dim 64) |
| E0s | fixed, from `e0s_wb97mv_tzvpd.json` (eV) |
| Optimiser | lr 1e-3, amsgrad, weight decay 1e-8, ReduceLROnPlateau (patience 25), EMA 0.99 |
| Loss weights | energy 10, forces 10 (1:1). Stage-two weighting is for any formal run, not this PoC. |
| Batch | 16 (train and valid) |
| Epochs | 15; validation every epoch |
| Precision / GPU | float32 / L4 |
| Timeout (cost cap) | 2.5 h for r6, 3.5 h for r12 |

The resulting `mace_run_train` command line is printed near the top of each launcher log (`code/modal_spice20k_r{6,12}.log`).

The launcher does the following:

1. Starts `caffeinate -w $$`.
2. Runs `modal run --detach ...::scratch`. `--detach` keeps the Modal app alive if the laptop client dies.
3. When it exits (success, failure or timeout), runs `modal volume get --force mace-ft scratch/<name> runs/scratch/`.

The entrypoint uploads `data/<data>_{train,valid}.xyz` to the volume the first time it sees them.

`code/run_modal_scratch.sh` in full:

```sh
#!/bin/zsh
caffeinate -w $$ &
cd /Users/cafierom/python_mac/mace
R=${1:?r_max}; EP=${2:?epochs}; TH=${3:?timeout_h}
DATA=${5:-omol25_wide20k}; NAME=${4:-omol_${DATA}_r${R}}
echo "=== modal scratch r_max=$R epochs=$EP timeout=${TH}h name=$NAME data=$DATA start: $(date) ==="
modal run --detach code/modal_finetune.py::scratch \
  --r-max $R --epochs $EP --timeout-h $TH --name $NAME --data $DATA "${@:6}"
echo "=== modal run exited ($?): $(date) -- pulling scratch/$NAME ==="
mkdir -p runs/scratch
modal volume get --force mace-ft scratch/$NAME runs/scratch/
ls -la runs/scratch/$NAME runs/scratch/$NAME/* 2>&1 | head -40
echo "=== modal scratch $NAME done: $(date) ==="
```

## 4. Persistence

Everything is written to the `mace-ft` volume under `scratch/<name>/`: the best checkpoint, the final `.model`, logs and results. `train_scratch` passes `--work_dir`, `--model_dir` and `--log_dir`, all on the volume. MACE's defaults point at the container's working directory, which dies with the container; that is how checkpoints were lost before. A background thread commits the volume every 300 s, so a killed container loses at most 5 minutes of work. MACE keeps only the best checkpoint, which is intended.

## 5. Monitoring and cost

- **Per-epoch table:** `.venv/bin/python longrange/epoch_table.py --spice` prints validation loss, RMSE_E and RMSE_F for each epoch. Report it at each check-in.
- **Cost:**
  - `modal billing report --for today --json` gives per-app cost;
  - `modal billing summary` gives the month against credits;
  - L4 all-in cost is about $0.89/h.
- **Measured epoch times:** about 8.1 min for r6 and 12.2 min for r12 (121.7 and 183.1 min for 15 epochs). The two spice20k runs cost about $4.46 together.

## 6. After the run

The launcher pulls the run to `runs/scratch/<name>/`. If the client died, pull it by hand:

```sh
modal volume get --force mace-ft scratch/<name> runs/scratch/
```

The `.model` was saved on CUDA and won't load on a CPU-only machine: e3nn's compiled pieces call `torch.jit.load` with no map_location. Make a CPU copy, which loads everywhere:

```sh
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/to_cpu.py runs/scratch/<name>/<name>.model
```

Use the `.cpu.model` for evaluation. Both are in git under `runs/scratch/omol_spice20k_r{6,12}/`. The `.pt` checkpoint is only for resuming training.

## 7. Resuming or changing the learning rate

`--restart_latest` is always on:

- **To resume,** re-run the launcher with the same name and a larger `epochs`. `epochs` is the total maximum, because MACE restarts at the checkpoint's epoch index.
- **The LR on a resume:** a resume restores the optimizer and scheduler state, so `--lr` is ignored.
- **To change the LR,** fork the checkpoint under a new name first:

  ```sh
  KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/fork_checkpoint.py <src_name> <new_name> <lr>
  zsh code/run_modal_scratch.sh <r_max> <total_epochs> <timeout_h> <new_name> omol25_spice20k --energy-weight 10 --forces-weight 10 --e0s-json ...
  ```

  Loss weights aren't stored in the checkpoint, so pass them again.

## 8. Evaluation

All evaluation runs locally on the CPU models. Use 2 threads while ORCA is running (the laptop has 6 cores; ORCA uses 4).

- `longrange/frame_errors.py <model> <label> omol25_spice20k`: per-frame validation errors by charge, multiplicity, element and size.
- `longrange/eval_cafquant.py <model> <label>`: cafquant's charge-separation sweep and DFT spot checks.
- `longrange/scan/make_scans.py`, `run_scan_seq.sh` and `eval_scans.py`: rigid distance scans against ORCA. Results are in `longrange/SCAN_NOTES.md`.

## 9. PolarMACE from scratch (2026-11-08)

Results: `SCAN_NOTES_polar.md`. Same 20k frames, E0s, lr 1e-3 and 10:10 weights as above; `--polar` selects PolarMACE with MACE-POLAR-1's electrostatics settings (`POLAR_ARGS` in `code/modal_finetune.py`) and no charge/spin embeddings.

- **Centre the data first.** mace forges a vacuum box (max|pos|+1)·5·r_max per non-periodic frame and PolarMACE's k-grid scales with its cube. `data/omol25_spice20k_c_{train,valid}.xyz` are the spice20k frames shifted to their centroids (energies and forces unchanged): `a.positions -= a.positions.mean(0)` for every frame.
- **k-grid patch.** Even centred, the unused k-grid ran out of GPU memory. For non-periodic frames PolarMACE takes the real-space path and never reads it, so `train_scratch_polar` shrinks it to k = 0 before calling mace's `run_train` (valid only for non-periodic data). `eval_scans.py` and `frame_errors.py` apply the same patch.
- **Image.** `polar_image` copies `graph_longrange` 0.4.0 and `nvalchemiops` 0.5.0 from cafquant's venv (not on PyPI at these versions) and sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- **Seed.** `--seed` (default 123). The 6 Å seed-123 run was unstable and did not recover at lr 3e-4; seed 7 converged. If a POLAR run spikes by 10–1000× in its first epochs, try another seed before changing the lr.
- **Launch** (detached as in step 3):

  ```sh
  zsh code/run_modal_scratch.sh 6 20 5 omol_spice20k_polar_r6_s7 omol25_spice20k_c --lr 1e-3 --energy-weight 10 --forces-weight 10 --e0s-json <e0s.json> --polar --seed 7
  zsh code/run_modal_scratch.sh 12 20 12 omol_spice20k_polar_r12 omol25_spice20k_c --lr 1e-3 --energy-weight 10 --forces-weight 10 --e0s-json <e0s.json> --polar
  ```

- **Measured:** 13.5 min/epoch (6 Å), 18.1 min/epoch (12 Å) on the L4.
- **Local evaluation** needs `graph_longrange`: use `/Users/cafierom/python_mac/cafquant/.venv/bin/python` for `to_cpu.py`, `eval_scans.py polar|polar1` and `frame_errors.py --polar`. Run one at a time on this laptop.
- **Monitoring:** `.venv/bin/python longrange/epoch_table.py --polar`.

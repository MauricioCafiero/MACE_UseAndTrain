# Protein binding-site benchmark (cafquant MBE) for the 6 Å vs 12 Å models

The four long-range PoC models run through cafquant's many-body benchmark: sult, ddc, comt and aldh active sites, 760 frames (ligand–residue pairs, residue–residue pairs, ligand + two-residue triples, complex/protein/ligand totals). The question is the **6 Å vs 12 Å difference within each model family**; the legacy ωB97XD sums from cafquant's `mbe/RESULTS.md` are the common yardstick.

| Method | Model |
|---|---|
| `lr-mace6` | plain charge/spin MACE, r_max 6, 30 epochs (`omol_spice20k_r6_ep30`) |
| `lr-mace12` | plain charge/spin MACE, r_max 12, 30 epochs (`omol_spice20k_r12_ep30`) |
| `lr-polar6` | PolarMACE, r_max 6, seed 7, epoch 17 (`omol_spice20k_polar_r6_s7`) |
| `lr-polar12` | PolarMACE, r_max 12, epoch 18 (`omol_spice20k_polar_r12`) |

## Files

- `run_lr_mbe.sh`: driver. Runs cafquant's `mbe/pipeline.py run <system> --method <m>` for each model and system with cafquant's `.venv/bin/python`, copies each `sp_<m>.json` here, then runs `pipeline.py assemble` and `table`.
- `out/<system>/sp_lr-*.json`: per-frame energies; `out/<system>/assembly.json`: cafquant's assembled sums (all backends).
- `table_all.txt`: `pipeline.py table` output (total / 2B / 2B+3B for every backend and system).

## Changes made in cafquant (2026-11-08, uncommitted there)

- `mbe/pipeline.py`: `lr-mace6`, `lr-mace12`, `lr-polar6`, `lr-polar12` added to `METHODS`.
- `code/cafquant/mlips.py`: one `make_calculator` branch that loads the four `.cpu.model` files from `../mace/runs/scratch/`. The PolarMACE models get `charges_key="charge"` and the non-periodic k-grid shrink (below).
- `dft_check/legacy_reference/score_backends.py`: the four names added to its `METHODS`; its `scores.json` was backed up before any re-run.

## k-grid: needed at inference too

PolarMACE builds a k-space grid from each frame's cell even for non-periodic frames, where it is never read. On these protein frames (up to 313 atoms, |pos| ≤ 13 Å) the 12 Å model without the shrink swapped the 8 GB laptop to a standstill (sult not finished after 17 min, 9 GB swap); with it, sult took 82 s. Scaling cafquant's probe number (13.8 GiB at a 2,880 Å box) predicted only 0.3 GiB here, which was wrong. Always apply the shrink for PolarMACE at r_max 12 on non-periodic frames.

## Timing (CPU, 4 threads)

Plain MACE about 3–4 min per model for all four systems; PolarMACE 6 Å about 2–4 min per system; PolarMACE 12 Å (patched) 1.5–5 min per system.

## Result (2026-11-08)

Total binding energy, kcal/mol (legacy ωB97XD from cafquant `mbe/RESULTS.md`):

| System | DFT | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|---|
| sult | −109.5 | −100.1 | +175.7 | −90.0 | −209.1 |
| ddc | −77.7 | −175.0 | −670.6 | −267.4 | −508.4 |
| comt | −172.1 | −261.6 | −266.5 | −301.2 | −408.2 |
| aldh | −50.4 | −64.6 | +233.2 | −60.5 | −174.5 |

12 Å is worse than 6 Å in every system for both families, including the all-neutral aldh. Likely cause: training frames have ≤ 60 atoms, so 12 Å neighbourhoods inside 200–300-atom clusters are far outside the training distribution, while 6 Å neighbourhoods are not. 2B and 2B+3B for all backends: `table_all.txt` (2B+3B does not converge for the charged systems, per cafquant).

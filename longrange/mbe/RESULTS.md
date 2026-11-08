# 6 Å vs 12 Å on the protein binding sites

Snapshot of cafquant `pipeline.py table` for the four long-range PoC backends (2026-11-08), with the legacy ωB97XD sums from cafquant `mbe/RESULTS.md`. All energies kcal/mol; net 3B = (2B+3B) − 2B. Backends: lr-mace6/12 = plain charge/spin MACE r_max 6/12 (30 epochs); lr-polar6/12 = from-scratch PolarMACE r_max 6 (seed 7)/12. All four trained on the same 20k OMol25 frames.

| system | method | total | 2B | 2B+3B | net 3B |
|---|---|---|---|---|---|
| sult | legacy DFT | -109.52 | -121.43 | -118.44 | 2.99 |
| sult | lr-mace6 | -100.12 | 87.18 | -199.89 | -287.07 |
| sult | lr-mace12 | 175.71 | -112.42 | -806.33 | -693.91 |
| sult | lr-polar6 | -90.03 | -101.93 | 102.89 | 204.82 |
| sult | lr-polar12 | -209.08 | -194.39 | -18.49 | 175.90 |
| ddc | legacy DFT | -77.74 | -83.09 | -194.88 | -111.79 |
| ddc | lr-mace6 | -174.95 | -169.97 | -973.88 | -803.91 |
| ddc | lr-mace12 | -670.64 | -401.02 | -1132.23 | -731.21 |
| ddc | lr-polar6 | -267.36 | -431.25 | 347.17 | 778.42 |
| ddc | lr-polar12 | -508.41 | -584.16 | 86.72 | 670.88 |
| comt | legacy DFT | -172.12 | -290.71 | 218.22 | 508.93 |
| comt | lr-mace6 | -261.62 | -223.43 | 1132.57 | 1356.00 |
| comt | lr-mace12 | -266.53 | -285.98 | -2536.14 | -2250.16 |
| comt | lr-polar6 | -301.19 | -490.91 | 104.17 | 595.08 |
| comt | lr-polar12 | -408.19 | -644.86 | 543.36 | 1188.22 |
| aldh | legacy DFT | -50.38 | -51.09 | — | — |
| aldh | lr-mace6 | -64.64 | -79.22 | -65.90 | 13.32 |
| aldh | lr-mace12 | 233.18 | -140.85 | 246.45 | 387.30 |
| aldh | lr-polar6 | -60.50 | -48.43 | -80.23 | -31.80 |
| aldh | lr-polar12 | -174.49 | -154.15 | -226.98 | -72.83 |

Read:

- **Total: 6 Å is closer to DFT than 12 Å in every system, for both
  families** (lr-mace: errors +9 / −97 / −90 / −14 at 6 Å against
  +285 / −593 / −94 / +284 at 12 Å; lr-polar: +20 / −190 / −129 / −10
  against −100 / −431 / −236 / −124). lr-mace12 has the wrong sign on
  sult and aldh.
- **2B: 6 Å closer in 6 of 8.** The exceptions are both lr-mace (sult
  −112.42 and comt −285.98 at 12 Å, within 9 and 5 of DFT).
- **2B+3B / net 3B** is far off for every backend; per cafquant the MBE
  truncated at 3B does not converge for the charged systems even with DFT.
- **Likely cause of the 12 Å failures:** training frames have ≤ 60 atoms,
  so 12 Å neighbourhoods inside these 200–313-atom clusters are far
  outside the training distribution (the all-neutral aldh fails too);
  6 Å neighbourhoods are not. This is a limit of these small-molecule
  models, not of a 12 Å cutoff in principle.

Per-frame energies: `out/<system>/sp_lr-*.json`; every backend: `table_all.txt`.

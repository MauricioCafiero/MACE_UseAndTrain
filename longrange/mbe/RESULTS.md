# Protein binding sites: 6 Å vs 12 Å

Total, 2-body (2B) and 2-body + 3-body (2B+3B) interaction energies for the four active sites, kcal/mol. DFT is the legacy ωB97XD reference from cafquant `mbe/RESULTS.md`. Models: plain charge/spin MACE (30 epochs) and from-scratch PolarMACE, each at r_max 6 and 12 Å, trained on the same 20k OMol25 frames.

## Total

| System | DFT | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|---|
| sult | −109.5 | −100.1 | +175.7 | −90.0 | −209.1 |
| ddc | −77.7 | −174.9 | −670.6 | −267.4 | −508.4 |
| comt | −172.1 | −261.6 | −266.5 | −301.2 | −408.2 |
| aldh | −50.4 | −64.6 | +233.2 | −60.5 | −174.5 |

Error vs DFT (model − DFT); the smaller of each 6/12 pair in bold:

| System | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|
| sult | **+9.4** | +285.2 | **+19.5** | −99.6 |
| ddc | **−97.2** | −592.9 | **−189.6** | −430.7 |
| comt | **−89.5** | −94.4 | **−129.1** | −236.1 |
| aldh | **−14.3** | +283.6 | **−10.1** | −124.1 |

## 2B

| System | DFT | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|---|
| sult | −121.4 | +87.2 | −112.4 | −101.9 | −194.4 |
| ddc | −83.1 | −170.0 | −401.0 | −431.2 | −584.2 |
| comt | −290.7 | −223.4 | −286.0 | −490.9 | −644.9 |
| aldh | −51.1 | −79.2 | −140.8 | −48.4 | −154.2 |

Error vs DFT (model − DFT); the smaller of each 6/12 pair in bold:

| System | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|
| sult | +208.6 | **+9.0** | **+19.5** | −73.0 |
| ddc | **−86.9** | −317.9 | **−348.2** | −501.1 |
| comt | +67.3 | **+4.7** | **−200.2** | −354.2 |
| aldh | **−28.1** | −89.8 | **+2.7** | −103.1 |

## 2B+3B

| System | DFT | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|---|
| sult | −118.4 | −199.9 | −806.3 | +102.9 | −18.5 |
| ddc | −194.9 | −973.9 | −1132.2 | +347.2 | +86.7 |
| comt | +218.2 | +1132.6 | −2536.1 | +104.2 | +543.4 |
| aldh | — | −65.9 | +246.4 | −80.2 | −227.0 |

Error vs DFT (model − DFT); the smaller of each 6/12 pair in bold:

| System | MACE 6 Å | MACE 12 Å | POLAR 6 Å | POLAR 12 Å |
|---|---|---|---|---|
| sult | **−81.4** | −687.9 | +221.3 | **+100.0** |
| ddc | **−779.0** | −937.4 | +542.0 | **+281.6** |
| comt | **+914.3** | −2754.4 | **−114.0** | +325.1 |

2B+3B does not converge for the charged systems even at the DFT level (cafquant `mbe/RESULTS.md`); compare it with care. Per-frame energies: `out/<system>/sp_lr-*.json`; full table for every backend: `table_all.txt`.

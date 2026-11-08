"""Per-epoch validation loss/errors for the scratch runs, from the launcher logs.

Continuation runs are appended after their parent run so the history reads
as one curve. Note: the lr1e-3 continuations use energy_weight 10 (was 1),
so their loss is on a different scale from the first 10 epochs; compare
RMSE_E / RMSE_F across the join, not loss. Columns: valid loss, RMSE_E
(meV/atom), RMSE_F (meV/A); * marks continuation epochs.

    .venv/bin/python longrange/epoch_table.py            # wide20k runs
    .venv/bin/python longrange/epoch_table.py --spice    # spice20k runs
    .venv/bin/python longrange/epoch_table.py --polar    # PolarMACE runs
"""
import re
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
RUNS = {"6 A": ["modal_wide20k_r6.log", "modal_wide20k_r6_lr1e-3.log"],
        "12 A": ["modal_wide20k_r12.log", "modal_wide20k_r12_lr1e-3.log"]}
# spice20k: SPICE elements, |q|<=2, metals forced in, ORCA E0s, lr1e-3, e:f 10:10
SPICE_RUNS = {"6 A spice20k": ["modal_spice20k_r6.log", "modal_spice20k_r6_ep30.log"],
              "12 A spice20k": ["modal_spice20k_r12.log", "modal_spice20k_r12_ep30.log"]}
# PolarMACE from scratch on the centred spice20k (omol25_spice20k_c), 2026-11-08
POLAR_RUNS = {"6 A polar": ["modal_spice20k_polar_r6.log", "modal_spice20k_polar_r6_lr3e-4.log"],
              "12 A polar": ["modal_spice20k_polar_r12.log"],
              "6 A polar seed 7": ["modal_spice20k_polar_r6_s7.log"]}
import sys
if "--spice" in sys.argv:
    RUNS = SPICE_RUNS
if "--polar" in sys.argv:
    RUNS = POLAR_RUNS
PAT = re.compile(r"(\d\d:\d\d:\d\d)\.\d+ INFO: (Initial|Epoch (\d+)): head: Default, "
                 r"loss=([\d.]+), RMSE_E_per_atom=\s*([\d.]+) meV, RMSE_F=\s*([\d.]+) meV")

# --polar: a continuation forked from an earlier checkpoint REPLACES the parent's
# epochs from its restart epoch on (the parent's later epochs were abandoned),
# and is printed without the * mark.
REPLACE = "--polar" in sys.argv

for cut, logs in RUNS.items():
    print(f"\n{cut}  epoch | loss   | RMSE_E | RMSE_F")
    rows = []                                 # (epoch or None, mark, loss, E, F)
    for i, log in enumerate(logs):
        p = CODE / log
        if not p.exists():
            continue
        new = []
        for m in PAT.finditer(p.read_text(errors="replace")):
            t, tag, ep, loss, re_, rf = m.groups()
            if tag == "Initial" and i > 0:
                continue                      # resume re-evaluates the parent's best
            mark = "*" if i > 0 and not REPLACE else " "   # * = continuation
            new.append((None if ep is None else int(ep), mark, loss, re_, rf))
        if REPLACE and i > 0 and new:
            rows = [r for r in rows if r[0] is None or r[0] < new[0][0]]
        rows += new
    for ep, mark, loss, re_, rf in rows:
        print(f"{'':5}{('init' if ep is None else ep):>5}{mark}| {float(loss):.4f} | "
              f"{float(re_):6.1f} | {float(rf):6.1f}")

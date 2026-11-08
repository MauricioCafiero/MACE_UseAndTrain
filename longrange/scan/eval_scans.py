"""Score the 6 vs 12 A models on the distance scans against the ORCA reference.

For each pair: dE(d) = E(pair) - E(A) - E(B) in kcal/mol for DFT (ORCA
wB97M-V/def2-TZVPD) and each model. Curves are also shifted to zero at the
farthest point (20 A) so only the SHAPE is compared -- that removes the
models' known constant charge-state offset. Reports, per pair:
  * shifted dE at selected gaps (DFT, r6, r12)
  * fraction of the DFT change between d and 20 A that each model recovers
  * the gap beyond which each model's curve is flat (< 0.1 kcal/mol change)
Writes longrange/scan/results/{scan_table.tsv,summary.txt,<pair>.png}.
Runs with 2 threads so it can share the laptop with ORCA (6 cores).

    OMP_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/scan/eval_scans.py
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from ase.io import read

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
KCAL_EH, KCAL_EV = 627.509474, 23.060548
MODELS = {"r6": REPO / "runs/scratch/omol_spice20k_r6/omol_spice20k_r6.cpu.model",
          "r12": REPO / "runs/scratch/omol_spice20k_r12/omol_spice20k_r12.cpu.model"}
PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
QM = re.compile(r"\*\s*xyzfile\s+(-?\d+)\s+(\d+)")


def dft(stem):
    o = HERE / "orca" / stem / f"{stem}.out"
    m = PAT.findall(o.read_text(errors="replace")) if o.exists() else []
    return float(m[-1]) * KCAL_EH if m else None


def charge(stem):
    return int(QM.search((HERE / "orca" / stem / f"{stem}.inp").read_text()).group(1))


def main():
    from mace.calculators import MACECalculator
    meta = json.loads((HERE / "meta.json").read_text())
    calcs = {k: MACECalculator(model_paths=[str(p)], device="cpu", default_dtype="float64")
             for k, p in MODELS.items()}
    cache = {}

    def mlip(stem, k):
        if (stem, k) not in cache:
            at = read(HERE / "frames" / f"{stem}.xyz")
            at.info["charge"], at.info["spin"] = charge(stem), 1
            at.calc = calcs[k]
            cache[(stem, k)] = float(at.get_potential_energy()) * KCAL_EV
        return cache[(stem, k)]

    rows = ["pair\ttype\tgap\td_com\tdft\tr6\tr12\tdft_shift\tr6_shift\tr12_shift"]
    summ = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for p in meta["pairs"]:
        gaps = np.array([pt["d_min"] for pt in p["points"]])
        eA = {"dft": dft(p["monoA"]), **{k: mlip(p["monoA"], k) for k in MODELS}}
        eB = {"dft": dft(p["monoB"]), **{k: mlip(p["monoB"], k) for k in MODELS}}
        cur = {}
        for k in ("dft", *MODELS):
            vals = []
            for pt in p["points"]:
                e = dft(pt["stem"]) if k == "dft" else mlip(pt["stem"], k)
                vals.append(None if e is None or eA[k] is None or eB[k] is None else e - eA[k] - eB[k])
            cur[k] = np.array([np.nan if v is None else v for v in vals])
        if np.isnan(cur["dft"]).any():
            summ.append(f"{p['label']}: DFT incomplete ({int(np.isnan(cur['dft']).sum())} points missing) -- skipped")
            continue
        sh = {k: v - v[-1] for k, v in cur.items()}
        for i, pt in enumerate(p["points"]):
            rows.append(f"{p['label']}\t{p['type']}\t{gaps[i]:.2f}\t{pt['d_com']:.2f}\t"
                        + "\t".join(f"{cur[k][i]:+.3f}" for k in ("dft", *MODELS)) + "\t"
                        + "\t".join(f"{sh[k][i]:+.3f}" for k in ("dft", *MODELS)))
        line = [f"\n{p['label']}  [{p['type']}]  q={p['qA']:+d}/{p['qB']:+d}"]
        line.append("   gap  | DFT shifted | r6 shifted | r12 shifted | r6 recov | r12 recov")
        for g in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0):
            i = int(np.argmin(abs(gaps - g)))
            d = sh["dft"][i]
            rec = {k: (sh[k][i] / d if abs(d) > 0.05 else np.nan) for k in MODELS}
            line.append(f"  {gaps[i]:5.1f} | {d:+10.2f} | {sh['r6'][i]:+10.2f} | {sh['r12'][i]:+11.2f} | "
                        f"{rec['r6']:8.2f} | {rec['r12']:8.2f}")
        for k in MODELS:
            moving = np.where(np.abs(np.diff(cur[k])) > 0.1)[0]
            line.append(f"  {k}: curve flat beyond {gaps[moving[-1] + 1] if len(moving) else gaps[0]:.1f} A; "
                        f"raw dE at 20 A {cur[k][-1]:+.2f} (DFT {cur['dft'][-1]:+.2f}) kcal/mol")
        summ += line
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for k, st in (("dft", "k-o"), ("r6", "C0-s"), ("r12", "C1-^")):
            ax.plot(gaps, sh[k], st, ms=3, lw=1.2, label={"dft": "DFT", "r6": "MACE r=6", "r12": "MACE r=12"}[k])
        for c in (6, 12):
            ax.axvline(c, color="0.7", ls=":", lw=1)
        ax.set_xlabel("min interfragment gap (A)"); ax.set_ylabel("dE - dE(20 A) (kcal/mol)")
        ax.set_title(f"{p['label']}  ({p['type']})", fontsize=9); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"{p['label'].replace('+', 'p').replace('-', 'm')}.png", dpi=130)
        plt.close(fig)
    (OUT / "scan_table.tsv").write_text("\n".join(rows) + "\n")
    (OUT / "summary.txt").write_text("\n".join(summ) + "\n")
    print("\n".join(summ))
    print(f"\n[written to {OUT}]")


if __name__ == "__main__":
    main()

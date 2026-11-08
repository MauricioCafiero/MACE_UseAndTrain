"""Score two MLIPs (by default the 6 vs 12 A models) on the distance scans against ORCA.

For each pair: dE(d) = E(pair) - E(A) - E(B) in kcal/mol for DFT (ORCA
wB97M-V/def2-TZVPD) and each model. Curves are also shifted to zero at the
farthest point (20 A) so only the SHAPE is compared -- that removes the
models' known constant charge-state offset. Reports, per pair:
  * shifted dE at selected gaps (DFT and the two models)
  * fraction of the DFT change between d and 20 A that each model recovers
  * the gap beyond which each model's curve is flat (< 0.1 kcal/mol change)
Writes longrange/scan/results[_<tag>]/{scan_table.tsv,summary.txt,<pair>.png}.
Runs with 2 threads so it can share the laptop with ORCA (6 cores).

    OMP_NUM_THREADS=2 KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/scan/eval_scans.py
    ... eval_scans.py ep30     # models omol_spice20k_r{6,12}_ep30, output results_ep30/
    ... eval_scans.py polar1   # pretrained MACE-POLAR-1 s/m, output results_polar1/
    ... eval_scans.py polar    # from-scratch PolarMACE r{6,12} (omol_spice20k_polar_r*)
The polar tags need graph_longrange, which this repo's .venv lacks: run them with
/Users/cafierom/python_mac/cafquant/.venv/bin/python.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else ""        # e.g. "ep30" -> *_ep30 models
SUF = f"_{TAG}" if TAG else ""
OUT = HERE / f"results{SUF}"
OUT.mkdir(exist_ok=True)
KCAL_EH, KCAL_EV = 627.509474, 23.060548
# key -> (model path, plot label); polar models read the total charge via charges_key
if TAG == "polar1":
    _C = Path.home() / ".cache" / "mace"
    MODELS = {"polar1s": (_C / "MACEPOLAR1Smodel", "MACE-POLAR-1-s"),
              "polar1m": (_C / "MACEPOLAR1Mmodel", "MACE-POLAR-1-m")}
elif TAG == "polar":
    # final from-scratch PolarMACE runs: 6 A = seed 7 (seed 123 was unstable), 12 A = seed 123
    MODELS = {f"pr{r}": (REPO / f"runs/scratch/omol_spice20k_polar_{n}/omol_spice20k_polar_{n}.cpu.model",
                         f"PolarMACE r={r}") for r, n in ((6, "r6_s7"), (12, "r12"))}
else:
    MODELS = {f"r{r}": (REPO / f"runs/scratch/omol_spice20k_r{r}{SUF}/omol_spice20k_r{r}{SUF}.cpu.model",
                        f"MACE r={r}") for r in (6, 12)}
K1, K2 = MODELS
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
    kw = {}
    if TAG.startswith("polar"):
        kw = {"charges_key": "charge"}
        # scan frames are non-periodic: PolarMACE takes the realspace path and never
        # reads its k-grid, which mace sizes from a forged (max|pos|+1)*5*r_max box
        import mace.modules.extensions as ext
        _kv = ext.compute_k_vectors_flat
        ext.compute_k_vectors_flat = lambda cut, cell, rcell: _kv(1e-6, cell, rcell)
    calcs = {k: MACECalculator(model_paths=[str(p)], device="cpu", default_dtype="float64", **kw)
             for k, (p, _) in MODELS.items()}
    cache = {}

    def mlip(stem, k):
        if (stem, k) not in cache:
            at = read(HERE / "frames" / f"{stem}.xyz")
            at.info["charge"], at.info["spin"] = charge(stem), 1
            at.calc = calcs[k]
            cache[(stem, k)] = float(at.get_potential_energy()) * KCAL_EV
        return cache[(stem, k)]

    rows = [f"pair\ttype\tgap\td_com\tdft\t{K1}\t{K2}\tdft_shift\t{K1}_shift\t{K2}_shift"]
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
        line.append(f"   gap  | DFT shifted | {K1} shifted | {K2} shifted | {K1} recov | {K2} recov")
        for g in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0):
            i = int(np.argmin(abs(gaps - g)))
            d = sh["dft"][i]
            rec = {k: (sh[k][i] / d if abs(d) > 0.05 else np.nan) for k in MODELS}
            line.append(f"  {gaps[i]:5.1f} | {d:+10.2f} | {sh[K1][i]:+10.2f} | {sh[K2][i]:+11.2f} | "
                        f"{rec[K1]:8.2f} | {rec[K2]:8.2f}")
        for k in MODELS:
            moving = np.where(np.abs(np.diff(cur[k])) > 0.1)[0]
            line.append(f"  {k}: curve flat beyond {gaps[moving[-1] + 1] if len(moving) else gaps[0]:.1f} A; "
                        f"raw dE at 20 A {cur[k][-1]:+.2f} (DFT {cur['dft'][-1]:+.2f}) kcal/mol")
        summ += line
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for k, st, lab in (("dft", "k-o", "DFT"), (K1, "C0-s", MODELS[K1][1]), (K2, "C1-^", MODELS[K2][1])):
            ax.plot(gaps, sh[k], st, ms=3, lw=1.2, label=lab)
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

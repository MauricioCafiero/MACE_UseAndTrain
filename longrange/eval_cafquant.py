"""Score a charge/spin-aware MACE model on cafquant's long-range test sets.

Two tests, both read (never written) from ../cafquant/dft_check:

* sweep/  -- rigid charge-separation sweep, Lys+/Asp- ("organic", pair q=0)
  and Mg(w)2+/Glu- ("mg"), 2.4 -> 24 A. dE(d) = E_pair - E_cat - E_an in
  kcal/mol, printed next to the point-charge reference and the backends
  already in sweep/sweep_compare.tsv.
* spot checks -- the six DFT-arbitrated delta3 triples (wB97M-V/def2-TZVP
  ORCA outputs) plus their three pair 2B interaction energies. Triple
  definitions and the ORCA energy reader are imported from dft_check/eval.py.

Charge and multiplicity go in atoms.info["charge"] / ["spin"], which
MACECalculator maps onto the model's total_charge / total_spin embeddings.
Outputs go to longrange/eval/<label>/ (energies JSON + tables).

    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/eval_cafquant.py \
        runs/scratch/omol_omol25_wide20k_r6/omol_omol25_wide20k_r6.model r6
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DFT = REPO.parent / "cafquant" / "dft_check"
KCAL = 23.060548          # eV -> kcal/mol
EH_KCAL = 627.509474


def main(model_path: str, label: str) -> None:
    from ase.io import read
    from mace.calculators import MACECalculator

    out = REPO / "longrange" / "eval" / label
    out.mkdir(parents=True, exist_ok=True)
    calc = MACECalculator(model_paths=[model_path], device="cpu",
                          default_dtype="float64")

    def energy(atoms, q, m):
        atoms.info["charge"], atoms.info["spin"] = q, m
        atoms.calc = calc
        return float(atoms.get_potential_energy())

    # ---- sweep -----------------------------------------------------------
    sw = DFT / "sweep"
    frames = read(sw / "frames.xyz", index=":", format="xyz")
    ch = json.loads((sw / "charges.json").read_text())
    meta = json.loads((sw / "meta.json").read_text())
    E = [energy(a, q, m) for a, q, m in zip(frames, ch["charges"], ch["mults"])]
    (out / "sweep_energies.json").write_text(json.dumps({"model": model_path,
                                                         "energies_eV": E}))
    ref = {}
    with open(sw / "sweep_compare.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ref[(r["set"], f"{float(r['d_min']):.2f}")] = r
    cols = ["E_point", "omol", "uma1p2", "polar1-m", "gfn2"]
    lines = ["set\td_min\t" + label + "\t" + "\t".join(cols)]
    base = 0
    for name, spec in meta["sets"].items():
        cat, an = spec["monomer_frames"]
        for i, p in enumerate(spec["pairs"]):
            de = (E[base + 2 + i] - E[cat] - E[an]) * KCAL
            r = ref.get((name, f"{p['d_min']:.2f}"), {})
            lines.append(f"{name}\t{p['d_min']:.2f}\t{de:+.2f}\t"
                         + "\t".join(r.get(c, "") for c in cols))
        base += 2 + len(spec["pairs"])
    (out / "sweep_compare.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- DFT spot checks --------------------------------------------------
    spec = importlib.util.spec_from_file_location("cafquant_dft_eval", DFT / "eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    qm = re.compile(r"\*\s*xyzfile\s+(-?\d+)\s+(\d+)")
    cache: dict[str, tuple[float, float | None]] = {}

    def both(stem):                       # (MLIP eV, DFT Eh) for one frame
        if stem not in cache:
            d = DFT / stem.split("_")[0]
            q, m = map(int, qm.search((d / f"{stem}.inp").read_text()).groups())
            o = d / f"{stem}.out"
            cache[stem] = (energy(read(d / f"{stem}.xyz"), q, m),
                           ev.final_energy_eh(o) if o.exists() else None)
        return cache[stem]

    rows = ["triple\tterm\tdft\t" + label]
    for tname, parts in ev.TRIPLES.items():
        e = {role: both(stem) for role, stem in parts.items()}
        if any(v[1] is None for v in e.values()):
            rows.append(f"{tname}\t(missing DFT output)\t\t")
            continue
        terms = {"2B la": ("la", "l", "a"), "2B lb": ("lb", "l", "b"),
                 "2B ab": ("ab", "a", "b")}
        for t, (pr, x, y) in terms.items():
            dft = (e[pr][1] - e[x][1] - e[y][1]) * EH_KCAL
            mlp = (e[pr][0] - e[x][0] - e[y][0]) * KCAL
            rows.append(f"{tname}\t{t}\t{dft:+.2f}\t{mlp:+.2f}")
        s = {"lab": 1, "la": -1, "lb": -1, "ab": -1, "l": 1, "a": 1, "b": 1}
        dft3 = sum(s[k] * e[k][1] for k in s) * EH_KCAL
        mlp3 = sum(s[k] * e[k][0] for k in s) * KCAL
        rows.append(f"{tname}\tdelta3\t{dft3:+.2f}\t{mlp3:+.2f}")
    (out / "spotcheck.tsv").write_text("\n".join(rows) + "\n")
    (out / "spot_energies.json").write_text(json.dumps(
        {k: {"mlip_eV": v[0], "dft_Eh": v[1]} for k, v in cache.items()}, indent=1))
    print("\n".join(rows))
    print(f"[written to {out}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

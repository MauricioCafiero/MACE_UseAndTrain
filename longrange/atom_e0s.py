"""Isolated-atom reference energies (E0s) for the SPICE-element set, with ORCA.

Level matches OMol25: wB97M-V/def2-TZVPD, no RI/COSX approximations, tight
SCF, defgrid3. Neutral atoms in their ground-state multiplicity (UKS when
M > 1). Two steps, with the run done by run_atoms_seq.sh (copied from
moltui/surface_gen_scripts/run_opts_seq.sh, the proven local ORCA driver):

    .venv/bin/python longrange/atom_e0s.py make       # write <el>/<el>.inp
    caffeinate -i bash longrange/atom_e0s/run_atoms_seq.sh
    .venv/bin/python longrange/atom_e0s.py collect    # -> e0s_wb97mv_tzvpd.json

collect writes Hartree and eV values; the eV dict is what --E0s takes.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent / "atom_e0s"
EV = 27.211386245988
# SPICE 2 element set, ground-state spin multiplicity of the neutral atom
MULT = {"H": 2, "Li": 2, "B": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Na": 2,
        "Mg": 1, "Si": 3, "P": 4, "S": 3, "Cl": 2, "K": 2, "Ca": 1, "Br": 2, "I": 2}
Z = {"H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
     "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Br": 35, "I": 53}
PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def make(el: str) -> None:
    d = HERE / el.lower()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{el.lower()}.inp").write_text(
        "! wB97M-V def2-TZVPD NoRI NoCOSX TightSCF DEFGRID3 SlowConv\n"
        "%maxcore 2000\n"
        "%scf MaxIter 500 end\n"
        f"* xyz 0 {MULT[el]}\n{el} 0.0 0.0 0.0\n*\n")


def collect(el: str) -> float:
    out = HERE / el.lower() / f"{el.lower()}.out"
    m = PAT.findall(out.read_text(errors="replace")) if out.exists() else []
    if not m:
        raise RuntimeError(f"{el}: no FINAL SINGLE POINT ENERGY in {out}")
    return float(m[-1])


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "make":
        for el in MULT:
            make(el)
        (HERE / "jobs.txt").write_text("\n".join(e.lower() for e in MULT) + "\n")
        print("wrote", len(MULT), "inputs + jobs.txt in", HERE)
    else:
        res = {el: collect(el) for el in MULT}
        for el, eh in res.items():
            print(f"{el:2s} M={MULT[el]}  {eh:.10f} Eh  {eh * EV:.6f} eV")
        (HERE / "e0s_wb97mv_tzvpd.json").write_text(json.dumps({
            "level": "wB97M-V/def2-TZVPD NoRI NoCOSX TightSCF DEFGRID3 (ORCA 6.1.1), neutral ground-state atoms",
            "multiplicity": MULT,
            "hartree": {Z[e]: res[e] for e in MULT},
            "ev": {Z[e]: res[e] * EV for e in MULT}}, indent=1))
        print("written", HERE / "e0s_wb97mv_tzvpd.json")

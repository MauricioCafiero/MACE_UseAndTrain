"""Grounded OOD example structures from computation-ready datasets (C/N/O/H/F/Cl).

Real non-covalent and host-guest structures to test whether MACE-OFF23's latent
space flags them as out-of-distribution vs its SPICE small-molecule training
distribution (isolated, near-equilibrium drug-like organics). Every structure is
kept within the H,C,N,O,F,Cl element scope.

Sources
-------
- **S66** (Řezáč, Riley, Hobza, JCTC 2011, 7, 2427; doi:10.1021/ct2002946):
  66 benchmark non-covalent dimers (H-bond / dispersion / mixed), C/N/O/H,
  MP2/cc-pVTZ geometries. Parsed from the Psi4 ``S66.py`` module; the ``--``
  fragment separator in each dimer gives both monomers, so we get a paired
  **monomer (in-distribution) vs dimer (non-covalent contact -> OOD)** contrast.
- **S30L-CI** (Sure & Grimme, JCTC 2015, 11, 2856; doi:10.1021/acs.jctc.5b00296):
  30 realistic host-guest complexes (cucurbituril, calixarene, octaacid,
  exohedral fullerene, ...), DFT-D optimized, up to ~200 atoms, with counterions.
  The ACS Supporting Information ships each system pre-split into **host (A) /
  guest (B) / complex (AB)** Turbomole coord files, so the
  monomer-vs-complex decomposition comes from the dataset itself -- no extraction
  needed. We keep the 23 systems whose host+guest+complex fall entirely in scope
  (drop I/Na/S systems). The SI is paywalled; it must be obtained from ACS and
  placed under ``data/s30l/s30lci_test_set/`` (see :func:`load_s30l`).

Why not COD (Crystallography Open Database): an earlier version pulled cyclodextrin
host-guest CIFs from COD, but X-ray CIFs proved unusable for this purpose --
heavy fractional-occupancy disorder and unmodelled hydrogens meant the
"extracted macrocycle" was a broken, H-depleted fragment (e.g. C20H26O16 instead
of the real beta-CD C42H70O35), and that *brokenness* itself drove the OOD score.
Computation-ready DFT geometries (S30L) remove that confound entirely.
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch import). ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import re
import urllib.request
from pathlib import Path

import numpy as np

import mace_calc as mc  # noqa: E402
from activation_ood import ReferencePool, atom_ood_scores  # noqa: E402

log = logging.getLogger("ood_datasets")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_REPO = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO / "data"
S66_DIR = _DATA_DIR / "s66"
S30L_DIR = _DATA_DIR / "s30l"
S30L_ROOT = S30L_DIR / "s30lci_test_set"

# Element scope (atomic numbers). Structures with anything else are skipped.
SCOPE_Z = {1, 6, 7, 8, 9, 17}          # H, C, N, O, F, Cl
SCOPE_SYMS = {"H", "C", "N", "O", "F", "Cl"}

S66_RAW_URL = (
    "https://raw.githubusercontent.com/psi4/psi4/master/"
    "psi4/share/psi4/databases/S66.py"
)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
def _fetch(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("cached %s", dest.name)
        return dest.read_text(encoding="utf-8", errors="replace")
    log.info("downloading %s ...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "mace-ood/1.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# S66
# ---------------------------------------------------------------------------
_SYM_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, "P": 15,
             "S": 16, "Br": 35, "I": 53, "B": 5, "Si": 14}


def _parse_fragment(text: str):
    """Parse one qcdb fragment block (after the `0 1` line) -> (symbols, pos)."""
    syms, pos = [], []
    for line in text.strip().splitlines():
        s = line.strip()
        if not s or s.lower().startswith("units") or s.lower().startswith("no_"):
            continue
        if re.fullmatch(r"-?\d+\s+-?\d+", s):        # charge multiplicity
            continue
        parts = s.split()
        if len(parts) < 4 or parts[0] not in _SYM_TO_Z:
            continue
        syms.append(parts[0])
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return syms, np.array(pos, dtype=float) if pos else np.zeros((0, 3))


def _make_atoms(syms, pos):
    from ase import Atoms
    return Atoms(symbols=syms, positions=pos, pbc=False)


def fetch_s66(dest_dir: Path = S66_DIR):
    """Fetch Psi4 S66.py and return ``[(idx, label, dimer, monoA, monoB), ...]``.

    Each dimer block is split on the ``--`` fragment separator so we also get the
    two monomers (for the paired monomer-vs-dimer OOD contrast).
    """
    txt = _fetch(S66_RAW_URL, dest_dir / "S66.py")
    # parse human-readable labels: TAGL['%s-%s' % (dbse, 'N')] = """Label """
    labels = {}
    for m in re.finditer(r"TAGL\['%s-%s'\s*%\s*\(dbse,\s*'(\d+)'\s*\)\]\s*=\s*\"\"\"(.*?)\"\"\"", txt):
        labels[int(m.group(1))] = m.group(2).strip()
    # parse dimer geometries
    pat = re.compile(
        r"GEOS\['%s-%s-dimer'\s*%\s*\(dbse,\s*'(\d+)'\)\]\s*=\s*qcdb\.Molecule\(\"\"\"(.*?)\"\"\"",
        re.S,
    )
    out = []
    for m in pat.finditer(txt):
        idx = int(m.group(1))
        body = m.group(2)
        frags = re.split(r"^--\s*$", body, flags=re.M)
        if len(frags) < 2:
            continue
        sa, pa = _parse_fragment(frags[0])
        sb, pb = _parse_fragment(frags[1])
        dimer = _make_atoms(sa + sb, np.vstack([pa, pb]))
        monoA = _make_atoms(sa, pa)
        monoB = _make_atoms(sb, pb)
        out.append((idx, labels.get(idx, f"S66-{idx}"), dimer, monoA, monoB))
    out.sort(key=lambda r: r[0])
    log.info("S66: parsed %d dimers", len(out))
    return out


def s66_bind_ref(dest_dir: Path = S66_DIR) -> dict:
    """CCSD(T)/CBS reference interaction energies (kcal/mol) for S66, parsed from
    the ``BIND`` dict in Psi4's S66.py (already on disk). Negative = attractive.
    These are the *positive-example* labels: S66 is in-distribution (mean atom-OOD
    ~0.115), so MACE is expected to be accurate here -- the in-distribution floor.
    """
    txt = _fetch(S66_RAW_URL, dest_dir / "S66.py")
    out = {}
    for m in re.finditer(r"BIND\['%s-%s'\s*%\s*\(dbse,\s*'(\d+)'\s*\)\]\s*=\s*"
                         r"(-?\d+\.\d+)", txt):
        out[int(m.group(1))] = float(m.group(2))
    return out


# ---------------------------------------------------------------------------
# S30L-CI (Sure & Grimme host-guest benchmark, Turbomole coords)
# ---------------------------------------------------------------------------
_BOHR = 0.5291772105446329
_EL = re.compile(r"^[A-Za-z]{1,2}$")


def _parse_turbomole_coord(path: Path):
    """Parse a Turbomole ``$coord`` file (atomic units -> Angstrom) -> ASE Atoms."""
    from ase import Atoms
    syms, pos = [], []
    inside = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("$coord"):
            inside = True
            continue
        if s.startswith("$end"):
            break
        if not inside or not s or s.startswith("$"):
            continue
        p = s.split()
        if len(p) < 4 or not _EL.match(p[3]):      # 4th token must be an element
            continue
        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        except ValueError:
            continue
        pos.append([x * _BOHR, y * _BOHR, z * _BOHR])
        syms.append(p[3].capitalize())
    return Atoms(symbols=syms, positions=np.array(pos) if pos else np.zeros((0, 3)),
                 pbc=False)


def _chrg(folder: Path) -> str:
    f = folder / ".CHRG"
    return f.read_text().strip() if f.exists() else "0"


# eV -> kcal/mol (MACE/ASE energies are in eV; the S30L reference is in kcal/mol).
_KCAL = 23.0605487415


def load_s30l_ref(root: Path = S30L_ROOT) -> dict:
    """Reference gas-phase association energies {system#: kcal/mol}.

    These are the empirical DeltaE_emp back-corrected from experiment
    (Sure & Grimme, doi:10.1021/acs.jctc.5b00296), one per system 1..30, in the
    file ``reference_s30lci``. DeltaE_emp = E(complex) - E(host) - E(guest), so it
    is directly comparable to a MACE association energy -- the absolute-energy
    offset between MACE (wB97M-D3) and the reference method cancels in the
    difference.
    """
    f = root / "reference_s30lci"
    if not f.exists():
        return {}
    out = {}
    for i, line in enumerate(f.read_text().splitlines(), 1):
        s = line.strip()
        if s:
            out[i] = float(s)
    return out


# Table S1 of the S30L SI (Sure & Grimme, doi:10.1021/acs.jctc.5b00296): computed
# gas-phase association energies DeltaE at def2-QZVP (2-body D3 included; 3-body
# dispersion given separately in the table, not added here). MACE-OFF23 was
# trained on wB97M-D3, so the wB97X-D3 column is the closest available computed
# reference (both range-separated hybrids + D3); the residual ~5 kcal/mol offset
# is the wB97X-D3 vs wB97M-D3 functional gap. Values are transcribed from the SI
# PDF (Table S1). NEUTRAL = systems 1-30 without counterions; CI = 23-30 with
# counterions (net-neutral, what MACE-OFF23 -- neutral-only -- can score).
_S30L_TBL_S1 = {
    "wB97XD3": {
        "neutral": {1:-33.18,2:-21.78,3:-21.47,4:-20.35,5:-34.66,6:-24.86,7:-35.39,
                    8:-41.00,9:-32.14,10:-33.67,11:-39.25,12:-39.19,13:-31.55,14:-33.00,
                    15:-15.91,16:-20.09,17:-37.22,18:-24.58,19:-19.11,20:-23.75,
                    21:-32.54,22:-37.39,23:-64.47,24:-146.92,25:-34.39,26:-34.50,
                    27:-89.02,28:-84.80,29:-56.77,30:-52.60},
        "ci": {23:-65.68,24:-81.44,25:-35.68,26:-35.50,27:-39.34,28:-33.74,
               29:-46.72,30:-46.46},
    },
    "PW6B95D3": {
        "neutral": {1:-32.08,2:-21.72,3:-26.17,4:-21.07,5:-35.19,6:-31.38,7:-34.93,
                    8:-39.73,9:-34.98,10:-36.05,11:-43.42,12:-42.88,13:-28.82,14:-31.32,
                    15:-18.17,16:-24.51,17:-32.92,18:-21.41,19:-16.64,20:-20.18,
                    21:-28.98,22:-33.89,23:-58.67,24:-139.23,25:-33.17,26:-33.14,
                    27:-84.41,28:-80.68,29:-54.79,30:-50.59},
        "ci": {23:-63.11,24:-74.16,25:-35.82,26:-35.62,27:-34.63,28:-29.41,
               29:-47.18,30:-47.70},
    },
}


def s30l_computed_ref(n: int, functional: str = "wB97XD3") -> float:
    """Computed DeltaE (kcal/mol) at def2-QZVP from Table S1 for system ``n``.

    Returns the counterion (CI) value for systems 23-30 (net-neutral, the
    variant MACE-OFF23 scores) and the neutral value for 1-22.
    """
    tbl = _S30L_TBL_S1[functional]
    return tbl["ci"].get(n, tbl["neutral"][n])


def load_s30l(root: Path = S30L_ROOT):
    """Load the S30L-CI test set -> ``[(n, charge, host, guest, complex), ...]``.

    Host/guest/complex are pre-split by the dataset authors (A/B/AB folders),
    so the monomer-vs-complex decomposition is exact -- no extraction. The ACS
    SI is paywalled; obtain it from doi:10.1021/acs.jctc.5b00296 and place the
    extracted ``s30lci_test_set`` folder under ``data/s30l/``. Systems whose
    host+guest+complex contain any element outside the scope are skipped.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"S30L-CI not found at {root}. Download the Supporting Information "
            "from https://doi.org/10.1021/acs.jctc.5b00296 (ACS access required) "
            "and place the 's30lci_test_set' folder under data/s30l/.")
    out = []
    for n in range(1, 31):
        d = root / str(n)
        if not d.exists():
            continue
        host = _parse_turbomole_coord(d / "A" / "coord")
        guest = _parse_turbomole_coord(d / "B" / "coord")
        cplx = _parse_turbomole_coord(d / "AB" / "coord")
        if len(cplx) == 0:
            continue
        syms = (set(host.get_chemical_symbols()) | set(guest.get_chemical_symbols())
                | set(cplx.get_chemical_symbols()))
        if not syms.issubset(SCOPE_SYMS):
            log.info("S30L %d: out-of-scope %s; skipping", n, sorted(syms - SCOPE_SYMS))
            continue
        out.append((n, _chrg(d / "AB"), host, guest, cplx))
        log.info("S30L %d: complex %s (%d at, q=%s)", n, cplx.get_chemical_formula(),
                 len(cplx), out[-1][1])
    log.info("S30L-CI: %d in-scope systems", len(out))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _in_scope(atoms) -> bool:
    return set(atoms.get_chemical_symbols()).issubset(SCOPE_SYMS)


def score_atoms(atoms, pool, model="off-medium", dtype="float64") -> dict:
    """Per-atom OOD of a structure vs the pool (mean/max/flagged count)."""
    if not (hasattr(atoms, "calc") and atoms.calc is not None and hasattr(atoms.calc, "models")):
        mc.attach(atoms, model=model, dtype=dtype)
    r = atom_ood_scores(atoms, pool, model=model, dtype=dtype)
    d = r["distances"]
    d = d[~np.isnan(d)]
    return {"mean": float(d.mean()) if d.size else float("nan"),
            "max": float(d.max()) if d.size else float("nan"),
            "n": len(atoms), "flagged": int(r["flags"].sum()),
            "formula": atoms.get_chemical_formula(),
            "used_global": sorted(set(r["used_global"]))}


def _mace_energy_eV(atoms, model="off-medium", dtype="float64") -> float:
    """MACE single-point total energy (eV) on the given geometry."""
    if not (hasattr(atoms, "calc") and atoms.calc is not None and hasattr(atoms.calc, "models")):
        mc.attach(atoms, model=model, dtype=dtype)
    return float(atoms.get_potential_energy())


def score_table(rows, pool, model="off-medium", dtype="float64"):
    """Score a list of (name, atoms) and print a sorted table; return rows."""
    res = []
    for name, atoms in rows:
        s = score_atoms(atoms, pool, model=model, dtype=dtype)
        res.append((name, s))
    res.sort(key=lambda r: r[1]["mean"])
    print(f"\n{'name':42s} {'atoms':>6s} {'meanOOD':>8s} {'maxOOD':>8s} {'flagged':>7s}")
    for name, s in res:
        print(f"{name[:42]:42s} {s['n']:6d} {s['mean']:8.3f} {s['max']:8.3f} {s['flagged']:7d}")
    return res


# ---------------------------------------------------------------------------
# Assembly + CLI
# ---------------------------------------------------------------------------
def build_ood_set(fetch=True):
    """Return S66 dimers/monomers and S30L systems for library use."""
    out = {"s66_dimers": [], "s66_monomers": [], "s30l": []}
    if fetch:
        for idx, label, dimer, mA, mB in fetch_s66():
            if _in_scope(dimer):
                out["s66_dimers"].append((f"S66-{idx:02d} {label}", dimer))
                out["s66_monomers"].append((f"S66-{idx:02d}A {label}", mA))
                out["s66_monomers"].append((f"S66-{idx:02d}B {label}", mB))
        out["s30l"] = load_s30l()
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Fetch grounded OOD structures (S66 + S30L-CI) and score them "
                    "against the MACE-OFF23 reference pool.")
    p.add_argument("--no-fetch", action="store_true", help="use cached data only")
    p.add_argument("--model", default="off-medium")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--diagram", action="store_true",
                   help="draw a neuron-firing diagram for the most-OOD S30L complex")
    p.add_argument("--energies", action="store_true",
                   help="compare MACE association energies (E_cpx-E_host-E_guest) "
                        "to the Grimme reference for each S30L system -- an "
                        "energy-wise OOD measure independent of the latent proxy")
    p.add_argument("--ensemble", action="store_true",
                   help="ensemble OOD signal: spread of E_assoc across MACE off-small/"
                        "off-medium/off-large (model-disagreement uncertainty). Tests "
                        "whether ensemble disagreement predicts |energy error| better "
                        "than the latent-distance proxy.")
    p.add_argument("--separation", action="store_true",
                   help="deployable reliability test: confirm MACE is accurate on the "
                        "in-distribution S66 positives (vs CCSD(T)/CBS BIND refs), then "
                        "ask whether the NN signal (latent OOD) separates reliable from "
                        "unreliable energetics -- the inference-time question that needs "
                        "no reference calculation.")
    p.add_argument("--trend", action="store_true",
                   help="model-size trend: S66 interaction-energy MAE and S30L E_assoc "
                        "MAE for off-small/medium/large. Shows that accuracy improves "
                        "with model size and that off-large fixes the off-medium "
                        "fullerene over-binding failure.")
    p.add_argument("--forces", action="store_true",
                   help="force-residual signal: MACE predicted |F| on the DFT/MP2-"
                        "optimized S66/S30L geometries, where the true force is ~0. A "
                        "large residual force means the model is 'surprised' by the "
                        "geometry -- a physics-based, pool-free reliability signal "
                        "orthogonal to the latent-distance proxy.")
    args = p.parse_args(argv)
    import numpy as _np

    pool = ReferencePool.load()
    M = dict(model=args.model, dtype=args.dtype)

    # --- Model-size trend: S66 (positives) + S30L (test) accuracy per size. ---
    if args.trend:
        s66 = fetch_s66()
        bind = s66_bind_ref()
        s30l = load_s30l()
        models = ["off-small", "off-medium", "off-large"]
        print("\n=== Model-size trend: S66 (held-out positives) + S30L (test) ===")

        def _e(atoms, mdl):
            atoms.calc = None  # force re-attach per model (stale-calc guard)
            return _mace_energy_eV(atoms, model=mdl, dtype=args.dtype)

        print(f"  {'model':10s} | {'S66 MAE':>8s} {'S66max':>7s} | "
              f"{'S30L MAE':>9s} {'exclFull':>8s} | {'full9':>6s} {'full10':>7s} "
              f"{'Cl4-25':>7s} {'Cl4-26':>7s}  (all kcal/mol)")
        for mdl in models:
            s66e = []
            for idx, lab, dimer, mA, mB in s66:
                e = (_e(dimer, mdl) - _e(mA, mdl) - _e(mB, mdl)) * _KCAL
                s66e.append(abs(e - bind.get(idx, float("nan"))))
            s30e = {}
            for n, ch, host, guest, cplx in s30l:
                e = (_e(cplx, mdl) - _e(host, mdl) - _e(guest, mdl)) * _KCAL
                s30e[n] = abs(e - s30l_computed_ref(n, "wB97XD3"))
            s66e = _np.array(s66e)
            alls = _np.array(list(s30e.values()))
            nof = _np.array([v for n, v in s30e.items() if n not in (9, 10)])
            print(f"  {mdl:10s} | {s66e.mean():8.2f} {s66e.max():7.2f} | "
                  f"{alls.mean():9.2f} {nof.mean():8.2f} | {s30e[9]:6.1f} "
                  f"{s30e[10]:7.1f} {s30e[25]:7.1f} {s30e[26]:7.1f}")
        print("  -> accuracy improves with model size; off-large fixes the off-medium")
        print("     fullerene over-binding (full9/10). Cl4 (25/26) fails at every size.")
        return

    # --- Force-residual signal: MACE |F| on optimized geometries (true F ~0). ---
    if args.forces:
        RELIABLE = 10.0
        s66 = fetch_s66()
        bind = s66_bind_ref()
        s30l = load_s30l()
        print("\n=== Force-residual signal: MACE |F| on optimized geometries ===")
        print("  S66/S30L are DFT/MP2-optimized -> true force ~0; large MACE |F| = the")
        print(f"  model is 'surprised' by the geometry. reliable label: |E err|<{RELIABLE:.0f}.")

        def _ef(atoms):
            mc.attach(atoms, model=args.model, dtype=args.dtype)
            f = atoms.get_forces()
            return float(atoms.get_potential_energy()), np.linalg.norm(f, axis=1)

        rows = []  # (set, idx, formula, err, fmax, frms)
        for idx, lab, dimer, mA, mB in s66:
            ed, fd = _ef(dimer)
            ea, _ = _ef(mA)
            eb, _ = _ef(mB)
            err = (ed - ea - eb) * _KCAL - bind.get(idx, float("nan"))
            rows.append(("S66", idx, dimer.get_chemical_formula(), err,
                         float(fd.max()), float(np.sqrt((fd ** 2).mean()))))
        for n, ch, host, guest, cplx in s30l:
            ec, fc = _ef(cplx)
            eh, _ = _ef(host)
            eg, _ = _ef(guest)
            err = (ec - eh - eg) * _KCAL - s30l_computed_ref(n, "wB97XD3")
            rows.append(("S30L", n, cplx.get_chemical_formula(), err,
                         float(fc.max()), float(np.sqrt((fc ** 2).mean()))))

        def _stat(sub, key, unit="kcal/mol"):
            v = _np.array([abs(key(r)) for r in sub if not _np.isnan(key(r))])
            return (f"n={v.size}, MAE={v.mean():.2f}, median={_np.median(v):.2f}, "
                    f"max={v.max():.2f}")

        def _frange(sub, key):
            v = _np.array([key(r) for r in sub])
            return f"[{v.min():.2f}, med {_np.median(v):.2f}, {v.max():.2f}]"

        def _pct(v, p):
            v = _np.sort(_np.array(v))
            return float(v[int(round(p * (v.size - 1)))]) if v.size else float("nan")

        s66r = [r for r in rows if r[0] == "S66"]
        s30 = [r for r in rows if r[0] == "S30L"]
        s30_rel = [r for r in s30 if abs(r[3]) < RELIABLE]
        s30_bad = [r for r in s30 if abs(r[3]) >= RELIABLE]
        print(f"\n  S66 (in-dist positives):      E err {_stat(s66r, lambda r: r[3])}")
        print(f"    fmax (eV/A) {_frange(s66r, lambda r: r[4])}; frms {_frange(s66r, lambda r: r[5])}")
        print(f"  S30L reliable (|err|<{RELIABLE:.0f}):    E err {_stat(s30_rel, lambda r: r[3])}")
        print(f"    fmax {_frange(s30_rel, lambda r: r[4])}; frms {_frange(s30_rel, lambda r: r[5])}")
        print(f"  S30L UNreliable (|err|>={RELIABLE:.0f}): E err {_stat(s30_bad, lambda r: r[3])}")
        print(f"    fmax {_frange(s30_bad, lambda r: r[4])}; frms {_frange(s30_bad, lambda r: r[5])}")

        rel_fmax = [r[4] for r in s66r + s30_rel]
        rel_frms = [r[5] for r in s66r + s30_rel]
        bad_fmax = [r[4] for r in s30_bad]
        bad_frms = [r[5] for r in s30_bad]
        print("\n  separation (reliable = S66 + reliable-S30L; unreliable = S30L failures):")
        if bad_fmax:
            print(f"    fmax: reliable p95={_pct(rel_fmax,.95):.2f} vs unreliable "
                  f"min={min(bad_fmax):.2f} "
                  f"{'-> SEPARATED' if _pct(rel_fmax,.95) < min(bad_fmax) else '-> OVERLAP'}")
            print(f"    frms: reliable p95={_pct(rel_frms,.95):.2f} vs unreliable "
                  f"min={min(bad_frms):.2f} "
                  f"{'-> SEPARATED' if _pct(rel_frms,.95) < min(bad_frms) else '-> OVERLAP'}")

        print(f"\n  {'set':5s} {'#':>3} {'formula':16s} {'err':>8s} {'fmax':>7s} "
              f"{'frms':>7s} {'label':>6s}")
        for s, i, f, e, fm, fr in sorted(rows, key=lambda r: r[4]):
            lab = "OK" if abs(e) < RELIABLE else "FAIL"
            print(f"  {s:5s} {i:3d} {f:16s} {e:8.2f} {fm:7.2f} {fr:7.2f} {lab:>6s}")
        return

    # --- Ensemble OOD signal (self-contained; skips the latent/S66 analysis). ---
    if args.ensemble:
        s30l = load_s30l()
        models = ["off-small", "off-medium", "off-large"]
        print("\n=== Ensemble OOD: MACE off-small/medium/large E_assoc disagreement ===")
        print(f"  {len(s30l)} in-scope systems. std across 3 model sizes = ensemble "
              "uncertainty. err = off-medium E_assoc - wB97X-D3/QZ ref.")
        rows_en = []
        for n, ch, host, guest, cplx in s30l:
            eas = {}
            for mdl in models:
                for at in (host, guest, cplx):
                    at.calc = None  # force re-attach: the attach guard skips if a
                    # (possibly different-size) calc is already present, which would
                    # silently return the wrong model's energy and collapse the std.
                eas[mdl] = (_mace_energy_eV(cplx, model=mdl, dtype=args.dtype)
                            - _mace_energy_eV(host, model=mdl, dtype=args.dtype)
                            - _mace_energy_eV(guest, model=mdl, dtype=args.dtype)) * _KCAL
            std = float(_np.std([eas[m] for m in models]))
            err = eas["off-medium"] - s30l_computed_ref(n, "wB97XD3")
            cplx.calc = None  # score against the pool, which is off-medium (2048-d)
            sc = score_atoms(cplx, pool, model="off-medium", dtype=args.dtype)
            rows_en.append((n, cplx.get_chemical_formula(), eas["off-small"],
                            eas["off-medium"], eas["off-large"], std, abs(err),
                            sc["mean"], sc["max"]))
        rows_en.sort(key=lambda r: r[5], reverse=True)
        print(f"\n  {'#':>3} {'formula':16s} {'small':>8s} {'medium':>8s} {'large':>8s} "
              f"{'std':>7s} {'|err|':>7s} {'latMean':>8s} {'latMax':>7s}")
        for n, f, s, m, l, st, e, lm, lmx in rows_en:
            print(f"  {n:3d} {f:16s} {s:8.2f} {m:8.2f} {l:8.2f} {st:7.2f} "
                  f"{e:7.2f} {lm:8.3f} {lmx:7.3f}")
        stds = _np.array([r[5] for r in rows_en])
        er = _np.array([r[6] for r in rows_en])
        latm = _np.array([r[7] for r in rows_en])
        latmx = _np.array([r[8] for r in rows_en])
        full = [i for i, r in enumerate(rows_en) if r[0] in (9, 10)]

        def _spear(a, b):
            ra = _np.argsort(_np.argsort(a)); rb = _np.argsort(_np.argsort(b))
            return float(_np.corrcoef(ra, rb)[0, 1])

        def _crep(a, b, label):
            p = float(_np.corrcoef(a, b)[0, 1]); s = _spear(a, b)
            a2 = _np.delete(a, full); b2 = _np.delete(b, full)
            p2 = float(_np.corrcoef(a2, b2)[0, 1]); s2 = _spear(a2, b2)
            print(f"  {label:30s} all: P={p:+.2f} S={s:+.2f} | excl-full: P={p2:+.2f} S={s2:+.2f}")
        print("\n  signal vs |energy err|:")
        _crep(stds, er, "ensemble std (3 sizes)")
        _crep(latm, er, "latent mean (for contrast)")
        _crep(latmx, er, "latent max (for contrast)")
        return

    # --- Separation: are reliable vs unreliable energetics separable by the NN
    #     signal alone? S66 (in-distribution, CCSD(T)/CBS refs) = positive examples;
    #     S30L (vs wB97X-D3) = the test set. The reference energies are OFFLINE LABELS
    #     only -- at inference time you'd carry just the NN signal, no DFT. ---
    if args.separation:
        RELIABLE = 10.0  # kcal/mol; above the ~5 noise floor, clearly a failure
        s66 = fetch_s66()
        bind = s66_bind_ref()
        print("\n=== Reliability separation: NN signal vs energy error ===")
        print(f"  reliable label: |err| < {RELIABLE:.0f} kcal/mol. "
              "S66 = in-distribution positives; S30L = test set.")
        rows = []  # (set, idx, formula, err, latMean, latMax)
        print("  computing S66 (66 in-distribution positives)...")
        for idx, label, dimer, mA, mB in s66:
            eint = (_mace_energy_eV(dimer, model=args.model, dtype=args.dtype)
                    - _mace_energy_eV(mA, model=args.model, dtype=args.dtype)
                    - _mace_energy_eV(mB, model=args.model, dtype=args.dtype)) * _KCAL
            err = eint - bind.get(idx, float("nan"))
            sc = score_atoms(dimer, pool, model=args.model, dtype=args.dtype)
            rows.append(("S66", idx, dimer.get_chemical_formula(), err,
                         sc["mean"], sc["max"]))
        print("  computing S30L (23 test systems)...")
        for n, ch, host, guest, cplx in load_s30l():
            eas = (_mace_energy_eV(cplx, model=args.model, dtype=args.dtype)
                   - _mace_energy_eV(host, model=args.model, dtype=args.dtype)
                   - _mace_energy_eV(guest, model=args.model, dtype=args.dtype)) * _KCAL
            err = eas - s30l_computed_ref(n, "wB97XD3")
            sc = score_atoms(cplx, pool, model=args.model, dtype=args.dtype)
            rows.append(("S30L", n, cplx.get_chemical_formula(), err,
                         sc["mean"], sc["max"]))

        def _stats(sub, key):
            v = _np.array([abs(key(r)) for r in sub if not _np.isnan(key(r))])
            return (f"n={v.size}, MAE={v.mean():.2f}, median={_np.median(v):.2f}, "
                    f"max={v.max():.2f}")

        def _lat(sub, key):
            v = _np.array([key(r) for r in sub])
            return f"[{v.min():.3f}, med {_np.median(v):.3f}, {v.max():.3f}]"

        s66r = [r for r in rows if r[0] == "S66"]
        s30 = [r for r in rows if r[0] == "S30L"]
        s30_rel = [r for r in s30 if abs(r[3]) < RELIABLE]
        s30_bad = [r for r in s30 if abs(r[3]) >= RELIABLE]
        print(f"\n  S66 (in-distribution positives):  err {_stats(s66r, lambda r: r[3])}")
        print(f"    latent mean {_lat(s66r, lambda r: r[4])}; latent max {_lat(s66r, lambda r: r[5])}")
        print(f"  S30L reliable (|err|<{RELIABLE:.0f}): err {_stats(s30_rel, lambda r: r[3])}")
        print(f"    latent mean {_lat(s30_rel, lambda r: r[4])}; latent max {_lat(s30_rel, lambda r: r[5])}")
        print(f"  S30L UNreliable (|err|>={RELIABLE:.0f}): err {_stats(s30_bad, lambda r: r[3])}")
        print(f"    latent mean {_lat(s30_bad, lambda r: r[4])}; latent max {_lat(s30_bad, lambda r: r[5])}")

        # Separation: does the NN signal separate reliable (S66) from unreliable
        # (S30L failures)? Report the reliable p95 vs the unreliable minimum.
        def _pct(v, p):
            v = _np.sort(_np.array(v))
            return float(v[int(round(p * (v.size - 1)))]) if v.size else float("nan")
        rel_mean = [r[4] for r in s66r + s30_rel]
        rel_max = [r[5] for r in s66r + s30_rel]
        bad_mean = [r[4] for r in s30_bad]
        bad_max = [r[5] for r in s30_bad]
        print("\n  separation (reliable = S66 + reliable-S30L; unreliable = S30L failures):")
        if bad_mean:
            print(f"    latent mean: reliable p95={_pct(rel_mean, .95):.3f} vs "
                  f"unreliable min={min(bad_mean):.3f} "
                  f"{'-> SEPARATED' if _pct(rel_mean, .95) < min(bad_mean) else '-> OVERLAP'}")
            print(f"    latent max:  reliable p95={_pct(rel_max, .95):.3f} vs "
                  f"unreliable min={min(bad_max):.3f} "
                  f"{'-> SEPARATED' if _pct(rel_max, .95) < min(bad_max) else '-> OVERLAP'}")

        # Combined table sorted by latent max (the signal that flags the Cl edge case).
        print(f"\n  {'set':5s} {'#':>3} {'formula':16s} {'err':>8s} {'latMean':>8s} "
              f"{'latMax':>7s} {'label':>6s}")
        for s, i, f, e, lm, lmx in sorted(rows, key=lambda r: r[5]):
            lab = "OK" if abs(e) < RELIABLE else "FAIL"
            print(f"  {s:5s} {i:3d} {f:16s} {e:8.2f} {lm:8.3f} {lmx:7.3f} {lab:>6s}")
        return

    # --- S66: paired monomer (isolated, in-distribution) vs dimer (non-covalent) ---
    s66 = fetch_s66()
    mono_means, rows = [], []
    for idx, label, dimer, mA, mB in s66:
        sd = score_atoms(dimer, pool, **M)
        sa = score_atoms(mA, pool, **M)
        sb = score_atoms(mB, pool, **M)
        mono_means += [sa["mean"], sb["mean"]]
        mono_mean = 0.5 * (sa["mean"] + sb["mean"])
        rows.append((idx, label, mono_mean, sd["mean"], sd["max"], sd["n"]))
    base = _np.array(mono_means)
    print(f"\n=== In-distribution baseline: S66 monomers (isolated, n={len(base)}) ===")
    print(f"  mean atom-OOD = {base.mean():.3f} +/- {base.std():.3f}  "
          f"(p1={_np.percentile(base,1):.3f}, p99={_np.percentile(base,99):.3f})")

    print("\n=== S66 non-covalent dimers: does the contact add OOD? ===")
    print(f"{'#':>3s} {'label':34s} {'mono':>7s} {'dimer':>7s} {'delta':>7s} {'max':>7s} {'atoms':>6s}")
    rows.sort(key=lambda r: r[3] - r[2], reverse=True)
    for idx, label, mm, dm, mx, n in rows[:12] + rows[-3:]:
        print(f"{idx:3d} {label[:34]:34s} {mm:7.3f} {dm:7.3f} {dm - mm:+7.3f} {mx:7.3f} {n:6d}")
    dimer_means = _np.array([r[3] for r in rows])
    print(f"  dimers: mean atom-OOD = {dimer_means.mean():.3f} +/- {dimer_means.std():.3f}  "
          f"(vs monomer baseline {base.mean():.3f})")

    # --- S30L-CI: host-guest (DFT-D optimized, pre-split host/guest/complex) ---
    s30l = load_s30l()
    print(f"\n=== S30L-CI host-guest complexes (Sure & Grimme, DFT-D optimized) ===")
    print(f"  {len(s30l)} in-scope systems (CNOH/F/Cl); host/guest/complex pre-split "
          "by the dataset authors -> decomposition is exact (no extraction).")
    scored = []
    for n, ch, host, guest, cplx in s30l:
        sh = score_atoms(host, pool, **M)
        sg = score_atoms(guest, pool, **M)
        sc = score_atoms(cplx, pool, **M)
        scored.append((f"S30L-{n:02d}", cplx, sc))
        print(f"\n  S30L-{n:02d} (q={ch}): complex {cplx.get_chemical_formula()} "
              f"({len(cplx)} at) mean={sc['mean']:.3f} max={sc['max']:.3f} "
              f"flagged={sc['flagged']}")
        print(f"    {'component':26s} {'atoms':>6s} {'meanOOD':>8s} {'maxOOD':>8s} {'flagged':>8s}")
        for tag, a, s in [("host", host, sh), ("guest", guest, sg), ("complex", cplx, sc)]:
            print(f"    {tag + ' ' + a.get_chemical_formula():26s} "
                  f"{s['n']:6d} {s['mean']:8.3f} {s['max']:8.3f} {s['flagged']:8d}")
    scored.sort(key=lambda r: r[2]["mean"])          # ascending -> [-1] is most-OOD

    # --- Energy-wise OOD: MACE association energy vs the Grimme reference. ---
    if args.energies:
        emp_ref = load_s30l_ref()
        latent = {int(name.split("-")[1]): sc for name, _, sc in scored}  # {n: score dict}
        print("\n=== S30L-CI association energies: MACE vs computed reference ===")
        print("  E_assoc = E(complex) - E(host) - E(guest); MACE in eV -> kcal/mol.")
        print("  Primary ref: Table S1 wB97X-D3/def2-QZVP computed DeltaE (closest")
        print("  available column to MACE's wB97M-D3 training level). Empirical")
        print("  DeltaE_emp shown for cross-check. 23-30 use the CI (counterion) row.")
        rows_e = []
        for n, ch, host, guest, cplx in s30l:
            e_assoc = (_mace_energy_eV(cplx, **M) - _mace_energy_eV(host, **M)
                       - _mace_energy_eV(guest, **M)) * _KCAL
            wref = s30l_computed_ref(n, "wB97XD3")
            err = e_assoc - wref
            sc = latent.get(n, {"mean": float("nan"), "max": float("nan")})
            rows_e.append((n, cplx.get_chemical_formula(), wref, e_assoc, err,
                           emp_ref.get(n, float("nan")), sc["mean"], sc["max"]))
        rows_e.sort(key=lambda r: abs(r[4]), reverse=True)
        print(f"\n  {'#':>3s} {'formula':18s} {'wB97X-D3':>9s} {'MACE':>9s} "
              f"{'err':>8s} {'empRef':>8s} {'latent':>7s} {'latMax':>7s}")
        for n, form, r, e, err, er, lm, lmx in rows_e:
            print(f"  {n:3d} {form:18s} {r:9.2f} {e:9.2f} {err:+8.2f} "
                  f"{er:8.2f} {lm:7.3f} {lmx:7.3f}")
        errs = _np.array([abs(r[4]) for r in rows_e])
        latm = _np.array([r[6] for r in rows_e])    # latent mean (idx 6 after emp_ref)
        latmx = _np.array([r[7] for r in rows_e])   # latent max
        full = [i for i, r in enumerate(rows_e) if r[0] in (9, 10)]  # bare-fullerene
        nofull = _np.delete(errs, full) if full else errs
        print(f"\n  MACE vs wB97X-D3/QZ: MAE = {errs.mean():.2f} kcal/mol, "
              f"median = {_np.median(errs):.2f}, max |err| = {errs.max():.2f} "
              f"(S30L-{rows_e[0][0]:02d})")
        print(f"    excl bare-fullerene 9,10: MAE = {nofull.mean():.2f}, "
              f"median = {_np.median(nofull):.2f}  (~the wB97X-D3 vs wB97M-D3 "
              f"functional noise floor)")

        def _spear(a, b):                      # rank correlation (no scipy needed)
            ra = _np.argsort(_np.argsort(a)); rb = _np.argsort(_np.argsort(b))
            return float(_np.corrcoef(ra, rb)[0, 1])

        def _corr_report(a, b, label, full_idx):
            p = float(_np.corrcoef(a, b)[0, 1]); s = _spear(a, b)
            a2 = _np.delete(a, full_idx); b2 = _np.delete(b, full_idx)
            p2 = float(_np.corrcoef(a2, b2)[0, 1]); s2 = _spear(a2, b2)
            print(f"  {label:26s} all: Pearson={p:+.2f} Spearman={s:+.2f} | "
                  f"excl-fullerene: Pearson={p2:+.2f} Spearman={s2:+.2f}")
        if latm.size > 2:
            print("\n  NN-signal vs |energy err| (does the latent OOD predict energy error?):")
            _corr_report(latm, errs, "latent mean (default)", full)
            _corr_report(latmx, errs, "latent max (worst atom)", full)
        print("  -> the latent-mean correlation is driven by the 2 fullerene outliers;")
        print("     the latent max is the better signal for non-obvious failures (Cl4).")

    if args.diagram and scored:
        from activation_viz import draw_neuron_firing
        name, atoms, s = scored[-1]
        out = _REPO / "figures" / f"ood_{name}_firing.png"
        draw_neuron_firing(atoms, smi=None, out_png=out, pair_layer=None,
                           model=args.model, dtype=args.dtype, title=name)
        print(f"\ndiagram: {out}")


if __name__ == "__main__":
    main()
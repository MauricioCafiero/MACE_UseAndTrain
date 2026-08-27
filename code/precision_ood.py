"""Precision-disagreement OOD signal: |E(low precision) - E(float64)|.

Rationale. The energy-space signals of :mod:`energy_ood` failed because they
score atoms *individually*: an association-energy error is a coherent sum over
all atoms (the fullerene's -85 kcal/mol is only ~0.015 eV/atom, 30x below the
per-atom scatter), so per-atom decompositions are blind to it. Precision
disagreement is a **system-level** signal -- the whole energy is computed twice
and compared once -- so a collective offset survives the aggregation.

The idea: the model's prediction is a large cancellation (total energies are
~10^3 eV; the quantities of interest are ~10^-2-10^-1 eV differences). Running
the same forward at reduced precision (float32; bfloat16 was tried but the
graph-build path calls an op torch does not implement for bf16 on CPU)
perturbs every intermediate sum; the output-level disagreement
|E_low - E_64| measures how much *internal amplification* the input's
cancellation structure produces. The hypothesis is that out-of-distribution
inputs -- where the network extrapolates and intermediate activations run
large -- amplify rounding more, so precision disagreement is a proxy for how
far the network is working outside its comfort zone. Cheap: two forwards, no
reference pool, fully deployable.

As with every signal in this repo, the decisive test is ``--separation``:
S66 (held-out, in-distribution, CCSD(T)/CBS refs) as reliable positives vs
S30L-CI (vs wB97X-D3/QZ), asking whether reliable p95 falls below the
unreliable minimum.

Example
-------
>>> import mace_calc as mc
>>> from precision_ood import score_molecule
>>> a = mc.smiles_to_atoms("CC(=O)Nc1ccc(cc1)O")
>>> score_molecule(a, model="off-medium", low_dtype="bfloat16")
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch import). ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging

import numpy as np

import mace_calc as mc  # noqa: E402

log = logging.getLogger("precision_ood")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _energy(atoms, calc) -> float:
    """Total energy (eV) with the given (shared) calculator, cache-reset."""
    atoms.calc = calc
    calc.reset()                      # shared calc: drop the previous structure's cache
    return float(atoms.get_potential_energy())


def _calcs(model: str, low_dtype: str, dtype: str = "float64"):
    return (mc.get_calculator(model=model, dtype=dtype),
            mc.get_calculator(model=model, dtype=low_dtype))


def precision_scores(atoms, calc64, calc_low) -> dict:
    """E at both precisions + disagreement, for one structure."""
    e64 = _energy(atoms, calc64)
    e_low = _energy(atoms, calc_low)
    return {"e64": e64, "e_low": e_low, "dE": abs(e64 - e_low)}


def score_molecule(atoms, model: str = "off-medium", low_dtype: str = "bfloat16",
                   dtype: str = "float64") -> dict:
    """Score one molecule and print the precision disagreement."""
    calc64, calc_low = _calcs(model, low_dtype, dtype)
    r = precision_scores(atoms, calc64, calc_low)
    n = len(atoms)
    print(f"{atoms.get_chemical_formula()}  ({n} atoms)")
    print(f"  E(float64)      = {r['e64']:16.6f} eV")
    print(f"  E({low_dtype:9s}) = {r['e_low']:16.6f} eV")
    print(f"  |dE|            = {r['dE']:16.3e} eV  ({r['dE'] * 23.0605:.4f} kcal/mol, "
          f"{r['dE'] / n:.2e} eV/atom)")
    return r


# ---------------------------------------------------------------------------
# Decisive test: S66/S30L separation verdict for the precision signal
# ---------------------------------------------------------------------------
def separation(model: str = "off-medium", dtype: str = "float64",
               low_dtype: str = "bfloat16", reliable: float = 10.0):
    """Association-energy precision disagreement vs reliability, S66 + S30L."""
    import ood_datasets as od

    calc64, calc_low = _calcs(model, low_dtype, dtype)

    def assoc(atoms_list):
        """(E_assoc at f64, E_assoc at low precision) for [cplx, host, guest]."""
        es = [(lambda a: (_energy(a, calc64), _energy(a, calc_low)))(a)
              for a in atoms_list]
        e64 = es[0][0] - es[1][0] - es[2][0]
        e_low = es[0][1] - es[1][1] - es[2][1]
        return e64, abs(e64 - e_low)

    rows = []  # (set, idx, formula, err, dE_assoc_kcal, dE_rel, dE_per_atom)
    print(f"computing S66 (66 in-distribution positives, {model}, "
          f"{dtype} vs {low_dtype})...", flush=True)
    bind = od.s66_bind_ref()
    for idx, label, dimer, mA, mB in od.fetch_s66():
        e64, dE = assoc([dimer, mA, mB])
        err = e64 * od._KCAL - bind.get(idx, float("nan"))
        n = len(dimer)
        rows.append(("S66", idx, dimer.get_chemical_formula(), err,
                     dE * od._KCAL, dE * od._KCAL / max(abs(e64 * od._KCAL), 1e-9),
                     dE / n))
    print("computing S30L (23 test systems)...", flush=True)
    for n_, ch, host, guest, cplx in od.load_s30l():
        e64, dE = assoc([cplx, host, guest])
        err = e64 * od._KCAL - od.s30l_computed_ref(n_, "wB97XD3")
        n = len(cplx)
        rows.append(("S30L", n_, cplx.get_chemical_formula(), err,
                     dE * od._KCAL, dE * od._KCAL / max(abs(e64 * od._KCAL), 1e-9),
                     dE / n))

    keys = ("dE_assoc", "dE_rel", "dE_per_atom")

    def _pct(v, p):
        v = np.sort(np.array(v))
        return float(v[int(round(p * (v.size - 1)))]) if v.size else float("nan")

    def _rng(sub, ki):
        v = np.array([r[4 + ki] for r in sub])
        return f"[{v.min():.4g}, med {np.median(v):.4g}, {v.max():.4g}]"

    s66 = [r for r in rows if r[0] == "S66"]
    s30 = [r for r in rows if r[0] == "S30L"]
    s30_bad = [r for r in s30 if abs(r[3]) >= reliable]
    s30_rel = [r for r in s30 if abs(r[3]) < reliable]
    rel = s66 + s30_rel

    errs = np.array([abs(r[3]) for r in s66] + [abs(r[3]) for r in s30_rel])
    bad_errs = np.array([abs(r[3]) for r in s30_bad])

    def _stats(v):
        return f"n={v.size}, MAE={v.mean():.2f}, median={np.median(v):.2f}, max={v.max():.2f}"

    print(f"\n=== Precision-disagreement vs reliability ({model}, {low_dtype}) ===")
    print(f"  reliable label: |err| < {reliable:.0f} kcal/mol")
    print(f"  S66 (in-dist positives):        err {_stats(errs[:len(s66)])}")
    print(f"  S30L reliable:                  err {_stats(errs[len(s66):])}")
    print(f"  S30L UNreliable (|err|>={reliable:.0f}): err {_stats(bad_errs)}")
    for ki, key in enumerate(keys):
        print(f"  {key:11s} S66 {_rng(s66, ki)}  reliable {_rng(rel, ki)}  "
              f"unreliable {_rng(s30_bad, ki)}")
    print("\n  separation (reliable = S66 + reliable-S30L; unreliable = S30L failures):")
    if s30_bad:
        for ki, key in enumerate(keys):
            p95 = _pct([r[4 + ki] for r in rel], .95)
            mn = min(r[4 + ki] for r in s30_bad)
            print(f"    {key:11s}: reliable p95={p95:.4g} vs unreliable min={mn:.4g} "
                  f"{'-> SEPARATED' if p95 < mn else '-> OVERLAP'}")

    print(f"\n  {'set':5s} {'#':>3} {'formula':16s} {'err':>8s} {'dE_assoc':>10s} "
          f"{'dE_rel':>10s} {'dE/at':>10s} {'label':>6s}")
    for s, i, f_, e, da, dr, dn in sorted(rows, key=lambda r: r[6]):
        lab = "OK" if abs(e) < reliable else "FAIL"
        print(f"  {s:5s} {i:3d} {f_:16s} {e:8.2f} {da:10.4g} {dr:10.3g} "
              f"{dn:10.3g} {lab:>6s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Precision-disagreement OOD signal for MACE "
                    "(|E(low precision) - E(float64)|, system-level).")
    p.add_argument("molecule", nargs="?", default="CC(=O)Nc1ccc(cc1)O",
                   help="ASE molecule name or SMILES (default: paracetamol)")
    p.add_argument("--model", default="off-medium")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--low-dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="low-precision forward to disagree with (default float32; "
                        "bfloat16/float16 fail: lu_cpu not implemented for them)")
    p.add_argument("--separation", action="store_true",
                   help="run the S66/S30L reliability-separation verdict")
    args = p.parse_args(argv)

    if args.separation:
        separation(model=args.model, dtype=args.dtype, low_dtype=args.low_dtype)
        return

    from activation_ood import _build_atoms  # ASE-name-or-SMILES helper
    score_molecule(_build_atoms(args.molecule), model=args.model,
                   low_dtype=args.low_dtype, dtype=args.dtype)


if __name__ == "__main__":
    main()
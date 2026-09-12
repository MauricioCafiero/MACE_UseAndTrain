"""Rotaxane interaction-energy benchmark -- the target-chemistry test set.

S66/S30L (``forgetting.py``) ask whether a fine-tune kept the *foundation's*
chemistry. This asks the complementary question: does it get the chemistry it
was fine-tuned FOR right? Two sets, both from the rotaxanes-results study
(https://github.com/MauricioCafiero/rotaxanes-results):

**stacking** -- capped wheel/rod fragment pairs at the center-ring stacked well
and at both stoppers, with a **DLPNO-CCSD(T)/aug-cc-pVTZ** reference plus DFT,
GFN2-xTB and stock MACE-OFF23 columns from that study
(``QM_comparisons/README.md``, ``DETHREADING_REPORT.md`` §4.1/§4.1a). The GFN2
column matters most here: it is the level these checkpoints were trained toward,
so a good fine-tune should land on it.

**whole** -- four ~128-atom double-stack geometries scored without capping,
referenced against UMA (``ROT3QM/README.md`` §4a). UMA agrees with CCSD(T) to
~0.7 kcal/mol on the stacking set, so it is a usable reference; the ordering of
the four and the size of the central-ring/iso-side outlier gap are the
scientifically meaningful quantities.

    python rotaxane_bench.py --models off-medium ft-medium \
        --ft-model ft-dimer=../runs/rot250Md_replay.model
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_BENCH = _REPO / "data" / "rotaxane_bench"
_KCAL = 23.0605487415
OUT = _REPO / "data" / "rotaxane_bench_results.json"

# DETHREADING_REPORT.md §4.1 / §4.1a and QM_comparisons/README.md
STACK_REF = {
    "center_4F":          {"ccsdt": -10.10, "dft": -10.09, "gfn2": -10.26, "mace": -10.67},
    "center_2F":          {"ccsdt":  -9.63, "dft":  -9.30, "gfn2":  -9.90, "mace":  -9.77},
    "center_0F":          {"ccsdt":  -7.59, "dft":  -7.28, "gfn2":  -8.58, "mace":  -8.17},
    "weak_stopper_real":  {"ccsdt":  -8.09, "dft":  -8.23, "gfn2":  -7.84, "mace":  -8.09},
    "strong_stopper_real": {"ccsdt":  None, "dft": -11.53, "gfn2": -11.71, "mace": -11.51},
    "strong_stopper":     {"ccsdt":  None, "dft": -11.52, "gfn2": -12.33, "mace": -10.71},
    "weak_stopper":       {"ccsdt":  None, "dft":  -9.50, "gfn2": -10.52, "mace":  -8.99},
}
# ROT3QM/README.md §4a (UMA whole-structure)
WHOLE_REF = {"cf3_ring": -38.17, "iso_ring": -36.73,
             "central_cf3side": -36.05, "central_isoside": -30.62}

FT_MODELS = {"ft-medium": _REPO / "runs" / "rot250M_replay.model",
             "ft-large": _REPO / "runs" / "rot250L_replay.model"}


def _calc(name):
    import mace_calc as mc
    if name in FT_MODELS:
        import torch
        from mace.calculators import MACECalculator
        _orig = torch.jit.load
        torch.jit.load = lambda *a, **kw: _orig(
            *a, **{**kw, "map_location": kw.get("map_location", "cpu")})
        return MACECalculator(model_paths=str(FT_MODELS[name]), device="cpu",
                              default_dtype="float64")
    return mc.get_calculator(model=name, dtype="float64")


def _eint(d, calc):
    from ase.io import read
    es = []
    for n in ("dimer", "rod", "wheel"):
        at = read(d / f"{n}.xyz")
        at.calc = calc
        es.append(float(at.get_potential_energy()))
        at.calc = None
    return (es[0] - es[1] - es[2]) * _KCAL


def run(models, sets=("stacking", "whole")):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in models:
        calc = None
        for sub, ref in (("stacking", STACK_REF), ("whole", WHOLE_REF)):
            if sub not in sets:
                continue
            res.setdefault(sub, {}).setdefault(name, {})
            for key in ref:
                d = _BENCH / sub / key
                if not d.exists() or key in res[sub][name]:
                    continue
                calc = calc or _calc(name)
                res[sub][name][key] = _eint(d, calc)
                print(f"  {name:12s} {sub:8s} {key:20s} {res[sub][name][key]:8.2f}",
                      flush=True)
                OUT.write_text(json.dumps(res, indent=1))
    return res


def report(models):
    res = json.loads(OUT.read_text())

    print("\n=== stacking fragments (kcal/mol) — CCSD(T)/aug-cc-pVTZ reference ===")
    hdr = f"  {'pairing':22s} {'CCSD(T)':>8s} {'DFT':>7s} {'GFN2':>7s} {'OFF23':>7s}"
    print(hdr + "".join(f" {m:>12s}" for m in models))
    for key, r in STACK_REF.items():
        if key not in res.get("stacking", {}).get(models[0], {}):
            continue
        c = f"{r['ccsdt']:8.2f}" if r["ccsdt"] is not None else f"{'--':>8s}"
        print(f"  {key:22s} {c} {r['dft']:7.2f} {r['gfn2']:7.2f} {r['mace']:7.2f}"
              + "".join(f" {res['stacking'][m][key]:12.2f}" for m in models))
    for label, col in (("vs CCSD(T)", "ccsdt"), ("vs GFN2 (trained level)", "gfn2")):
        line = f"  {label:22s} {'':8s} {'':7s} {'':7s} {'':7s}"
        for m in models:
            e = [res["stacking"][m][k] - STACK_REF[k][col] for k in STACK_REF
                 if STACK_REF[k][col] is not None
                 and k in res["stacking"].get(m, {})]
            line += f" {np.abs(e).mean():12.2f}" if e else f" {'--':>12s}"
        print(line + "   <- MAE")

    if "whole" in res:
        print("\n=== whole-structure double stacks (kcal/mol) — UMA reference ===")
        order = sorted(WHOLE_REF, key=lambda k: WHOLE_REF[k], reverse=True)
        print(f"  {'geometry':22s} {'UMA':>8s}" + "".join(f" {m:>12s}" for m in models))
        for key in order:
            print(f"  {key:22s} {WHOLE_REF[key]:8.2f}"
                  + "".join(f" {res['whole'][m][key]:12.2f}" for m in models))
        gap = lambda d: d["central_isoside"] - d["central_cf3side"]
        print(f"  {'outlier gap':22s} {gap(WHOLE_REF):8.2f}"
              + "".join(f" {gap(res['whole'][m]):12.2f}" for m in models))
        print(f"  {'ordering matches UMA':22s} {'--':>8s}"
              + "".join(f" {str(sorted(res['whole'][m], key=lambda k: res['whole'][m][k], reverse=True) == order):>12s}"
                        for m in models))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*",
                   default=["off-medium", "off-large", "ft-medium", "ft-large"])
    p.add_argument("--ft-model", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--sets", nargs="*", default=["stacking", "whole"])
    a = p.parse_args(argv)
    for spec in a.ft_model:
        name, _, path = spec.partition("=")
        FT_MODELS[name] = Path(path)
    run(a.models, sets=tuple(a.sets))
    report(a.models)


if __name__ == "__main__":
    main()

"""Candidate-layer sweep: is another latent layer a better OOD signal?

The deployed signal scores per-atom latent embeddings from the deepest
interaction block (``interactions.1.linear``) by cosine distance to a
reference pool. This sweep asks whether a *different* per-atom layer separates
reliable from unreliable S30L energetics better, using the same protocol as
``_large_ood.py``:

  for each candidate layer L (off-large):
    1. build a 2500-frame OFF23 test-split reference pool at layer L
       (data/off23_pool_layer_<L>.npz, cached)
    2. score every S66 dimer and S30L complex: per-atom cosine distance to the
       nearest same-element reference atom -> molecule mean/max
    3. compare separation: S66 range (in-distribution) vs S30L energy
       failures (|E_assoc err| >= 10 kcal/mol vs wB97X-D3/QZ)

Energies are model-level (identical for every layer), so they are computed
once and reused across layers. Results -> data/layer_sweep_results.json.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# hold off macOS idle sleep for the whole run (child dies with this process)
try:
    import subprocess
    subprocess.Popen(["caffeinate", "-w", str(os.getpid())],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass

import json
import time
from pathlib import Path

import numpy as np

from activation_ood import ReferencePool, atom_ood_scores
import mace_calc as mc
from ood_datasets import (fetch_s66, s66_bind_ref, load_s30l, s30l_computed_ref,
                          _KCAL, _mace_energy_eV)

_REPO = Path(__file__).resolve().parent.parent
MODEL = "off-large"
REL = 10.0
N_FRAMES = 2500
# Deployed baseline first (pool cached), then candidates shallow -> deep.
LAYERS = [
    "interactions.1.linear",      # deployed baseline (pool cached)
    "node_embedding.linear",      # input embedding
    "interactions.0.linear",      # first interaction block
    "readouts.1.linear_1",        # deep readout hidden layer
]
OUT_JSON = _REPO / "data" / "layer_sweep_results.json"


def _pool_path(layer: str) -> Path:
    return _REPO / "data" / ("off23_pool_layer_"
                             + layer.replace(".", "_") + ".npz")


def _score(atoms, pool, layer: str) -> dict:
    # one calc for the whole run (same model for every layer); capture_activations
    # resets ASE's cache itself, so no detach needed here.
    if not (hasattr(atoms, "calc") and atoms.calc is not None
            and hasattr(atoms.calc, "models")):
        mc.attach(atoms, model=MODEL, dtype="float64")
    r = atom_ood_scores(atoms, pool, model=MODEL, dtype="float64", layer=layer)
    d = r["distances"]
    d = d[~np.isnan(d)]
    return {"mean": float(d.mean()) if d.size else float("nan"),
            "max": float(d.max()) if d.size else float("nan")}


def main():
    # --- systems + energies (layer-independent, computed once) ----------------
    print("collecting systems + energies ...", flush=True)
    systems = []   # (set, idx, formula, n_atoms, err_kcal, atoms)
    bind = s66_bind_ref()
    for idx, lab, dimer, mA, mB in fetch_s66():
        e = (_mace_energy_eV(dimer, model=MODEL, dtype="float64")
             - _mace_energy_eV(mA, model=MODEL, dtype="float64")
             - _mace_energy_eV(mB, model=MODEL, dtype="float64")) * _KCAL
        systems.append(("S66", idx, dimer.get_chemical_formula(),
                        len(dimer), e - bind.get(idx, float("nan")), dimer))
    for n, ch, host, guest, cplx in load_s30l():
        e = (_mace_energy_eV(cplx, model=MODEL, dtype="float64")
             - _mace_energy_eV(host, model=MODEL, dtype="float64")
             - _mace_energy_eV(guest, model=MODEL, dtype="float64")) * _KCAL
        systems.append(("S30L", n, cplx.get_chemical_formula(),
                        len(cplx), e - s30l_computed_ref(n, "wB97XD3"), cplx))
    print(f"  {len(systems)} systems scored.", flush=True)

    # --- per-layer pool + latent scores ---------------------------------------
    results = {}
    for layer in LAYERS:
        slug = layer.replace(".", "_")
        pool_path = _REPO / "data" / f"off23_pool_layer_{slug}.npz"
        if pool_path.exists():
            print(f"\n[{layer}] using cached pool {pool_path.name}", flush=True)
            pool = ReferencePool.load(pool_path)
        else:
            print(f"\n[{layer}] building {N_FRAMES}-frame pool -> "
                  f"{pool_path.name} ...", flush=True)
            t0 = time.time()
            pool = ReferencePool.build(n_frames=N_FRAMES, model=MODEL,
                                       dtype="float64", atom_layer=layer,
                                       out=pool_path)
            print(f"[{layer}] pool built in {(time.time()-t0)/60:.1f} min "
                  f"({pool.atom_vecs.shape[0]} atoms).", flush=True)

        rows = []
        for si, (setname, idx, formula, nat, err, atoms) in enumerate(systems, 1):
            sc = _score(atoms, pool, layer)
            rows.append({"set": setname, "idx": idx, "formula": formula,
                         "n_atoms": nat, "err_kcal": err,
                         "lat_mean": sc["mean"], "lat_max": sc["max"]})
            if si % 20 == 0:
                print(f"[{layer}]   {si}/{len(systems)} scored", flush=True)

        s66 = [r for r in rows if r["set"] == "S66"]
        s30 = [r for r in rows if r["set"] == "S30L"]
        bad = [r for r in s30 if abs(r["err_kcal"]) >= REL]
        ok = [r for r in s30 if abs(r["err_kcal"]) < REL]
        s66range = [min(r["lat_mean"] for r in s66),
                    float(np.median([r["lat_mean"] for r in s66])),
                    max(r["lat_mean"] for r in s66)]
        summ = {
            "s66_mean_range": s66range,
            "s30l_reliable_p95_mean": float(np.percentile(
                [r["lat_mean"] for r in ok], 95)),
            "s30l_reliable_max_mean": max(r["lat_mean"] for r in ok),
            "s30l_failures": [{"idx": r["idx"], "err": r["err_kcal"],
                               "lat_mean": r["lat_mean"],
                               "lat_max": r["lat_max"]} for r in bad],
        }
        fmins = [r["lat_mean"] for r in bad]
        summ["mean_separates_s30l"] = bool(fmins) and min(fmins) > summ["s30l_reliable_max_mean"]
        summ["mean_separates_s66_and_s30l"] = bool(fmins) and min(fmins) > s66range[2]
        results[layer] = {"summary": summ, "rows": rows}
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[{layer}] S66 mean range [{s66range[0]:.3f}, {s66range[2]:.3f}]"
              f"  failures: " + (", ".join(
                  f"{b['idx']}(err{b['err_kcal']:+.0f},m{b['lat_mean']:.3f})"
                  for b in bad) or "none")
              + f"  mean-separates: {summ['mean_separates_s30l']}", flush=True)

    # --- final comparison table ------------------------------------------------
    print("\n=== layer sweep summary (off-large) ===", flush=True)
    print(f"{'layer':28s} {'S66 mean [min,med,max]':26s} {'relP95':>7s} "
          f"{'failMin':>8s} {'sep':>5s}")
    for layer, res in results.items():
        s = res["summary"]
        r = s["s66_mean_range"]
        fmins = [b["lat_mean"] for b in s["s30l_failures"]] or [float("nan")]
        rng = f"[{r[0]:.3f},{r[1]:.3f},{r[2]:.3f}]"
        print(f"{layer:28s} {rng:26s} {s['s30l_reliable_p95_mean']:7.3f} "
              f"{min(fmins):8.3f} {str(s['mean_separates_s30l']):>5s}", flush=True)
    print("\ndone -> data/layer_sweep_results.json", flush=True)


if __name__ == "__main__":
    main()
"""Energy-space OOD signals: per-atom interaction-energy residual + latent-kNN consistency.

The latent-distance proxy (:mod:`activation_ood`) measures *input novelty* --
how unusual an atom's deep features are vs the training distribution. This
module adds signals that look at what the network *outputs* for each atom, or
at the system level:

Per-atom energy residual (``eZ``)
    MACE decomposes the total energy per atom: ``E = sum(E0_i) + sum(f_i)``,
    where ``f_i`` is the per-atom interaction energy -- literally the output of
    the model's ``scale_shift`` block. Reference mean/std of ``f_i`` are built
    per element from the OFF23 test split. Two molecule-level statistics:
    ``ezMean`` (mean |z|, the novelty-style score) and ``ezSign`` (signed mean
    z). ``ezSign`` matters because a *coherent* per-atom energy shift -- which
    is what a collective binding error looks like -- survives averaging with
    sqrt(N) better statistics, while |z| throws the sign away and cannot see it.

Latent-kNN energy consistency (``kZ``)
    For each atom, predict ``f_i`` from the k nearest same-element atoms in
    latent space and score ``|f_i - f_kNN| / sd_resid_el`` (calibrated
    leave-one-out on the reference). "Does the model assign this atom the
    energy its latent neighbours usually get?"

Layer-contribution profile (``profZ`` / ``fracDeep``, system-level)
    MACE's readouts decompose each atom's interaction energy by depth:
    ``f_i = scale * (r_shallow_i + r_deep_i)`` (verified to ~1e-7 eV; the
    shallow readout is ``readouts.0.linear``, the deep one
    ``readouts.1.linear_2``). The molecule-mean of each contribution, and the
    deep fraction ``mean(r_deep)/mean(f)``, are compared to the reference
    frame distribution. A system whose prediction is carried by a different
    layer balance than anything in training is suspicious even if every
    individual atom looks normal -- and because these are molecule-level
    statistics they can in principle see coherent (collective) errors.

All signals are one forward pass at inference (plus a cached reference file).
As with the latent proxy, scores are relative to the reference they were built
against -- compare molecules scored against the same reference only.

The decisive test (``--separation``) asks whether any signal separates MACE's
reliable from unreliable association energies on S66 (held-out,
in-distribution positives vs CCSD(T)/CBS) + S30L-CI (vs wB97X-D3/QZ).

Example
-------
>>> import mace_calc as mc
>>> from energy_ood import EnergyReference, score_molecule
>>> ref = EnergyReference.load_or_build()            # builds once (~6 min for 2500 frames)
>>> a = mc.smiles_to_atoms("CC(=O)Nc1ccc(cc1)O")
>>> score_molecule(a, "CC(=O)Nc1ccc(cc1)O", ref)
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch import). ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import collections
import logging
import random
from pathlib import Path

import numpy as np

import mace_calc as mc  # noqa: E402  (sets OMP guards, loads mace/torch)
from activation_ood import (  # noqa: E402
    ATOM_LAYER, _iter_frames, _nn_cos_dist, _unit_normalize,
    download_off23_test_data,
)
from inspect_activations import capture_activations  # noqa: E402

log = logging.getLogger("energy_ood")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# The per-atom interaction energy f_i = E_i - E0_i is the output of the model's
# ScaleShiftBlock (see ScaleShiftMACE.forward: node_inter_es after scale_shift,
# summed into the total energy alongside the E0s).
ENERGY_LAYER = "scale_shift"
# Per-atom readout contributions by depth: f_i = scale * (r_shallow + r_deep)
# (+ shift, 0 for the OFF23 models) -- verified numerically to <1e-6 eV.
DEEP_LAYERS = ["readouts.0.linear", "readouts.1.linear_2"]
K = 8                       # kNN neighbours for the consistency signal
_CHUNK = 2048               # query block size for the chunked kNN matmuls

_REPO = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO / "data"


def _pool_path(model: str) -> Path:
    return _DATA_DIR / f"energy_pool_{model}.npz"


# ---------------------------------------------------------------------------
# Reference: per-atom (latent vector, f_i, element, layer contributions) + stats
# ---------------------------------------------------------------------------
class EnergyReference:
    """Cached energy-space reference built from OFF23 test-split frames.

    Stores, per reference atom: the unit-normalized latent vector (ATOM_LAYER),
    the per-atom interaction energy ``f_i``, the per-depth readout
    contributions, and the element label; plus per-element calibration
    (f mean/std, self-kNN residual scale) and molecule-level calibration for
    the layer profile (mean contribution / deep fraction over frames).
    """

    def __init__(self, vecs, f, elements, f_stats, resid_stats, lat_stats,
                 deep=None):
        self.vecs = vecs                  # (Na, D) float32, unit-normalized
        self.f = f                        # (Na,) float64 interaction energies (eV)
        self.elements = np.asarray(elements, dtype=object)
        self.f_stats = f_stats            # {el: {mu, sd, p99}}  of f_i
        self.resid_stats = resid_stats    # {el: {sd, p99}}    of |f - f_kNN| (LOO)
        self.lat_stats = lat_stats        # {el: {p99}}        of latent NN distance
        self.deep = deep or {}            # see _build_deep below

    # -- build --------------------------------------------------------------
    @classmethod
    def build(cls, n_frames: int = 2500, max_atoms: int = 100,
              model: str = "off-medium", dtype: str = "float64",
              k: int = K, seed: int = 0, out: Path | None = None) -> "EnergyReference":
        xyz_dir = download_off23_test_data()
        calc = mc.get_calculator(model=model, dtype=dtype)  # load model ONCE
        rng = random.Random(seed)

        # reservoir sampling over the lazy frame stream
        reservoir, seen, skipped = [], 0, 0
        for frame in _iter_frames(xyz_dir):
            seen += 1
            if len(frame) > max_atoms:
                skipped += 1
                continue
            if len(reservoir) < n_frames:
                reservoir.append(frame)
            else:
                j = rng.randint(0, seen - 1)
                if j < n_frames:
                    reservoir[j] = frame
        log.info("Sampled %d frames (%d seen, %d skipped for >%d atoms).",
                 len(reservoir), seen, skipped, max_atoms)

        from ase.data import chemical_symbols

        vecs, f_list, els = [], [], []
        deep_lists = {layer: [] for layer in DEEP_LAYERS}
        spans = []                       # (start, end) atom rows per frame
        n_done = 0
        for kk, frame in enumerate(reservoir, 1):
            frame.calc = calc
            try:
                acts = capture_activations(
                    frame, layers=[ATOM_LAYER, ENERGY_LAYER] + DEEP_LAYERS,
                    model=model, dtype=dtype)
            except Exception as exc:
                log.warning("frame %d failed (%s); skipping", kk, exc)
                continue
            na = len(frame)
            av, fv = acts[ATOM_LAYER].numpy(), acts[ENERGY_LAYER].numpy()
            if av.shape[0] != na or fv.shape[0] != na:
                log.warning("frame %d shape mismatch; skipping", kk)
                continue
            vecs.append(_unit_normalize(av))
            f_list.append(fv.reshape(na, -1)[:, 0])           # (n_atoms,) head 0
            els.extend(chemical_symbols[int(z)] for z in frame.get_atomic_numbers())
            for layer in DEEP_LAYERS:
                deep_lists[layer].append(acts[layer].reshape(na, -1)[:, 0])
            end = sum(len(x) for x in f_list)
            spans.append((end - na, end))
            if kk % 250 == 0:
                log.info("  processed %d/%d frames", kk, len(reservoir))
        n_done = len(f_list)

        vecs = np.concatenate(vecs, 0)
        fs = np.concatenate(f_list, 0).astype(np.float64)
        els = np.array(els, dtype=object)
        deep_vals = {layer: np.concatenate(deep_lists[layer]).astype(np.float64)
                     for layer in DEEP_LAYERS}
        log.info("Reference: %d atoms, %d frames.", vecs.shape[0], n_done)

        f_stats, resid_stats, lat_stats = cls._calibrate(vecs, fs, els, k)
        deep = cls._build_deep(deep_vals, els, fs, spans, n_done)
        ref = cls(vecs, fs, els, f_stats, resid_stats, lat_stats, deep)
        out = Path(out or _pool_path(model))
        out.parent.mkdir(parents=True, exist_ok=True)
        ref.save(out)
        log.info("Saved energy reference -> %s", out)
        return ref

    @staticmethod
    def _calibrate(vecs, f, els, k):
        """Per-element stats: f mean/sd, LOO-kNN residual scale, latent NN p99."""
        idx_by_el = collections.defaultdict(list)
        for i, e in enumerate(els):
            idx_by_el[e].append(i)

        f_stats, resid_stats, lat_stats = {}, {}, {}
        for el, idx in idx_by_el.items():
            idx = np.array(idx)
            fi = f[idx]
            mu, sd = float(fi.mean()), float(fi.std())
            # self-kNN distance percentiles (in-dist latent scale, as in the pool)
            V = vecs[idx]
            nn_d = []
            for s in range(0, len(idx), 2048):
                sims = V[s:s + 2048] @ V.T
                np.fill_diagonal(sims[:, s:s + 2048], -1.0)   # exclude self
                nn_d.append(1.0 - sims.max(axis=1))
            nn_d = np.concatenate(nn_d)
            lat_stats[el] = {"p95": float(np.percentile(nn_d, 95)),
                             "p99": float(np.percentile(nn_d, 99)),
                             "n": int(len(idx))}

            # LOO kNN energy prediction -> residual scale (calibration for kZ)
            f_pred = EnergyReference._knn_f(vecs[idx], els[idx], vecs, f, els,
                                            k=k, exclude=idx)
            resid = fi - f_pred
            rsd = float(np.abs(resid).std())
            resid_stats[el] = {"sd": rsd,
                               "p95": float(np.percentile(np.abs(resid), 95)),
                               "p99": float(np.percentile(np.abs(resid), 99)),
                               "n": int(len(idx))}
            f_stats[el] = {"mu": mu, "sd": sd,
                           "p99": float(np.percentile(np.abs(fi - mu), 99)),
                           "n": int(len(idx))}
        return f_stats, resid_stats, lat_stats

    @staticmethod
    def _build_deep(deep_vals, els, f, spans, n_frames):
        """Stats for the layer-contribution profile.

        Returns {"layers", "vals", "el_stats", "mol", "frac"}: per-layer
        per-atom values, per-element mean/sd, and -- the system-level part --
        the distribution over reference frames of each molecule-mean
        contribution (``mol``) and of the deep fraction
        ``mean(r_deep)/mean(f)`` (``frac``).
        """
        layers = list(deep_vals)
        deep_layer = layers[-1]

        def _zs(v):
            mu, sd = float(v.mean()), float(v.std())
            z = (v - mu) / sd
            return {"mu": mu, "sd": sd, "p99": float(np.percentile(np.abs(z), 99))}

        mol = {}
        for layer in layers:
            means = np.array([deep_vals[layer][a:b].mean() for a, b in spans])
            mol[layer] = _zs(means)
        frac = _zs(np.array([deep_vals[deep_layer][a:b].mean() / f[a:b].mean()
                             for a, b in spans]))

        el_stats = {layer: {} for layer in layers}
        idx_by_el = collections.defaultdict(list)
        for i, e in enumerate(els):
            idx_by_el[e].append(i)
        for layer in layers:
            vals = deep_vals[layer]
            for el, idx in idx_by_el.items():
                v = vals[idx]
                el_stats[layer][el] = {"mu": float(v.mean()),
                                       "sd": float(v.std()), "n": int(len(idx))}
        return {"layers": layers, "vals": deep_vals, "el_stats": el_stats,
                "mol": mol, "frac": frac}

    @staticmethod
    def _knn_f(q, q_els, ref_vecs, ref_f, ref_els, k=K, exclude=None):
        """Predict each query row's f from the mean f of its k nearest
        same-element reference vectors (cosine). ``exclude`` gives, per query,
        a reference row index to drop from its own candidate set (LOO)."""
        nq = q.shape[0]
        f_pred = np.full(nq, np.nan)
        groups = collections.defaultdict(list)
        for i, e in enumerate(q_els):
            groups[e].append(i)
        ref_idx = collections.defaultdict(list)
        for j, e in enumerate(ref_els):
            ref_idx[e].append(j)

        for el, rows in groups.items():
            m = np.array(ref_idx.get(el, []), dtype=int)
            fallback = m.size == 0
            if fallback:
                m = np.arange(len(ref_f))          # element unseen: global pool
            fr = ref_f[m]
            R = ref_vecs[m]
            for s in range(0, len(rows), _CHUNK):
                ch = np.array(rows[s:s + _CHUNK])
                sims = q[ch] @ R.T
                if exclude is not None and not fallback:
                    # drop each query's own column (LOO)
                    cols = np.searchsorted(m, exclude[ch])
                    sims[np.arange(len(ch)), cols] = -2.0
                kk = min(k, sims.shape[1])
                if kk < sims.shape[1]:
                    part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
                else:
                    part = np.tile(np.arange(sims.shape[1]), (len(ch), 1))
                f_pred[ch] = fr[part].mean(axis=1)
        return f_pred

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path | str):
        np.savez_compressed(
            path, vecs=self.vecs, f=self.f, elements=self.elements,
            f_stats=np.array(self.f_stats, dtype=object),
            resid_stats=np.array(self.resid_stats, dtype=object),
            lat_stats=np.array(self.lat_stats, dtype=object),
            deep=np.array(self.deep, dtype=object),
        )

    @classmethod
    def load(cls, path: Path | str) -> "EnergyReference":
        d = np.load(path, allow_pickle=True)
        return cls(d["vecs"], d["f"].astype(np.float64), d["elements"],
                   d["f_stats"].item(), d["resid_stats"].item(),
                   d["lat_stats"].item(),
                   d["deep"].item() if "deep" in d.files else None)

    @classmethod
    def load_or_build(cls, model: str = "off-medium", **build_kw) -> "EnergyReference":
        p = _pool_path(model)
        if p.exists():
            log.info("Loading cached energy reference from %s", p)
            return cls.load(p)
        return cls.build(model=model, **build_kw)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def energy_ood_scores(atoms, ref: EnergyReference, model: str = "off-medium",
                      dtype: str = "float64", k: int = K) -> dict:
    """Per-atom energy-space OOD scores for one structure.

    Returns per-atom arrays: ``f`` (interaction energy, eV), ``ez`` (|f-mu_el|
    / sd_el), ``kz`` (|f - f_kNN| / sd_resid_el, LOO-calibrated), and ``lat``
    (latent NN distance, for contrast with :mod:`activation_ood`); plus the
    molecule-level scalars ``ezs`` (signed mean z of f -- the statistic that
    can see coherent shifts) and ``prof`` (per-layer z of the molecule-mean
    readout contribution) / ``frac`` (z of the deep fraction).
    """
    from ase.data import chemical_symbols

    layers = [ATOM_LAYER, ENERGY_LAYER] + list(ref.deep.get("layers", []))
    if not (hasattr(atoms, "calc") and atoms.calc is not None
            and hasattr(atoms.calc, "models")):
        mc.attach(atoms, model=model, dtype=dtype)
    acts = capture_activations(atoms, layers=layers, model=model, dtype=dtype)
    na = len(atoms)
    f = acts[ENERGY_LAYER].reshape(na, -1)[:, 0].numpy().astype(np.float64)
    q = _unit_normalize(acts[ATOM_LAYER].numpy())
    els = [chemical_symbols[int(z)] for z in atoms.get_atomic_numbers()]

    # signal 1: element-wise energy residual z-scores (abs + signed mean)
    ez = np.full(na, np.nan)
    for i, e in enumerate(els):
        st = ref.f_stats.get(e)
        if st and st["sd"] > 0:
            ez[i] = abs(f[i] - st["mu"]) / st["sd"]
    zsigned = (f - np.array([ref.f_stats.get(e, {}).get("mu", np.nan) for e in els]))
    zsigned /= np.array([ref.f_stats.get(e, {}).get("sd", np.nan) for e in els])
    ezs = float(np.nanmean(zsigned))

    # signal 2: latent-kNN energy consistency
    f_pred = EnergyReference._knn_f(q, els, ref.vecs, ref.f, ref.elements, k=k)
    kz = np.full(na, np.nan)
    for i, e in enumerate(els):
        st = ref.resid_stats.get(e)
        if st and st["sd"] > 0:
            kz[i] = abs(f[i] - f_pred[i]) / st["sd"]

    # signal 3 (contrast): latent NN distance to same-element refs
    lat = np.full(na, np.nan)
    by_el = collections.defaultdict(list)
    for j, e in enumerate(ref.elements):
        by_el[e].append(j)
    idx_by_el = collections.defaultdict(list)
    for i, e in enumerate(els):
        idx_by_el[e].append(i)
    for e, idxs in idx_by_el.items():
        m = np.array(by_el.get(e, []), dtype=int)
        if m.size == 0:
            m = np.arange(ref.vecs.shape[0])
        lat[idxs] = _nn_cos_dist(q[idxs], ref.vecs[m])

    # signal 4: layer-contribution profile (system level)
    prof, frac = {}, float("nan")
    for layer in ref.deep.get("layers", []):
        r = acts[layer].reshape(na, -1)[:, 0].numpy().astype(np.float64)
        mol = ref.deep.get("mol", {}).get(layer)
        if mol and mol["sd"] > 0:
            prof[layer] = float((r.mean() - mol["mu"]) / mol["sd"])
    deep_layer = ref.deep.get("layers", [None])[-1]
    if deep_layer is not None and deep_layer in acts:
        fr_stats = ref.deep.get("frac", {})
        r_deep = acts[deep_layer].reshape(na, -1)[:, 0].numpy().astype(np.float64).mean()
        if abs(f.mean()) > 1e-9 and fr_stats.get("sd", 0) > 0:
            ratio = r_deep / f.mean()
            frac = float((ratio - fr_stats["mu"]) / fr_stats["sd"])

    return {"f": f, "ez": ez, "kz": kz, "lat": lat, "elements": els,
            "ezs": ezs, "zsigned": zsigned, "prof": prof, "frac": frac}


def score_molecule(atoms, smi, ref: EnergyReference, model: str = "off-medium",
                   dtype: str = "float64", k: int = K, top: int = 5) -> dict:
    """Score a molecule with all signals and print a summary."""
    r = energy_ood_scores(atoms, ref, model=model, dtype=dtype, k=k)
    print(f"\n{atoms.get_chemical_formula()}  E = {atoms.get_potential_energy():.4f} eV")
    print(f"  ezMean (|z|)     : {np.nanmean(r['ez']):.3f}   "
          f"ezSign (coherent): {r['ezs']:+.3f}")
    print(f"  kNN consistency  : mean = {np.nanmean(r['kz']):.3f}  "
          f"max = {np.nanmax(r['kz']):.3f}")
    print(f"  latent dist      : mean = {np.nanmean(r['lat']):.3f}  "
          f"max = {np.nanmax(r['lat']):.3f}")
    for layer, z in r["prof"].items():
        tag = "deep" if layer == ref.deep["layers"][-1] else "shallow"
        print(f"  layer profile    : {tag:5s} mean-z = {r['prof'][layer]:+.3f}")
    print(f"  deep fraction z  : {r['frac']:+.3f}")
    order = np.argsort(-np.abs(np.nan_to_num(r["zsigned"], nan=-np.inf)))
    print(f"  top-{top} atoms by |signed ez|:")
    for i in order[:top]:
        print(f"    #{i:3d} {r['elements'][i]:>2s}  f={r['f'][i]:9.3f} eV  "
              f"ez={r['ez'][i]:.2f}  kz={r['kz'][i]:.2f}  lat={r['lat'][i]:.3f}")
    return r


# ---------------------------------------------------------------------------
# Decisive test: does the signal separate reliable from unreliable energies?
# ---------------------------------------------------------------------------
def separation(model: str = "off-medium", dtype: str = "float64",
               reliable: float = 10.0, k: int = K, n_frames: int = 2500):
    """S66 (in-distribution positives, CCSD(T)/CBS refs) + S30L-CI (vs wB97X-D3).

    For each system: the MACE association-energy error in kcal/mol (an OFFLINE
    label -- not available at inference) and the NN signals on the
    dimer/complex. The verdict per signal: does the reliable set's p95 fall
    below the unreliable set's minimum? Prints the full table sorted by the
    layer-profile z (the new system-level signal).
    """
    import ood_datasets as od

    ref = EnergyReference.load_or_build(model=model, n_frames=n_frames)
    deep_layer = ref.deep.get("layers", [None])[-1]

    def sig(atoms):
        atoms.calc = None  # stale-calc guard
        r = energy_ood_scores(atoms, ref, model=model, dtype=dtype, k=k)
        prof = r["prof"].get(deep_layer, float("nan"))
        # (ezMean, ezSign, profDeep, fracDeep, kzMax, latMean, latMax)
        return (float(np.nanmean(r["ez"])), r["ezs"], prof, r["frac"],
                float(np.nanmax(r["kz"])), float(np.nanmean(r["lat"])),
                float(np.nanmax(r["lat"])))

    rows = []  # (set, idx, formula, err, signals)
    print(f"computing S66 (66 in-distribution positives, {model})...", flush=True)
    bind = od.s66_bind_ref()
    for idx, label, dimer, mA, mB in od.fetch_s66():
        eint = (od._mace_energy_eV(dimer, model=model, dtype=dtype)
                - od._mace_energy_eV(mA, model=model, dtype=dtype)
                - od._mace_energy_eV(mB, model=model, dtype=dtype)) * od._KCAL
        err = eint - bind.get(idx, float("nan"))
        rows.append(("S66", idx, dimer.get_chemical_formula(), err, sig(dimer)))
    print("computing S30L (23 test systems)...", flush=True)
    for n, ch, host, guest, cplx in od.load_s30l():
        eas = (od._mace_energy_eV(cplx, model=model, dtype=dtype)
               - od._mace_energy_eV(host, model=model, dtype=dtype)
               - od._mace_energy_eV(guest, model=model, dtype=dtype)) * od._KCAL
        err = eas - od.s30l_computed_ref(n, "wB97XD3")
        rows.append(("S30L", n, cplx.get_chemical_formula(), err, sig(cplx)))

    keys = ("ezMean", "ezSign", "profDeep", "fracDeep", "kzMax", "latMean", "latMax")

    def _pct(v, p):
        v = np.sort(np.array(v))
        return float(v[int(round(p * (v.size - 1)))]) if v.size else float("nan")

    def _rng(sub, ki):
        v = np.array([r[4][ki] for r in sub], dtype=float)
        return f"[{v.min():+.3f}, med {np.median(v):+.3f}, {v.max():+.3f}]"

    s66 = [r for r in rows if r[0] == "S66"]
    s30 = [r for r in rows if r[0] == "S30L"]
    s30_bad = [r for r in s30 if abs(r[3]) >= reliable]
    s30_rel = [r for r in s30 if abs(r[3]) < reliable]
    rel = s66 + s30_rel
    errs = np.array([abs(r[3]) for r in s66] + [abs(r[3]) for r in s30_rel])
    bad_errs = np.array([abs(r[3]) for r in s30_bad])

    def _stats(v):
        return f"n={v.size}, MAE={v.mean():.2f}, median={np.median(v):.2f}, max={v.max():.2f}"

    print(f"\n=== Energy-space OOD vs reliability ({model}) ===")
    print(f"  reliable label: |err| < {reliable:.0f} kcal/mol")
    print(f"  S66 (in-dist positives):        err {_stats(errs[:len(s66)])}")
    print(f"  S30L reliable:                  err {_stats(errs[len(s66):])}")
    print(f"  S30L UNreliable (|err|>={reliable:.0f}): err {_stats(bad_errs)}")
    for ki, key in enumerate(keys):
        print(f"  {key:8s}  S66 {_rng(s66, ki)}  reliable {_rng(rel, ki)}  "
              f"unreliable {_rng(s30_bad, ki)}")
    print("\n  separation (reliable = S66 + reliable-S30L; unreliable = S30L failures):")
    if s30_bad:
        for ki, key in enumerate(keys):
            rel_v = np.abs([r[4][ki] for r in rel])
            bad_v = np.abs([r[4][ki] for r in s30_bad])
            p95, mn = _pct(rel_v, .95), bad_v.min()
            print(f"    {key:8s}: reliable p95={p95:.3f} vs unreliable min={mn:.3f} "
                  f"{'-> SEPARATED' if p95 < mn else '-> OVERLAP'}")

    hdr = "  {:5s} {:>3} {:16s} {:>8s} " + " ".join(f"{k_:>8s}" for k_ in keys)
    print(hdr.format("set", "#", "formula", "err", *keys))
    fmt = "  {:5s} {:3} {:16s} {:8.2f} " + " ".join(["{:8.3f}"] * len(keys))
    for s, i, f_, e, sg in sorted(rows, key=lambda r: abs(r[4][2])):
        lab = "OK" if abs(e) < reliable else "FAIL"
        print(fmt.format(s, i, f_, e, *sg) + f"  {lab:>5s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_atoms(arg: str):
    from ase.build import molecule as _molecule
    try:
        return _molecule(arg)
    except (KeyError, ValueError):
        return mc.smiles_to_atoms(arg)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Energy-space OOD scoring (per-atom interaction-energy "
                    "residual + latent-kNN consistency + layer profile) for MACE.")
    p.add_argument("molecule", nargs="?", default="CC(=O)Nc1ccc(cc1)O",
                   help="ASE molecule name or SMILES (default: paracetamol)")
    p.add_argument("--model", default="off-medium")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--n-frames", type=int, default=2500)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--build-ref", action="store_true",
                   help="(re)build the energy reference from the OFF23 test split")
    p.add_argument("--separation", action="store_true",
                   help="run the S66/S30L reliability-separation verdict (the "
                        "decisive test: does an energy-space signal separate "
                        "reliable from unreliable association energies?)")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)

    if args.separation:
        separation(model=args.model, dtype=args.dtype, k=args.k,
                   n_frames=args.n_frames)
        return

    if args.build_ref or not _pool_path(args.model).exists():
        ref = EnergyReference.build(n_frames=args.n_frames, model=args.model,
                                    dtype=args.dtype)
    else:
        ref = EnergyReference.load(_pool_path(args.model))

    atoms = _build_atoms(args.molecule)
    score_molecule(atoms, args.molecule, ref, model=args.model,
                   dtype=args.dtype, k=args.k, top=args.top)


if __name__ == "__main__":
    main()
"""Activation-based out-of-distribution (OOD) scoring for MACE-OFF23.

A molecule is "out-of-distribution" for MACE-OFF23 when it contains atoms or
local motifs the model rarely saw in training. With a single deterministic
checkpoint we cannot compute ensemble/force uncertainty (the gold standard for
ML-potential OOD); instead we use the standard **latent-distance proxy**.

Per-atom OOD
    The deep per-atom activation vector (``interactions.1.linear``, 2048-d) is
    scored by cosine distance to the nearest reference atom **of the same
    element** in a pool built from real MACE-OFF23 training data. This is
    well-calibrated: the pool's own per-element nearest-neighbor distance
    percentiles (p50/p95/p99) define a meaningful "in-distribution" scale, and a
    query atom is flagged when it exceeds its element's p99.

Per-bond OOD
    The radial-MLP (``interactions.0.conv_tp_weights``) output is a deterministic
    function of the bond length alone, so its latent distance to the nearest
    same-pair-type reference edge is ~0 whenever the exact length is common in
    training and larger for unusual lengths. That makes a *calibrated threshold*
    degenerate (standard lengths all match exactly), so the bond score is
    reported as a continuous "how unusual is this bond length" value and the
    flag is **relative within the molecule** (the top-10% most-unusual bonds),
    not an absolute verdict. The absolute OOD verdict comes from the atoms.

The reference pool is built (once, then cached) from the public MACE-OFF23 test
split -- the same SPICE/wB97M-D3(BJ) distribution the model was trained on, just
the held-out 5%, so it is a clean "in-distribution" sample. Source:
``doi.org/10.17863/CAM.107498`` (Cambridge data repository), 81 MB.

This is a **heuristic exploratory tool**, not a calibrated detector. Scores are
*relative* -- meaningful when molecules are scored against the same pool -- and
the pool is sampled (a few hundred frames), not exhaustive. Use it to localize
and explore which parts of a molecule are unusual for MACE, not as a binary
in/out verdict.

Example
-------
>>> from activation_ood import ReferencePool, score_molecule
>>> import mace_calc as mc
>>> pool = ReferencePool.load_or_build()           # one-time ~81 MB download + build
>>> a = mc.smiles_to_atoms("CC(=O)Nc1ccc(cc1)O")   # paracetamol
>>> score_molecule(a, "CC(=O)Nc1ccc(cc1)O", pool)
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guards (must precede torch import). ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import logging
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

import mace_calc as mc  # noqa: E402  (sets OMP guards, loads mace/torch)
from inspect_activations import capture_activations, mace_edge_index  # noqa: E402

log = logging.getLogger("activation_ood")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Public MACE-OFF23 test split (Cambridge repository, 81 MB, MIT-licensed).
# Same SPICE/wB97M-D3 distribution as training, held out -> in-distribution ref.
OFF23_TEST_URL = (
    "https://www.repository.cam.ac.uk/bitstreams/"
    "cb8351dd-f09c-413f-921c-67a702a7f0c5/download"
)
OFF23_TEST_TARBALL = "test_large_neut_no_bad_clean.tar.gz"

# Layers used for the latent representation. Per-atom: deep mixing layer (2048-d).
# Per-pair: the radial/bond MLP (512-d at layer 0). Both are flat invariant vectors.
ATOM_LAYER = "interactions.1.linear"
PAIR_LAYER = "interactions.0.conv_tp_weights"

# Anchor data/ to the repo root (parent of code/) so the path is stable whether
# the script is run from the repo root or from code/.
_REPO = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO / "data"
_POOL_PATH = _DATA_DIR / "off23_pool.npz"


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_off23_test_data(dest: Path | str = _DATA_DIR / "off23_test") -> Path:
    """Download + extract the 81 MB MACE-OFF23 test split once. Returns the dir.

    Skips the download if the extracted directory already contains extxyz files.
    """
    dest = Path(dest)
    xyzs = sorted(glob.glob(str(dest / "**" / "*.xyz"), recursive=True)) if dest.exists() else []
    if xyzs:
        log.info("OFF23 test data already present (%d files in %s), skipping download.",
                 len(xyzs), dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    tgz = dest / OFF23_TEST_TARBALL
    if not tgz.exists():
        log.info("Downloading MACE-OFF23 test split (~81 MB) from Cambridge repository...")
        req = urllib.request.Request(OFF23_TEST_URL, headers={"User-Agent": "mace-ood-tool/1.0"})
        with urllib.request.urlopen(req) as r, open(tgz, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:6.1f} / {total/1e6:6.1f} MB", end="", flush=True)
            print()
        log.info("Downloaded %s (%.1f MB).", tgz, tgz.stat().st_size / 1e6)

    log.info("Extracting %s ...", tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(dest, filter="data")
    xyzs = sorted(glob.glob(str(dest / "**" / "*.xyz"), recursive=True))
    log.info("Extracted %d extxyz files into %s.", len(xyzs), dest)
    return dest


def _iter_frames(xyz_dir: Path | str):
    """Lazily yield ASE Atoms frames from all extxyz files in ``xyz_dir``."""
    from ase.io import iread

    paths = sorted(glob.glob(str(Path(xyz_dir) / "**" / "*.xyz"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no .xyz files found under {xyz_dir}")
    for p in paths:
        try:
            for frame in iread(p, format="extxyz"):
                yield frame
        except Exception as exc:  # skip unreadable files/frames
            log.warning("could not read %s as extxyz (%s); skipping", p, exc)


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (x / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Reference pool
# ---------------------------------------------------------------------------
class ReferencePool:
    """Cached latent reference built from MACE-OFF23 training-distribution frames.

    Stores unit-normalized per-atom activation vectors (``ATOM_LAYER``) with
    element labels and per-element nearest-neighbor distance percentiles (the
    calibrated in-distribution scale used to flag atoms), plus a bounded set of
    per-pair radial-MLP vectors (``PAIR_LAYER``) with pair-type labels for bond
    scoring.
    """

    def __init__(self, atom_vecs, atom_elements, atom_stats, pair_vecs, pair_types):
        self.atom_vecs = atom_vecs            # (Na, D) float32, unit-normalized
        self.atom_elements = np.asarray(atom_elements, dtype=object)
        self.atom_stats = atom_stats          # {element: {p50,p95,p99,mean,n}}
        self.pair_vecs = pair_vecs            # (Np, D) float32, unit-normalized
        self.pair_types = np.asarray(pair_types, dtype=object)

    @classmethod
    def build(cls, xyz_dir=None, n_frames: int = 250, max_atoms: int = 100,
              max_pairs_per_frame: int = 100, model: str = "off-medium",
              dtype: str = "float64", atom_layer: str = ATOM_LAYER,
              pair_layer: str = PAIR_LAYER, out: Path | str = _POOL_PATH,
              seed: int = 0) -> "ReferencePool":
        """Sample ``n_frames`` from the OFF23 test split and build the pool.

        ``max_atoms`` skips frames larger than this (keeps the pool drug-like and
        the build fast -- introduces a small-molecule bias; raise it for
        peptide-scale coverage). ``max_pairs_per_frame`` bounds the pool size by
        subsampling each frame's edges. Vectors are unit-normalized for cosine
        distance.
        """
        if xyz_dir is None:
            xyz_dir = download_off23_test_data()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        calc = mc.get_calculator(model=model, dtype=dtype)  # load model ONCE, reuse
        rng = random.Random(seed)

        # reservoir sampling over the lazy stream -> uniform n_frames without
        # needing the total count upfront
        reservoir: list = []
        seen = skipped = 0
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

        atom_vecs, atom_elements, pair_vecs, pair_types = [], [], [], []
        for k, frame in enumerate(reservoir, 1):
            frame.calc = calc
            try:
                acts = capture_activations(frame, layers=[atom_layer, pair_layer],
                                           model=model, dtype=dtype)
                ei = mace_edge_index(frame, model=model, dtype=dtype)
            except Exception as exc:
                log.warning("frame %d failed (%s); skipping", k, exc)
                continue
            from ase.data import chemical_symbols

            syms = [chemical_symbols[int(z)] for z in frame.get_atomic_numbers()]
            av = acts[atom_layer].numpy()
            pv = acts[pair_layer].numpy()
            if av.shape[0] != len(syms) or pv.shape[0] != ei.shape[1]:
                log.warning("frame %d shape mismatch; skipping", k)
                continue
            atom_vecs.append(_unit_normalize(av))
            atom_elements.extend(syms)
            # subsample this frame's edges to bound pool size
            E = ei.shape[1]
            keep = range(E) if E <= max_pairs_per_frame else rng.sample(range(E), max_pairs_per_frame)
            pair_vecs.append(_unit_normalize(pv[list(keep)]))
            for e in keep:
                a, b = int(ei[0, e]), int(ei[1, e])
                pair_types.append("-".join(sorted((syms[a], syms[b]))))
            if k % 25 == 0:
                log.info("  processed %d/%d frames", k, len(reservoir))

        atom_vecs = np.concatenate(atom_vecs, 0) if atom_vecs else np.zeros((0, 1), np.float32)
        pair_vecs = np.concatenate(pair_vecs, 0) if pair_vecs else np.zeros((0, 1), np.float32)
        atom_elements = np.array(atom_elements, dtype=object)
        pair_types = np.array(pair_types, dtype=object)
        log.info("Pool: %d atoms, %d pairs.", atom_vecs.shape[0], pair_vecs.shape[0])

        atom_stats = cls._self_stats(atom_vecs, atom_elements)
        pool = cls(atom_vecs, atom_elements, atom_stats, pair_vecs, pair_types)
        pool.save(out)
        log.info("Saved reference pool -> %s", out)
        return pool

    @staticmethod
    def _self_stats(vecs, labels, cap=2000):
        """Per-label nearest-other-neighbor distance percentiles (in-dist scale)."""
        import collections

        groups = collections.defaultdict(list)
        for i, lab in enumerate(labels):
            groups[lab].append(i)
        stats = {}
        for lab, idx in groups.items():
            idx = idx[:cap]  # cap pairwise cost for very common labels
            if len(idx) < 2:
                stats[lab] = {"p50": np.nan, "p95": np.nan, "p99": np.nan,
                              "mean": np.nan, "n": len(idx)}
                continue
            V = vecs[idx]
            sims = V @ V.T
            np.fill_diagonal(sims, -1.0)        # exclude self
            nn = 1.0 - sims.max(axis=1)          # nearest-other cosine distance
            stats[lab] = {"p50": float(np.percentile(nn, 50)),
                          "p95": float(np.percentile(nn, 95)),
                          "p99": float(np.percentile(nn, 99)),
                          "mean": float(nn.mean()), "n": len(idx)}
        return stats

    def save(self, path: Path | str):
        np.savez_compressed(
            path, atom_vecs=self.atom_vecs, atom_elements=self.atom_elements,
            atom_stats=np.array(self.atom_stats, dtype=object),
            pair_vecs=self.pair_vecs, pair_types=self.pair_types,
        )

    @classmethod
    def load(cls, path: Path | str = _POOL_PATH) -> "ReferencePool":
        d = np.load(path, allow_pickle=True)
        return cls(d["atom_vecs"], d["atom_elements"], d["atom_stats"].item(),
                   d["pair_vecs"], d["pair_types"])

    @classmethod
    def load_or_build(cls, **build_kw) -> "ReferencePool":
        if Path(_POOL_PATH).exists():
            log.info("Loading cached pool from %s", _POOL_PATH)
            return cls.load(_POOL_PATH)
        return cls.build(**build_kw)

    def _by_label(self, vecs, labels):
        import collections
        g = collections.defaultdict(list)
        for i, lab in enumerate(labels):
            g[lab].append(i)
        return {k: vecs[v] for k, v in g.items()}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _nn_cos_dist(query_unit, ref_matrix):
    """Cosine distance from each query row to its nearest ref row (both unit-norm)."""
    if ref_matrix.shape[0] == 0:
        return np.full(query_unit.shape[0], np.nan)
    sims = query_unit @ ref_matrix.T          # (nq, nref)
    return 1.0 - sims.max(axis=1)


def atom_ood_scores(atoms, pool: ReferencePool, model: str = "off-medium",
                    dtype: str = "float64", layer: str = ATOM_LAYER) -> dict:
    """Per-atom OOD = cosine distance to nearest same-element reference atom.

    ``flags`` marks atoms whose distance exceeds that element's p99 in the pool
    (a calibrated in-distribution threshold).
    """
    from ase.data import chemical_symbols

    if not hasattr(atoms, "calc") or atoms.calc is None or not hasattr(atoms.calc, "models"):
        mc.attach(atoms, model=model, dtype=dtype)
    acts = capture_activations(atoms, layers=[layer], model=model, dtype=dtype)
    q = _unit_normalize(acts[layer].numpy())
    els = [chemical_symbols[int(z)] for z in atoms.get_atomic_numbers()]

    by_el = pool._by_label(pool.atom_vecs, pool.atom_elements)
    dist = np.full(len(els), np.nan)
    used_global = []
    # batch by element: one matmul per element instead of per atom (matters for
    # ~1000-atom crystals; identical results to the per-atom loop).
    import collections
    idx_by_el = collections.defaultdict(list)
    for i, e in enumerate(els):
        idx_by_el[e].append(i)
    for e, idxs in idx_by_el.items():
        ref = by_el.get(e)
        if ref is not None and ref.shape[0] > 0:
            dist[idxs] = _nn_cos_dist(q[idxs], ref)
        else:
            used_global.append(e)
            dist[idxs] = _nn_cos_dist(q[idxs], pool.atom_vecs)
    flags = np.zeros(len(els), dtype=bool)
    for i, e in enumerate(els):
        thr = pool.atom_stats.get(e, {}).get("p99", np.nan)
        if not np.isnan(thr) and not np.isnan(dist[i]) and dist[i] > thr:
            flags[i] = True
    return {"distances": dist, "elements": els, "flags": flags,
            "used_global": used_global}


def pair_ood_scores(atoms, smi: str, pool: ReferencePool, model: str = "off-medium",
                    dtype: str = "float64", pair_layer: str = PAIR_LAYER,
                    top_pct: float = 10.0) -> dict:
    """Per-covalent-bond OOD = cosine distance to nearest same-pair-type ref edge.

    Because the radial MLP depends only on bond length, this distance is ~0 for
    lengths common in training and larger for unusual lengths -- a continuous
    "how unusual is this bond length" score. There is no calibrated absolute
    threshold (standard lengths all match exactly), so ``flags`` marks the
    ``top_pct`` most-unusual bonds *within this molecule* (relative), not an
    absolute verdict. The absolute OOD verdict comes from the atoms.
    """
    from rdkit import Chem
    from ase.data import chemical_symbols

    if not hasattr(atoms, "calc") or atoms.calc is None or not hasattr(atoms.calc, "models"):
        mc.attach(atoms, model=model, dtype=dtype)
    acts = capture_activations(atoms, layers=[pair_layer], model=model, dtype=dtype)
    pair_acts = acts[pair_layer].numpy()
    ei = mace_edge_index(atoms, model=model, dtype=dtype)
    edge_map = {frozenset((int(ei[0, e]), int(ei[1, e]))): e for e in range(ei.shape[1])}

    mol = Chem.MolFromSmiles(smi)            # heavy-atom mol; indices match ASE
    syms = [chemical_symbols[int(z)] for z in atoms.get_atomic_numbers()]
    by_type = pool._by_label(pool.pair_vecs, pool.pair_types)

    bonds, dists, types = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        e = edge_map.get(frozenset((i, j)))
        if e is None:
            continue  # bond longer than r_max (shouldn't happen for covalent)
        ptype = "-".join(sorted((syms[i], syms[j])))
        q = _unit_normalize(pair_acts[e:e + 1])
        ref = by_type.get(ptype)
        if ref is not None and ref.shape[0] > 0:
            d = _nn_cos_dist(q, ref)[0]
        else:
            d = _nn_cos_dist(q, pool.pair_vecs)[0]  # pair type unseen in pool
        dists.append(float(d))
        types.append(ptype)
        bonds.append((i, j))
    dists = np.array(dists)
    flags = np.zeros(len(dists), dtype=bool)
    if len(dists):
        thr = np.nanpercentile(dists, 100 - top_pct)
        flags = dists >= thr
    return {"bonds": bonds, "distances": dists, "pair_types": types, "flags": flags}


def score_molecule(atoms, smi, pool: ReferencePool, model: str = "off-medium",
                  dtype: str = "float64", top: int = 5) -> dict:
    """Score a molecule's atoms and bonds for OOD-ness and print a summary."""
    a = atom_ood_scores(atoms, pool, model=model, dtype=dtype)
    p = {"bonds": [], "distances": np.array([]), "pair_types": [], "flags": np.array([], bool)}
    if smi:
        p = pair_ood_scores(atoms, smi, pool, model=model, dtype=dtype)

    ad, af = a["distances"], a["flags"]
    pd_, pf = p["distances"], p["flags"]
    print(f"\n{atoms.get_chemical_formula()}  E = {atoms.get_potential_energy():.4f} eV")
    print(f"  atoms: {len(ad)}  mean OOD = {np.nanmean(ad):.3f}  "
          f"max = {np.nanmax(ad):.3f}  flagged(>element p99) = {int(af.sum())}")
    if a["used_global"]:
        print(f"  ! elements not in reference pool (strong OOD signal): "
              f"{sorted(set(a['used_global']))}")
    if len(pd_):
        print(f"  bonds: {len(pd_)}  mean OOD = {np.nanmean(pd_):.3f}  "
              f"max = {np.nanmax(pd_):.3f}  flagged(top-10% in molecule) = {int(pf.sum())}")

    order = np.argsort(-np.where(np.isnan(ad), -np.inf, ad))
    print(f"  top-{top} most-OOD atoms:")
    for i in order[:top]:
        if np.isnan(ad[i]):
            break
        print(f"    #{i:3d} {a['elements'][i]:>2s}  dist={ad[i]:.3f}  "
              f"{'** flagged' if af[i] else ''}")
    if len(pd_):
        porder = np.argsort(-np.where(np.isnan(pd_), -np.inf, pd_))
        print(f"  top-{top} most-OOD bonds:")
        for k in porder[:top]:
            if np.isnan(pd_[k]):
                break
            i, j = p["bonds"][k]
            print(f"    {a['elements'][i]:>2s}(#{i})-{a['elements'][j]:>2s}(#{j})  "
                  f"[{p['pair_types'][k]}]  dist={pd_[k]:.3f}  "
                  f"{'** flagged' if pf[k] else ''}")
    return {"atom": a, "pair": p}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_atoms(arg: str):
    from ase.build import molecule as _molecule
    try:
        return _molecule(arg)
    except (KeyError, ValueError):
        return mc.smiles_to_atoms(arg)


def _looks_like_ase_name(arg: str) -> bool:
    from ase.build import molecule as _m
    try:
        _m(arg)
        return True
    except Exception:
        return False


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Activation-based OOD scoring for MACE-OFF23 "
                    "(latent distance to a reference pool from the OFF23 test split).",
    )
    p.add_argument("molecule", nargs="?", default="CC(=O)Nc1ccc(cc1)O",
                   help="ASE molecule name or SMILES (default: paracetamol)")
    p.add_argument("--model", default="off-medium")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--build-pool", action="store_true",
                   help="(re)build the reference pool from the OFF23 test split")
    p.add_argument("--n-frames", type=int, default=250,
                   help="number of frames to sample for the pool (default 250)")
    p.add_argument("--max-atoms", type=int, default=100,
                   help="skip pool frames larger than this (default 100)")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)

    if args.build_pool or not _POOL_PATH.exists():
        pool = ReferencePool.build(n_frames=args.n_frames, max_atoms=args.max_atoms,
                                   model=args.model, dtype=args.dtype)
    else:
        pool = ReferencePool.load(_POOL_PATH)

    atoms = _build_atoms(args.molecule)
    smi = args.molecule if not _looks_like_ase_name(args.molecule) else None
    score_molecule(atoms, smi, pool, model=args.model, dtype=args.dtype, top=args.top)


if __name__ == "__main__":
    main()
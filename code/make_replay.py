"""Build the replay (``--pt_train_file``) set that keeps a fine-tune from forgetting.

Why this file exists
--------------------
``mace_run_train`` has ``--multiheads_finetuning`` ON by default: it trains the
new level of theory on one head while *replaying* foundation-level data through
a second head, which is what stops the shared layers from drifting. For a
Materials-Project foundation model MACE downloads that replay data itself. For
**MACE-OFF it cannot**, and it says so and silently continues single-head:

    WARNING: Using multiheads finetuning with a foundation model that is not a
    Materials Project model, need to provied a path to a pretraining file with
    --pt_train_file.

That warning is exactly what the rotaxane fine-tunes hit (``runs/probe.log``),
so they ran with no replay at all -- and destroyed the base model's non-covalent
chemistry (``forgetting.py``: S66 MAE 0.22 -> 7.22 kcal/mol).

This script builds the missing file from the local OFF23 test split
(``data/off23_test/test_large_neut_all.xyz``, 50195 frames with wB97M-D3 labels,
13908 of them ``DES370K Dimers`` -- the very non-covalent chemistry that was
lost). Keys are rewritten to ``REF_energy``/``REF_forces`` to match the GFN2
training file, because ``--energy_key``/``--forces_key`` apply to every head.

Caveat: this is the OFF23 *test* split (the only OFF23 data we hold locally);
ideal replay is a subsample of the actual SPICE training split. Same
distribution, but note that the latent reference pool is drawn from this file
too, so pool and replay are not independent samples.

    python make_replay.py --n 4000 --out ../data/off23_replay
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger("make_replay")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_REPO = Path(__file__).resolve().parent.parent
SRC = _REPO / "data" / "off23_test" / "test_large_neut_all.xyz"
N_TOTAL = 50195          # frames in SRC (grep -c 'Properties=')


def build(n: int = 4000, out: Path | str = _REPO / "data" / "off23_replay",
          src: Path = SRC, valid_frac: float = 0.1, max_atoms: int = 60):
    """Evenly subsample ``src`` -> ``<out>_train.xyz`` / ``<out>_valid.xyz``.

    Even striding (not random) keeps the config_type mix of the source, so the
    dimers stay proportionally represented. ``max_atoms`` keeps the replay
    frames small enough to batch cheaply next to the 144-atom rotaxane frames.
    """
    from ase.io import iread, write

    out = Path(out)
    stride = max(1, N_TOTAL // n)
    kept, types = [], Counter()
    for i, at in enumerate(iread(src, index=":", format="extxyz")):
        if i % stride:
            continue
        if len(at) > max_atoms:
            continue
        # ASE's extxyz reader moves energy/forces off info/arrays and onto a
        # SinglePointCalculator, so read them back through the calculator.
        try:
            e = float(at.get_potential_energy())
            f = at.get_forces()
        except Exception:
            continue
        at.info["REF_energy"] = float(e)
        at.arrays["REF_forces"] = f
        at.info.pop("MACE_energy", None)
        at.arrays.pop("MACE_forces", None)
        at.calc = None
        types[at.info.get("config_type", "SPICE (untyped)")] += 1
        kept.append(at)
        if len(kept) >= n:
            break
        if len(kept) % 500 == 0:
            log.info("kept %d frames (source frame %d)", len(kept), i)

    n_val = max(1, int(len(kept) * valid_frac))
    write(f"{out}_valid.xyz", kept[:n_val], format="extxyz")
    write(f"{out}_train.xyz", kept[n_val:], format="extxyz")
    log.info("replay written: %d train / %d valid -> %s_{train,valid}.xyz",
             len(kept) - n_val, n_val, out)
    for k, v in types.most_common():
        log.info("  %-24s %5d", k, v)
    els = sorted({s for at in kept for s in at.get_chemical_symbols()})
    log.info("  elements kept: %s", els)
    return Path(f"{out}_train.xyz"), Path(f"{out}_valid.xyz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--out", default=str(_REPO / "data" / "off23_replay"))
    p.add_argument("--max-atoms", type=int, default=60)
    a = p.parse_args()
    build(n=a.n, out=a.out, max_atoms=a.max_atoms)

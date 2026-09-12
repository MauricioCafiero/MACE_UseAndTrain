# GFN2-xTB finetuned MACE-OFF23 checkpoints

Two checkpoints, one per deployed foundation size, fine-tuned on 225 GFN2-xTB
labelled frames of a rotaxane (25 held out) **with a replay head**, so they
learn the new level of theory without losing the foundation model's chemistry.

| file | foundation | size | fit to GFN2 target (held-out) |
|---|---|---|---|
| `rot250M_replay.model` | MACE-OFF23 medium | 8.9 MB | 0.5 meV/atom, 30.6 meV/Å |
| `rot250L_replay.model` | MACE-OFF23 large | 26.6 MB | 0.6 meV/atom, 25.7 meV/Å |

`*_train.txt` are the full per-epoch training records.

## Loading

Both carry **two heads**, `['pt_head', 'Default']`. `Default` is the GFN2 head
you want; `pt_head` is the replay head holding the foundation's original
wB97M-D3 level. `MACECalculator` selects `Default` automatically:

```python
from mace.calculators import MACECalculator

calc = MACECalculator(model_paths="finetuned/rot250L_replay.model",
                      device="cpu", default_dtype="float64")
# -> "Using head Default out of ['pt_head', 'Default']"
```

They were trained on a GPU in float32; MACE upcasts on load. On a CPU-only
torch build the CUDA-saved TorchScript buffers need a map_location shim:

```python
import torch
_orig = torch.jit.load
torch.jit.load = lambda *a, **kw: _orig(*a, **{**kw, "map_location": kw.get("map_location", "cpu")})
```

Both keep the full 10-element OFF23 `z_table` (H, C, N, O, F, P, S, Cl, Br, I).

## How they were made

`modal run code/modal_finetune.py --foundation off-medium|off-large` —
20 epochs, `lr=5e-4`, batch 4, `energy_weight=forces_weight=100`, EMA +
ReduceLROnPlateau, float32. Two settings are load-bearing:

* `--pt_train_file` (built by `code/make_replay.py`) — MACE cannot self-supply
  replay data for a MACE-OFF foundation model and silently trains single-head
  without it, which destroys non-covalent chemistry.
* `e0s=finetune_mace.GFN2_E0S` — real isolated-atom energies. `--E0s=average`
  is degenerate on a single-molecule training set.

## Benchmarks (kcal/mol)

Interaction energies vs reference, from `python code/forgetting.py --stage report`:

| model | S66 (CCSD(T)/CBS) | S30L* (wB97X-D3) | rotaxane vs GFN2 |
|---|---|---|---|
| stock off-medium | 0.26 | 3.76 | 6.36 |
| **rot250M_replay** | 0.90 | 3.16 | 2.19 |
| stock off-large | 0.22 | 3.02 | 6.22 |
| **rot250L_replay** | 1.72 | 7.27 | 1.87 |
| GFN2-xTB (target level) | 0.73 | 5.55 | — |

\* excluding S30L 9/10, exohedral fullerenes that stock off-medium already
fails by 70–85 kcal/mol before any fine-tuning.

Validate any new fine-tune with `code/forgetting.py`, not with the latent OOD
score — see `OOD_NOTES.md`.

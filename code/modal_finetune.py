"""Fine-tune MACE-OFF23 on GFN2 labels on a Modal GPU, *with* the replay head.

This is the rewritten fine-tune: the Colab runs forgot their base chemistry
(S66 interaction-energy MAE 0.22 -> 7.22 kcal/mol, see ``forgetting.py``)
because ``mace_run_train`` cannot self-supply replay data for a MACE-OFF
foundation model and silently fell back to a single head. Here the replay file
built by ``make_replay.py`` is passed explicitly, and the atomic baseline comes
from real GFN2 isolated-atom energies instead of the degenerate ``E0s=average``
fit.

    modal run code/modal_finetune.py                    # A10G, replay on, then gate
    modal run code/modal_finetune.py --gpu L4 --epochs 40
    modal run code/modal_finetune.py --no-replay        # reproduce the forgetting

The gate re-runs the S66 / S30L-CI benchmark on the new checkpoint on the same
GPU (~minutes, vs ~40 min on the laptop CPU) and prints it next to the stock and
GFN2 rows carried up in ``data/forget_*.json``, so a run says immediately
whether the fix held.
"""
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
_CODE, _DATA = _REPO / "code", _REPO / "data"
# The foundation checkpoint is uploaded rather than downloaded in-container:
# 53 MB, and it removes a network dependency (and the MACE license prompt) from
# every cold start.
_CACHE = Path.home() / ".cache" / "mace"
# Both sizes ship in the image so one app covers medium and large runs.
_FOUNDATIONS = {"off-medium": _CACHE / "MACE-OFF23_medium.model",
                "off-large": _CACHE / "MACE-OFF23_large.model"}

app = modal.App("mace-gfn2-finetune")
vol = modal.Volume.from_name("mace-ft", create_if_missing=True)

_MODULES = ["finetune_mace.py", "forgetting.py", "ood_datasets.py",
            "activation_ood.py", "mace_calc.py"]
_INPUTS = ["rot250_gfn2_train.xyz", "rot250_gfn2_valid.xyz",
           "off23_replay_train.xyz", "off23_replay_valid.xyz",
           "forget_gfn2.json", "forget_mace.json"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("mace-torch==0.3.16", "ase==3.29.0", "e3nn==0.4.4",
                 "numpy==2.5.1", "torch==2.13.0")
    .add_local_file(_FOUNDATIONS["off-medium"],
                    "/root/foundation/MACE-OFF23_medium.model")
    .add_local_file(_FOUNDATIONS["off-large"],
                    "/root/foundation/MACE-OFF23_large.model")
    .add_local_dir(_DATA / "s66", "/root/data/s66")
    .add_local_dir(_DATA / "s30l", "/root/data/s30l")
)
for _m in _MODULES:
    image = image.add_local_file(_CODE / _m, f"/root/code/{_m}")
for _f in _INPUTS:
    image = image.add_local_file(_DATA / _f, f"/root/data/{_f}")

TRAIN_H, GATE_H = 4 * 3600, 3600


@app.function(image=image, gpu="A10G", timeout=TRAIN_H,
              volumes={"/vol": vol})
def train(gpu: str = "A10G", epochs: int = 30, lr: float = 1e-4,
          weight_pt_head: float = 1.0, replay: bool = True,
          name: str = "rot250L_replay", batch_size: int = 4,
          foundation: str = "off-large", dataset: str = "rot250") -> str:
    """Run mace_run_train on the GPU; returns the checkpoint path on the volume."""
    import subprocess
    import sys

    sys.path.insert(0, "/root/code")
    from finetune_mace import build_train_command, GFN2_E0S

    out = Path("/vol/runs")
    out.mkdir(parents=True, exist_ok=True)
    kw = {}
    if replay:
        kw = {"pt_train_file": "/root/data/off23_replay_train.xyz",
              "pt_valid_file": "/root/data/off23_replay_valid.xyz",
              "weight_pt_head": weight_pt_head}

    cmd = build_train_command(
        f"/root/data/{dataset}_gfn2_train.xyz",
        f"/root/data/{dataset}_gfn2_valid.xyz",
        foundation_model=("/root/foundation/MACE-OFF23_"
                          f"{'medium' if foundation == 'off-medium' else 'large'}.model"),
        e0s=GFN2_E0S,                      # NOT "average": that fit is degenerate
        name=name, results_dir=str(out), checkpoints_dir=str(out),
        max_num_epochs=epochs, lr=lr,
        energy_weight=100.0, forces_weight=100.0,
        scheduler="ReduceLROnPlateau", device="cuda", default_dtype="float32",
        extra=(f"--batch_size={batch_size}", f"--valid_batch_size={batch_size}",
               "--eval_interval=2"),
        **kw,
    )
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    models = sorted(out.glob(f"{name}_run-*.model"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise RuntimeError(f"training produced no checkpoint in {out}")
    final = models[-1]
    vol.commit()          # results_dir IS the volume: logs are already beside it
    print("checkpoint:", final, flush=True)
    return str(final)


@app.function(image=image, gpu="A10G", timeout=GATE_H, volumes={"/vol": vol})
def gate(model_path: str, label: str = "ft-replay") -> str:
    """Re-run the S66 / S30L-CI benchmark on the new checkpoint and report."""
    import io
    import sys
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/code")
    import forgetting as fg

    fg.FT_MODELS[label] = Path(model_path)
    fg.stage_mace(models=[label])          # stock + GFN2 rows come up in the JSONs
    buf = io.StringIO()
    with redirect_stdout(buf):
        fg.stage_report()
    # Carry the merged results back so the laptop copy stays authoritative.
    import shutil
    shutil.copy(fg.MACE_JSON, "/vol/runs/forget_mace.json")
    vol.commit()
    return buf.getvalue()


@app.local_entrypoint()
def main(gpu: str = "A10G", epochs: int = 30, lr: float = 1e-4,
         weight_pt_head: float = 1.0, no_replay: bool = False,
         name: str = None, batch_size: int = 4,
         skip_gate: bool = False, foundation: str = "off-large",
         dataset: str = "rot250"):
    if foundation not in _FOUNDATIONS:
        raise SystemExit(f"foundation must be one of {sorted(_FOUNDATIONS)}")
    name = name or ("rot250M_replay" if foundation == "off-medium"
                    else "rot250L_replay")
    for f in [_FOUNDATIONS[foundation]] + [_DATA / x for x in _INPUTS]:
        if not f.exists():
            raise SystemExit(
                f"missing input: {f}\n"
                "  rot250_gfn2_*.xyz  -> code/run_label250.sh (GFN2 labels)\n"
                "  off23_replay_*.xyz -> python code/make_replay.py\n"
                "  forget_*.json      -> python code/forgetting.py --stage gfn2/mace")

    opts = {"gpu": gpu} if gpu != "A10G" else {}
    ckpt = train.with_options(**opts).remote(
        gpu=gpu, epochs=epochs, lr=lr, weight_pt_head=weight_pt_head,
        replay=not no_replay, name=name, batch_size=batch_size,
        foundation=foundation, dataset=dataset)

    local = _REPO / "runs" / f"{name}.model"
    local.parent.mkdir(exist_ok=True)
    rel = ckpt.split("/vol/", 1)[1]
    with open(local, "wb") as fh:
        for chunk in vol.read_file(rel):
            fh.write(chunk)
    print(f"\ncheckpoint downloaded -> {local}")

    if not skip_gate:
        print(gate.with_options(**opts).remote(ckpt, label=f"ft-{name}"))

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

From-scratch, charge/spin-aware runs (no foundation, so ``--r_max`` is honoured;
a fine-tune cannot change it -- see ``build_train_command``). Data is the
OMol25 validation-mirror subset ``data/<data>_{train,valid}.xyz`` (default
``omol25_wide20k``: the 20k widest-fragment-gap frames of cafquant's
``make_replay_omol25.py --n 200000`` build), uploaded once to the volume:

    zsh code/run_modal_scratch.sh 6 10 2.0      # r_max, epochs, timeout (h) cap
    zsh code/run_modal_scratch.sh 12 10 3.5     # launcher also pulls the run down

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

TRAIN_H, GATE_H, SCRATCH_H = 4 * 3600, 3600, 20 * 3600

# total_charge spans -9..+9 in the OMol25 subset; offset 100 / 201 classes and
# 101 spin classes follow the arg_parser example (MACE-OMOL convention).
EMBEDDING_SPECS = (
    "{'total_charge': {'type': 'categorical', 'per': 'graph', "
    "'num_classes': 201, 'offset': 100, 'emb_dim': 64}, "
    "'total_spin': {'type': 'categorical', 'per': 'graph', "
    "'num_classes': 101, 'emb_dim': 64}}")
COMMIT_S = 300          # volume commit cadence: a killed container loses <= 5 min


def _scratch_files(data: str) -> list[str]:
    """``data`` is a file prefix in data/, e.g. omol25_scratch or omol25_wide20k."""
    return [f"{data}_train.xyz", f"{data}_valid.xyz"]


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


@app.function(image=image, gpu="L4", timeout=SCRATCH_H, volumes={"/vol": vol})
def train_scratch(r_max: float, epochs: int, name: str, data: str,
                  lr: float = 5e-3, batch_size: int = 16, num_channels: int = 128,
                  energy_weight: float = 1.0, forces_weight: float = 10.0,
                  e0s: str = "average",
                  max_L: int = 1, num_interactions: int = 2) -> str:
    """From-scratch ScaleShiftMACE with charge/spin embeddings at a given r_max.

    Persistence: ``--work_dir`` is the run folder on the volume, so the final
    .model, logs, results and checkpoints all land there (MACE's default
    model_dir/log_dir is the container cwd, which dies with the container).
    MACE keeps only the best checkpoint (saved on each validation
    improvement; the previous one is deleted). The volume is committed every
    ``COMMIT_S``.

    Continuing: ``--restart_latest`` is on, so re-running the same ``name`` with
    a larger ``epochs`` resumes from the best checkpoint (model, optimizer and
    LR-scheduler state) instead of starting over.
    """
    import subprocess
    import sys
    import threading
    import time

    sys.path.insert(0, "/root/code")
    from finetune_mace import build_train_command

    out = Path("/vol/scratch") / name
    out.mkdir(parents=True, exist_ok=True)
    tr, va = _scratch_files(data)
    cmd = build_train_command(
        f"/vol/data/{tr}", f"/vol/data/{va}",
        foundation_model=None, model="ScaleShiftMACE", e0s=e0s,
        name=name, results_dir=str(out), checkpoints_dir=str(out),
        max_num_epochs=epochs, lr=lr, energy_weight=energy_weight,
        forces_weight=forces_weight,
        scheduler="ReduceLROnPlateau", device="cuda", default_dtype="float32",
        extra=(f"--r_max={r_max}", f"--num_channels={num_channels}",
               f"--max_L={max_L}", f"--num_interactions={num_interactions}",
               "--correlation=3", "--max_ell=3",
               "--embedding_specs", EMBEDDING_SPECS,
               f"--batch_size={batch_size}", f"--valid_batch_size={batch_size}",
               "--eval_interval=1", "--restart_latest", "--amsgrad",
               f"--work_dir={out}", f"--model_dir={out}",
               f"--log_dir={out}/logs"),
    )
    print(" ".join(cmd), flush=True)

    stop = threading.Event()

    def _commit_loop():
        while not stop.wait(COMMIT_S):
            vol.commit()

    th = threading.Thread(target=_commit_loop, daemon=True)
    th.start()
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        th.join(timeout=60)
        print(f"wall time: {(time.time() - t0) / 60:.1f} min", flush=True)
        vol.commit()
    models = sorted(out.glob(f"{name}*.model"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise RuntimeError(f"training produced no model in {out}")
    return str(models[-1])


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


@app.local_entrypoint()
def scratch(r_max: float = 6.0, epochs: int = 10, name: str = None,
            data: str = "omol25_wide20k", lr: float = 5e-3,
            batch_size: int = 16, gpu: str = "L4", timeout_h: float = 2.0,
            energy_weight: float = 1.0, forces_weight: float = 10.0,
            e0s_json: str = ""):
    """Upload the OMol25 subset to the volume (once), then train from scratch.

    Launch with ``modal run --detach`` so the run survives the laptop client;
    everything stays on the volume under ``scratch/<name>/`` and
    ``run_modal_scratch.sh`` pulls that folder down afterwards. ``timeout_h``
    is the hard cost cap for the container.
    """
    name = name or f"omol_{data}_r{r_max:g}"
    have = {e.path.rsplit("/", 1)[-1] for e in vol.listdir("data")} \
        if any(e.path == "data" for e in vol.listdir("/")) else set()
    todo = [f for f in _scratch_files(data) if f not in have]
    if todo:
        with vol.batch_upload() as up:
            for f in todo:
                up.put_file(_DATA / f, f"data/{f}")
        print("uploaded:", todo)
    # --e0s-json: fixed isolated-atom E0s (e.g. longrange/atom_e0s/e0s_wb97mv_tzvpd.json,
    # "ev" dict keyed by Z) instead of the least-squares "average" fit.
    e0s = "average"
    if e0s_json:
        import json
        ev = json.loads(Path(e0s_json).read_text())["ev"]
        e0s = "{" + ", ".join(f"{int(z)}: {v}" for z, v in sorted(ev.items(), key=lambda x: int(x[0]))) + "}"
        print("E0s:", e0s)
    opts = {"timeout": int(timeout_h * 3600)}
    if gpu != "L4":
        opts["gpu"] = gpu
    print("model:", train_scratch.with_options(**opts).remote(
        r_max=r_max, epochs=epochs, name=name, data=data, lr=lr,
        batch_size=batch_size, energy_weight=energy_weight,
        forces_weight=forces_weight, e0s=e0s))

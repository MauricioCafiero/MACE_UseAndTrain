"""Fork a finished scratch run into a new run name with a new learning rate.

mace_run_train --restart_latest restores the optimizer and LR-scheduler state
from the checkpoint (mace/tools/checkpoint.py), so a resumed run silently keeps
the old lr and ignores --lr. This copies the run's best checkpoint to
runs/scratch/<new>/<new>_run-<seed>_epoch-<N>.pt with the lr rewritten in
every optimizer param group and in the scheduler's _last_lr, then uploads it
to the mace-ft volume as scratch/<new>/. The source run is not modified.
Loss weights are NOT in the checkpoint: pass the new ones on the resume.

    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/fork_checkpoint.py \
        omol_omol25_wide20k_r6 omol_omol25_wide20k_r6_lr1e-3 1e-3
"""
import re
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
src_name, new_name, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
src_dir = REPO / "runs" / "scratch" / src_name
ckpts = sorted(src_dir.glob(f"{src_name}_run-*_epoch-*.pt"),
               key=lambda p: int(re.search(r"_epoch-(\d+)\.pt$", p.name).group(1)))
src = ckpts[-1]
seed_epoch = src.name[len(src_name):]                  # "_run-123_epoch-9.pt"

c = torch.load(src, map_location="cpu", weights_only=False)
for g in c["optimizer"]["param_groups"]:
    g["lr"] = lr
c["lr_scheduler"]["_last_lr"] = [lr] * len(c["optimizer"]["param_groups"])

dst_dir = REPO / "runs" / "scratch" / new_name
dst_dir.mkdir(parents=True, exist_ok=True)
dst = dst_dir / f"{new_name}{seed_epoch}"
torch.save(c, dst)
print(f"{src.name} -> {dst.relative_to(REPO)} (lr -> {lr})")
subprocess.run(["modal", "volume", "put", "mace-ft", str(dst),
                f"scratch/{new_name}/{dst.name}"], check=True)
print(f"uploaded to mace-ft:scratch/{new_name}/{dst.name}")

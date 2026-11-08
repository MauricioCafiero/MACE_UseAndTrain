#!/bin/zsh
# From-scratch charge/spin-aware MACE on an OMol25 subset, one r_max per call.
#   zsh code/run_modal_scratch.sh <r_max> <epochs> <timeout_h> [name] [data] [extra flags]
# extra flags pass through to the scratch entrypoint, e.g. --energy-weight 10
# --forces-weight 10. On a resume, epochs is the TOTAL max (MACE restarts at
# the checkpoint's epoch index, so best ckpt epoch-9 + epochs 19 = 10 more).
# --lr is ignored on a resume (optimizer state is restored): fork the
# checkpoint with longrange/fork_checkpoint.py to change it.
# --detach keeps the Modal app running if this client dies; caffeinate holds
# off laptop sleep while the client is attached. Everything (best checkpoint,
# final .model, logs) is on the mace-ft volume under scratch/<name>/,
# and is pulled to runs/scratch/<name>/ when modal run exits -- success,
# failure or timeout alike.
#
# Continue a run: same name, larger epochs (--restart_latest resumes it):
#   zsh code/run_modal_scratch.sh 12 20 3.5 omol_omol25_wide20k_r12
# Manual pull if this client died:
#   modal volume get --force mace-ft scratch/<name> runs/scratch/
caffeinate -w $$ &
cd /Users/cafierom/python_mac/mace
R=${1:?r_max}; EP=${2:?epochs}; TH=${3:?timeout_h}
DATA=${5:-omol25_wide20k}; NAME=${4:-omol_${DATA}_r${R}}
echo "=== modal scratch r_max=$R epochs=$EP timeout=${TH}h name=$NAME data=$DATA start: $(date) ==="
modal run --detach code/modal_finetune.py::scratch \
  --r-max $R --epochs $EP --timeout-h $TH --name $NAME --data $DATA "${@:6}"
echo "=== modal run exited ($?): $(date) -- pulling scratch/$NAME ==="
mkdir -p runs/scratch
modal volume get --force mace-ft scratch/$NAME runs/scratch/
ls -la runs/scratch/$NAME runs/scratch/$NAME/* 2>&1 | head -40
echo "=== modal scratch $NAME done: $(date) ==="

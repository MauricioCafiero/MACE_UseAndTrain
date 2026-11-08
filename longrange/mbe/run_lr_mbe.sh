#!/bin/zsh
# Run the four long-range PoC models through cafquant's protein binding-site
# MBE benchmark (sult, ddc, comt, aldh: LD pairs, residue pairs, triples,
# totals), one model at a time, then copy the per-frame results and the
# assembled table back into this repo (longrange/mbe/).
#
#   zsh longrange/mbe/run_lr_mbe.sh            (launch detached, see TRAINING.md)
#
# Methods lr-mace6/12, lr-polar6/12 are registered in cafquant's
# mbe/pipeline.py METHODS and code/cafquant/mlips.py (LR_MODELS).
caffeinate -w $$ &
CQ=/Users/cafierom/python_mac/cafquant
HERE=/Users/cafierom/python_mac/mace/longrange/mbe
PY=$CQ/.venv/bin/python
export OMP_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE
cd $CQ/mbe
for m in lr-mace6 lr-mace12 lr-polar6 lr-polar12; do
  for s in sult ddc comt aldh; do
    echo "=== $m $s start: $(date) ==="
    $PY pipeline.py run $s --method $m || echo "!!! $m $s FAILED"
    mkdir -p $HERE/out/$s
    cp -p out/$s/sp_$m.json $HERE/out/$s/ 2>/dev/null
  done
done
echo "=== assemble: $(date) ==="
$PY pipeline.py assemble all
for s in sult ddc comt aldh; do cp -p out/$s/assembly.json $HERE/out/$s/ 2>/dev/null; done
$PY pipeline.py table > $HERE/table_all.txt 2>&1
echo "=== done: $(date) ==="

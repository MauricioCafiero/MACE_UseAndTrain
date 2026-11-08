#!/usr/bin/env bash
# Sequential ORCA driver for the distance scans. Copied from
# moltui/surface_gen_scripts/run_opts_seq.sh (same ORCA/OpenMPI environment,
# one job at a time, timing log), with cafquant/dft_check/run_orca.sh's
# skip-if-finished check so a rerun resumes. Inputs + jobs.txt (smallest
# first) come from make_scans.py. Each input uses %pal nprocs 4.
# Launch detached (CLAUDE.md pattern); caffeinate dies with the script.
set -u
cd "$(dirname "$0")"
caffeinate -w $$ &

export PATH="$HOME/opt/orca-6.1.1:$HOME/opt/openmpi-4.1.8/bin:$PATH"
ORCA="$HOME/opt/orca-6.1.1/orca"

LOG=run_log.txt
N=$(grep -c . jobs.txt)
echo "=== scan driver started $(date '+%Y-%m-%d %H:%M:%S') ($N jobs) ===" | tee -a "$LOG"
I=0
while read -r STEM; do
  [ -z "$STEM" ] && continue
  I=$((I + 1))
  if grep -q "FINAL SINGLE POINT ENERGY" "orca/$STEM/$STEM.out" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] SKIP $I/$N ${STEM} (done)" | tee -a "$LOG"
    continue
  fi
  T0=$(date +%s)
  # ORCA (parallel) needs the absolute binary path; output names follow the input basename.
  # < /dev/null: ORCA's internal mpirun otherwise swallows the rest of jobs.txt
  # from the loop's stdin (first launch stopped after 1 job).
  (cd "orca/$STEM" && "$ORCA" "$STEM.inp" > "$STEM.out" 2> "$STEM.err" < /dev/null)
  RC=$?
  echo "[$(date '+%H:%M:%S')] DONE $I/$N ${STEM} rc=$RC wall=$(( $(date +%s) - T0 ))s" | tee -a "$LOG"
done < jobs.txt
echo "=== scan driver finished $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

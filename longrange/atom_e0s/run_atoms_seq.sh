#!/usr/bin/env bash
# Sequential ORCA driver for the isolated-atom E0s. Copied from
# moltui/surface_gen_scripts/run_opts_seq.sh (same ORCA/OpenMPI environment,
# one job at a time, timing log); the deadline gate is dropped (atoms take
# seconds) and cafquant/dft_check/run_orca.sh's skip-if-finished check added,
# so a rerun resumes. Inputs and jobs.txt come from `atom_e0s.py make`.
# Launch via `caffeinate -i bash longrange/atom_e0s/run_atoms_seq.sh`.
set -u
cd "$(dirname "$0")"

export PATH="$HOME/opt/orca-6.1.1:$HOME/opt/openmpi-4.1.8/bin:$PATH"
ORCA="$HOME/opt/orca-6.1.1/orca"

LOG=run_log.txt
echo "=== atom E0 driver started $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

while read -r EL; do
  [ -z "$EL" ] && continue
  if grep -q "FINAL SINGLE POINT ENERGY" "$EL/$EL.out" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] SKIP ${EL} (done)" | tee -a "$LOG"
    continue
  fi
  echo "[$(date '+%H:%M:%S')] LAUNCH ${EL}" | tee -a "$LOG"
  T0=$(date +%s)
  # ORCA run; it derives output names from the input basename.
  (cd "$EL" && "$ORCA" "$EL.inp" > "$EL.out" 2> "$EL.err")
  RC=$?
  WALL=$(( $(date +%s) - T0 ))
  echo "[$(date '+%H:%M:%S')] DONE ${EL} rc=$RC wall=${WALL}s" | tee -a "$LOG"
done < jobs.txt

echo "=== atom E0 driver finished $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

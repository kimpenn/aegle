#!/bin/bash
# nohup bash /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/launcher/run_analysis_uterus.sh > logs/run_analysis_uterus_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# [0] Maximum number of parallel jobs (default: 8)
MAX_PARALLEL=${MAX_PARALLEL:-8}

# [1] Set experiment set name
EXP_SET_NAME="analysis_uterus_hb"

# [2] Define the base directory (adjust as needed)
ROOT_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"

# [3] Define the path to the inner script that launches run_analysis.py
RUN_FILE="${ROOT_DIR}/scripts/run_analysis.sh"

# [4] Define input data directory
DATA_DIR="${ROOT_DIR}/out/main/main_uterus_hb"

# [5] Directory where your config files for analysis live
#     Example: each experiment has its own subfolder with an analysis config.
CONFIG_DIR="${ROOT_DIR}/exps/configs/analysis/${EXP_SET_NAME}"

# [6] Define the output and log directories for analysis
LOG_DIR="${ROOT_DIR}/logs/analysis/${EXP_SET_NAME}"
OUT_DIR="${ROOT_DIR}/out/analysis/${EXP_SET_NAME}"

# [7] Create them if they don't exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# [8] Define an array of experiment identifiers
#     In your main pipeline, you might have 'D18_Scan1_0', etc.
#     Adjust as needed for whichever analysis you are doing:
declare -a EXPERIMENTS=(
  "D11_img0018_0"
  "D11_img0018_1"
  "D11_img0018_2"
  "D11_img0018_3"
  "D11_img0020_0"
  "D11_img0020_1"
  "D13_img0022_0"
  "D13_img0022_1"
  "D13_img0022_2"
  "D13_img0022_3"
  "D13_img0024_0"
  "D13_img0024_1"
  "D14_img0026_0"
  "D14_img0026_1"
  "D14_img0026_2"
  "D14_img0026_3"
  "D14_img0028_0"
  "D14_img0028_1"
  "D15_img0030_0"
  "D15_img0030_1"
  "D15_img0030_2"
  "D15_img0030_3"
  "D15_img0032_0"
  "D15_img0032_1"
  "D16_img0034_0"
  "D16_img0034_1"
  "D16_img0034_2"
  "D16_img0034_3"
  "D16_img0036_0"
  "D16_img0036_1"
  "D17_img0038_0"
  "D17_img0038_1"
  "D17_img0038_2"
  "D17_img0038_3"
  "D17_img0040_0"
  "D17_img0040_1"
  "D18_img0042_0"
  "D18_img0042_1"
  "D18_img0042_2"
  "D18_img0044_0"
  "D18_img0044_1"
)

# [9] Function to run a single experiment
run_experiment() {
  local EXP_ID="$1"
  local ANALYSIS_CONFIG="${CONFIG_DIR}/${EXP_ID}/config.yaml"
  local EXP_DATA_DIR="${DATA_DIR}/${EXP_ID}"
  local LOG_FILE="${LOG_DIR}/${EXP_ID}.log"

  if [ ! -f "${ANALYSIS_CONFIG}" ]; then
    echo "[WARN] Skipping ${EXP_ID}: missing config ${ANALYSIS_CONFIG}" >&2
    return 1
  fi
  if [ ! -d "${EXP_DATA_DIR}" ]; then
    echo "[WARN] Skipping ${EXP_ID}: missing data directory ${EXP_DATA_DIR}" >&2
    return 1
  fi

  {
    echo "Current time: $(date)"
    echo "Running analysis for $EXP_ID"
    time bash "${RUN_FILE}" "$EXP_SET_NAME" "$EXP_ID" "$ROOT_DIR" "$DATA_DIR" "$CONFIG_DIR" "$OUT_DIR"
    echo "Experiment $EXP_ID analysis completed at $(date)."
  } > "$LOG_FILE" 2>&1
}

# [10] Parallel execution with job limit
echo "Starting parallel analysis (MAX_PARALLEL=${MAX_PARALLEL}) at $(date)"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "-----------------------------------------------------------"

job_count=0
launched=0
for EXP_ID in "${EXPERIMENTS[@]}"; do
  echo "Launching analysis for $EXP_ID (running: $job_count/$MAX_PARALLEL)"
  run_experiment "$EXP_ID" &
  ((job_count++))
  ((launched++))

  # Wait if we hit the parallel limit
  if (( job_count >= MAX_PARALLEL )); then
    wait -n  # Wait for any one job to finish
    ((job_count--))
  fi
done

# Wait for remaining jobs
echo "-----------------------------------------------------------"
echo "All $launched experiments launched. Waiting for remaining jobs..."
wait

echo "-----------------------------------------------------------"
echo "All analysis experiments completed at $(date)."

#!/bin/bash

# [1] Set experiment set name (matches exps/configs/analysis/<EXP_SET_NAME>)
EXP_SET_NAME="analysis_ft_hb"

# [2] Define the base directory
ROOT_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"

# [3] Python entry point
PYTHON_RUNNER="${ROOT_DIR}/src/run_annotation.py"

# [4] Config + output roots
CONFIG_DIR="${ROOT_DIR}/exps/configs/analysis/${EXP_SET_NAME}"
OUT_DIR="${ROOT_DIR}/out/analysis/${EXP_SET_NAME}"
LOG_DIR="${ROOT_DIR}/logs/annotation/${EXP_SET_NAME}"

mkdir -p "${LOG_DIR}"

# [5] Experiments to annotate (must already have analysis outputs)
declare -a EXPERIMENTS=(
  "D10_0"
  # Add additional experiment IDs as needed
)

for EXP_ID in "${EXPERIMENTS[@]}"; do
  CONFIG_PATH="${CONFIG_DIR}/${EXP_ID}/config.yaml"
  ANALYSIS_OUT="${OUT_DIR}/${EXP_ID}"
  PAYLOAD_PATH="${ANALYSIS_OUT}/differential_expression/llm_cluster_payload.json"
  LOG_FILE="${LOG_DIR}/${EXP_ID}.log"

  if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[WARN] Skipping ${EXP_ID}: config not found at ${CONFIG_PATH}" >&2
    continue
  fi
  if [ ! -d "${ANALYSIS_OUT}" ]; then
    echo "[WARN] Skipping ${EXP_ID}: analysis output missing at ${ANALYSIS_OUT}" >&2
    continue
  fi
  if [ ! -f "${PAYLOAD_PATH}" ]; then
    echo "[WARN] Skipping ${EXP_ID}: cluster payload missing at ${PAYLOAD_PATH}" >&2
    continue
  fi

  echo "Running LLM annotation for ${EXP_ID}"
  {
    echo "Current time: $(date)"
    time python "${PYTHON_RUNNER}" \
      --config_file "${CONFIG_PATH}" \
      --analysis_out_dir "${ANALYSIS_OUT}"
  } > "${LOG_FILE}" 2>&1

done

echo "Annotation runs completed."

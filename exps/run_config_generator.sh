#!/usr/bin/env bash
# Utility script for generating pipeline configs from a CSV experiment set.
#
# Supply the required environment variables before invoking this helper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPS_DIR="${REPO_DIR}/exps"
TEMPLATES_DIR="${EXPS_DIR}/templates"
SCHEMAS_DIR="${EXPS_DIR}/schemas"
GENERATOR="${EXPS_DIR}/config_generator.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Configuration (must be provided by the caller via environment variables)
# ---------------------------------------------------------------------------
: "${ANALYSIS_STEP:?Missing required env ANALYSIS_STEP}"   # preprocess | main | analysis
: "${EXPERIMENT_SET:?Missing required env EXPERIMENT_SET}"
: "${CSV_PATH:?Missing required env CSV_PATH}"
: "${OUTPUT_DIR:?Missing required env OUTPUT_DIR}"

TEMPLATE_PATH=${TEMPLATE_PATH:-${TEMPLATES_DIR}/${ANALYSIS_STEP}_template.yaml}
SCHEMA_PATH=${SCHEMA_PATH:-${SCHEMAS_DIR}/${ANALYSIS_STEP}.yaml}

# ---------------------------------------------------------------------------

die() {
  echo "Error: $*" >&2
  exit 1
}

[[ -f "${GENERATOR}" ]]    || die "Generator script not found: ${GENERATOR}"
[[ -f "${CSV_PATH}" ]]      || die "Design table CSV not found: ${CSV_PATH}"
[[ -f "${TEMPLATE_PATH}" ]] || die "Template not found: ${TEMPLATE_PATH}"
[[ -f "${SCHEMA_PATH}" ]]   || die "Schema not found: ${SCHEMA_PATH}"

mkdir -p "${OUTPUT_DIR}"

echo "Generating configs using:"
echo "  analysis_step  = ${ANALYSIS_STEP}"
echo "  experiment_set = ${EXPERIMENT_SET}"
echo "  csv            = ${CSV_PATH}"
echo "  template       = ${TEMPLATE_PATH}"
echo "  schema         = ${SCHEMA_PATH}"
echo "  output_dir     = ${OUTPUT_DIR}"
echo

"${PYTHON_BIN}" "${GENERATOR}" \
  --analysis-step "${ANALYSIS_STEP}" \
  --experiment-set "${EXPERIMENT_SET}" \
  --base-dir "${EXPS_DIR}" \
  --csv "${CSV_PATH}" \
  --template "${TEMPLATE_PATH}" \
  --schema "${SCHEMA_PATH}" \
  --output-dir "${OUTPUT_DIR}"

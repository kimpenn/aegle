from __future__ import annotations
import argparse
import copy
import csv
import os
import sys
from collections.abc import MutableMapping

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from schema_validator import (
    SchemaLoadingError,
    SchemaValidationError,
    load_schema,
)

# -----------------------------------------------------------------------------
# Default configuration (can be overridden via CLI arguments)
# -----------------------------------------------------------------------------
DEFAULT_EXPERIMENT_SET = "main_ft"
DEFAULT_ANALYSIS_STEP = "main"  # "preprocess", "main", or "analysis"
BASE_DIR = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pipeline configs from CSV + schema")
    parser.add_argument("--experiment-set", default=DEFAULT_EXPERIMENT_SET, help="Experiment set name")
    parser.add_argument(
        "--analysis-step",
        default=DEFAULT_ANALYSIS_STEP,
        choices=["preprocess", "main", "analysis"],
        help="Pipeline stage this config targets",
    )
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base directory for experiment assets")
    parser.add_argument("--csv", dest="design_table", help="Path to design table CSV")
    parser.add_argument("--template", dest="template_path", help="Path to template YAML")
    parser.add_argument("--schema", dest="schema_path", help="Path to schema YAML")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory for generated configs")
    return parser.parse_args()


def read_csv(file_path: str) -> tuple[list[dict[str, str]], list[str]]:
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def read_yaml(file_path: str):
    yaml = YAML()
    with open(file_path, "r") as yamlfile:
        return yaml.load(yamlfile)


def write_yaml(data, file_path: str) -> None:
    def represent_null(self, value):
        return self.represent_scalar("tag:yaml.org,2002:null", "NULL")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.representer.add_representer(type(None), represent_null)
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(file_path, "w") as yamlfile:
        yaml.dump(data, yamlfile)


def update_nested_dict(container: MutableMapping, keys: list[str], value) -> None:
    current = container
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = CommentedMap() if isinstance(current, CommentedMap) else {}
        current = current[key]
        if not isinstance(current, MutableMapping):  # pragma: no cover - guardrail
            raise TypeError(f"Cannot assign nested key through non-mapping for '{'::'.join(keys)}'")
    final_key = keys[-1]
    current[final_key] = value


def generate_experiments_string(exp_ids: list[str]) -> str:
    lines = ["experiments=("]
    for exp_id in exp_ids:
        lines.append(f'  "{exp_id}"')
    lines.append(")")
    return "\n".join(lines)


def build_output_paths(base_dir: str, analysis_step: str, experiment_set: str) -> dict[str, str]:
    configs_dir = os.path.join(base_dir, "configs", analysis_step, experiment_set)
    csv_path = os.path.join(base_dir, "csvs", f"{experiment_set}.csv")
    template_path = os.path.join(base_dir, f"{analysis_step}_template.yaml")
    schema_path = os.path.join(base_dir, "schemas", f"{analysis_step}.yaml")
    return {
        "csv": csv_path,
        "template": template_path,
        "schema": schema_path,
        "output": configs_dir,
    }


def apply_post_processing(analysis_step: str, row_results) -> None:
    if analysis_step != "preprocess":
        return

    DEFAULT_DOWNSCALE = 64
    for result in row_results:
        visualize = result.final_values.get("tissue_extraction::visualize")
        downscale = result.final_values.get("tissue_extraction::downscale_factor")

        needs_default = False
        if isinstance(downscale, (int, float)):
            needs_default = downscale <= 0
        elif downscale in (None, ""):
            needs_default = True

        if visualize and needs_default:
            result.final_values["tissue_extraction::downscale_factor"] = DEFAULT_DOWNSCALE
            result.values_to_set["tissue_extraction::downscale_factor"] = DEFAULT_DOWNSCALE


def write_configs(row_results, template_config, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for result in row_results:
        exp_dir = os.path.join(output_dir, result.exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        config_obj = copy.deepcopy(template_config)
        config_obj["exp_id"] = result.exp_id

        for full_key, value in result.values_to_set.items():
            key_parts = full_key.split("::")
            update_nested_dict(config_obj, key_parts, value)

        write_yaml(config_obj, os.path.join(exp_dir, "config.yaml"))


def main() -> int:
    args = parse_args()

    base_dir = args.base_dir
    experiment_set = args.experiment_set
    analysis_step = args.analysis_step

    paths = build_output_paths(base_dir, analysis_step, experiment_set)
    design_table_path = args.design_table or paths["csv"]
    template_path = args.template_path or paths["template"]
    schema_path = args.schema_path or paths["schema"]
    output_dir = args.output_dir or paths["output"]

    try:
        design_rows, csv_columns = read_csv(design_table_path)
    except FileNotFoundError:
        print(f"CSV not found: {design_table_path}", file=sys.stderr)
        return 1

    try:
        template_config = read_yaml(template_path)
    except FileNotFoundError:
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 1

    if template_config is None:
        print(f"Template is empty: {template_path}", file=sys.stderr)
        return 1

    try:
        schema = load_schema(schema_path)
    except SchemaLoadingError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        schema.validate_columns(csv_columns)
        row_results = schema.validate_rows(design_rows)
    except SchemaValidationError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not row_results:
        print("No rows found in design table; nothing to generate.")
        return 0

    apply_post_processing(analysis_step, row_results)

    write_configs(row_results, template_config, output_dir)

    exp_ids = [result.exp_id for result in row_results]
    print(f"Generated {len(exp_ids)} configs under {output_dir}")
    print(generate_experiments_string(exp_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())

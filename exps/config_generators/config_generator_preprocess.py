# config_generator_preprocess.py

import os
from generator_utils import (
    read_csv,
    read_yaml,
    write_yaml,
    parse_string_value,
    update_nested_dict,
)


def generate_preprocess_configs(design_csv, default_yaml, output_dir):
    """
    For each row in design_csv, load the default_yaml config,
    override fields, and save to output_dir/<exp_id>/config.yaml.
    """
    # 1) Read the design table
    rows = read_csv(design_csv)

    # 2) Load the default config
    default_config = read_yaml(default_yaml)

    for row in rows:
        exp_id = row.get("exp_id", "exp_undefined")

        # Create the experiment-specific output folder
        exp_config_dir = os.path.join(output_dir, exp_id)
        os.makedirs(exp_config_dir, exist_ok=True)

        # Fresh copy of the default config
        config = default_config.copy()

        # Apply overrides from each CSV column
        for column_key, raw_value in row.items():
            if column_key == "exp_id":
                continue  # We can store it in config if we want
            parsed_value = parse_string_value(raw_value)

            # "data::file_name" => ["data", "file_name"]
            nested_keys = column_key.split("::")
            update_nested_dict(config, nested_keys, parsed_value)

        # Optionally embed exp_id into config
        config["exp_id"] = exp_id

        # Save updated config
        out_config_path = os.path.join(exp_config_dir, "config.yaml")
        write_yaml(config, out_config_path)
        print(f"[Preprocess] Created config for '{exp_id}' at {out_config_path}")


def main():
    experiment_set_name = "test0206_preprocess"

    # CSV and default for the main pipeline
    module_name = "preprocess"
    base_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps"
    design_csv = os.path.join(base_dir, "csvs", f"{experiment_set_name}.csv")
    default_yaml = os.path.join(base_dir, f"default_{module_name}_config.yaml")

    # Output directory for generated configs
    output_dir = os.path.join(base_dir, "configs", module_name, experiment_set_name)

    # Run the generator
    generate_preprocess_configs(design_csv, default_yaml, output_dir)


if __name__ == "__main__":
    main()

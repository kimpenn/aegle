# config_generator_main.py

import os
from generator_utils import (
    read_csv,
    read_yaml,
    write_yaml,
    parse_string_value,
    update_nested_dict,
)


def generate_main_configs(design_csv, default_yaml, output_dir):
    """
    Generate config files for the main pipeline,
    using a separate CSV and default config from the preprocess pipeline.
    """
    rows = read_csv(design_csv)
    default_config = read_yaml(default_yaml)

    for row in rows:
        exp_id = row.get("exp_id", "exp_undefined")
        exp_config_dir = os.path.join(output_dir, exp_id)
        os.makedirs(exp_config_dir, exist_ok=True)

        config = default_config.copy()

        for column_key, raw_value in row.items():
            if column_key == "exp_id":
                continue
            parsed_value = parse_string_value(raw_value)

            # Special case for channels::wholecell_channel
            # If the parsed_value is still a string and contains commas, split into a list
            if column_key == "channels::wholecell_channel" and isinstance(
                parsed_value, str
            ):
                parsed_value = [item.strip() for item in parsed_value.split(",")]

            nested_keys = column_key.split("::")
            update_nested_dict(config, nested_keys, parsed_value)

        config["exp_id"] = exp_id

        out_config_path = os.path.join(exp_config_dir, "config.yaml")
        write_yaml(config, out_config_path)
        print(f"[Main] Created config for '{exp_id}' at {out_config_path}")


def main():
    experiment_set_name = "test0206_main"

    # CSV and default for the main pipeline
    module_name = "main"
    base_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps"
    design_csv = os.path.join(base_dir, "csvs", f"{experiment_set_name}.csv")
    default_yaml = os.path.join(base_dir, f"default_{module_name}_config.yaml")

    # Output directory for generated configs
    output_dir = os.path.join(base_dir, "configs", module_name, experiment_set_name)

    # Run the generator
    generate_main_configs(design_csv, default_yaml, output_dir)


if __name__ == "__main__":
    main()

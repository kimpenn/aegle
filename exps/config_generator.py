import csv
import os
import ast
from ruamel.yaml import YAML

##############################
# (A) USER-EDITABLE SETTINGS #
##############################

experiment_set_name = "preprocess/test0206"

module_name = experiment_set_name.split("/")[0]
base_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps"
design_table_path = f"{base_dir}/csvs/{experiment_set_name}.csv"
default_config_path = f"{base_dir}/{module_name}_config.yaml"

output_dir = f"/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/{experiment_set_name}"

##############################
# (B) HELPER FUNCTIONS       #
##############################


def read_csv(file_path):
    """Read rows from a CSV into a list of dicts."""
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def read_yaml(file_path):
    """Load a YAML file using ruamel.yaml."""
    yaml = YAML()
    with open(file_path, "r") as yamlfile:
        return yaml.load(yamlfile)


def write_yaml(data, file_path):
    """Write a Python dict to YAML (with some formatting preferences)."""

    def my_represent_none(self, data):
        return self.represent_scalar("tag:yaml.org,2002:null", "null")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.representer.add_representer(type(None), my_represent_none)
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(file_path, "w") as yamlfile:
        yaml.dump(data, yamlfile)


def update_nested_dict(d, keys, value):
    """
    Recursively navigate the nested dict `d` via `keys`,
    then set its final key to `value`.
      Example: update_nested_dict(config, ["tissue_extraction", "n_tissue"], 4)
    """
    if len(keys) > 1:
        if keys[0] not in d:
            d[keys[0]] = {}
        update_nested_dict(d[keys[0]], keys[1:], value)
    else:
        d[keys[0]] = value


def str_to_bool(s):
    """Convert string to a Python boolean if it looks like 'true'/'false'."""
    s = s.strip().lower()
    if s in ["yes", "true", "t", "y", "1"]:
        return True
    elif s in ["no", "false", "f", "n", "0"]:
        return False
    raise ValueError(f"Cannot convert {s} to a bool")


def parse_string_value(value):
    """
    Attempt to parse the string `value` into Python bool, None, int, float, or leave as string.
    """
    if value is None:
        return None
    v = value.strip()

    # Handle 'None'
    if v in ["None", "null", "NULL", "Null", ""]:
        return None

    # Handle bool
    try:
        return str_to_bool(v)
    except ValueError:
        pass  # not a boolean

    # Handle numeric (int or float)
    try:
        # if it has a decimal point => float
        if "." in v:
            return float(v)
        else:
            return int(v)
    except ValueError:
        pass

    # Default: string
    return v


##############################
# (C) MAIN LOGIC             #
##############################


def generate_config_files(design_table_path, default_config_path, output_dir):
    """
    For each row in `design_table_path` (CSV), load a default config,
    override the specified keys, and save to a new config in `output_dir/exp_id/config.yaml`.
    """
    design_table = read_csv(design_table_path)
    default_config = read_yaml(default_config_path)

    # For each experiment row
    for row in design_table:
        exp_id = row["exp_id"]

        # Prepare output folder for this exp_id
        exp_config_dir = os.path.join(output_dir, exp_id)
        os.makedirs(exp_config_dir, exist_ok=True)

        # Make a fresh copy of the default config
        config = default_config.copy()

        # Overwrite any fields in 'row' -> config
        #
        # Example CSV columns:
        #   "exp_id"
        #   "data::file_name"
        #   "tissue_extraction::n_tissue"
        #   "tissue_extraction::downscale_factor"
        #   "tissue_extraction::min_area"
        #   "tissue_extraction::visualize"
        #
        for column_key, raw_value in row.items():
            # We skip 'exp_id' itself, or we can store it in config if you want
            if column_key == "exp_id":
                continue

            # Convert CSV string to appropriate Python type
            parsed_value = parse_string_value(raw_value)

            # Column key format: "data::file_name" => keys = ["data", "file_name"]
            nested_keys = column_key.split("::")
            update_nested_dict(config, nested_keys, parsed_value)

        # Optionally set config["exp_id"] to exp_id if you use that in your pipeline:
        config["exp_id"] = exp_id

        # Write out the updated config
        config_filename = os.path.join(exp_config_dir, "config.yaml")
        write_yaml(config, config_filename)
        print(f"Created config: {config_filename}")

    print("All configuration files have been created.")


if __name__ == "__main__":
    generate_config_files(design_table_path, default_config_path, output_dir)

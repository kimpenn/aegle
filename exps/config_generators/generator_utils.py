# generator_utils.py
import os
import csv
from ruamel.yaml import YAML


def read_csv(file_path):
    """Read rows from a CSV into a list of dicts."""
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def read_yaml(file_path):
    """Load a YAML file using ruamel.yaml."""
    yaml = YAML()
    with open(file_path, "r") as f:
        return yaml.load(f)


def write_yaml(data, file_path):
    """Write a Python dict to a YAML file with some formatting preferences."""

    def my_represent_none(self, value):
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
    """Convert a string to a Python boolean if it looks like 'true' or 'false'."""
    s = s.strip().lower()
    if s in ["yes", "true", "t", "y", "1"]:
        return True
    elif s in ["no", "false", "f", "n", "0"]:
        return False
    raise ValueError(f"Cannot convert {s} to bool")


def parse_string_value(value):
    """
    Attempt to parse the string `value` into Python bool, None, int, float,
    or leave it as a string.
    """
    if value is None:
        return None
    v = value.strip()
    # Handle 'None' or empty
    if v in ["None", "null", "NULL", "Null", ""]:
        return None

    # Handle bool
    try:
        return str_to_bool(v)
    except ValueError:
        pass

    # Handle numeric
    try:
        if "." in v:
            return float(v)
        else:
            return int(v)
    except ValueError:
        pass

    # Default: string
    return v

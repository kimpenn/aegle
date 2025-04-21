import csv
import os
from ruamel.yaml import YAML
import ast

# TODO:: Change this according to the csv file
experiment_set_name = "test0206_main"
# experiment_set_name = "test_analysis"
# TODO:: Change this according to the analysis step
# "preprocess", "main", "analysis"
analysis_step = "main"  
# analysis_step = "analysis"  
base_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps"
# Path to the input CSV file
design_table_path = (
    f"{base_dir}/csvs/{experiment_set_name}.csv"
)
# Path to the YAML template
default_config_path = (
    f"{base_dir}/{analysis_step}_template.yaml"
)

# Base path for saving the configs
output_dir = f"{base_dir}/configs/{analysis_step}/{experiment_set_name}"
print(f"Output directory: {output_dir}")

def read_csv(file_path):
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def read_yaml(file_path):
    yaml = YAML()
    with open(file_path, "r") as yamlfile:
        return yaml.load(yamlfile)


def write_yaml(data, file_path):
    def my_represent_none(self, data):
        return self.represent_scalar("tag:yaml.org,2002:null", "NULL")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.representer.add_representer(type(None), my_represent_none)
    # ref: https://yaml.readthedocs.io/en/latest/detail/
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(file_path, "w") as yamlfile:
        yaml.dump(data, yamlfile)


def generate_experiments_string(design_table_path):
    # Generate a string in the following format for name column in design table.
    # This string can be used to set up experiments in a bash script.
    # experiments=(
    #   "exp3-1 0"
    #   "exp3-2 0"
    #   "exp3-3 0"
    # )
    design_table = read_csv(design_table_path)
    experiments = [row["exp_id"] for row in design_table]
    experiments_string = "experiments=(\n"
    for i, experiment in enumerate(experiments):
        cuda_device = f"cuda:{i % 2}"
        # experiments_string += f'  "{experiment} {cuda_device}"\n'
        experiments_string += f'  "{experiment}"\n'
    experiments_string += ")"
    return experiments_string


def update_nested_dict(d, keys, value):
    if len(keys) > 1:
        if keys[0] not in d:
            d[keys[0]] = {}
        update_nested_dict(d[keys[0]], keys[1:], value)
    else:
        d[keys[0]] = value


def str_to_bool(s):
    if s.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif s.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise ValueError("Cannot convert {} to a bool".format(s))


def generate_config_files(design_table_path, default_config_path, output_dir):
    # Read design table and default config
    design_table = read_csv(design_table_path)
    default_config = read_yaml(default_config_path)

    for row in design_table:
        # Extract the experiment ID
        exp_id = row["exp_id"]

        # Create directory for the config if it doesn't exist
        exp_config_dir = os.path.join(output_dir, exp_id)
        os.makedirs(exp_config_dir, exist_ok=True)

        # Copy template and update with values from the row
        config = default_config.copy()
        config["exp_id"] = exp_id  # Update exp_id from the CSV file

        for k, v in row.items():
            print(f"Processing key: {k}, value: {v}")
            keys = k.split("::")
            if keys[-1] in ["wholecell_channel"]:
                # transform string "phylodist,gaussian_noise,pca" into list of strings
                v = v.split(",")
                if keys[-1] == "assign_sizes":
                    v = [float(i) for i in v]
            elif keys[-1] in [
                "patch_width",
                "patch_height",
                "patch_index"
            ]:
                v = int(v)
            elif keys[-1] == "output_dim":
                try:
                    v = int(v)
                except ValueError:
                    pass  # v is a string, leave it as is
            elif keys[-1] == "hidden_dims":
                v = ast.literal_eval(v)
            elif keys[-1] in [
                "generate_channel_stats", "visualize_whole_sample", 
                "visualize_patches", "save_all_channel_patches", 
                "visualize_segmentation", "save_segmentation_images", 
                "save_segmentation_pickle", "save_disrupted_patches", 
                "compute_metrics", "skip_viz", "enhance_contrast"
                ]:
                v = str_to_bool(v)
            elif v == "None":
                v = None
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass  # v is a string, leave it as is
            update_nested_dict(config, keys, v)

        # Save the new config as a YAML file
        config_filename = os.path.join(exp_config_dir, "config.yaml")
        # print(config)
        write_yaml(config, config_filename)

    print("Configuration files have been created.")


if __name__ == "__main__":

    generate_config_files(design_table_path, default_config_path, output_dir)
    print(f"experiment_set_name: {experiment_set_name}")
    print(generate_experiments_string(design_table_path))

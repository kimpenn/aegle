#!/usr/bin/env python

import os
import sys
import argparse
import logging
import yaml
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd

##################################################
# (A) ARGUMENT AND CONFIG HANDLING               #
##################################################


def parse_args():
    """
    Parse command-line arguments for the antibody extraction script.
    """
    parser = argparse.ArgumentParser(
        description="Extract channel/antibody info from a QPTIFF using Bio-Formats showinf."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file with 'data.file_name' and optionally 'showinf_path'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspaces/codex-analysis/data",
        help="Directory containing the QPTIFF file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output_antibody_info",
        help="Directory to save the resulting antibody.tsv.",
    )
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"config: {config}")
    return config


##################################################
# (B) CORE EXTRACTION FUNCTIONS                  #
##################################################


def extract_xml(image_path, showinf_path="/workspaces/codex-analysis/bftools/showinf"):
    """
    Extract the OME-XML metadata from the QPTIFF using the Bio-Formats 'showinf' command.
    If the .xml file already exists alongside the .qptiff, it is re-used.

    Args:
        image_path (str): Path to the .qptiff file.
        showinf_path (str): Path to the `showinf` executable.

    Returns:
        str: Path to the extracted (or existing) .xml file.
    """
    output_xml_path = image_path.replace(".qptiff", ".xml")

    # If the XML is already present, skip extraction
    if os.path.exists(output_xml_path):
        logging.info(f"XML file already exists: {output_xml_path}")
        return output_xml_path

    # Construct the showinf command
    command = [
        showinf_path,
        "-no-upgrade",
        "-nopix",
        "-omexml-only",
        "-series",
        "0",
        image_path,
    ]

    logging.info(f"Running showinf to extract XML: {' '.join(command)}")

    # Run the showinf command, capturing the output
    with open(output_xml_path, "w") as xml_file:
        try:
            subprocess.run(command, stdout=xml_file, stderr=subprocess.PIPE, check=True)
            logging.info(f"XML file successfully extracted to: {output_xml_path}")
        except subprocess.CalledProcessError as e:
            logging.error(
                f"Error occurred while extracting XML: {e.stderr.decode('utf-8')}"
            )
        except FileNotFoundError:
            logging.error(
                f"Error: The `showinf` executable was not found at {showinf_path}"
            )
    return output_xml_path


def extract_channel_names(xml_file):
    """
    Parse the extracted OME-XML to get channel IDs and names.

    Args:
        xml_file (str): Path to the extracted OME-XML (.xml).

    Returns:
        pd.DataFrame: DataFrame with columns ['version', 'channel_id', 'antibody_name'].
    """
    namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the first Image element (ID="Image:0") in the OME metadata
    image_elem = root.find(".//ome:Image[@ID='Image:0']", namespace)

    if image_elem is None:
        logging.warning("No Image element found in XML.")
        return pd.DataFrame(columns=["version", "channel_id", "antibody_name"])

    channels = image_elem.findall(".//ome:Channel", namespace)
    data = []
    for c in channels:
        channel_id = c.get("ID")
        channel_name = c.get("Name")
        data.append(
            {"version": 2, "channel_id": channel_id, "antibody_name": channel_name}
        )
    df = pd.DataFrame(data)
    return df


##################################################
# (C) MAIN PIPELINE LOGIC                        #
##################################################


def run_antibody_extraction(config, args):
    """
    Main function to:
      1. Grab the QPTIFF path from config["data"]["file_name"],
      2. Optionally read a custom 'showinf_path',
      3. Extract the OME-XML,
      4. Parse channels and write 'antibody.tsv' to out_dir/extra/.
    """
    # 1) Find QPTIFF path
    file_name = config["data"]["file_name"]
    qptiff_path = os.path.join(args.data_dir, file_name)

    if not os.path.exists(qptiff_path):
        logging.error(f"QPTIFF file not found: {qptiff_path}")
        sys.exit(1)

    # 2) Extract showinf path from config if present, else default
    showinf_path = config.get("bftools", {}).get(
        "showinf_path", "/workspaces/codex-analysis/bftools/showinf"
    )

    # 3) Extract or locate existing XML
    xml_file_path = extract_xml(qptiff_path, showinf_path=showinf_path)

    # 4) Parse channels
    df_channels = extract_channel_names(xml_file_path)
    logging.info(f"Parsed {len(df_channels)} channels from XML.")

    # Create an "extra" folder inside the user-specified --out_dir/exp_name/ or similar
    # Or mirror the data path's folder structure; for now we just create "extra" under out_dir
    out_extra_dir = os.path.join(args.out_dir, "extra")
    os.makedirs(out_extra_dir, exist_ok=True)

    # Save the tsv
    antibody_tsv_path = os.path.join(out_extra_dir, "antibody.tsv")
    df_channels.to_csv(antibody_tsv_path, sep="\t", index=False)
    logging.info(f"Saved antibody info to {antibody_tsv_path}.")


def main():
    """
    Command-line entrypoint.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    qptiff_path = config["data"]["file_name"]
    parent_dir = os.path.dirname(qptiff_path)
    args.out_dir = os.path.join(args.out_dir, parent_dir)

    # Run main logic
    run_antibody_extraction(config, args)
    logging.info("Antibody extraction completed. Exiting.")


if __name__ == "__main__":
    main()

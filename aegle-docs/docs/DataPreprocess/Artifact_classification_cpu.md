# Artifact Classification (CPU and GPU) (Optional)

## Overview
This is an optional step and is not required for the standard AEGLE workflow.

The Artifact Classification module is designed to automatically identify imaging artifacts in CODEX multiplexed images. It uses a DeepSets CNN model to classify small image tiles into predefined categories, such as normal tissue or various types of artifacts (e.g., bubbles, debris, folds, smears).

The CPU-specific shell script (`launcher/artifact_classification_cpu.sh`) wraps the inference process so that you can easily run tile classification entirely on your CPU, which is useful for environments without an available GPU.

If your environment has an available GPU, you can use `launcher/artifact_classification.sh` instead. It is recommended to use GPU for faster inference. (note that Blackwell architecture of GPUs are not supported)

By classifying tiles, you can systematically flag or filter out problematic regions before downstream analysis, ensuring that your spatial and single-cell profiling results are not skewed by imaging artifacts.

## How it works
The `launcher/artifact_classification_cpu.sh` script orchestrates the classification process. It uses `aegle/artifact_CNN/classify_tiles_cpu.py` to run the inference. 

Given an input `.qptiff` image, the script expects image tiles to be generated (or assumes they exist in a matching `_tiles` subdirectory) and then loads the trained DeepSets model to assign a class label to each tile.

The output is a detailed CSV report (`classification_report.csv`) that lists the path of each tile, its predicted label, and the raw model logits for every class.

## Usage

### 1. Using the Shell Wrapper (Recommended)
You can use the shell wrapper to execute the classification pipeline and specify your input files through command line arguments. 

```bash
bash launcher/artifact_classification_cpu.sh \
  --codex_image "/path/to/image.qptiff" \
  --tissue_regions "/path/to/regions.geojson" \
  --antibody_names "/path/to/channels.json" \
  --output_dir "/path/to/output_dir" \
  --model_path "/path/to/model.pt" \
  --class_names "Normal Bubble Debris Fold Smear"
```

#### Command Line Arguments
* **`--codex_image`**: Path to the raw CODEX `.qptiff` image. Note: The script will look for an associated `_tiles` folder in the same directory.
* **`--tissue_regions`**: Path to the tissue regions `.geojson` file. It is highly recommended for users to provide the `.geojson` file that provides the tissue region information via making annotation in `QuPath` or similar software, so that artifact classification is not performed in image regions that does not contain tissue.
* **`--antibody_names`**: Path to the JSON file containing antibody/channel names corresponding to the image. (see `DataPreprocess/antibody_marker_example.json` for the expected format).
* **`--output_dir`**: The directory where the classification results will be saved.
* **`--model_path`**: Path to the trained DeepSets model weights (`.pt` file).
* **`--class_names`**: Space-separated list of classes. These must exactly match the model's training order. Default is `"Normal Bubble Debris Fold Smear"`.

### 2. Running the Python Script Directly
If you have already generated the image tiles and simply want to run the model inference, you can directly invoke the Python script. Be sure to include the `aegle/artifact_CNN` module in your `PYTHONPATH`.

```bash
PYTHONPATH="aegle/artifact_CNN:$PYTHONPATH" python aegle/artifact_CNN/classify_tiles_cpu.py \
    --data_dir "/path/to/image_tiles/Unlabeled" \
    --metadata_path "/path/to/channels.json" \
    --model_path "/path/to/model.pt" \
    --csv_path "/path/to/output_dir/classification_report.csv" \
    --class_names Normal Bubble Debris Fold Smear
```

* **`--data_dir`**: Directory containing the unlabeled tile images.
* **`--metadata_path`**: JSON file containing the channel names.
* **`--model_path`**: The `.pt` file with model weights.
* **`--csv_path`**: Where to save the detailed results CSV file.
* **`--class_names`**: Expected class names matching the model.

## Output
Upon successful completion, the script will generate a **`classification_report.csv`** file in your specified `--output_dir`. This file contains:
* **`Path`**: The absolute path to the tile image.
* **`Predicted_Label`**: The final classification decision.
* **`Logit_<Class>`**: The raw logit score for each possible class.

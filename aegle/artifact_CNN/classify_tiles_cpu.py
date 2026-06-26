#!/usr/bin/env python3
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")  # Non-interactive backend
from aegle.artifact_CNN.deep_sets_model import DeepSetsCNN
from aegle.artifact_CNN.metadata_utils import build_vocabulary, load_metadata
from aegle.artifact_CNN.tiff_dataloader import build_dataloaders, deep_sets_collate_fn


def infer_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list,
    device: str = "cuda",
    csv_path: str = None,
):
    device_t = "cpu"
    model.to(device_t)
    model.eval()

    all_preds = []
    all_paths = []
    all_logits = []

    print(f"Running inference using {device_t}...")
    with torch.no_grad():
        for xb, _, ids, batch_indices, paths in loader:
            xb = xb.to(device_t, non_blocking=True)
            ids = ids.to(device_t, non_blocking=True)
            batch_indices = batch_indices.to(device_t, non_blocking=True)

            logits = model(xb, ids, batch_indices)
            preds = logits.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_paths.extend(paths)
            all_logits.extend(logits.cpu().numpy())

    if csv_path:
        print(f"\nSaving detailed results to {csv_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        import csv

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["Path", "Predicted_Label"] + [
                f"Logit_{c}" for c in class_names
            ]
            writer.writerow(header)
            for i in range(len(all_paths)):
                row = [
                    all_paths[i],
                    class_names[all_preds[i]],
                ]
                row.extend(all_logits[i].tolist())
                writer.writerow(row)

        print("Done.")

    return all_preds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference with DeepSetsCNN on tiled images")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing tile class subfolders")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to JSON file with antibody names")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to save detailed results CSV")
    parser.add_argument("--class_names", type=str, nargs='+', help="Space-separated list of class names (must match model training order)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_ratio", type=float, default=1.0, help="Ratio of data to use for testing")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    METADATA_PATH = args.metadata_path
    MODEL_PATH = args.model_path
    CSV_PATH = args.csv_path

    BATCH_SIZE = args.batch_size
    TEST_RATIO = args.test_ratio
    # -------------------------------------------

    # Check paths
    if not os.path.exists(DATA_DIR) or not os.path.exists(METADATA_PATH):
        print(f"ERROR: Data or metadata not found at specified paths.")
        print(f"DATA_DIR: {DATA_DIR}")
        print(f"METADATA_PATH: {METADATA_PATH}")
        print("Please edit the CONFIGURATION section in the script.")
        exit(1)

    if not os.path.exists(MODEL_PATH):
        # Fallback to local file if not found at absolute path
        local_model = "deepsets_model.pt"
        if os.path.exists(local_model):
            print(f"Model not found at {MODEL_PATH}, using local {local_model}")
            MODEL_PATH = local_model
        else:
            print(f"ERROR: Model not found at {MODEL_PATH}")
            exit(1)

    print(f"Loading data from {DATA_DIR}...")

    # Reconstruct DataLoaders (Crucial: same seed/ratios for consistent split)
    # Note: We pass unk_dropout_prob=0.1 just to match the signature/behavior logic,
    # but the test loader forces it to 0.0.
    _, _, test_loader, class_to_idx, vocab = build_dataloaders(
        root_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4,
        val_ratio=0.0,
        test_ratio=TEST_RATIO,
        hflip=False,  # No augmentation for test
        vflip=False,
        to_float32=True,
        json_path=METADATA_PATH,
        unk_dropout_prob=0.1,
        collate_fn=deep_sets_collate_fn,
        seed=42,
    )

    if test_loader is None:
        print("Error: test_loader is None. Check TEST_RATIO.")
        exit(1)

    # Load Weights first to determine num_classes dynamically
    print(f"Loading weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    
    if 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
    else:
        num_classes = len(class_to_idx)
        print(f"Warning: Could not infer num_classes from state_dict. Falling back to dataloader subfolders ({num_classes}).")

    if 'antibody_embedding.weight' in state_dict:
        num_antibodies = state_dict['antibody_embedding.weight'].shape[0]
    else:
        num_antibodies = len(vocab) if vocab else 1
        print(f"Warning: Could not infer num_antibodies from state_dict. Falling back to vocab length ({num_antibodies}).")

    if args.class_names:
        class_names = args.class_names
        if len(class_names) != num_classes:
            print(f"Warning: --class_names provided {len(class_names)} classes, but model expects {num_classes}.")
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    print(f"Classes: {class_names}")
    print(f"Inference Set Size: {len(test_loader.dataset)}")

    # Initialize Model
    print("Initializing model...")

    model = DeepSetsCNN(
        num_antibodies=num_antibodies,
        num_classes=num_classes,
        backbone_name="resnet50",
        use_peft=True,
    )

    # Load Weights into model
    model.load_state_dict(state_dict)

    # Run Inference
    infer_model(
        model, test_loader, class_names, csv_path=CSV_PATH
    )

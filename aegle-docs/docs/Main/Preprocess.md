---
sidebar_position: 2
---

# Preprocess

This document describes the preprocessing pipeline for CODEX images in Aegle. The preprocessing consists of two main stages: **Full Image Preprocessing** and **Patched Image Preprocessing**, as implemented in `aegle/pipeline.py` under the `run_pipeline` function.

Key outputs include:


## A. Full Image Preprocessing

This stage prepares the full CODEX image for downstream patch-level processing.

### Step A1: Initialize `CodexImage`
- Constructs a `CodexImage` object using configuration parameters and command-line arguments.
- Loads the image and its corresponding antibody metadata.
- If enabled in the configuration (`data.generate_channel_stats: true`), calculates statistics across channels.

```python
codex_image = CodexImage(config, args)
codex_image.calculate_channel_statistics()
```

### Step A2: Extract Target Channels
- Uses configuration to extract biologically relevant channels (e.g. membrane or nucleus markers).

```python
codex_image.extract_target_channels()
```

### Optional: Visualize Whole Sample
- Saves RGB visualizations of extracted channels to the output directory if `visualization.visualize_whole_sample` is set to `true`.

```python
save_image_rgb(codex_image.extended_extracted_channel_image, ...)
save_image_rgb(codex_image.extracted_channel_image, ...)
```

---

## B. Patched Image Preprocessing

This stage slices the image into overlapping patches and optionally applies perturbations for robustness testing.

### Step B1: Extend Image
- Pads the original image to ensure that all patches are fully contained within the image bounds.

```python
codex_image.extend_image()
```

### Step B2: Initialize `CodexPatches` and Save Patches
- Initializes a `CodexPatches` object using the extended `CodexImage`, then generates and saves patches.
- Patch metadata is also saved.

```python
codex_patches = CodexPatches(codex_image, config, args)
codex_patches.save_patches(...)
codex_patches.save_metadata()
```

### Optional: Add Disruptions
- If configured, adds synthetic noise or perturbations to patches for testing robustness.
- Disruption parameters include `type` and `level`, and the disrupted patches can be optionally saved and visualized.

```python
disruption_type = disruption_config.get("type")
disruption_level = disruption_config.get("level", 1)
codex_patches.add_disruptions(disruption_type, disruption_level)
codex_patches.save_disrupted_patches()
```

### Optional: Visualize Patches
- RGB visualizations of either the original or disrupted patches are saved if `visualization.visualize_patches` is set to `true`.

```python
save_patches_rgb(codex_patches.extracted_channel_patches, ...)
```

---

This preprocessing module prepares both the full and patched images for subsequent segmentation and analysis steps. All outputs are saved in a structured format for downstream reproducibility.


#!/usr/bin/env python3
"""
PyTorch DataLoader for categorized TIFF tiles (preserves original HxW and channel count).

Directory layout (produced by your tiler):
    root_dir/
      bubble/
      fold/
      smear/
      unlabeled/     # optional

Key behaviors
- Loads TIFFs with tifffile (no resizing, no RGB conversion).
- Preserves channel count exactly as on disk.
- Returns tensors shaped (C, H, W).
- Optional float32 scaling for integer dtypes (configurable).
- Simple, size-preserving augmentations (random flips).
- Deterministic stratified train/val(/test) split.
- **New**: Metadata integration for Deep Sets (antibody IDs).

Note: Many CNN backbones expect a fixed number of channels (often 3). If your tiles
have C != 3, choose an architecture that supports arbitrary input channels, or add
a 1x1 conv stem to adapt channels. This loader intentionally does NOT alter channels.
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Import metadata utils
try:
    from aegle.artifact_CNN.metadata_utils import load_metadata, build_vocabulary, build_metadata_map, get_antibody_ids, UNK_TOKEN
except ImportError:
    # Fallback if running from a different directory or not yet set up
    print("Warning: metadata_utils not found. Deep Sets features will be disabled.")
    load_metadata = None

try:
    import tifffile as tiff
    _HAVE_TIFFFILE = True
except Exception:
    tiff = None  # type: ignore
    _HAVE_TIFFFILE = False

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    Image = None  # type: ignore
    _HAVE_PIL = False


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(root: Path) -> List[Tuple[Path, int]]:
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    
    if not classes:
        # Check if there are images directly in the root
        images_in_root = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if images_in_root:
            return [(p, 0) for p in images_in_root]
        raise RuntimeError(f"No class subfolders or images found under {root}. Running inference mode.")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Tuple[Path, int]] = []
    for cls in classes:
        cdir = root / cls
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((p, class_to_idx[cls]))
    if not items:
        raise RuntimeError(f"No images found under {root}. "
                           "Expected files like root/Inner_Medulla/*.tiff, etc.")
    return items


def _ensure_channel_last(arr: np.ndarray) -> np.ndarray:
    """Heuristic: if channel-first with small C and large last dim, move channel to last."""
    if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[-1] > 8:
        return np.moveaxis(arr, 0, -1)
    return arr


def _to_tensor_preserve(arr: np.ndarray, to_float32: bool = True) -> torch.Tensor:
    """
    Convert numpy image array to torch tensor (C,H,W) without resizing or channel changes.
    - If 2D => add channel dim (1,H,W)
    - If channel-last => permute to (C,H,W)
    - If integer dtype and to_float32=True => scale to [0,1] float32
    - Else keep dtype (e.g., float32) as-is
    """
    if arr.ndim == 2:
        arr = arr[:, :, None]  # (H,W,1)

    # Now expect (H,W,C)
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported image shape {arr.shape}")

    # Make contiguous and convert
    if to_float32:
        if np.issubdtype(arr.dtype, np.integer):
            maxv = np.iinfo(arr.dtype).max
            arr = (arr.astype(np.float32) / float(maxv))
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
    else:
        # Keep original dtype
        pass

    # channel-last -> channel-first
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)  # (C,H,W)
    return tensor


def _load_image_tensor(path: Path, to_float32: bool = True) -> torch.Tensor:
    """Load image as torch tensor (C,H,W), preserving original HxW and channels."""
    suffix = path.suffix.lower()

    # Always use tifffile for TIFFs; PIL cannot handle many multi-sample TIFFs
    if suffix in {".tif", ".tiff"}:
        if not _HAVE_TIFFFILE:
            raise RuntimeError(
                f"tifffile is required to read TIFFs, but not available. "
                f"Install with: pip install tifffile imagecodecs"
            )
        try:
            import tifffile as tiff  # ensure local symbol
            # Prefer imread; falls back internally and is concise
            arr = tiff.imread(str(path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to read TIFF via tifffile: {path}\n"
                f"Make sure 'imagecodecs' is installed for compressed TIFFs.\n"
                f"Original error: {e}"
            ) from e

        arr = _ensure_channel_last(np.asarray(arr))
        return _to_tensor_preserve(arr, to_float32=to_float32)

    # Non-TIFFs: PIL is fine
    if not _HAVE_PIL:
        raise RuntimeError("Pillow is required to read non-TIFF images: pip install pillow")
    from PIL import Image  # local import to avoid module being None at import time
    img = Image.open(path)
    arr = np.array(img)  # (H,W) or (H,W,C) — keep channels as-is
    return _to_tensor_preserve(arr, to_float32=to_float32)


@dataclass
class SplitConfig:
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 42
    stratify: bool = True


class TileFolderDataset(Dataset):
    """Dataset returning (tensor(C,H,W), label:int, antibody_ids:tensor(C), path:str)."""
    def __init__(
        self,
        items: Sequence[Tuple[Path, int]],
        to_float32: bool = True,
        hflip: bool = True,
        vflip: bool = False,
        rotate: bool = False,
        metadata_map: Optional[Dict[str, Dict[str, str]]] = None,
        vocab: Optional[Dict[str, int]] = None,
        unk_dropout_prob: float = 0.0,
    ):
        self.items = list(items)
        self.to_float32 = to_float32
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.metadata_map = metadata_map
        self.vocab = vocab
        self.unk_dropout_prob = unk_dropout_prob

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, target = self.items[idx]
        x = _load_image_tensor(path, to_float32=self.to_float32)  # (C,H,W)

        # Size-preserving augmentations
        if self.hflip and random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # flip width (W)
        if self.vflip and random.random() < 0.5:
            x = torch.flip(x, dims=[1])  # flip height (H)
        if self.rotate:
            # Random 0, 90, 180, 270 rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])
            
        # Metadata / Antibody IDs
        antibody_ids = torch.zeros(x.shape[0], dtype=torch.long) # Default to 0 (<UNK>)
        
        if self.metadata_map and self.vocab:
            # Extract image_id from filename (e.g., "img0001_tile_0_0.tif" -> "img0001")

            filename = path.stem # "img0001_tile_0_0"
            image_id = filename.split('_')[0] 
            
            if image_id in self.metadata_map:
                ids = get_antibody_ids(image_id, self.metadata_map, self.vocab)
                
                # Check if channel count matches
                if len(ids) != x.shape[0]:
                    print(f"Warning: Channel count mismatch for {image_id}: expected {x.shape[0]}, got {len(ids)}")
                    if len(ids) > x.shape[0]: # longer metadata antibody list provided than actual image channels
                        ids = ids[:x.shape[0]]
                        print(f"Truncated antibody IDs to {len(ids)}")
                    elif len(ids) < x.shape[0]:
                        # Pad with UNK
                        ids.extend([0] * (x.shape[0] - len(ids)))
                        print(f"Padded antibody IDs to {len(ids)}")

                # Apply Channel Dropout
                if self.unk_dropout_prob > 0:
                    for i in range(len(ids)):
                        if random.random() < self.unk_dropout_prob:
                            # Drop the antibody ID
                            ids[i] = 0
                            # Also zero out the actual image pixel tensor for this channel
                            # x is (C, H, W)
                            if i < x.shape[0]:
                                x[i, :, :] = 0.0
                            
                antibody_ids = torch.tensor(ids, dtype=torch.long)
            else:
                print(f"Warning: Image ID {image_id} not found in metadata.")
                pass

        return x, target, antibody_ids, str(path)


def stratified_split(
    items: List[Tuple[Path, int]],
    config: SplitConfig,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(config.seed)
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for p, y in items:
        by_class.setdefault(y, []).append((p, y))

    train, val, test = [], [], []
    for y, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * config.test_ratio))
        n_val = int(round((n - n_test) * config.val_ratio))
        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    if not config.stratify:
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * config.test_ratio))
        n_val = int(round((n - n_test) * config.val_ratio))
        test = items[:n_test]
        val = items[n_test:n_test + n_val]
        train = items[n_test + n_val:]

    return train, val, test


def _class_weights(items: List[Tuple[Path, int]], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, y in items:
        counts[y] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights


def build_dataloaders(
    root_dir: str | os.PathLike,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 42,
    hflip: bool = True,
    vflip: bool = False,
    rotate: bool = False,
    to_float32: bool = True,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True,
    json_path: Optional[str] = None,
    unk_dropout_prob: float = 0.0,
    collate_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, int], Optional[Dict[str, int]]]:
    """
    Build (train_loader, val_loader, test_loader, class_to_idx, vocab) for a folder dataset.
    """
    root = Path(root_dir)
    items = _list_images(root)

    classes = sorted({p.parent.name for p, _ in items})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = [(p, class_to_idx[p.parent.name]) for p, _ in items]

    train_items, val_items, test_items = stratified_split(
        items, SplitConfig(val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, stratify=True)
    )
    
    # Load Metadata
    metadata_map = None
    vocab = None
    if json_path:
        if load_metadata is None:
             raise RuntimeError("metadata_utils not available but json_path provided.")
        metadata = load_metadata(json_path)
        vocab = build_vocabulary(metadata)
        metadata_map = build_metadata_map(metadata)
        print(f"Loaded metadata. Vocab size: {len(vocab)}")

    # Train DS has dropout
    train_ds = TileFolderDataset(
        train_items, 
        to_float32=to_float32, 
        hflip=hflip, 
        vflip=vflip,
        rotate=rotate,
        metadata_map=metadata_map,
        vocab=vocab,
        unk_dropout_prob=unk_dropout_prob
    )
    
    # Val/Test DS has NO dropout and NO rotation/flips
    val_ds = TileFolderDataset(
        val_items, 
        to_float32=to_float32, 
        hflip=False, 
        vflip=False,
        rotate=False,
        metadata_map=metadata_map,
        vocab=vocab,
        unk_dropout_prob=0.0
    )
    
    test_ds = TileFolderDataset(
        test_items, 
        to_float32=to_float32, 
        hflip=False, 
        vflip=False,
        metadata_map=metadata_map,
        vocab=vocab,
        unk_dropout_prob=0.0
    ) if test_items else None

    train_sampler = None
    if use_weighted_sampler:
        weights = _class_weights(train_items, num_classes=len(classes))
        sample_weights = [weights[y].item() for _, y in train_items]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn
    ) if len(train_ds) > 0 else None
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn
    ) if len(val_ds) > 0 else None
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn
        )

    return train_loader, val_loader, test_loader, class_to_idx, vocab


def deep_sets_collate_fn(batch):
    """
    Collate function for Deep Sets with variable number of channels.
    
    Args:
        batch: List of tuples (x, y, antibody_ids, path)
            x: (C, H, W) tensor
            y: int
            antibody_ids: (C,) tensor
            path: str
            
    Returns:
        x_flat: (Total_C, 1, H, W) - All channels from all images flattened
        y: (B,) - Labels
        ids_flat: (Total_C,) - Antibody IDs for all channels
        batch_indices: (Total_C,) - Index of the image in the batch (0..B-1) for each channel
        paths: List[str] - Paths
    """
    xs = []
    ys = []
    ids_list = []
    batch_indices = []
    paths = []
    
    for i, (x, y, ids, path) in enumerate(batch):
        # x is (C, H, W)
        C, H, W = x.shape
        
        # Reshape x to (C, 1, H, W) and append
        xs.append(x.unsqueeze(1))
        
        ys.append(y)
        ids_list.append(ids)
        paths.append(path)
        
        # Create batch indices for this image: [i, i, ..., i] (C times)
        batch_indices.append(torch.full((C,), i, dtype=torch.long))
        
    # Concatenate everything
    x_flat = torch.cat(xs, dim=0) # (Total_C, 1, H, W)
    y = torch.tensor(ys, dtype=torch.long) # (B,)
    ids_flat = torch.cat(ids_list, dim=0) # (Total_C,)
    batch_indices = torch.cat(batch_indices, dim=0) # (Total_C,)
    
    return x_flat, y, ids_flat, batch_indices, paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Dataloaders for categorized TIFF tiles (preserve size & channels)")
    ap.add_argument("--root", required=True, help="Root directory containing class subfolders")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--json_path", type=str, help="Path to metadata JSON")
    args = ap.parse_args()

    tl, vl, tl2, class_to_idx, vocab = build_dataloaders(
        root_dir=args.root,
        batch_size=args.batch_size,
        json_path=args.json_path,
        unk_dropout_prob=0.1,
        collate_fn=deep_sets_collate_fn
    )

    print("Classes:", class_to_idx)
    if vocab:
        print("Vocab sample:", list(vocab.items())[:5])
        
    xb, yb, ids, batch_indices, paths = next(iter(tl))
    print("Train batch:", xb.shape, yb.shape, ids.shape, batch_indices.shape)
    print("IDs sample:", ids[0])


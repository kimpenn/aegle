import json
import os
from typing import List, Dict, Any

UNK_TOKEN = "<UNK>"

def load_metadata(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads metadata from a JSON file.
    Expected format: List of dicts, each having "image_id" and "CHANNELS".
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def build_vocabulary(metadata: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Builds a vocabulary of antibody names from the metadata.
    Index 0 is reserved for <UNK>.
    """
    vocab = {UNK_TOKEN: 0}
    idx = 1
    
    # Collect all unique channel names
    unique_names = set()
    for item in metadata:
        channels = item.get("CHANNELS", {})
        # channels is a dict like {"Channel:0:0": "DAPI", ...}
        for name in channels.values():
            unique_names.add(name)
            
    # Sort for deterministic ordering
    for name in sorted(list(unique_names)):
        vocab[name] = idx
        idx += 1
        
    return vocab

def get_antibody_ids(image_id: str, metadata_map: Dict[str, Dict[str, str]], vocab: Dict[str, int]) -> List[int]:
    """
    Returns a list of antibody IDs for a given image ID.
    If a channel name is not in the vocab, it maps to 0 (<UNK>).
    """
    if image_id not in metadata_map:
        raise ValueError(f"Image ID '{image_id}' not found in metadata.")
        
    channels_dict = metadata_map[image_id]
    # Parse keys to get index
    # Key format: "Channel:0:X"
    sorted_items = []
    for key, name in channels_dict.items():
        try:
            # Extract the last number
            parts = key.split(':')
            idx = int(parts[-1])
            sorted_items.append((idx, name))
        except ValueError:
            print(f"Warning: Could not parse channel key '{key}' for image '{image_id}'")
            continue
            
    # Sort by index
    sorted_items.sort(key=lambda x: x[0])
    
    # Convert names to IDs
    ids = []
    for _, name in sorted_items:
        ids.append(vocab.get(name, 0)) # Default to 0 (<UNK>)
        
    return ids

def build_metadata_map(metadata: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Converts list of metadata items to a dict keyed by image_id for fast lookup.
    """
    mapping = {}
    for item in metadata:
        img_id = item.get("image_id")
        if img_id:
            mapping[img_id] = item.get("CHANNELS", {})
    return mapping

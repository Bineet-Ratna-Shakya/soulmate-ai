#!/usr/bin/env python3
import os
import sys
import json
import pickle
from pathlib import Path

import chess
import torch

from Core.monte_carlo_tree_search import MCTS
from Core.Core_Soul import SoulmateModel

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(project_root, relative_path)

# Define the project root directory
project_root = Path(__file__).parent.absolute()

def load_model_with_dataparallel_compatibility(model_path, device=None):
    """Load a model with DataParallel compatibility (robust version)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} to {device}")
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # If checkpoint is a dict with 'model_state_dict', use it
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and all(isinstance(v, dict) for v in checkpoint.values()):
        # Try to find a state_dict-like key
        for k in checkpoint:
            if 'state_dict' in k:
                state_dict = checkpoint[k]
                break
        else:
            raise RuntimeError(f"No valid model state_dict found in checkpoint: {model_path}")
    else:
        state_dict = checkpoint
    # Ensure state_dict is a dict of tensors
    if not (isinstance(state_dict, dict) and all(hasattr(v, 'size') for v in state_dict.values())):
        raise RuntimeError(f"Extracted state_dict is not valid. Please check the checkpoint format: {model_path}")
    # Fix state dict keys for DataParallel compatibility
    fixed_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        fixed_state_dict[name] = v
    # Detect the number of residual blocks
    num_res_blocks = 0
    for key in fixed_state_dict.keys():
        if 'res_blocks.' in key:
            try:
                block_num = int(key.split('res_blocks.')[1].split('.')[0])
                num_res_blocks = max(num_res_blocks, block_num + 1)
            except Exception:
                pass
    print(f"Detected {num_res_blocks} residual blocks in the state dict")
    # Detect the number of output classes
    policy_fc_key = None
    for k in fixed_state_dict.keys():
        if 'policy_fc.weight' in k:
            policy_fc_key = k
            break
    if policy_fc_key:
        num_classes = fixed_state_dict[policy_fc_key].shape[0]
        print(f"Detected {num_classes} classes from state dict")
    else:
        num_classes = 1977
        print(f"Could not detect number of classes, using default: {num_classes}")
    # Create the model
    model = SoulmateModel(num_res_blocks=num_res_blocks, num_classes=num_classes).to(device)
    # Try to load the state dict
    try:
        model.load_state_dict(fixed_state_dict)
        print("Successfully loaded state dict")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(fixed_state_dict.keys())
        missing_keys = model_keys - state_dict_keys
        unexpected_keys = state_dict_keys - model_keys
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        # Try loading with only matching keys
        matching_state_dict = {k: v for k, v in fixed_state_dict.items() if k in model_keys}
        print(f"Created matching state dict with {len(matching_state_dict)} keys")
        print("Attempting to load with strict=False...")
        model.load_state_dict(matching_state_dict, strict=False)
        print("Loaded state dict with strict=False")
    # Wrap with DataParallel if using multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.eval()
    return model

def load_mapping(mapping_path):
    """
    Load a mapping file (usually created with pickle)
    
    Args:
        mapping_path (str): Path to the mapping file
        
    Returns:
        dict: The loaded mapping dictionary
    """
    try:
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        print(f"Mapping loaded successfully from {mapping_path}")
        return mapping
    except Exception as e:
        print(f"Error loading mapping: {e}")
        sys.exit(1)

def main():
    # Default paths based on your project structure
    model_path = get_resource_path("Hot Models/final_hybrid_model.pth")
    mapping_path = get_resource_path("Hot Mappings/100kmap.pth")


    # Load mapping first
    mapping = load_mapping(mapping_path)

    # Choose device for model loading (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with DataParallel compatibility
    model = load_model_with_dataparallel_compatibility(model_path, device)

    # Accept FEN from sys.argv or stdin
    if len(sys.argv) > 1:
        fen = sys.argv[1]
    else:
        try:
            fen = input("Enter FEN string (or leave blank for initial position): ").strip()
        except EOFError:
            fen = ''
    if not fen:
        chess_state = chess.Board()  # Start from the initial chess position
    else:
        try:
            chess_state = chess.Board(fen)
        except Exception as e:
            print(json.dumps({"error": f"Invalid FEN: {fen}", "details": str(e)}))
            sys.exit(1)

    mcts = MCTS(model)
    best_move = mcts.get_move(chess_state)
    # Output as JSON or clean string
    result = {"fen": chess_state.fen(), "best_move": str(best_move)}
    try:
        print(json.dumps(result), flush=True)
    except Exception as e:
        print(json.dumps({"error": "Failed to output JSON", "details": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

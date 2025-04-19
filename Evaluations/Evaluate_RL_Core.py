import os
import time
import random
import pickle
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
import chess
import chess.engine
import chess.pgn
from tqdm import tqdm
import psutil

from Core_RL_Logic import ChessEnvironment, DQNAgent
from Core.Core_Soul import SoulmateModel
from Core.monte_carlo_tree_search import MCTS
from Core.Neural_Augmentation import create_full_move_to_int

# Set number of worker processes based on CPU cores
NUM_WORKERS = max(1, mp.cpu_count() - 1)

def detect_num_res_blocks(state_dict):
    """Detect the number of residual blocks from a state dictionary"""
    num_res_blocks = 0
    for key in state_dict.keys():
        if 'res_blocks.' in key:
            # Extract the block number from keys like 'res_blocks.5.conv1.weight'
            block_num = int(key.split('res_blocks.')[1].split('.')[0])
            num_res_blocks = max(num_res_blocks, block_num + 1)
    return num_res_blocks  # Return the actual number of blocks (not +1)

def load_model_with_dataparallel_compatibility(model_path, device=None):
    """Load a model with DataParallel compatibility"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path} to {device}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    print(f"Loaded state dict with {len(state_dict)} keys")
    
    # Print the first few keys for debugging
    print("First few keys in state dict:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {i}: {key}")
    
    # Fix state dict keys for DataParallel compatibility
    fixed_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if it exists
        name = k[7:] if k.startswith('module.') else k
        fixed_state_dict[name] = v
    
    # Detect the number of residual blocks from the state dict
    num_res_blocks = detect_num_res_blocks(fixed_state_dict)
    print(f"Detected {num_res_blocks} residual blocks in the state dict")
    
    # Determine the number of classes from the state dict
    # Find the policy_fc.weight key to determine the output size
    policy_fc_key = None
    for k in fixed_state_dict.keys():
        if 'policy_fc.weight' in k:
            policy_fc_key = k
            break
    
    if policy_fc_key:
        num_classes = fixed_state_dict[policy_fc_key].shape[0]
        print(f"Detected {num_classes} classes from state dict")
    else:
        # Default to a reasonable number if we can't determine it
        num_classes = 1977  # Standard number of possible chess moves
        print(f"Could not detect number of classes, using default: {num_classes}")
    
    # Create the model with the detected number of residual blocks
    print(f"Creating model with {num_res_blocks} residual blocks and {num_classes} classes")
    model = SoulmateModel(num_res_blocks=num_res_blocks, num_classes=num_classes).to(device)
    
    # Print model's state dict keys for comparison
    print("Model's state dict keys:")
    for i, key in enumerate(list(model.state_dict().keys())[:5]):
        print(f"  {i}: {key}")
    
    # Try to load the state dict
    try:
        model.load_state_dict(fixed_state_dict)
        print("Successfully loaded state dict")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        
        # Try to identify missing or unexpected keys
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(fixed_state_dict.keys())
        
        missing_keys = model_keys - state_dict_keys
        unexpected_keys = state_dict_keys - model_keys
        
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Create a new state dict with only matching keys
        matching_state_dict = {}
        for k, v in fixed_state_dict.items():
            if k in model_keys:
                matching_state_dict[k] = v
        
        print(f"Created matching state dict with {len(matching_state_dict)} keys")
        
        # Try loading with strict=False
        print("Attempting to load with strict=False...")
        model.load_state_dict(matching_state_dict, strict=False)
        print("Loaded state dict with strict=False")
    
    # Wrap with DataParallel if using multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.eval()  # Set to evaluation mode
    return model

def load_models(original_model_path, rl_model_path, move_to_int_path):
    """Load both the original model and the RL-trained model"""
    # Load move mapping
    with open(move_to_int_path, "rb") as file:
        move_to_int = pickle.load(file)
    
    # Load original model with DataParallel compatibility
    original_model = load_model_with_dataparallel_compatibility(original_model_path)
    
    # Load RL model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_shape = (18, 8, 8)
    num_classes = len(move_to_int)
    
    # Load the RL model state dict to detect number of residual blocks
    rl_state_dict = torch.load(rl_model_path, map_location=device)
    print(f"Loaded RL model state dict with {len(rl_state_dict)} keys")
    
    # Check if the state dict is nested (contains 'model_state_dict')
    if 'model_state_dict' in rl_state_dict:
        print("RL model state dict contains 'model_state_dict' key")
        rl_state_dict = rl_state_dict['model_state_dict']
        print(f"Extracted model state dict with {len(rl_state_dict)} keys")
    
    # Print the first few keys for debugging
    print("First few keys in RL model state dict:")
    for i, key in enumerate(list(rl_state_dict.keys())[:5]):
        print(f"  {i}: {key}")
    
    # Fix state dict keys for DataParallel compatibility
    fixed_rl_state_dict = {}
    for k, v in rl_state_dict.items():
        # Remove 'module.' prefix if it exists
        name = k[7:] if k.startswith('module.') else k
        fixed_rl_state_dict[name] = v
    
    # Detect the number of residual blocks from the RL model state dict
    num_res_blocks = detect_num_res_blocks(fixed_rl_state_dict)
    print(f"Detected {num_res_blocks} residual blocks in the RL model state dict")
    
    # Create RL agent with the detected number of residual blocks
    print(f"Creating RL agent with {num_res_blocks} residual blocks and {num_classes} classes")
    rl_agent = DQNAgent(state_shape, num_classes, num_res_blocks=num_res_blocks)
    
    # Print RL model's state dict keys for comparison
    print("RL model's state dict keys:")
    for i, key in enumerate(list(rl_agent.model.state_dict().keys())[:5]):
        print(f"  {i}: {key}")
    
    # Try to load the state dict
    try:
        rl_agent.load(rl_model_path)
        print("Successfully loaded RL model state dict")
    except Exception as e:
        print(f"Error loading RL model state dict: {e}")
        
        # Try to identify missing or unexpected keys
        model_keys = set(rl_agent.model.state_dict().keys())
        state_dict_keys = set(fixed_rl_state_dict.keys())
        
        missing_keys = model_keys - state_dict_keys
        unexpected_keys = state_dict_keys - model_keys
        
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Create a new state dict with only matching keys
        matching_state_dict = {}
        for k, v in fixed_rl_state_dict.items():
            if k in model_keys:
                matching_state_dict[k] = v
        
        print(f"Created matching state dict with {len(matching_state_dict)} keys")
        
        # Try loading with strict=False
        print("Attempting to load RL model with strict=False...")
        rl_agent.model.load_state_dict(matching_state_dict, strict=False)
        rl_agent.target_model.load_state_dict(matching_state_dict, strict=False)
        print("Loaded RL model state dict with strict=False")
    
    rl_agent.model.eval()
    
    # Move models to GPU if available
    if torch.cuda.is_available():
        original_model = original_model.to(device)
        rl_agent.model = rl_agent.model.to(device)
        rl_agent.target_model = rl_agent.target_model.to(device)
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
    
    return original_model, rl_agent, move_to_int

def play_game_original_vs_rl(original_model, rl_agent, move_to_int, mcts_original, mcts_rl, temp=0.1):
    """Play a game between the original model and the RL-trained model"""
    board = chess.Board()
    moves = []
    move_number = 1
    
    print("\nStarting new game: Original Model (White) vs RL Model (Black)")
    print("Initial board position:")
    print(board)
    
    while not board.is_game_over():
        try:
            if board.turn == chess.WHITE:
                # Original model's move
                move = mcts_original.get_move(board, temp=temp)
                if move not in board.legal_moves:
                    print(f"Warning: Original model generated illegal move {move}. Using random legal move instead.")
                    move = random.choice(list(board.legal_moves))
                moves.append(move)
                board.push(move)
                print(f"\nMove {move_number} (White - Original Model): {move}")
            else:
                # RL model's move
                env = ChessEnvironment()
                env.board = board.copy()
                state = env._get_state()
                action = rl_agent.act(state, board, training=False)
                move = env._index_to_move(action)
                if move not in board.legal_moves:
                    print(f"Warning: RL model generated illegal move {move}. Using random legal move instead.")
                    move = random.choice(list(board.legal_moves))
                moves.append(move)
                board.push(move)
                print(f"\nMove {move_number} (Black - RL Model): {move}")
            
            move_number += 1
            print("\nCurrent board position:")
            print(board)
            print(f"FEN: {board.fen()}")
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            print(f"Current board state: {board.fen()}")
            print(f"Legal moves: {list(board.legal_moves)}")
            # Use a random legal move as fallback
            move = random.choice(list(board.legal_moves))
            board.push(move)
            print(f"Fallback move: {move}")
            print("\nBoard after fallback move:")
            print(board)
    
    # Generate PGN
    game = chess.pgn.Game.from_board(board)
    pgn = str(game)
    
    print("\nGame Over!")
    print(f"Result: {board.result()}")
    print("\nPGN for the game:")
    print(pgn)
    
    return board.result()

def play_game_rl_vs_stockfish(rl_agent, stockfish_path, temp=0.1):
    """Play a game between the RL-trained model and Stockfish"""
    board = chess.Board()
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    moves = []
    move_number = 1
    
    print("\nStarting new game: RL Model (White) vs Stockfish (Black)")
    print("Initial board position:")
    print(board)
    
    while not board.is_game_over():
        try:
            if board.turn == chess.WHITE:
                # RL model's move
                env = ChessEnvironment()
                env.board = board.copy()
                state = env._get_state()
                action = rl_agent.act(state, board, training=False)
                move = env._index_to_move(action)
                if move not in board.legal_moves:
                    print(f"Warning: RL model generated illegal move {move}. Using random legal move instead.")
                    move = random.choice(list(board.legal_moves))
                moves.append(move)
                board.push(move)
                print(f"\nMove {move_number} (White - RL Model): {move}")
            else:
                # Stockfish's move
                result = stockfish.play(board, chess.engine.Limit(time=0.1))
                moves.append(result.move)
                board.push(result.move)
                print(f"\nMove {move_number} (Black - Stockfish): {result.move}")
            
            move_number += 1
            print("\nCurrent board position:")
            print(board)
            print(f"FEN: {board.fen()}")
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            print(f"Current board state: {board.fen()}")
            print(f"Legal moves: {list(board.legal_moves)}")
            # Use a random legal move as fallback
            move = random.choice(list(board.legal_moves))
            board.push(move)
            print(f"Fallback move: {move}")
            print("\nBoard after fallback move:")
            print(board)
    
    stockfish.close()
    
    # Generate PGN
    game = chess.pgn.Game.from_board(board)
    pgn = str(game)
    
    print("\nGame Over!")
    print(f"Result: {board.result()}")
    print("\nPGN for the game:")
    print(pgn)
    
    return board.result()

def evaluate_models(original_model_path, rl_model_path, move_to_int_path, stockfish_path, num_games=10):
    """Evaluate the RL model against the original model and Stockfish with parallel processing"""
    # Print system information
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Using {NUM_WORKERS} worker processes")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load models
    original_model, rl_agent, move_to_int = load_models(original_model_path, rl_model_path, move_to_int_path)
    
    # Create MCTS instances with move mapping
    # Get the base model if using DataParallel
    base_original_model = original_model.module if isinstance(original_model, torch.nn.DataParallel) else original_model
    base_rl_model = rl_agent.model.module if isinstance(rl_agent.model, torch.nn.DataParallel) else rl_agent.model
    
    mcts_original = MCTS(base_original_model, num_simulations=200, exploration_weight=1.0)
    mcts_original.move_to_int = move_to_int  # Set move mapping for original model MCTS
    
    mcts_rl = MCTS(base_rl_model, num_simulations=200, exploration_weight=1.0)
    mcts_rl.move_to_int = move_to_int  # Set move mapping for RL model MCTS
    
    # Play games between original model and RL model
    print("\nPlaying games between original model and RL model...")
    original_vs_rl_results = []
    
    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        result = play_game_original_vs_rl(original_model, rl_agent, move_to_int, mcts_original, mcts_rl)
        original_vs_rl_results.append(result)
        print(f"Game {game_num + 1} result: {result}")
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Play games between RL model and Stockfish
    print("\nPlaying games between RL model and Stockfish...")
    rl_vs_stockfish_results = []
    
    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        result = play_game_rl_vs_stockfish(rl_agent, stockfish_path)
        rl_vs_stockfish_results.append(result)
        print(f"Game {game_num + 1} result: {result}")
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate win rates
    original_wins = sum(1 for r in original_vs_rl_results if r == '1-0')
    rl_wins = sum(1 for r in original_vs_rl_results if r == '0-1')
    draws = sum(1 for r in original_vs_rl_results if r == '1/2-1/2')
    
    rl_vs_stockfish_wins = sum(1 for r in rl_vs_stockfish_results if r == '1-0')
    stockfish_wins = sum(1 for r in rl_vs_stockfish_results if r == '0-1')
    rl_vs_stockfish_draws = sum(1 for r in rl_vs_stockfish_results if r == '1/2-1/2')
    
    print("\nFinal Results:")
    print(f"Original model vs RL model:")
    print(f"Original model wins: {original_wins}")
    print(f"RL model wins: {rl_wins}")
    print(f"Draws: {draws}")
    
    print(f"\nRL model vs Stockfish:")
    print(f"RL model wins: {rl_vs_stockfish_wins}")
    print(f"Stockfish wins: {stockfish_wins}")
    print(f"Draws: {rl_vs_stockfish_draws}")

if __name__ == "__main__":
    # Define base path for models
    base_path = "/Users/soul/Desktop/soulmate-ai/Hot Models"
    
    evaluate_models(
        original_model_path=os.path.join(base_path, "100kepochs.pth"),
        rl_model_path=os.path.join(base_path, "final_hybrid_model.pth"),
        move_to_int_path=os.path.join(base_path, "100kmap.pth"),
        stockfish_path="stockfish-windows-x86-64-avx2.exe",
        num_games=1
    )
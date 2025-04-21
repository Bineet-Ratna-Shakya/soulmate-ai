from datetime import datetime
import json
import sys
import os

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import torch
import chess
import chess.engine
import chess.pgn
import chess.polyglot
from tqdm import tqdm
import pickle
import time
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
import seaborn as sns
from collections import Counter, defaultdict

from Core.Core_Soul import SoulmateModel
from Core.monte_carlo_tree_search import MCTS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Core_Reinforcement_Learning')))
from Core_RL_Logic import DQNAgent, ChessEnvironment
from Core.Neural_Augmentation import create_full_move_to_int

# Set number of worker processes based on CPU cores
NUM_WORKERS = max(1, mp.cpu_count() - 1)

# Helper functions for chess metrics
def get_piece_value(piece):
    """Get the value of a chess piece in centipawns"""
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000  # High value for the king
    }
    return piece_values.get(piece.piece_type, 0)

def calculate_material_difference(board):
    """Calculate material difference between white and black in centipawns"""
    white_material = sum(get_piece_value(board.piece_at(square)) 
                          for square in chess.SQUARES 
                          if board.piece_at(square) and board.piece_at(square).color == chess.WHITE)
    
    black_material = sum(get_piece_value(board.piece_at(square)) 
                          for square in chess.SQUARES 
                          if board.piece_at(square) and board.piece_at(square).color == chess.BLACK)
    
    return white_material - black_material  # Positive value means white has advantage

def identify_opening(moves, num_moves=10):
    """Identify the opening based on first few moves (simplified)"""
    # This is a simplified approach; for production, consider using an openings database
    opening_prefix = " ".join([move for move in moves[:min(num_moves, len(moves))]])
    
    # Very basic opening identification - could be expanded with a proper database
    common_openings = {
        "e2e4 e7e5": "Open Game",
        "e2e4 c7c5": "Sicilian Defense",
        "e2e4 e7e6": "French Defense",
        "d2d4 d7d5": "Queen's Pawn Game",
        "d2d4 g8f6": "Indian Defense",
        "c2c4": "English Opening",
        "g1f3": "Réti Opening",
    }
    
    for prefix, name in common_openings.items():
        if opening_prefix.startswith(prefix):
            return name
    
    return "Unknown Opening"

def determine_game_termination(board):
    """Determine how the game ended"""
    if board.is_checkmate():
        return "checkmate"
    elif board.is_stalemate():
        return "stalemate"
    elif board.is_insufficient_material():
        return "insufficient_material"
    elif board.is_fifty_moves():
        return "fifty_moves"
    elif board.is_repetition():
        return "repetition"
    else:
        return "other" 

def calculate_blunder_rate(move_qualities, threshold=100):
    """Calculate the percentage of moves with centipawn loss above threshold"""
    if not move_qualities:
        return 0
    blunders = sum(1 for q in move_qualities if abs(q) > threshold)
    return (blunders / len(move_qualities)) * 100

def find_first_error_move(move_qualities, threshold=100):
    """Find the move number of the first significant error"""
    for i, quality in enumerate(move_qualities):
        if abs(quality) > threshold:
            return i + 1  # Move numbers are 1-indexed
    return None  # No significant error found

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

def load_rl_model(rl_model_path, move_to_int_path):
    """Load the RL-trained model"""
    # Load move mapping
    with open(move_to_int_path, "rb") as file:
        move_to_int = pickle.load(file)
    
    # Load RL model
    state_shape = (18, 8, 8)
    num_classes = len(move_to_int)
    
    # Load the RL model state dict to detect number of residual blocks
    rl_state_dict = torch.load(rl_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Set model to training mode
    rl_agent.model.train()
    
    return rl_agent, move_to_int

def analyze_game_quality(board_history, stockfish_path):
    """Analyze the quality of moves using Stockfish with unlimited time"""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    move_qualities = []
    move_times = {"rl": [], "stockfish": []}
    material_imbalance = []
    
    for i, board in enumerate(board_history):
        if not board.is_game_over():
            start_time = time.time()
            # Get Stockfish's evaluation
            info = engine.analyse(board, chess.engine.Limit(depth=2))
            end_time = time.time()
            
            # Track time taken for analysis
            if i % 2 == 0:  # RL model's move (white)
                move_times["rl"].append(end_time - start_time)
            else:  # Stockfish's move (black)
                move_times["stockfish"].append(end_time - start_time)
            
            # Track move quality
            move_qualities.append(info["score"].relative.score(mate_score=10000) / 100.0)
            
            # Track material imbalance
            material_imbalance.append(calculate_material_difference(board))
    
    engine.close()
    return {
        "move_qualities": move_qualities,
        "move_times": move_times,
        "material_imbalance": material_imbalance
    }

def play_game_rl_vs_stockfish(rl_agent, stockfish_path, temp=0.1):
    """Play a game between the RL-trained model and Stockfish with learning enabled"""
    board = chess.Board()
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    moves = []
    move_times = {"rl": [], "stockfish": []}
    move_number = 1
    board_history = [board.copy()]
    env = ChessEnvironment()
    opening_moves = []
    first_error_detected = False
    first_error_move = None
    
    print("\nStarting new game: RL Model (White) vs Stockfish (Black)")
    print("Initial board position:")
    print(board)
    
    while not board.is_game_over():
        try:
            if board.turn == chess.WHITE:
                # RL model's move with training enabled
                start_time = time.time()
                env.board = board.copy()
                state = env._get_state()
                action = rl_agent.act(state, board, training=True)  # Enable training
                move = env._index_to_move(action)
                if move not in board.legal_moves:
                    print(f"Warning: RL model generated illegal move {move}. Using random legal move instead.")
                    move = random.choice(list(board.legal_moves))
                end_time = time.time()
                
                # Record move time
                move_times["rl"].append(end_time - start_time)
                
                # Record move
                moves.append(move.uci())
                opening_moves.append(move.uci())
                board.push(move)
                print(f"\nMove {move_number} (White - RL Model): {move}")
                
                # Store experience in replay buffer
                next_state = env._get_state()
                env.board = board.copy()
                reward = env._calculate_reward()
                done = board.is_game_over()
                rl_agent.memory.push(state, action, reward, next_state, done)
                
                # Train the model
                if len(rl_agent.memory) >= rl_agent.batch_size:
                    loss = rl_agent.train()
                    if loss is not None:
                        print(f"Training loss: {loss:.4f}")
            else:
                # Stockfish's move with limited strength
                print("\nStockfish is thinking...")
                start_time = time.time()
                stockfish.configure({
                    'Skill Level': 1,
                })
                result = stockfish.play(board, chess.engine.Limit(depth=1))
                end_time = time.time()
                
                # Record move time
                move_times["stockfish"].append(end_time - start_time)
                
                # Record move
                moves.append(result.move.uci())
                opening_moves.append(result.move.uci())
                board.push(result.move)
                print(f"\nMove {move_number} (Black - Stockfish): {result.move}")
            
            move_number += 1
            board_history.append(board.copy())
            print("\nCurrent board position:")
            print(board)
            print(f"FEN: {board.fen()}")
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            print(f"Current board state: {board.fen()}")
            print(f"Legal moves: {list(board.legal_moves)}")
            # Use a random legal move as fallback
            move = random.choice(list(board.legal_moves))
            moves.append(move.uci())
            opening_moves.append(move.uci())
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
    
    # Analyze game quality
    analysis_results = analyze_game_quality(board_history, stockfish_path)
    move_qualities = analysis_results["move_qualities"]
    
    # Calculate blunder rate
    blunder_rate = calculate_blunder_rate(move_qualities, threshold=100)
    
    # Find first significant error
    if not first_error_detected:
        first_error_move = find_first_error_move(move_qualities, threshold=100)
        first_error_detected = first_error_move is not None
    
    # Identify opening
    opening = identify_opening(opening_moves)
    
    # Determine game termination type
    termination_type = determine_game_termination(board)
    
    return {
        'result': board.result(),
        'moves': moves,
        'move_qualities': move_qualities,
        'game_length': len(moves),
        'pgn': pgn,
        'blunder_rate': blunder_rate,
        'first_error_move': first_error_move,
        'move_times': move_times,
        'material_imbalance': analysis_results["material_imbalance"],
        'opening': opening,
        'termination_type': termination_type
    }

def plot_learning_curves(results_history, save_path=None):
    """Plot learning curves from evaluation results"""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    games = range(1, len(results_history) + 1)
    rl_wins = [sum(1 for r in results_history[:i] if r['result'] == '1-0') for i in games]
    stockfish_wins = [sum(1 for r in results_history[:i] if r['result'] == '0-1') for i in games]
    draws = [sum(1 for r in results_history[:i] if r['result'] == '1/2-1/2') for i in games]
    
    # Plot win rates
    plt.subplot(2, 2, 1)
    plt.plot(games, rl_wins, label='RL Wins')
    plt.plot(games, stockfish_wins, label='Stockfish Wins')
    plt.plot(games, draws, label='Draws')
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Count')
    plt.title('Game Results Over Time')
    plt.legend()
    
    # Plot average game length
    plt.subplot(2, 2, 2)
    avg_lengths = [np.mean([r['game_length'] for r in results_history[:i]]) for i in games]
    plt.plot(games, avg_lengths)
    plt.xlabel('Game Number')
    plt.ylabel('Average Game Length')
    plt.title('Average Game Length Over Time')
    
    # Plot move quality
    plt.subplot(2, 2, 3)
    avg_qualities = [np.mean([np.mean(r['move_qualities']) for r in results_history[:i]]) for i in games]
    plt.plot(games, avg_qualities)
    plt.xlabel('Game Number')
    plt.ylabel('Average Move Quality (centipawns)')
    plt.title('Average Move Quality Over Time')
    
    # Plot win rate percentage
    plt.subplot(2, 2, 4)
    win_rates = [(w / i) * 100 for w, i in zip(rl_wins, games)]
    plt.plot(games, win_rates)
    plt.xlabel('Game Number')
    plt.ylabel('Win Rate (%)')
    plt.title('RL Model Win Rate Over Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    else:
        plt.show()

def evaluate_models(rl_model_path, move_to_int_path, stockfish_path, num_games=10):
    """Evaluate and train the RL model against Stockfish with parallel processing"""
    # Print system information
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Using {NUM_WORKERS} worker processes")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load RL model
    rl_agent, move_to_int = load_rl_model(rl_model_path, move_to_int_path)
    
    # Play games between RL model and Stockfish
    print("\nPlaying games between RL model and Stockfish...")
    results_history = []
    
    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        result = play_game_rl_vs_stockfish(rl_agent, stockfish_path)
        results_history.append(result)
        print(f"Game {game_num + 1} result: {result['result']}")
        
        # Save the model after each game
        save_path = f"rl_model_after_game_{game_num + 1}.pth"
        rl_agent.save(save_path)
        print(f"Model saved to {save_path}")
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate win rates
    rl_vs_stockfish_wins = sum(1 for r in results_history if r['result'] == '1-0')
    stockfish_wins = sum(1 for r in results_history if r['result'] == '0-1')
    rl_vs_stockfish_draws = sum(1 for r in results_history if r['result'] == '1/2-1/2')
    
    # Calculate additional metrics
    avg_game_length = np.mean([r['game_length'] for r in results_history])
    avg_move_quality = np.mean([np.mean(r['move_qualities']) for r in results_history])
    
    print("\nFinal Results:")
    print(f"RL model vs Stockfish:")
    print(f"RL model wins: {rl_vs_stockfish_wins}")
    print(f"Stockfish wins: {stockfish_wins}")
    print(f"Draws: {rl_vs_stockfish_draws}")
    print(f"Average game length: {avg_game_length:.2f} moves")
    print(f"Average move quality: {avg_move_quality:.2f} centipawns")
    
    # Save results and plot learning curves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results to JSON
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'rl_wins': rl_vs_stockfish_wins,
            'stockfish_wins': stockfish_wins,
            'draws': rl_vs_stockfish_draws,
            'avg_game_length': avg_game_length,
            'avg_move_quality': avg_move_quality,
            'results_history': results_history
        }, f, indent=2)
    
    # Plot and save learning curves
    plot_path = os.path.join(results_dir, f"learning_curves_{timestamp}.png")
    plot_learning_curves(results_history, save_path=plot_path)
    
    # Plot advanced metrics
    plot_advanced_metrics(results_history, results_dir)
    
    return results_history

def plot_advanced_metrics(results_history, save_dir):
    """Plot advanced metrics for RL model evaluation"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Blunder Rate Plot
    plot_blunder_rate(results_history, os.path.join(save_dir, f"blunder_rate_{timestamp}.png"))
    
    # 2. Average Centipawn Loss Plot
    plot_centipawn_loss(results_history, os.path.join(save_dir, f"centipawn_loss_{timestamp}.png"))
    
    # 3. Move Accuracy Distribution Plot
    plot_move_accuracy_distribution(results_history, os.path.join(save_dir, f"move_accuracy_{timestamp}.png"))
    
    # 4. Win/Loss Streak Plot
    plot_win_loss_streaks(results_history, os.path.join(save_dir, f"win_loss_streaks_{timestamp}.png"))
    
    # 5. Opening Diversity Plot
    plot_opening_diversity(results_history, os.path.join(save_dir, f"opening_diversity_{timestamp}.png"))
    
    # 6. Time per Move Plot
    plot_time_per_move(results_history, os.path.join(save_dir, f"time_per_move_{timestamp}.png"))
    
    # 7. Material Imbalance Plot (for a sample game)
    if results_history:
        # Plot material imbalance for the last game as an example
        plot_material_imbalance(results_history[-1], os.path.join(save_dir, f"material_imbalance_sample_{timestamp}.png"))
    
    # 8. Game Termination Type Plot
    plot_game_termination_types(results_history, os.path.join(save_dir, f"game_termination_{timestamp}.png"))
    
    # 9. First Error Move Plot
    plot_first_error_moves(results_history, os.path.join(save_dir, f"first_error_move_{timestamp}.png"))

def plot_blunder_rate(results_history, save_path=None):
    """Plot blunder rate over games"""
    blunder_rates = [r['blunder_rate'] for r in results_history]
    games = range(1, len(blunder_rates) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.bar(games, blunder_rates, color='crimson', alpha=0.7)
    plt.axhline(y=np.mean(blunder_rates), color='r', linestyle='-', label=f'Average: {np.mean(blunder_rates):.2f}%')
    
    plt.xlabel('Game Number')
    plt.ylabel('Blunder Rate (%)')
    plt.title('Blunder Rate per Game (Threshold: 100cp)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Blunder rate plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_centipawn_loss(results_history, save_path=None):
    """Plot average centipawn loss per game"""
    avg_loss = [np.abs(np.mean(r['move_qualities'])) for r in results_history]
    games = range(1, len(avg_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, avg_loss, 'o-', color='blue', markersize=8)
    plt.axhline(y=np.mean(avg_loss), color='navy', linestyle='-', 
                label=f'Average: {np.mean(avg_loss):.2f} cp')
    
    plt.xlabel('Game Number')
    plt.ylabel('Average Centipawn Loss')
    plt.title('Average Centipawn Loss per Game')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Centipawn loss plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_move_accuracy_distribution(results_history, save_path=None):
    """Plot histogram of move qualities for RL and Stockfish"""
    all_qualities = []
    for r in results_history:
        qualities = r['move_qualities']
        all_qualities.extend(qualities)
    
    plt.figure(figsize=(12, 7))
    
    # Create histogram with different bin sizes
    n, bins, patches = plt.hist(all_qualities, bins=50, alpha=0.7, color='teal')
    
    # Add vertical line at x=0 (perfect play)
    plt.axvline(x=0, color='r', linestyle='--', label='Perfect Play')
    
    # Add annotations for mean and median
    mean_quality = np.mean(all_qualities)
    median_quality = np.median(all_qualities)
    plt.axvline(x=mean_quality, color='orange', linestyle='-', 
                label=f'Mean: {mean_quality:.2f} cp')
    plt.axvline(x=median_quality, color='green', linestyle='-', 
                label=f'Median: {median_quality:.2f} cp')
    
    plt.xlabel('Move Quality (Centipawns)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Move Qualities')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Move accuracy distribution saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_win_loss_streaks(results_history, save_path=None):
    """Plot longest win/loss streaks"""
    results = [r['result'] for r in results_history]
    
    # Calculate win/loss streaks
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    win_streaks = []
    loss_streaks = []
    
    for i, result in enumerate(results):
        if result == '1-0':  # RL win
            if current_streak >= 0:
                current_streak += 1
            else:
                loss_streaks.append(abs(current_streak))
                current_streak = 1
        elif result == '0-1':  # RL loss
            if current_streak <= 0:
                current_streak -= 1
            else:
                win_streaks.append(current_streak)
                current_streak = -1
        else:  # Draw
            if current_streak > 0:
                win_streaks.append(current_streak)
            elif current_streak < 0:
                loss_streaks.append(abs(current_streak))
            current_streak = 0
    
    # Capture the final streak
    if current_streak > 0:
        win_streaks.append(current_streak)
    elif current_streak < 0:
        loss_streaks.append(abs(current_streak))
    
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    
    plt.figure(figsize=(12, 6))
    
    # Create a bar plot for streak distribution
    win_counts = Counter(win_streaks)
    loss_counts = Counter(loss_streaks)
    
    max_streak = max(max(win_counts.keys()) if win_counts else 0, 
                     max(loss_counts.keys()) if loss_counts else 0)
    
    all_streaks = range(1, max_streak + 1)
    win_heights = [win_counts.get(s, 0) for s in all_streaks]
    loss_heights = [loss_counts.get(s, 0) for s in all_streaks]
    
    width = 0.35
    x = np.arange(len(all_streaks))
    
    plt.bar(x - width/2, win_heights, width, label='Win Streaks', color='green', alpha=0.7)
    plt.bar(x + width/2, loss_heights, width, label='Loss Streaks', color='red', alpha=0.7)
    
    plt.xlabel('Streak Length')
    plt.ylabel('Frequency')
    plt.title(f'Win/Loss Streak Distribution (Max Win: {max_win_streak}, Max Loss: {max_loss_streak})')
    plt.xticks(x, all_streaks)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Win/loss streak plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_opening_diversity(results_history, save_path=None):
    """Plot pie chart of openings played"""
    openings = [r['opening'] for r in results_history]
    opening_counts = Counter(openings)
    
    # Sort by frequency
    labels = [f"{opening} ({count})" for opening, count in opening_counts.most_common()]
    sizes = [count for _, count in opening_counts.most_common()]
    
    plt.figure(figsize=(12, 8))
    
    # Create a pie chart with percentage labels
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            shadow=True, explode=[0.1 if i == 0 else 0 for i in range(len(sizes))])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title(f'Opening Diversity (Total: {len(set(openings))} unique openings)')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Opening diversity plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_time_per_move(results_history, save_path=None):
    """Plot average time per move for RL and Stockfish"""
    rl_times = []
    stockfish_times = []
    
    for r in results_history:
        if 'move_times' in r:
            rl_times.append(np.mean(r['move_times']['rl']) if r['move_times']['rl'] else 0)
            stockfish_times.append(np.mean(r['move_times']['stockfish']) if r['move_times']['stockfish'] else 0)
    
    if not rl_times or not stockfish_times:
        print("No time data available")
        return
    
    games = range(1, len(rl_times) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(games, rl_times, 'o-', color='blue', label='RL Model')
    plt.plot(games, stockfish_times, 'o-', color='red', label='Stockfish')
    
    plt.axhline(y=np.mean(rl_times), color='navy', linestyle='--', 
                label=f'RL Avg: {np.mean(rl_times):.4f}s')
    plt.axhline(y=np.mean(stockfish_times), color='darkred', linestyle='--',
                label=f'SF Avg: {np.mean(stockfish_times):.4f}s')
    
    plt.xlabel('Game Number')
    plt.ylabel('Average Time per Move (seconds)')
    plt.title('Average Time per Move')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Time per move plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_material_imbalance(game_result, save_path=None):
    """Plot material imbalance over move number for a single game"""
    material_diffs = game_result['material_imbalance']
    moves = range(1, len(material_diffs) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(moves, material_diffs, 'o-', color='purple')
    
    # Fill areas based on material advantage
    plt.fill_between(moves, material_diffs, 0, where=(np.array(material_diffs) > 0), 
                     color='green', alpha=0.3, label='White Advantage')
    plt.fill_between(moves, material_diffs, 0, where=(np.array(material_diffs) < 0), 
                     color='red', alpha=0.3, label='Black Advantage')
    
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xlabel('Move Number')
    plt.ylabel('Material Difference (centipawns)')
    plt.title('Material Imbalance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Material imbalance plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_game_termination_types(results_history, save_path=None):
    """Plot pie chart of game termination types"""
    terminations = [r['termination_type'] for r in results_history]
    termination_counts = Counter(terminations)
    
    labels = [f"{term.title()} ({count})" for term, count in termination_counts.most_common()]
    sizes = [count for _, count in termination_counts.most_common()]
    
    plt.figure(figsize=(10, 8))
    
    # Define attractive colors for each termination type
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
    
    # Create a pie chart with percentage labels
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
            shadow=True, colors=colors[:len(sizes)])
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Game Termination Types')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Game termination types plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_first_error_moves(results_history, save_path=None):
    """Plot histogram of first significant error moves"""
    error_moves = [r['first_error_move'] for r in results_history if r['first_error_move'] is not None]
    
    if not error_moves:
        print("No error move data available")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(error_moves, bins=range(1, max(error_moves) + 2), 
                                alpha=0.7, color='orange', rwidth=0.8)
    
    # Add mean line
    mean_error_move = np.mean(error_moves)
    plt.axvline(x=mean_error_move, color='red', linestyle='--',
                label=f'Mean: Move {mean_error_move:.1f}')
    
    plt.xlabel('Move Number')
    plt.ylabel('Frequency')
    plt.title('First Significant Error Move Distribution')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(1, max(error_moves) + 1))
    
    if save_path:
        plt.savefig(save_path)
        print(f"First error move plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def estimate_elo(results_history, stockfish_elo=1500, K=32):
    """Estimate RL agent's Elo rating based on performance against Stockfish"""
    rl_elo = 1500  # Starting Elo
    
    # Process results
    for game in results_history:
        result = game['result']
        
        # Calculate expected score based on Elo difference
        expected = 1 / (1 + 10 ** ((stockfish_elo - rl_elo) / 400))
        
        # Determine actual score
        if result == '1-0':  # RL win
            actual = 1
        elif result == '0-1':  # RL loss
            actual = 0
        else:  # Draw
            actual = 0.5
        
        # Update Elo
        rl_elo += K * (actual - expected)
    
    return rl_elo

def plot_elo_estimation(results_history, save_path=None, stockfish_base_elo=1000):
    """Plot estimated Elo rating progress over games"""
    games = range(1, len(results_history) + 1)
    elo_progression = []
    
    current_elo = stockfish_base_elo
    for i in range(1, len(results_history) + 1):
        current_elo = estimate_elo(results_history[:i], stockfish_elo=stockfish_base_elo)
        elo_progression.append(current_elo)
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, elo_progression, 'o-', color='blue', markersize=8)
    
    # Add horizontal line for starting Elo
    plt.axhline(y=stockfish_base_elo, color='gray', linestyle='--', 
                label=f'Stockfish Elo: {stockfish_base_elo}')
    
    # Add final Elo annotation
    plt.annotate(f'Final Elo: {elo_progression[-1]:.1f}', 
                xy=(games[-1], elo_progression[-1]),
                xytext=(games[-1] - 1, elo_progression[-1] + 50),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12)
    
    plt.xlabel('Game Number')
    plt.ylabel('Estimated Elo Rating')
    plt.title('RL Agent Elo Rating Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Elo estimation plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # Define base path for models
    base_path = "Hot Models"
    
    evaluate_models(
        rl_model_path=os.path.join(base_path, "final_hybrid_model.pth"),
        move_to_int_path=os.path.join(base_path, "100kmap.pth"),
        stockfish_path="/Users/soul/Desktop/soulmate-ai/ExistingModels/stockfish-8-64",
        num_games=10  # Increased to get better statistics
    )
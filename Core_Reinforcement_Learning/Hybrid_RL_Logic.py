# train_pgn_then_rl.py
# This script first trains the model on PGN data and then uses that trained model
# as a starting point for reinforcement learning.

import os
import numpy as np
import time
import gc
import pickle
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from chess import pgn, Board, Move
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

# Local imports (absolute, as per project structure)
from Core.Core_Soul import SoulmateModel
from Core.Neural_Augmentation import board_to_matrix, create_input_for_nn, create_full_move_to_int
from Core.monte_carlo_tree_search import MCTS
from Core_RL_Logic import DQNAgent, ChessEnvironment, self_play_training, train_with_mcts, train_against_stockfish

# Set number of worker processes based on CPU cores
NUM_WORKERS = max(1, mp.cpu_count() - 1)
# Set batch size based on available GPU memory
BATCH_SIZE = 32 if torch.cuda.is_available() else 64

def load_pgn(file_path):
    print(f"Loading PGN file: {file_path}")
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    print(f"Loaded {len(games)} games from {file_path}")
    return games

def calculate_top_k_accuracy(outputs, targets, k=3):
    """Calculate Top-K Accuracy."""
    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
    return correct.sum().item() / targets.size(0)

def calculate_kl_divergence(predicted_policy, target_policy):
    """Calculate KL Divergence between predicted and target policies."""
    predicted_policy = F.log_softmax(predicted_policy, dim=1)
    target_policy = F.softmax(target_policy, dim=1)
    return F.kl_div(predicted_policy, target_policy, reduction='batchmean').item()

def calculate_value_prediction_correlation(predicted_values, actual_values):
    """Calculate Pearson correlation between predicted and actual values."""
    return pearsonr(predicted_values.cpu().numpy(), actual_values.cpu().numpy())[0]

def log_class_frequency(labels, num_classes):
    """Log the frequency of each class in the training labels."""
    class_counts = Counter(labels.cpu().numpy())
    frequencies = {f"Class {i}": class_counts.get(i, 0) for i in range(num_classes)}
    print("Class Frequencies:", frequencies)
    return frequencies

def train_on_pgn_data(pgn_dir, num_epochs=2, batch_size=64, save_path=None):
    """
    Train the model on PGN data.
    
    Args:
        pgn_dir: Directory containing PGN files
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save the trained model
        
    Returns:
        Trained model and move_to_int mapping
    """
    print("Starting PGN data training...")
    
    # Load PGN files
    files = [file for file in os.listdir(pgn_dir) if file.endswith(".pgn")]
    LIMIT_OF_FILES = min(len(files), 5)  # Limit to 5 files for faster training
    print(f"Found {len(files)} PGN files, processing up to {LIMIT_OF_FILES} files.")
    
    games = []
    i = 1
    for file in tqdm(files[:LIMIT_OF_FILES]):
        games.extend(load_pgn(os.path.join(pgn_dir, file)))
        if i >= LIMIT_OF_FILES:
            break
        i += 1
    
    print(f"NUMBER OF GAMES: {len(games)}")
    
    # Create training data from PGN games
    print("Creating input for neural network...")
    X, y_policy, y_value = create_input_for_nn(games)
    print(f"NUMBER OF SAMPLES: {len(y_policy)}")
    
    # Limit dataset size if needed
    max_samples = 1000000  # Adjust based on available memory
    if len(y_policy) > max_samples:
        print(f"Limiting dataset to {max_samples} samples.")
        X = X[:max_samples]
        y_policy = y_policy[:max_samples]
        y_value = y_value[:max_samples]
    
    # Create full move mapping
    print("Creating full move mapping...")
    move_to_int = create_full_move_to_int()
    num_classes = len(move_to_int)
    print(f"Number of classes: {num_classes}")
    
    # Verify that 'e7e5' is in the mapping
    assert 'e7e5' in move_to_int, "Move 'e7e5' not found in move_to_int mapping"
    
    # Convert moves to integers using the full mapping
    y_policy_encoded = []
    for move in y_policy:
        if move in move_to_int:
            y_policy_encoded.append(move_to_int[move])
        else:
            print(f"Warning: Move {move} not found in mapping, skipping")
            y_policy_encoded.append(num_classes - 1)  # Assign to the last class as a fallback
    
    y_policy_encoded = np.array(y_policy_encoded, dtype=np.int64)
    
    # Convert data to tensors
    print("Converting data to tensors...")
    X = np.array(X, dtype=np.float32)
    X = torch.from_numpy(X)
    y_policy = torch.tensor(y_policy_encoded, dtype=torch.long)
    y_value = torch.tensor(y_value, dtype=torch.float32)
    
    # Create dataset and dataloader
    from Data_Feeder import ChessDataset
    
    print("Creating dataset and dataloader...")
    dataset = ChessDataset(X, y_policy, y_value)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Initialize model
    print("Initializing model...")
    model = SoulmateModel(num_res_blocks=5, num_classes=num_classes).to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Separate parameters for policy and value heads
    policy_params = []
    value_params = []
    
    # Handle parameter collection for both DataParallel and regular models
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    policy_params.extend([
        *base_model.conv_input.parameters(),
        *base_model.res_blocks.parameters(),
        *base_model.policy_conv.parameters(),
        *base_model.policy_bn.parameters(),
        *base_model.policy_fc.parameters()
    ])
    
    value_params.extend([
        *base_model.value_conv.parameters(),
        *base_model.value_bn.parameters(),
        *base_model.value_fc1.parameters(),
        *base_model.value_fc2.parameters(),
        *base_model.value_fc3.parameters()
    ])
                   
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': policy_params, 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
        {'params': value_params, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    # Initialize variables for best model checkpoint
    best_accuracy = 0.0
    best_model_state = None

    # Training loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_policy_loss = 0.0
        running_value_loss = 0.0
        running_total_loss = 0.0
        all_preds = []
        all_labels = []
        all_value_preds = []
        all_value_targets = []
    
        # Create a tqdm progress bar for the dataloader
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for inputs, policy_targets, value_targets in progress_bar:
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                policy_outputs, value_outputs = model(inputs)
                policy_loss = criterion(policy_outputs, policy_targets)
                value_loss = nn.SmoothL1Loss()(value_outputs.squeeze(), value_targets.float())
                total_loss = policy_loss + value_loss
            
            # Backward pass and optimize with mixed precision
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Accumulate losses
            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()
            running_total_loss += total_loss.item()
            
            # Collect predictions and targets for accuracy calculation
            _, predicted = torch.max(policy_outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(policy_targets.cpu().numpy())
            all_value_preds.extend(value_outputs.squeeze().cpu().numpy())
            all_value_targets.extend(value_targets.cpu().numpy())
            
            # Update the progress bar
            progress_bar.set_postfix({
                'Policy Loss': f'{policy_loss.item():.4f}',
                'Value Loss': f'{value_loss.item():.4f}'
            })
        
        # Calculate metrics for the epoch
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        top_3_accuracy = calculate_top_k_accuracy(policy_outputs, policy_targets, k=3)
        top_5_accuracy = calculate_top_k_accuracy(policy_outputs, policy_targets, k=5)
        kl_divergence = calculate_kl_divergence(policy_outputs, policy_targets)
        value_corr = calculate_value_prediction_correlation(
            torch.tensor(all_value_preds), torch.tensor(all_value_targets)
        )
        class_frequencies = log_class_frequency(torch.tensor(all_labels), num_classes)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Policy', running_policy_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/Value', running_value_loss / len(dataloader), epoch)
        writer.add_scalar('Accuracy/Top-1', epoch_accuracy, epoch)
        writer.add_scalar('Accuracy/Top-3', top_3_accuracy, epoch)
        writer.add_scalar('Accuracy/Top-5', top_5_accuracy, epoch)
        writer.add_scalar('KL Divergence', kl_divergence, epoch)
        writer.add_scalar('Value Prediction Correlation', value_corr, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time) - minutes * 60
        
        # Compute average losses
        avg_policy_loss = running_policy_loss / len(dataloader)
        avg_value_loss = running_value_loss / len(dataloader)
        avg_total_loss = running_total_loss / len(dataloader)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Policy Loss: {avg_policy_loss:.4f}, '
              f'Value Loss: {avg_value_loss:.4f}, '
              f'Total Loss: {avg_total_loss:.4f}, '
              f'Accuracy: {epoch_accuracy:.4f}, '
              f'Top-3 Accuracy: {top_3_accuracy:.4f}, '
              f'Top-5 Accuracy: {top_5_accuracy:.4f}, '
              f'KL Divergence: {kl_divergence:.4f}, '
              f'Value Correlation: {value_corr:.4f}, '
              f'Time: {minutes}m{seconds}s')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_state = model.state_dict()
    
    # Save the best model checkpoint
    if save_path and best_model_state:
        best_model_path = save_path.replace(".pth", "_best.pth")
        print(f"Saving the best model to {best_model_path}...")
        torch.save(best_model_state, best_model_path)
    
    # Save the move_to_int mapping
    mapping_path = os.path.join(os.path.dirname(save_path), "pgn_trained_map.pth")
    print(f"Saving the move_to_int mapping to {mapping_path}...")
    with open(mapping_path, "wb") as file:
        pickle.dump(move_to_int, file)
    
    print("PGN training complete.")
    return model, move_to_int

def create_rl_agent_from_pgn_model(model, move_to_int, device=None):
    """
    Create a DQNAgent using the weights from a PGN-trained model.
    
    Args:
        model: The PGN-trained model
        move_to_int: The move mapping
        device: The device to use for the agent
        
    Returns:
        A DQNAgent initialized with the PGN-trained model weights
    """
    print("Creating RL agent from PGN-trained model...")
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the state dict from the source model
    source_model = model.module if isinstance(model, nn.DataParallel) else model
    source_state_dict = source_model.state_dict()
    
    print(f"Source model has {len(source_state_dict)} parameters")
    
    # Print the first few keys to help with debugging
    print("First 5 keys in source state dict:")
    for i, k in enumerate(list(source_state_dict.keys())[:5]):
        print(f"  {i}: {k}")
    
    # Detect the number of residual blocks from the source model
    num_res_blocks = 0
    for key in source_state_dict.keys():
        if 'res_blocks.' in key:
            # Extract the block number from keys like 'res_blocks.4.conv1.weight'
            block_num = int(key.split('res_blocks.')[1].split('.')[0])
            num_res_blocks = max(num_res_blocks, block_num + 1)
    
    print(f"Detected {num_res_blocks} residual blocks in the source model")
    
    # Create agent with the same number of residual blocks
    state_shape = (18, 8, 8)  # Chess board representation
    action_size = len(move_to_int)
    agent = DQNAgent(state_shape, action_size, device=device, num_res_blocks=num_res_blocks)
    
    # Fix state dict keys for compatibility
    fixed_state_dict = {}
    for k, v in source_state_dict.items():
        # Remove 'module.' prefix if it exists
        name = k[7:] if k.startswith('module.') else k
        fixed_state_dict[name] = v
    
    print(f"Fixed state dict has {len(fixed_state_dict)} keys")
    
    # Get the base models if using DataParallel
    target_model = agent.model.module if isinstance(agent.model, nn.DataParallel) else agent.model
    target_model_2 = agent.target_model.module if isinstance(agent.target_model, nn.DataParallel) else agent.target_model
    
    # Print target model structure to help with debugging
    print("Target model structure:")
    for name, _ in target_model.named_parameters():
        print(f"  {name}")
    
    # Load state dicts
    try:
        # First try with strict=True
        target_model.load_state_dict(fixed_state_dict)
        target_model_2.load_state_dict(fixed_state_dict)
        print("Successfully loaded state dict with strict=True")
    except Exception as e:
        print(f"Error loading state dict with strict=True: {e}")
        print("Attempting to load with strict=False...")
        
        # Try to load state dict with strict=False
        try:
            target_model.load_state_dict(fixed_state_dict, strict=False)
            target_model_2.load_state_dict(fixed_state_dict, strict=False)
            print("Successfully loaded state dict with strict=False")
        except Exception as e:
            print(f"Error loading state dict with strict=False: {e}")
            
            # Try to identify missing and unexpected keys
            target_state_dict = target_model.state_dict()
            missing_keys = set(target_state_dict.keys()) - set(fixed_state_dict.keys())
            unexpected_keys = set(fixed_state_dict.keys()) - set(target_state_dict.keys())
            
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            # Try to create a new state dict with only matching keys
            matching_state_dict = {}
            for k, v in fixed_state_dict.items():
                if k in target_state_dict and v.shape == target_state_dict[k].shape:
                    matching_state_dict[k] = v
            
            print(f"Created matching state dict with {len(matching_state_dict)} keys")
            target_model.load_state_dict(matching_state_dict, strict=False)
            target_model_2.load_state_dict(matching_state_dict, strict=False)
            print("Loaded matching state dict")
    
    # Set a lower initial epsilon for the RL agent since it's pre-trained
    agent.epsilon = max(0.1, agent.epsilon * 0.99)
    
    print("RL agent created successfully.")
    return agent

def load_model_with_dataparallel_compatibility(model_path, device=None):
    """
    Load a model with DataParallel compatibility.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model to
        
    Returns:
        The loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path} to {device}")
    
    # Load the state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        print(f"Successfully loaded state dict with {len(state_dict)} keys")
        
        # Print the first few keys to help with debugging
        print("First 5 keys in state dict:")
        for i, k in enumerate(list(state_dict.keys())[:5]):
            print(f"  {i}: {k}")
    except Exception as e:
        print(f"Error loading state dict from file: {e}")
        raise
    
    # Fix state dict keys for DataParallel compatibility
    fixed_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if it exists
        name = k[7:] if k.startswith('module.') else k
        fixed_state_dict[name] = v
    
    print(f"Fixed state dict has {len(fixed_state_dict)} keys")
    
    # Detect the number of residual blocks from the state dict
    num_res_blocks = 0
    for key in fixed_state_dict.keys():
        if 'res_blocks.' in key:
            # Extract the block number from keys like 'res_blocks.4.conv1.weight'
            block_num = int(key.split('res_blocks.')[1].split('.')[0])
            num_res_blocks = max(num_res_blocks, block_num + 1)
    
    print(f"Detected {num_res_blocks} residual blocks in the state dict")
    
    # Create model with the same architecture
    # We need to determine the number of classes from the state dict
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
    model = SoulmateModel(num_res_blocks=num_res_blocks, num_classes=num_classes).to(device)
    
    # Print model structure to help with debugging
    print("Model structure:")
    for name, _ in model.named_parameters():
        print(f"  {name}")
    
    # Load the fixed state dict
    try:
        # First try with strict=True
        model.load_state_dict(fixed_state_dict)
        print("Successfully loaded state dict with strict=True")
    except Exception as e:
        print(f"Error loading state dict with strict=True: {e}")
        print("Attempting to load with strict=False...")
        
        # Try to load state dict with strict=False
        try:
            model.load_state_dict(fixed_state_dict, strict=False)
            print("Successfully loaded state dict with strict=False")
        except Exception as e:
            print(f"Error loading state dict with strict=False: {e}")
            
            # Try to identify missing and unexpected keys
            model_state_dict = model.state_dict()
            missing_keys = set(model_state_dict.keys()) - set(fixed_state_dict.keys())
            unexpected_keys = set(fixed_state_dict.keys()) - set(model_state_dict.keys())
            
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            # Try to create a new state dict with only matching keys
            matching_state_dict = {}
            for k, v in fixed_state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    matching_state_dict[k] = v
            
            print(f"Created matching state dict with {len(matching_state_dict)} keys")
            model.load_state_dict(matching_state_dict, strict=False)
            print("Loaded matching state dict")
    
    # Wrap with DataParallel if using multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.eval()  # Set to evaluation mode
    return model

def main():
    # Print system information
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Using {NUM_WORKERS} worker processes")
    print(f"Batch Size: {BATCH_SIZE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Define base path for models
    base_path = "/Users/soul/Desktop/soulmate-ai/Hot Models"
    pgn_dir = "C:/Users/bridi/OneDrive/Desktop/AIpipeline/pgns"
    
    # Check if we already have a trained model
    pgn_model_path = os.path.join(base_path, "pgn_trained_model.pth")
    mapping_path = os.path.join(base_path, "pgn_trained_map.pth")
    
    if os.path.exists(pgn_model_path) and os.path.exists(mapping_path):
        print("Loading existing PGN-trained model...")
        # Load the move mapping
        with open(mapping_path, "rb") as file:
            move_to_int = pickle.load(file)
        
        # Load the model with DataParallel compatibility
        model = load_model_with_dataparallel_compatibility(pgn_model_path)
        print("Model loaded successfully.")
        
        # Save the model and move_to_int mapping in the format required for evaluation
        print("Saving model and move_to_int mapping in evaluation format...")
        eval_model_path = os.path.join(base_path, "100kepochs.pth")
        eval_mapping_path = os.path.join(base_path, "100kmap.pth")
        
        # Save the model state dict
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), eval_model_path)
        else:
            torch.save(model.state_dict(), eval_model_path)
        
        # Save the move_to_int mapping
        with open(eval_mapping_path, "wb") as file:
            pickle.dump(move_to_int, file)
        
        print(f"Saved evaluation model to {eval_model_path}")
        print(f"Saved evaluation mapping to {eval_mapping_path}")
    else:
        # Step 1: Train on PGN data
        print("No existing model found. Training on PGN data...")
        model, move_to_int = train_on_pgn_data(
            pgn_dir=pgn_dir,
            num_epochs=10,
            batch_size=BATCH_SIZE,
            save_path=pgn_model_path
        )
        
        # Save the model and move_to_int mapping in the format required for evaluation
        print("Saving model and move_to_int mapping in evaluation format...")
        eval_model_path = os.path.join(base_path, "100kepochs.pth")
        eval_mapping_path = os.path.join(base_path, "100kmap.pth")
        
        # Save the model state dict
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), eval_model_path)
        else:
            torch.save(model.state_dict(), eval_model_path)
        
        # Save the move_to_int mapping
        with open(eval_mapping_path, "wb") as file:
            pickle.dump(move_to_int, file)
        
        print(f"Saved evaluation model to {eval_model_path}")
        print(f"Saved evaluation mapping to {eval_mapping_path}")
    
    # Step 2: Create RL agent from PGN-trained model
    agent = create_rl_agent_from_pgn_model(model, move_to_int)
    
    # Step 3: Create environment
    env = ChessEnvironment()
    
    # Step 4: Create MCTS with the RL agent's model
    mcts = MCTS(agent.model, num_simulations=20)
    agent_move = mcts.get_move(env.board)
    
    # Step 5: Train the agent with reinforcement learning
    print("Starting self-play training...")
    self_play_training(agent, env, num_episodes=50, target_update=5, save_interval=10)
    
    print("Starting MCTS-guided training...")
    train_with_mcts(agent, env, mcts, num_episodes=50, target_update=5, save_interval=10)
    
    # Step 6: Train against Stockfish if available
    stockfish_path = "stockfish_14053109_x64.exe"
    if os.path.exists(stockfish_path):
        print("Starting training against Stockfish...")
        train_against_stockfish(
            agent, 
            stockfish_path=stockfish_path,
            num_episodes=100,
            initial_skill_level=1,
            max_skill_level=15,
            skill_increment=1,
            skill_increment_interval=10,
            time_limit=0.1
        )
    else:
        print(f"Stockfish not found at {stockfish_path}. Skipping Stockfish training.")
    
    # Save the final model
    final_model_path = os.path.join(base_path, "final_hybrid_model.pth")
    agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print("Training complete!")

if __name__ == "__main__":
    main()
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess import pgn, Board, Move
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score  # For accuracy calculation

# Local files
from Core_Soul import SoulmateModel
from Neural_Augmentation import board_to_matrix, create_input_for_nn, create_full_move_to_int

def load_pgn(file_path):
    """
    Loads PGN games from the given file path.

    Args:
        file_path (str): Path to the PGN file.

    Returns:
        list: List of chess.pgn.Game objects.
    """
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

# Load all PGN files from the dataset directory
files = [file for file in os.listdir("/Users/soul/Desktop/soulmate-ai/Portable Game Notations/Feed Ready Data") if file.endswith(".pgn")]
LIMIT_OF_FILES = min(len(files), 28)
print(f"Found {len(files)} PGN files, processing up to {LIMIT_OF_FILES} files.")
games = []
i = 1
for file in tqdm(files):
    games.extend(load_pgn(f"/Users/soul/Desktop/soulmate-ai/Portable Game Notations/Feed Ready Data{file}"))
    if i >= LIMIT_OF_FILES:
        break
    i += 1

print(f"NUMBER OF GAMES: {len(games)}")

# Create training samples from PGN data
print("Creating input for neural network...")
X, y_policy, y_value = create_input_for_nn(games)
print(f"NUMBER OF SAMPLES: {len(y_policy)}")

# Limit dataset size for memory constraints or faster training
X = X[:2500000]
y_policy = y_policy[:2500000]
y_value = y_value[:2500000]
print(f"Dataset limited to {len(y_policy)} samples.")

# Create mapping from moves to integer indices for policy head classification
print("Creating full move mapping...")
move_to_int = create_full_move_to_int()
num_classes = len(move_to_int)
print(f"Number of classes: {num_classes}")

# Check if key move is in the mapping
assert 'e7e5' in move_to_int, "Move 'e7e5' not found in move_to_int mapping"

# Convert policy labels to integers
y_policy_encoded = []
for move in y_policy:
    if move in move_to_int:
        y_policy_encoded.append(move_to_int[move])
    else:
        print(f"Warning: Move {move} not found in mapping, skipping")
        y_policy_encoded.append(num_classes - 1)  # Assign unknown moves to the last class

y_policy_encoded = np.array(y_policy_encoded, dtype=np.int64)

# Convert all data to PyTorch tensors
print("Converting data to tensors...")
X = torch.tensor(X, dtype=torch.float32)
y_policy = torch.tensor(y_policy_encoded, dtype=torch.long)
y_value = torch.tensor(y_value, dtype=torch.float32)

# Load custom dataset and dataloader
from Data_Feeder import ChessDataset

print("Creating dataset and dataloader...")
dataset = ChessDataset(X, y_policy, y_value)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Detect device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Initialize Soulmate-style model
print("Initializing model...")
model = SoulmateModel(num_res_blocks=5, num_classes=num_classes).to(device)

# Group model parameters for policy and value heads (separate learning rates)
policy_params = list(model.conv_input.parameters()) + \
               list(model.res_blocks.parameters()) + \
               list(model.policy_conv.parameters()) + \
               list(model.policy_bn.parameters()) + \
               list(model.policy_fc.parameters())

value_params = list(model.value_conv.parameters()) + \
               list(model.value_bn.parameters()) + \
               list(model.value_fc1.parameters()) + \
               list(model.value_fc2.parameters()) + \
               list(model.value_fc3.parameters())

# Define loss functions and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
    {'params': policy_params, 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
    {'params': value_params, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4}
])
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min((epoch + 1) / 10, 1.0))

# Training parameters
num_epochs = 2

# Begin training loop
print("Starting training loop...")
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_total_loss = 0.0
    all_preds = []
    all_labels = []

    # TQDM progress bar for visual feedback
    progress_bar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    
    for inputs, policy_targets, value_targets in progress_bar:
        inputs, policy_targets, value_targets = inputs.to(device), policy_targets.to(device), value_targets.to(device)
        
        # Forward pass
        policy_outputs, value_outputs = model(inputs)
        
        # Compute policy loss (classification)
        policy_loss = criterion(policy_outputs, policy_targets)
        
        # Compute value loss (regression)
        value_loss = nn.SmoothL1Loss()(value_outputs.squeeze(), value_targets.float())
        total_loss = policy_loss + value_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track loss values
        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()
        running_total_loss += total_loss.item()
        
        # Collect predictions and targets for accuracy
        _, predicted = torch.max(policy_outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(policy_targets.cpu().numpy())
        
        # Update TQDM bar
        progress_bar.set_postfix({
            'Policy Loss': f'{policy_loss.item():.4f}',
            'Value Loss': f'{value_loss.item():.4f}'
        })
    
    # Accuracy computation after epoch
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time) - minutes * 60
    
    # Log epoch metrics
    avg_policy_loss = running_policy_loss / len(dataloader)
    avg_value_loss = running_value_loss / len(dataloader)
    avg_total_loss = running_total_loss / len(dataloader)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Policy Loss: {avg_policy_loss:.4f}, '
          f'Value Loss: {avg_value_loss:.4f}, '
          f'Total Loss: {avg_total_loss:.4f}, '
          f'Accuracy: {epoch_accuracy:.4f}, '
          f'Time: {minutes}m{seconds}s')

# Save trained model weights
print("Saving the model...")
torch.save(model.state_dict(), "/Users/soul/Desktop/soulmate-ai/Hot Models/100kepochs.pth")

# Save move-to-int dictionary
print("Saving the move_to_int mapping...")
with open("/Users/soul/Desktop/soulmate-ai/Hot Mappings/100kmap.pth", "wb") as file:
    pickle.dump(move_to_int, file)

print("Training complete.")

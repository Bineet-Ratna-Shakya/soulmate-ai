# reinforcement_learning.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import random
import pickle
import time
import gc
import multiprocessing as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import psutil
from tqdm import tqdm

from chess import Board, Move
import chess.engine

# Local imports (absolute, as per project structure)
from Core.Core_Soul import SoulmateModel
from Core.Neural_Augmentation import board_to_matrix, create_full_move_to_int
from Core.monte_carlo_tree_search import MCTS

# Set number of worker processes based on CPU cores
NUM_WORKERS = max(1, mp.cpu_count() - 1)
# Set batch size based on available GPU memory
BATCH_SIZE = 32 if torch.cuda.is_available() else 64
# Set replay buffer size based on available RAM (in GB)
MEMORY_SIZE = min(10000, int(psutil.virtual_memory().available / (1024**3) * 10000))

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.lock = mp.Lock()
    
    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        with self.lock:
            return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)

class ChessEnvironment:
    def __init__(self, stockfish_path=None, stockfish_skill_level=5, stockfish_time_limit=0.1):
        self.board = Board()
        self.reset()
        self.illegal_move_penalty = -1.0  # Penalty for illegal moves
        
        # Stockfish integration
        self.stockfish_path = stockfish_path
        self.stockfish_skill_level = stockfish_skill_level
        self.stockfish_time_limit = stockfish_time_limit
        self.engine = None
        
        # Initialize Stockfish engine if path is provided
        if self.stockfish_path and os.path.exists(self.stockfish_path):
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                # Set skill level (0-20)
                self.engine.configure({"Skill Level": self.stockfish_skill_level})
                print(f"Stockfish engine initialized with skill level {self.stockfish_skill_level}")
            except Exception as e:
                print(f"Error initializing Stockfish engine: {e}")
                self.engine = None
    
    def reset(self):
        self.board = Board()
        return self._get_state()
    
    def step(self, action):
        # Convert action (move index) to chess move
        move = self._index_to_move(action)
        
        # Check if move is legal
        if move not in self.board.legal_moves:
            # Return negative reward for illegal move
            return self._get_state(), self.illegal_move_penalty, True, {"error": "Illegal move"}
        
        # Make the move
        self.board.push(move)
        
        # Check game status
        done = self.board.is_game_over()
        reward = self._calculate_reward()
        
        return self._get_state(), reward, done, {}
    
    def step_against_stockfish(self, action):
        """
        Take a step in the environment against Stockfish.
        The agent makes a move, then Stockfish responds.
        
        Args:
            action: The action index for the agent's move
            
        Returns:
            next_state, reward, done, info
        """
        # Agent's move
        move = self._index_to_move(action)
        
        # Check if move is legal
        if move not in self.board.legal_moves:
            # Return negative reward for illegal move
            return self._get_state(), self.illegal_move_penalty, True, {"error": "Illegal move"}
        
        # Make the agent's move
        self.board.push(move)
        
        # Check if game is over after agent's move
        if self.board.is_game_over():
            reward = self._calculate_reward()
            return self._get_state(), reward, True, {}
        
        # Stockfish's move
        if self.engine:
            try:
                # Get Stockfish's move
                result = self.engine.play(self.board, chess.engine.Limit(time=self.stockfish_time_limit))
                stockfish_move = result.move
                
                # Make Stockfish's move
                self.board.push(stockfish_move)
                
                # Get evaluation from Stockfish
                info = self.engine.analyse(self.board, chess.engine.Limit(time=0.01))
                evaluation = info.get("score", None)
                
                # Convert evaluation to a reward signal
                if evaluation is not None:
                    # Check if it's a mate score
                    if evaluation.is_mate():
                        # Checkmate: positive if agent won, negative if Stockfish won
                        # For mate scores, positive means mate in N moves for the side to move
                        # Negative means mate in N moves for the opponent
                        mate_score = evaluation.relative.score()
                        if mate_score is not None and mate_score > 0:  # Agent checkmated Stockfish
                            reward = 1.0
                        else:  # Stockfish checkmated agent
                            reward = -1.0
                    else:
                        # Convert centipawns to a reward between -1 and 1
                        # Positive values favor the agent, negative values favor Stockfish
                        # For regular scores, positive means advantage for the side to move
                        # Negative means advantage for the opponent
                        score = evaluation.relative.score()
                        if score is not None:
                            reward = np.clip(score / 1000.0, -1.0, 1.0)
                        else:
                            # Fallback to standard reward calculation if score is None
                            reward = self._calculate_reward()
                else:
                    # Fallback to standard reward calculation
                    reward = self._calculate_reward()
                
                # Check if game is over after Stockfish's move
                done = self.board.is_game_over()
                
                return self._get_state(), reward, done, {"stockfish_eval": evaluation}
                
            except Exception as e:
                print(f"Error during Stockfish move: {e}")
                # Fallback to random move if Stockfish fails
                random_move = random.choice(list(self.board.legal_moves))
                self.board.push(random_move)
                done = self.board.is_game_over()
                reward = self._calculate_reward()
                return self._get_state(), reward, done, {"error": "Stockfish error"}
        else:
            # If Stockfish is not available, use a random move
            random_move = random.choice(list(self.board.legal_moves))
            self.board.push(random_move)
            done = self.board.is_game_over()
            reward = self._calculate_reward()
            return self._get_state(), reward, done, {"error": "Stockfish not available"}
    
    def _get_state(self):
        return board_to_matrix(self.board)
    
    def _index_to_move(self, action_index):
        # Get the move mapping
        move_to_int = create_full_move_to_int()
        int_to_move = {v: k for k, v in move_to_int.items()}
        
        # Try to convert the action index to a move
        if action_index in int_to_move:
            try:
                move = Move.from_uci(int_to_move[action_index])
                # Verify the move is legal
                if move in self.board.legal_moves:
                    return move
            except ValueError:
                pass
        
        # If we get here, either the action index is invalid or the move is illegal
        # Return a random legal move as fallback
        return random.choice(list(self.board.legal_moves))
    
    def _calculate_reward(self):
        if self.board.is_game_over():
            if self.board.is_checkmate():
                # Checkmate: win = 1, loss = -1
                return 1.0 if self.board.turn else -1.0
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                # Draw
                return 0.0
            elif self.board.is_fifty_moves() or self.board.is_repetition():
                # Draw by rule
                return 0.0
        else:
            # Small negative reward for each move to encourage efficiency
            return -0.01
    
    def get_stockfish_evaluation(self):
        """Get the current position evaluation from Stockfish"""
        if self.engine and not self.board.is_game_over():
            try:
                info = self.engine.analyse(self.board, chess.engine.Limit(time=0.01))
                return info.get("score", None)
            except Exception as e:
                print(f"Error getting Stockfish evaluation: {e}")
                return None
        return None
    
    def close(self):
        """Close the Stockfish engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None

class ChessRLDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.float32)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

class DQNAgent:
    def __init__(self, state_shape, action_size, device=None, num_res_blocks=3):
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Create model with specified number of residual blocks
        print(f"Creating DQNAgent with {num_res_blocks} residual blocks")
        self.model = SoulmateModel(num_res_blocks=num_res_blocks, num_classes=action_size).to(self.device)
        self.target_model = SoulmateModel(num_res_blocks=num_res_blocks, num_classes=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Enable CUDA optimizations if available
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.target_model = torch.nn.DataParallel(self.target_model)
            torch.backends.cudnn.benchmark = True
        
        # RL parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = BATCH_SIZE
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=MEMORY_SIZE)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        
        # Move mapping
        self.move_to_int = create_full_move_to_int()
    
    def act(self, state, board, training=True):
        """
        Select an action based on the current state and board.
        
        Args:
            state: The current state representation
            board: The current chess board
            training: Whether the agent is in training mode
            
        Returns:
            The selected action index
        """
        if training and random.random() < self.epsilon:
            # During exploration, choose a random legal move
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return random.randrange(self.action_size)
                
            # Convert legal moves to action indices
            legal_action_indices = []
            for move in legal_moves:
                uci_move = move.uci()
                if uci_move in self.move_to_int:
                    legal_action_indices.append(self.move_to_int[uci_move])
            
            # If we have legal action indices, choose one randomly
            if legal_action_indices:
                return random.choice(legal_action_indices)
            else:
                # Fallback to random action if no legal moves found
                return random.randrange(self.action_size)
        
        # During exploitation, use the model to select the best legal move
        with torch.no_grad():
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)

            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            policy, _ = self.model(state_tensor)
            policy = policy[0].cpu().numpy()
            
            # Create a mask for legal moves
            legal_move_mask = np.zeros_like(policy)
            for move in board.legal_moves:
                uci_move = move.uci()
                if uci_move in self.move_to_int:
                    idx = self.move_to_int[uci_move]
                    legal_move_mask[idx] = 1
            
            # Apply the mask to the policy
            masked_policy = policy * legal_move_mask
            
            # If no legal moves are in the mapping, use uniform policy
            if np.sum(legal_move_mask) == 0:
                return random.randrange(self.action_size)
            
            # Ensure all probabilities are non-negative
            masked_policy = np.maximum(masked_policy, 0)
            
            # Normalize the masked policy
            policy_sum = np.sum(masked_policy)
            if policy_sum > 0:
                masked_policy = masked_policy / policy_sum
            else:
                # If all legal moves have zero probability, use uniform distribution
                masked_policy = legal_move_mask / np.sum(legal_move_mask)
            
            # Double-check that probabilities are valid
            if not np.all(np.isfinite(masked_policy)) or np.any(masked_policy < 0):
                # Fallback to uniform distribution over legal moves
                masked_policy = legal_move_mask / np.sum(legal_move_mask)
            
            # Select action based on the masked policy
            try:
                return np.random.choice(self.action_size, p=masked_policy)
            except ValueError:
                # Fallback to uniform distribution if probability calculation fails
                return random.choice([i for i, m in enumerate(legal_move_mask) if m == 1])
    
    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return
        
        # Sample from replay buffer
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)  # Likely already a simple list/array
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        # Get current Q values
        current_policy, current_value = self.model(states)
        current_q_values = current_policy.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Get next Q values from target model
        with torch.no_grad():
            next_policy, next_value = self.target_model(next_states)
            next_q_values = next_policy.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        policy_loss = nn.MSELoss()(current_q_values, target_q_values)
        value_loss = nn.MSELoss()(current_value.squeeze(), rewards)
        total_loss = policy_loss + value_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_loss.item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, path):
        # Handle DataParallel models
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        target_model_to_save = self.target_model.module if isinstance(self.target_model, torch.nn.DataParallel) else self.target_model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'target_model_state_dict': target_model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle DataParallel models
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.module.load_state_dict(checkpoint['target_model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def __del__(self):
        self.thread_pool.shutdown()

def parallel_self_play(agent, env, num_simulations=1):
    """Run multiple self-play simulations in parallel"""
    results = []
    
    def run_simulation():
        env_copy = ChessEnvironment()
        state = env_copy.reset()
        done = False
        total_reward = 0
        trajectory = []
        
        while not done:
            action = agent.act(state, env_copy.board)
            next_state, reward, done, _ = env_copy.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
        
        return trajectory, total_reward
    
    # Run simulations in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_simulation) for _ in range(num_simulations)]
        for future in futures:
            trajectory, total_reward = future.result()
            results.append((trajectory, total_reward))
    
    return results

def parallel_stockfish_play(agent, env, num_simulations=1):
    """Run multiple games against Stockfish in parallel"""
    results = []
    
    def run_simulation():
        # Create a copy of the environment with Stockfish
        env_copy = ChessEnvironment(
            stockfish_path=env.stockfish_path,
            stockfish_skill_level=env.stockfish_skill_level,
            stockfish_time_limit=env.stockfish_time_limit
        )
        
        state = env_copy.reset()
        done = False
        total_reward = 0
        trajectory = []
        
        while not done:
            # Agent's move
            action = agent.act(state, env_copy.board)
            next_state, reward, done, _ = env_copy.step_against_stockfish(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
        
        # Close the Stockfish engine
        env_copy.close()
        return trajectory, total_reward
    
    # Run simulations in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_simulation) for _ in range(num_simulations)]
        for future in futures:
            trajectory, total_reward = future.result()
            results.append((trajectory, total_reward))
    
    return results

def self_play_training(agent, env, num_episodes=10, target_update=10, save_interval=100):
    """Train the agent through self-play with parallel processing"""
    move_to_int = create_full_move_to_int()
    action_size = len(move_to_int)
    
    # Create a pool of environments for parallel processing
    env_pool = [ChessEnvironment() for _ in range(NUM_WORKERS)]
    
    # Define base path for models
    base_path = "/Users/soul/Desktop/soulmate-ai/Hot Models"
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(base_path, "tensorboard_logs/self_play"))
    
    # Track metrics
    metrics = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "game_lengths": [],
        "top5_policy_accuracy": [],
        "value_losses": []
    }
    
    for episode in tqdm(range(num_episodes)):
        # Run parallel self-play simulations
        simulations_per_episode = max(1, NUM_WORKERS // 2)
        results = parallel_self_play(agent, env, num_simulations=simulations_per_episode)
        
        # Process results and update replay buffer
        total_reward = 0
        episode_game_lengths = []
        episode_top5_accuracy = []
        episode_value_losses = []
        
        for trajectory, reward in results:
            total_reward += reward
            
            # Track game outcome
            if reward > 0:
                metrics["wins"] += 1
            elif reward < 0:
                metrics["losses"] += 1
            else:
                metrics["draws"] += 1
                
            # Track game length
            game_length = len(trajectory)
            metrics["game_lengths"].append(game_length)
            episode_game_lengths.append(game_length)
            
            # Policy & value head metrics calculated during training
            for state, action, reward, next_state, done in trajectory:
                # Add experience to replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                
                # Calculate top-5 policy accuracy if game is done
                if done:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(np.array([state])).float().to(agent.device)
                        policy_out, value_out = agent.model(state_tensor)
                        
                        # Get top-5 moves
                        _, top5_moves = policy_out.topk(5, dim=1)
                        
                        # Check if actual action is in top-5
                        is_in_top5 = (action in top5_moves[0].cpu().numpy())
                        metrics["top5_policy_accuracy"].append(float(is_in_top5))
                        episode_top5_accuracy.append(float(is_in_top5))
                        
                        # Calculate value loss
                        value_loss = F.mse_loss(value_out.squeeze(), torch.tensor([reward], device=agent.device).float()).item()
                        metrics["value_losses"].append(value_loss)
                        episode_value_losses.append(value_loss)
        
        # Train the agent if enough samples are available
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train(agent.batch_size)
        else:
            loss = 0
            
        # Update target model periodically
        if episode % target_update == 0:
            agent.update_target_model()
            
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(os.path.join(base_path, f"rl_model_episode_{episode}.pth"))
            
        # Log metrics every 10 episodes or at the specified interval
        if episode % 10 == 0 or episode == num_episodes - 1:
            # Calculate win/loss/draw rates
            total_games = metrics["wins"] + metrics["losses"] + metrics["draws"]
            if total_games > 0:
                win_rate = metrics["wins"] / total_games
                loss_rate = metrics["losses"] / total_games
                draw_rate = metrics["draws"] / total_games
            else:
                win_rate = loss_rate = draw_rate = 0
                
            # Calculate average game length
            avg_game_length = sum(metrics["game_lengths"]) / len(metrics["game_lengths"]) if metrics["game_lengths"] else 0
            
            # Calculate top-5 policy accuracy
            top5_accuracy = sum(metrics["top5_policy_accuracy"]) / len(metrics["top5_policy_accuracy"]) if metrics["top5_policy_accuracy"] else 0
            
            # Calculate average value loss
            avg_value_loss = sum(metrics["value_losses"]) / len(metrics["value_losses"]) if metrics["value_losses"] else 0
            
            # Log to TensorBoard
            writer.add_scalar('Self-Play/WinRate', win_rate, episode)
            writer.add_scalar('Self-Play/LossRate', loss_rate, episode)
            writer.add_scalar('Self-Play/DrawRate', draw_rate, episode)
            writer.add_scalar('Self-Play/AverageGameLength', avg_game_length, episode)
            writer.add_scalar('Self-Play/Top5PolicyAccuracy', top5_accuracy, episode)
            writer.add_scalar('Self-Play/ValueLoss', avg_value_loss, episode)
            writer.add_scalar('Self-Play/Epsilon', agent.epsilon, episode)
            writer.add_scalar('Self-Play/TrainingLoss', loss, episode)
            
            # Print concise summary
            print(f"\nEpisode {episode} metrics:")
            print(f"  Win/Loss/Draw: {win_rate:.2f}/{loss_rate:.2f}/{draw_rate:.2f}")
            print(f"  Avg Game Length: {avg_game_length:.1f} moves")
            print(f"  Top-5 Policy Accuracy: {top5_accuracy:.2f}")
            print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Close TensorBoard writer
    writer.close()
    
    return agent

def train_against_stockfish(agent, stockfish_path, num_episodes=100, target_update=5, save_interval=20, 
                           initial_skill_level=5, max_skill_level=20, skill_increment=1, 
                           skill_increment_interval=10, time_limit=0.1):
    """
    Train the agent against Stockfish with adaptive difficulty.
    
    Args:
        agent: The DQN agent to train
        stockfish_path: Path to the Stockfish executable
        num_episodes: Number of training episodes
        target_update: How often to update the target network
        save_interval: How often to save the model
        initial_skill_level: Initial Stockfish skill level (0-20)
        max_skill_level: Maximum Stockfish skill level
        skill_increment: How much to increase skill level by
        skill_increment_interval: How often to increase skill level
        time_limit: Time limit for Stockfish moves in seconds
    """
    # Create environment with Stockfish
    env = ChessEnvironment(
        stockfish_path=stockfish_path,
        stockfish_skill_level=initial_skill_level,
        stockfish_time_limit=time_limit
    )
    
    # Define base path for models
    base_path = "/Users/soul/Desktop/soulmate-ai/Hot Models"
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(base_path, "tensorboard_logs/stockfish_training"))
    
    # Track performance metrics
    metrics = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "game_lengths": [],
        "top5_policy_accuracy": [],
        "value_losses": [],
        "highest_level_beaten": 0,
        "wins_per_level": {level: 0 for level in range(initial_skill_level, max_skill_level + 1)}
    }
    
    win_rate_history = []
    current_skill_level = initial_skill_level
    
    for episode in tqdm(range(num_episodes)):
        # Run parallel games against Stockfish
        simulations_per_episode = max(1, NUM_WORKERS // 2)
        results = parallel_stockfish_play(agent, env, num_simulations=simulations_per_episode)
        
        # Process results and update replay buffer
        total_reward = 0
        wins = 0
        episode_game_lengths = []
        episode_top5_accuracy = []
        episode_value_losses = []
        
        for trajectory, reward in results:
            total_reward += reward
            
            # Track game outcome
            if trajectory[-1][2] > 0:  # Last reward in trajectory is positive (win)
                wins += 1
                metrics["wins"] += 1
                metrics["wins_per_level"][current_skill_level] = metrics["wins_per_level"].get(current_skill_level, 0) + 1
                
                # Update highest level beaten
                if current_skill_level > metrics["highest_level_beaten"]:
                    metrics["highest_level_beaten"] = current_skill_level
            elif trajectory[-1][2] < 0:  # Loss
                metrics["losses"] += 1
            else:  # Draw
                metrics["draws"] += 1
                
            # Track game length
            game_length = len(trajectory)
            metrics["game_lengths"].append(game_length)
            episode_game_lengths.append(game_length)
            
            # Policy & value head metrics
            for state, action, reward, next_state, done in trajectory:
                # Add experience to replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                
                # Calculate top-5 policy accuracy if game is done
                if done:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(np.array([state])).float().to(agent.device)
                        policy_out, value_out = agent.model(state_tensor)
                        
                        # Get top-5 moves
                        _, top5_moves = policy_out.topk(5, dim=1)
                        
                        # Check if actual action is in top-5
                        is_in_top5 = (action in top5_moves[0].cpu().numpy())
                        metrics["top5_policy_accuracy"].append(float(is_in_top5))
                        episode_top5_accuracy.append(float(is_in_top5))
                        
                        # Calculate value loss
                        value_loss = F.mse_loss(value_out.squeeze(), torch.tensor([reward], device=agent.device).float()).item()
                        metrics["value_losses"].append(value_loss)
                        episode_value_losses.append(value_loss)
        
        # Calculate win rate
        win_rate = wins / simulations_per_episode
        win_rate_history.append(win_rate)
        
        # Train the agent multiple times per episode
        for _ in range(simulations_per_episode):
            loss = agent.train()
        
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_model()
        
        # Increase Stockfish skill level if agent is performing well
        if episode % skill_increment_interval == 0 and episode > 0:
            # Calculate average win rate over the last skill_increment_interval episodes
            avg_win_rate = sum(win_rate_history[-skill_increment_interval:]) / skill_increment_interval
            
            # If agent is winning more than 60% of games, increase difficulty
            if avg_win_rate > 0.6 and current_skill_level < max_skill_level:
                current_skill_level = min(current_skill_level + skill_increment, max_skill_level)
                env.stockfish_skill_level = current_skill_level
                if env.engine:
                    env.engine.configure({"Skill Level": current_skill_level})
                print(f"Increasing Stockfish skill level to {current_skill_level}")
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(os.path.join(base_path, f"rl_stockfish_model_episode_{episode}.pth"))
            
        # Log metrics every 10 episodes or at the specified interval
        if episode % 10 == 0 or episode == num_episodes - 1:
            # Calculate win/loss/draw rates
            total_games = metrics["wins"] + metrics["losses"] + metrics["draws"]
            if total_games > 0:
                win_rate = metrics["wins"] / total_games
                loss_rate = metrics["losses"] / total_games
                draw_rate = metrics["draws"] / total_games
            else:
                win_rate = loss_rate = draw_rate = 0
                
            # Calculate average game length
            avg_game_length = sum(metrics["game_lengths"]) / len(metrics["game_lengths"]) if metrics["game_lengths"] else 0
            
            # Calculate top-5 policy accuracy
            top5_accuracy = sum(metrics["top5_policy_accuracy"]) / len(metrics["top5_policy_accuracy"]) if metrics["top5_policy_accuracy"] else 0
            
            # Calculate average value loss
            avg_value_loss = sum(metrics["value_losses"]) / len(metrics["value_losses"]) if metrics["value_losses"] else 0
            
            # Calculate estimated ELO (very simple approximation)
            # Assume Stockfish skill levels correspond to ELO ratings (e.g., level 0 ~= 1000, level 20 ~= 3000)
            stockfish_base_elo = 1000
            stockfish_elo_per_level = 100  # Each level adds ~100 ELO
            stockfish_elo = stockfish_base_elo + current_skill_level * stockfish_elo_per_level
            
            # Adjust player's ELO based on win rate against current Stockfish
            k_factor = 32  # K-factor determines how quickly ratings change
            expected_score = 1 / (1 + 10**((stockfish_elo - 1600)/400))  # Initial player ELO of 1600
            elo_change = k_factor * (win_rate - expected_score)
            est_elo = 1600 + elo_change
            
            # Log to TensorBoard
            writer.add_scalar('Stockfish/WinRate', win_rate, episode)
            writer.add_scalar('Stockfish/LossRate', loss_rate, episode)
            writer.add_scalar('Stockfish/DrawRate', draw_rate, episode)
            writer.add_scalar('Stockfish/AverageGameLength', avg_game_length, episode)
            writer.add_scalar('Stockfish/Top5PolicyAccuracy', top5_accuracy, episode)
            writer.add_scalar('Stockfish/ValueLoss', avg_value_loss, episode)
            writer.add_scalar('Stockfish/CurrentSkillLevel', current_skill_level, episode)
            writer.add_scalar('Stockfish/HighestLevelBeaten', metrics["highest_level_beaten"], episode)
            writer.add_scalar('Stockfish/EstimatedELO', est_elo, episode)
            writer.add_scalar('Stockfish/Epsilon', agent.epsilon, episode)
            
            # Print concise summary
            print(f"\nEpisode {episode} metrics:")
            print(f"  Win/Loss/Draw: {win_rate:.2f}/{loss_rate:.2f}/{draw_rate:.2f}")
            print(f"  Avg Game Length: {avg_game_length:.1f} moves")
            print(f"  Top-5 Policy Accuracy: {top5_accuracy:.2f}")
            print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Current Stockfish Level: {current_skill_level}")
            print(f"  Highest Level Beaten: {metrics['highest_level_beaten']}")
            print(f"  Estimated ELO: {est_elo:.1f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Close the Stockfish engine
    env.close()
    
    # Close TensorBoard writer
    writer.close()
    
    return win_rate_history

def parallel_mcts_guided_training(agent, env, mcts, num_simulations=1):
    """Run multiple MCTS-guided training simulations in parallel"""
    results = []
    
    def run_simulation():
        env_copy = ChessEnvironment()
        state = env_copy.reset()
        done = False
        total_reward = 0
        trajectory = []
        
        while not done:
            # Use MCTS to get the best move
            mcts_move = mcts.get_move(env_copy.board)
            
            # Convert MCTS move to action index
            move_to_int = create_full_move_to_int()
            action = move_to_int.get(mcts_move.uci(), random.randrange(len(move_to_int)))
            
            # Take action in environment
            next_state, reward, done, _ = env_copy.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
        
        return trajectory, total_reward
    
    # Run simulations in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_simulation) for _ in range(num_simulations)]
        for future in futures:
            trajectory, total_reward = future.result()
            results.append((trajectory, total_reward))
    
    return results

def train_with_mcts(agent, env, mcts, num_episodes=100, target_update=5, save_interval=20):
    """Train the agent using MCTS for exploration with parallel processing"""
    # Create a pool of environments for parallel processing
    env_pool = [ChessEnvironment() for _ in range(NUM_WORKERS)]
    
    # Define base path for models
    base_path = "/Users/soul/Desktop/soulmate-ai/Hot Models"
    
    for episode in tqdm(range(num_episodes)):
        # Run parallel MCTS-guided training simulations
        simulations_per_episode = max(1, NUM_WORKERS // 2)
        results = parallel_mcts_guided_training(agent, env, mcts, num_simulations=simulations_per_episode)
        
        # Process results and update replay buffer
        total_reward = 0
        for trajectory, reward in results:
            total_reward += reward
            for state, action, reward, next_state, done in trajectory:
                agent.memory.push(state, action, reward, next_state, done)
        
        # Train the agent multiple times per episode
        for _ in range(simulations_per_episode):
            loss = agent.train()
        
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_model()
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(os.path.join(base_path, f"rl_mcts_model_episode_{episode}.pth"))
            
            # Clear CUDA cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"Episode {episode}, Total Reward: {total_reward/simulations_per_episode:.2f}, Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    # Print system information
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Using {NUM_WORKERS} worker processes")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Memory Buffer Size: {MEMORY_SIZE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Define base path for models
    base_path = "/Users/soul/Desktop/soulmate-ai/Hot Models"
    
    # Create environment
    env = ChessEnvironment()
    
    # Create agent
    state_shape = (18, 8, 8)  # Chess board representation
    move_to_int = create_full_move_to_int()
    action_size = len(move_to_int)
    agent = DQNAgent(state_shape, action_size)
    
    # Save the move mapping
    print(f"Saving move mapping to {os.path.join(base_path, '100kmap.pth')}")
    with open(os.path.join(base_path, "100kmap.pth"), "wb") as file:
        pickle.dump(move_to_int, file)
    
    # Create MCTS
    model = SoulmateModel(num_res_blocks=5, num_classes=action_size)
    if torch.cuda.is_available():
        model = model.cuda()
    mcts = MCTS(model, num_simulations=1)
    mcts.move_to_int = move_to_int  # Set the move mapping for MCTS
    
    # Save the original model
    print(f"Saving original model to {os.path.join(base_path, '100kepochs.pth')}")
    torch.save(model.state_dict(), os.path.join(base_path, "100kepochs.pth"))
    
    # Train the agent
    print("Starting self-play training...")
    self_play_training(agent, env, num_episodes=1)
    
    print("Starting MCTS-guided training...")
    train_with_mcts(agent, env, mcts, num_episodes=1)
    
    # Train against Stockfish if path is provided
    stockfish_path = "stockfish-windows-x86-64-avx2.exe"  # Update with your Stockfish path
    if os.path.exists(stockfish_path):
        print("Starting training against Stockfish...")
        train_against_stockfish(
            agent, 
            stockfish_path=stockfish_path,
            num_episodes=200,
            initial_skill_level=5,
            max_skill_level=15,
            skill_increment=1,
            skill_increment_interval=20,
            time_limit=0.1
        )
    else:
        print(f"Stockfish not found at {stockfish_path}. Skipping Stockfish training.")
    
    # Save the final model
    agent.save(os.path.join(base_path, "final_rl_model.pth"))
    print("Training complete!")
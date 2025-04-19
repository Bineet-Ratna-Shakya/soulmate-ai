# mcts.py
import numpy as np
import math
import torch
from chess import Board, Move
import pickle
import random

class Node:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.initialized = False

class MCTS:
    def __init__(self, model, num_simulations=800, exploration_weight=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.tree = {}
        self.move_to_int = {}
        # Get device from model
        self.device = next(model.parameters()).device
        print(f"MCTS using device: {self.device}")
    
    def get_move(self, board, temp=1):
        node = self._get_node(board)
        if not node.children:
            self._expand_node(node, board)
        
        for _ in range(self.num_simulations):
            self.simulate(board.copy())
        
        # Get visit counts for legal moves only
        visit_counts = np.array([child.visit_count for child in node.children.values()])
        move_list = list(node.children.keys())
        
        # Verify all moves in move_list are legal
        legal_moves = list(board.legal_moves)
        legal_move_list = [move for move in move_list if move in legal_moves]
        
        if not legal_move_list:
            # If no legal moves in children, expand the node again
            self._expand_node(node, board)
            visit_counts = np.array([child.visit_count for child in node.children.values()])
            move_list = list(node.children.keys())
            legal_move_list = [move for move in move_list if move in legal_moves]
            
            if not legal_move_list:
                # If still no legal moves, return a random legal move
                return random.choice(list(board.legal_moves))
        
        # Filter visit counts to only include legal moves
        legal_indices = [move_list.index(move) for move in legal_move_list]
        legal_visit_counts = visit_counts[legal_indices]
        
        if temp == 0:
            # Choose the move with the highest visit count
            best_idx = legal_indices[np.argmax(legal_visit_counts)]
            best_move = move_list[best_idx]
        else:
            # Apply temperature to visit counts and ensure non-negative
            legal_visit_counts = np.maximum(legal_visit_counts, 0)  # Ensure non-negative
            legal_visit_counts = legal_visit_counts ** (1/temp)
            
            # Normalize probabilities
            visit_sum = np.sum(legal_visit_counts)
            if visit_sum > 0:
                probs = legal_visit_counts / visit_sum
            else:
                # If all counts are zero, use uniform distribution
                probs = np.ones_like(legal_visit_counts) / len(legal_visit_counts)
            
            # Double-check that probabilities are valid
            if not np.all(np.isfinite(probs)) or np.any(probs < 0):
                # Fallback to uniform distribution
                probs = np.ones_like(legal_visit_counts) / len(legal_visit_counts)
            
            try:
                best_idx = legal_indices[np.random.choice(len(legal_indices), p=probs)]
                best_move = move_list[best_idx]
            except ValueError:
                # Fallback to uniform selection if probability calculation fails
                best_move = random.choice(legal_move_list)
        
        return best_move
    
    def _get_node(self, board):
        board_key = board.fen()
        if board_key not in self.tree:
            self.tree[board_key] = Node()
        return self.tree[board_key]

    def simulate(self, board):
        node = self._get_node(board)
        if board.is_game_over():
            return -1.0
            
        if not node.initialized:
            self._expand_node(node, board)
            return -self._evaluate_leaf(node, board)
            
        max_ucb = -float('inf')
        best_move = None
        
        for move in board.legal_moves:
            if move not in node.children:
                self._expand_node(node, board)
                return -self._evaluate_leaf(node.children[move], board.push(move))
                
            child = node.children[move]
            if child.visit_count == 0:
                ucb = float('inf')
            else:
                ucb = child.total_value / child.visit_count + \
                    self.exploration_weight * math.sqrt(math.log(child.parent.visit_count + 1) / child.visit_count)
            if ucb > max_ucb:
                max_ucb = ucb
                best_move = move
                
        board.push(best_move)
        result = self.simulate(board)
        board.pop()
        
        node.children[best_move].visit_count += 1
        node.children[best_move].total_value += result
        return -result

    def get_move_probs(self, board, temp=1):
        node = self._get_node(board)
        visit_counts = np.array([child.visit_count for child in node.children.values()])
        visit_counts = visit_counts ** (1/temp)
        probs = visit_counts / visit_counts.sum()
        return list(node.children.keys()), probs

    def _expand_node(self, node, board):
        input_tensor = self._board_to_tensor(board)
        # Move input tensor to the same device as the model
        input_tensor = input_tensor.to(self.device)
        policy, value = self.model(input_tensor.unsqueeze(0))
        
        legal_moves = list(board.legal_moves)
        policy = policy[0].detach().cpu().numpy()
        
        # Create a mask for legal moves
        legal_move_mask = np.zeros_like(policy)
        legal_move_indices = []
        unknown_moves = []
        for move in legal_moves:
            uci_move = move.uci()
            if uci_move in self.move_to_int:
                idx = self.move_to_int[uci_move]
                legal_move_mask[idx] = 1
                legal_move_indices.append(idx)
            else:
                unknown_moves.append(move)
        
        if np.sum(legal_move_mask) == 0:
            # If no legal moves are in the mapping, use uniform policy
            policy = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            # Apply the mask to the policy
            policy = policy * legal_move_mask
            policy = policy / policy.sum() if policy.sum() != 0 else policy
        
        # Add Dirichlet noise to priors
        if len(legal_moves) > 0 and np.sum(legal_move_mask) > 0:
            noise = np.random.dirichlet([0.03] * len(legal_moves))
            noise_policy = np.zeros_like(policy)
            for i, move in enumerate(legal_moves):
                uci_move = move.uci()
                if uci_move in self.move_to_int:
                    idx = self.move_to_int[uci_move]
                    noise_policy[idx] = noise[i]
            policy = (1 - 0.25) * policy + 0.25 * noise_policy
        
        for move in legal_moves:
            uci_move = move.uci()
            if uci_move in self.move_to_int:
                idx = self.move_to_int[uci_move]
                p = policy[idx]
                node.children[move] = Node(node, p)
            else:
                # Handle unknown moves by assigning a small probability
                node.children[move] = Node(node, 0.01)
        node.initialized = True
    
    def _ucb(self, child):
        if child.visit_count == 0:
            return float('inf')
        if child.parent.visit_count == 0:
            return float('inf')
        return child.total_value / child.visit_count + \
            self.exploration_weight * math.sqrt(math.log(child.parent.visit_count) / child.visit_count)
            
    def _board_to_tensor(self, board):
        tensor = np.zeros((18, 8, 8), dtype=np.float32)
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                color_offset = 0 if piece.color == board.turn else 7
                piece_type = piece.piece_type - 1
                plane = color_offset + piece_type
                row, col = divmod(square, 8)
                tensor[plane][row][col] = 1
        
        tensor[14] = 1.0 if board.has_queenside_castling_rights(board.turn) else 0.0
        tensor[15] = 1.0 if board.has_kingside_castling_rights(board.turn) else 0.0
        tensor[16] = 1.0 if board.ep_square else 0.0
        tensor[17] = board.fullmove_number / 100.0
        
        return torch.tensor(tensor)
    
    def _evaluate_leaf(self, node, board):
        # This method evaluates a leaf node and returns its value
        input_tensor = self._board_to_tensor(board)
        # Move input tensor to the same device as the model
        input_tensor = input_tensor.to(self.device)
        policy, value = self.model(input_tensor.unsqueeze(0))
        return value.item()
    
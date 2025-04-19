# auxiliary_func.py
import chess
import numpy as np
from chess import Board, Move

def create_full_move_to_int():
    # Generate all possible UCI moves
    all_moves = set()
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            if from_square != to_square:
                # Add normal moves
                move = chess.Move(from_square, to_square)
                all_moves.add(move.uci())
                
                # Add promoted moves
                for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    move = chess.Move(from_square, to_square, promotion=promotion)
                    all_moves.add(move.uci())
    
    # Sort and create mapping
    sorted_moves = sorted(all_moves)
    move_to_int = {move: idx for idx, move in enumerate(sorted_moves)}
    return move_to_int

def board_to_matrix(board: Board):
    matrix = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Piece planes (14 channels)
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == board.turn else 7
            piece_type = piece.piece_type - 1
            plane = color_offset + piece_type
            row, col = divmod(square, 8)
            matrix[plane][row][col] = 1
    
    # Additional features (4 channels)
    matrix[14] = 1.0 if board.has_queenside_castling_rights(board.turn) else 0.0
    matrix[15] = 1.0 if board.has_kingside_castling_rights(board.turn) else 0.0
    matrix[16] = 1.0 if board.ep_square else 0.0
    matrix[17] = board.fullmove_number / 100.0
    
    return matrix

def create_input_for_nn(games):
    X = []
    y_policy = []
    y_value = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y_policy.append(move.uci())
            # Add value target based on game outcome
            if board.turn == chess.WHITE:
                result = game.headers['Result']
                if result == '1-0':
                    value = 1.0
                elif result == '0-1':
                    value = -1.0
                else:
                    value = 0.0
            else:
                result = game.headers['Result']
                if result == '1-0':
                    value = -1.0
                elif result == '0-1':
                    value = 1.0
                else:
                    value = 0.0
            y_value.append(value)
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y_policy), np.array(y_value)

def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(sorted(set(moves)))}
    return np.array([move_to_int[move] for move in moves], dtype=np.int64), move_to_int

def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    for x, move in zip(X, y):
        # Apply 8 possible symmetries (4 rotations + 4 reflections)
        for k in range(4):
            rotated_x = np.rot90(x, k=k, axes=(1, 2))
            augmented_X.append(rotated_x)
            augmented_y.append(move)
            
            flipped_x = np.fliplr(rotated_x)
            augmented_X.append(flipped_x)
            augmented_y.append(move)
    return np.array(augmented_X), np.array(augmented_y)
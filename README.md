# Soulmate-AI

Soulmate-AI is a modular chess AI project that leverages deep reinforcement learning, neural augmentation, and Monte Carlo Tree Search (MCTS) to play and evaluate chess. The project is organized for extensibility and clarity, with each module handling a distinct aspect of the chess AI pipeline.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Core Modules](#core-modules)
  - [Core/](#core)
    - [Core_Soul.py](#coresoulpy)
    - [Neural_Augmentation.py](#neural_augmentationpy)
    - [monte_carlo_tree_search.py](#monte_carlo_tree_searchpy)
    - [Data_Feeder.py](#data_feederpy)
    - [Processor.py](#processorpy)
  - [Core_Reinforcement_Learning/](#core_reinforcement_learning)
    - [Core_RL_Logic.py](#core_rl_logicpy)
    - [Hybrid_RL_Logic.py](#hybrid_rl_logicpy)
  - [Evaluations/](#evaluations)
    - [Evaluate_Core.py](#evaluate_corepy)
    - [Evaluate_RL_Core.py](#evaluate_rl_corepy)
- [Main Entry Point](#main-entry-point)
  - [run_chess.py](#run_chesspy)
- [Model and Data Management](#model-and-data-management)
- [How the System Works: Flow Overview](#how-the-system-works-flow-overview)
- [Code Snippets and Examples](#code-snippets-and-examples)
- [How to Run](#how-to-run)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Project Structure

```
README.md
run_chess.py
run_chess.spec
Core/
    __init__.py
    Core_Soul.py
    Data_Feeder.py
    monte_carlo_tree_search.py
    Neural_Augmentation.py
    Processor.py
Core_Reinforcement_Learning/
    __init__.py
    Core_RL_Logic.py
    Hybrid_RL_Logic.py
Evaluations/
    Evaluate_Core.py
    Evaluate_RL_Core.py
ExistingModels/
Hot Mappings/
    100kmap.pth
Hot Models/
    100kepochs.pth
    100kmap.pth
    final_hybrid_model.pth
    Episodic Models/
        ...
Portable Game Notations/
    Feed Ready Data/
```

---

## Core Modules

### Core/

#### `Core_Soul.py`
- **Purpose:** Defines the main neural network model (`SoulmateModel`) for chess position evaluation and move selection.
- **Key Features:**
  - Implements a PyTorch neural network.
  - Used by both RL and MCTS modules.
- **Example:**
  ```python
  class SoulmateModel(nn.Module):
      def __init__(self, ...):
          super().__init__()
          # ... layers ...
      def forward(self, x):
          # ... forward pass ...
  ```

#### `Neural_Augmentation.py`
- **Purpose:** Provides utilities to convert chess board states to neural network-friendly formats.
- **Key Functions:**
  - `board_to_matrix(board)`: Converts a `chess.Board` to a matrix.
  - `create_full_move_to_int()`: Maps moves to integer indices for output layers.
- **Example:**
  ```python
  def board_to_matrix(board):
      # Converts board to a matrix representation
      ...
  ```

#### `monte_carlo_tree_search.py`
- **Purpose:** Implements the MCTS algorithm for move selection.
- **Key Class:**
  - `MCTS`: Handles tree search, node expansion, and backpropagation.
- **Example:**
  ```python
  class MCTS:
      def __init__(self, model, ...):
          # ...
      def search(self, board):
          # ...
  ```

#### `Data_Feeder.py`
- **Purpose:** Handles data loading and feeding for training and evaluation.
- **Key Features:**
  - Loads PGN files and prepares training batches.
  - May include custom PyTorch `Dataset` classes.

#### `Processor.py`
- **Purpose:** Additional data processing utilities, such as feature extraction or data cleaning.

---

### Core_Reinforcement_Learning/

#### `Core_RL_Logic.py`
- **Purpose:** Implements deep reinforcement learning logic for chess.
- **Key Classes:**
  - `DQNAgent`: Deep Q-Network agent for learning chess.
  - `ChessEnvironment`: Environment wrapper for chess games.
- **Features:**
  - Uses multiprocessing for self-play.
  - Integrates with TensorBoard for logging.
  - Handles experience replay, training loops, and model saving.
- **Example:**
  ```python
  class DQNAgent:
      def __init__(self, ...):
          # ...
      def select_action(self, state):
          # ...
      def train(self, ...):
          # ...
  ```

#### `Hybrid_RL_Logic.py`
- **Purpose:** (If present) Implements hybrid RL logic, possibly combining MCTS and DQN.

---

### Evaluations/

#### `Evaluate_Core.py`
- **Purpose:** Evaluates the core model (e.g., `SoulmateModel`) on test data or against other engines.
- **Features:**
  - Loads models and mappings.
  - Runs evaluation games or test suites.
- **Example:**
  ```python
  from Core.Core_Soul import SoulmateModel
  # Load model, run evaluation loop
  ```

#### `Evaluate_RL_Core.py`
- **Purpose:** Evaluates RL-trained models, possibly using different metrics or against different opponents.

---

## Main Entry Point

### `run_chess.py`
- **Purpose:** Main script to run the chess engine, load models, and play games.
- **Key Features:**
  - Loads models using robust PyTorch loading (with DataParallel compatibility).
  - Loads move mappings.
  - Sets up the chess environment and runs the main loop.
- **Example:**
  ```python
  from Core.monte_carlo_tree_search import MCTS
  from Core.Core_Soul import SoulmateModel

  def main():
      # Load model, set up board, run game loop
      ...
  if __name__ == "__main__":
      main()
  ```

---

## Model and Data Management

- **Hot Models/**: Stores trained model checkpoints.
- **Hot Mappings/**: Stores move-to-index mappings.
- **ExistingModels/**: May contain pre-trained or baseline models.
- **Portable Game Notations/**: Stores PGN files for training and evaluation.

---

## How the System Works: Flow Overview

1. **Data Preparation**
   - PGN files are processed by `Data_Feeder.py` and `Processor.py`.
   - Board states are converted to matrices using `Neural_Augmentation.py`.

2. **Model Training**
   - RL logic in `Core_RL_Logic.py` uses self-play and experience replay.
   - The neural network (`SoulmateModel`) is trained to predict move values or policies.

3. **Move Selection**
   - During play, MCTS (`monte_carlo_tree_search.py`) or DQN agent selects moves.
   - The model evaluates board states to guide search or direct move selection.

4. **Evaluation**
   - Scripts in `Evaluations/` run test games or benchmarks.
   - Results are logged and compared.

5. **Main Script**
   - `run_chess.py` ties everything together, loading models and running games.

---

## Code Snippets and Examples

### Loading a Model

```python
import torch
from Core.Core_Soul import SoulmateModel

model = SoulmateModel(...)
model.load_state_dict(torch.load('Hot Models/100kepochs.pth'))
model.eval()
```

### Board to Matrix Conversion

```python
from chess import Board
from Core.Neural_Augmentation import board_to_matrix

board = Board()
matrix = board_to_matrix(board)
```

### Running MCTS

```python
from Core.monte_carlo_tree_search import MCTS

mcts = MCTS(model)
best_move = mcts.search(board)
```

### RL Training Loop (Simplified)

```python
agent = DQNAgent(...)
env = ChessEnvironment()
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
```

---

## How to Run

1. **Install Dependencies**
   - Python 3.8+
   - PyTorch
   - python-chess
   - tqdm, psutil, tensorboard, etc.

2. **Train or Download a Model**
   - Use RL scripts in `Core_Reinforcement_Learning/` to train.
   - Or use a pre-trained model from `Hot Models/`.

3. **Run the Engine**
   ```bash
   python run_chess.py
   ```

4. **Evaluate a Model**
   ```bash
   python Evaluations/Evaluate_Core.py
   ```

---

## Troubleshooting

- **ModuleNotFoundError:**  
  Ensure you run scripts from the project root, or add the root to `sys.path` as done in the code:
  ```python
  import sys, os
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  ```

- **Directory Names:**  
  Avoid spaces in directory names for Python imports. Use underscores (e.g., `Core_Reinforcement_Learning`).

- **Model Loading Issues:**  
  Ensure model and mapping files exist and are compatible with your code version.

---


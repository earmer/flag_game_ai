# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Capture the Flag (CTF) game AI system** with multiple backend implementations competing in a real-time strategy game. The project includes:

- **Frontend**: Web-based game visualization (Phaser.js)
- **Backend**: Multiple AI implementations (Python-based)
- **Transformer Module**: Evolutionary training system for AI agents using Transformer networks, genetic algorithms, and adversarial training

## Quick Start Commands

### Setup
```bash
# Install dependencies (Python 3.13+)
uv sync

# Verify environment
uv run -c "import torch; print(torch.__version__)"
```

### Running AI Backends

**Simple Heuristic AI** (picks closest flag):
```bash
uv run backend/pick_closest_flag.py 8081
```

**Transformer AI** (evolutionary trained):
Not fully implemented yet, but working directory must be `backend/transformer`.
```
cd backend/transformer
uv run [script]
```

### Frontend
```bash
cd frontend
uv run -m http.server 8000
# Open http://localhost:8000/index.html in browser
```

### Transformer Training

**Quick test** (4 individuals, 5 generations, ~5 min):
```bash
cd backend/transformer
python train.py --quick-test
```

**Full training** (8 individuals, 50 generations, ~2-3 hours):
```bash
cd backend/transformer
python train.py
```

**Custom configuration**:
```bash
python train.py --population-size 16 --num-generations 100 --num-workers 8
```

**Run tests**:
```bash
cd backend/transformer
python test_modules.py
```

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Web UI)                     │
│              Phaser.js Game Visualization                │
└─────────────────────────────────────────────────────────┘
                    ↕ WebSocket
        ┌───────────────────────────────┐
        │   Game State Exchange (JSON)   │
        └───────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────────────┐
│                  Backend AI Servers                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Heuristic AI │  │  Tree AI     │  │Transformer AI│  │
│  │ (baseline)   │  │ (ML-based)   │  │ (evolved)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Backend Module Structure

```
backend/
├── transformer/              # Evolutionary training system
│   ├── train.py             # Main training loop
│   ├── transformer_model.py # Vision Transformer architecture
│   ├── population.py        # Population management
│   ├── genetic_ops.py       # Selection, crossover, mutation
│   ├── reward_system.py     # Reward calculation (sparse/dense/shaping)
│   ├── adversarial_trainer.py # Self-play training engine
│   ├── game_interface.py    # Game execution wrapper
│   ├── sim_env.py           # CTF game simulation
│   ├── encoding.py          # State encoding to tokens
│   ├── tokenizer.py         # Token vocabulary
│   ├── test_modules.py      # Unit tests
│   └── checkpoints/         # Saved models
│
├── lib/                      # Shared utilities
│   ├── game_engine.py       # Core game logic
│   ├── matrix_util.py       # State matrix conversion
│   ├── tiny_decision_tree.py # Decision tree classifier
│   └── tree_features.py     # Feature extraction
│
├── pick_closest_flag.py     # Heuristic baseline
├── pick_flag_tree_ai.py     # Tree-based AI
├── pick_flag_transformer_ai.py # Transformer inference
└── pick_flag_ai.py          # Alternative heuristic
```

## Core Components

### 1. Transformer Model (`backend/transformer/transformer_model.py`)

**Purpose**: Lightweight Vision Transformer for CTF policy learning

**Key Details**:
- ~130K parameters
- Input: Stacked game state (60 channels, 20×20 spatial)
- Output: 3 action heads (one per player), 5 actions each
- Architecture: Conv encoder → Positional embedding → Transformer encoder → Multi-head action outputs

**Usage**:
```python
from transformer_model import CTFTransformer
model = CTFTransformer()
actions = model(state_tensor)  # Returns list of 3 action logits
```

### 2. Population Management (`backend/transformer/population.py`)

**Purpose**: Manage 8 AI individuals across generations

**Key Classes**:
- `Individual`: Encapsulates model + fitness stats
- `Population`: Manages collection of individuals

**Key Methods**:
- `calculate_fitness()`: Multi-dimensional scoring (wins, flags, survival)
- `select_elite()`: Keep top performers
- `reset_epoch_stats()`: Clear per-generation counters

### 3. Genetic Operators (`backend/transformer/genetic_ops.py`)

**Purpose**: Evolution mechanisms (selection, crossover, mutation)

**Key Functions**:
- `tournament_selection()`: Select parents via tournament
- `crossover()`: Blend parent weights (alpha=0.5)
- `mutate()`: Add temperature-controlled Gaussian noise
- `update_temperature()`: Annealing schedule (cooling_rate=0.95)

**Temperature Decay**:
- Gen 0: 1.00 (high exploration)
- Gen 25: 0.28 (balanced)
- Gen 50: 0.08 (fine-tuning)

### 4. Reward System (`backend/transformer/reward_system.py`)

**Purpose**: Multi-faceted reward calculation with curriculum learning

**Three Reward Types**:

1. **Sparse Rewards**: Only at key events (win/loss, flag capture)
2. **Dense Rewards**: Every step (movement, positioning, teamwork)
3. **Reward Shaping**: Domain knowledge guidance (role-based, position value, timing)

**Curriculum Learning** (Gen-based transition):
- Gen 0-10: 80% dense + 20% sparse (fast learning)
- Gen 11-25: Linear transition
- Gen 26-40: 10% dense + 90% sparse
- Gen 41-50: 100% sparse (pure objectives)

**Shaping Decay**:
- Starts at 1.0, decays to 0.0 by Gen 30
- Encourages role specialization early, then fades

### 5. Adversarial Training (`backend/transformer/adversarial_trainer.py`)

**Purpose**: Self-play training with parallel game execution

**Key Features**:
- Matchup strategies: RoundRobin (early), Tournament (later)
- Parallel execution: 4 workers for concurrent games
- Fitness updates from game results
- Type-safe result tracking

### 6. Game Interface (`backend/transformer/game_interface.py`)

**Purpose**: Wrapper for game execution and agent interaction

**Key Classes**:
- `GameInterface`: Episode execution
- `PolicyAgent`: Abstract agent interface
- `TransformerAgent`: Transformer-based agent
- `RandomAgent`: Baseline for testing

### 7. Training Loop (`backend/transformer/train.py`)

**Purpose**: Main evolutionary training orchestration

**Key Classes**:
- `TrainingConfig`: Hyperparameter management
- `CheckpointManager`: Model persistence + cleanup
- `TrainingLogger`: CSV + text logging
- `EvolutionaryTrainer`: Main training loop

**Output Structure**:
```
checkpoints/ctf_evolution/
├── best_gen_0.pth
├── best_gen_5.pth
└── checkpoint_gen_10.pth

logs/ctf_evolution/
├── training_log.csv
└── training.log
```

### 8. State Encoding (`backend/transformer/encoding.py` + `sim_env.py`)

**Purpose**: Convert game state to token sequences for Transformer input

**Process**:
1. Game state → 20×20 matrix (20 channels)
2. Stack 3 time steps → 60 channels
3. Normalize and convert to tensor
4. Feed to Transformer

## Key Design Patterns

### Population-Based Training
- Maintains 8 diverse individuals
- Elite preservation (top 2 carry forward)
- Tournament selection (k=3) for parent selection
- Crossover + mutation for offspring

### Curriculum Learning
- Reward function evolves with training progress
- Dense rewards early (fast learning signal)
- Sparse rewards late (pure objective optimization)
- Shaping rewards fade out (encourage autonomy)

### Simulated Annealing
- Temperature controls mutation strength
- High temperature early (exploration)
- Low temperature late (exploitation)
- Prevents premature convergence

### Parallel Execution
- 4 concurrent game workers
- Reduces training time significantly
- Maintains reproducibility with seeding

## Important Files and Their Roles

| File | Purpose | Key Functions |
|------|---------|---------------|
| `train.py` | Main entry point | `EvolutionaryTrainer.train()` |
| `transformer_model.py` | Model architecture | `CTFTransformer` class |
| `population.py` | Population management | `Population`, `Individual` |
| `genetic_ops.py` | Evolution operators | `tournament_selection()`, `crossover()`, `mutate()` |
| `reward_system.py` | Reward calculation | `AdaptiveRewardSystem` |
| `adversarial_trainer.py` | Self-play training | `AdversarialTrainer` |
| `game_interface.py` | Game execution | `GameInterface`, `TransformerAgent` |
| `sim_env.py` | Game simulation | `CTFGameSimulator` |
| `encoding.py` | State encoding | `StateEncoder` |
| `test_modules.py` | Unit tests | Test suite for all modules |

## Common Development Tasks

### Running a Quick Test
```bash
cd backend/transformer
python train.py --quick-test
```
This runs 4 individuals for 5 generations with 100 steps/game (~5 min).

### Training a Full Model
```bash
cd backend/transformer
python train.py
```
Runs 8 individuals for 50 generations (~2-3 hours depending on hardware).

### Testing Individual Components
```bash
cd backend/transformer
python test_modules.py
```
Runs unit tests for all modules.

### Evaluating a Trained Model
```bash
python backend/pick_flag_transformer_ai.py 8081 --model checkpoints/ctf_evolution/best_gen_49.pth
```
Then connect frontend to port 8081.

### Debugging Training
- Check `logs/ctf_evolution/training.log` for detailed logs
- Check `logs/ctf_evolution/training_log.csv` for metrics
- Look for fitness trends, generation times, and error messages

## Hyperparameter Tuning

Key hyperparameters in `train.py`:

```python
TrainingConfig(
    population_size=8,           # Number of individuals
    num_generations=50,          # Training generations
    num_workers=4,               # Parallel game workers
    games_per_matchup=1,         # Games per pairing
    max_game_steps=1000,         # Max steps per game

    # Genetic algorithm
    elite_size=2,                # Top performers to keep
    tournament_size=3,           # Tournament selection size
    crossover_alpha=0.5,         # Weight blend ratio
    mutation_rate=0.1,           # Mutation probability
    initial_temperature=1.0,     # Annealing start
    cooling_rate=0.95,           # Temperature decay
)
```

**Tuning Tips**:
- Increase `population_size` for more diversity (slower)
- Increase `num_workers` to parallelize (if hardware allows)
- Decrease `cooling_rate` for faster convergence (may get stuck)
- Increase `num_generations` for better final performance

## Testing and Validation

### Unit Tests
```bash
cd backend/transformer
python test_modules.py
```

### Integration Testing
1. Train a small model: `python train.py --quick-test`
2. Run inference: `python backend/pick_flag_transformer_ai.py 8081 --model <checkpoint>`
3. Connect frontend and play manually

### Performance Metrics
- **Fitness**: Multi-dimensional score (wins, flags, survival)
- **Win Rate**: Percentage of games won
- **Flags Captured**: Average flags per game
- **Training Time**: Time per generation

## Debugging Tips

### Common Issues

**Issue**: Model not improving
- Check reward system is active (not all zeros)
- Verify game simulation is running correctly
- Check temperature decay (should decrease over time)

**Issue**: Training crashes
- Check GPU memory (if using CUDA)
- Verify all dependencies installed
- Check file permissions for checkpoints/logs

**Issue**: Slow training
- Increase `num_workers` for parallelization
- Reduce `max_game_steps` for faster games
- Use GPU if available

### Logging
- Main log: `logs/ctf_evolution/training.log`
- Metrics: `logs/ctf_evolution/training_log.csv`
- Checkpoints: `checkpoints/ctf_evolution/`

## Project Status

**Status**: Production Ready (Commit 6e3c487)

**Recent Fixes**:
- Type consistency in GameResult IDs
- Missing epoch attributes in Individual
- Module-level imports for error detection
- Unified constructor API
- RewardShaping A+B+D options
- Feature dimension validation
- Per-player reward breakdown

**Key Features**:
✅ Type safety with full annotations
✅ Runtime validation (feature dimensions)
✅ Per-player tracking
✅ Reward shaping with decay
✅ Curriculum learning
✅ Parallel execution
✅ Checkpoint management
✅ Comprehensive logging

## Documentation References

- `backend/transformer/README.md`: Detailed architecture overview
- `backend/transformer/IMPLEMENTATION_COMPLETE.md`: Component status
- `backend/transformer/Token_Convert.md`: State encoding details
- `backend/README.md`: Backend AI implementations
- `README.md`: Project overview and game rules

## Guidelines for Claude Code

1. For all commits, please only include files are related to each other to make a concise commit message, and you can commit in multiple times if needed.
2. Musn't add `Co-authored by Claude`, `By Claude Code` or any other similar text to the commit message, document, or any other file unless the user strongly requests it.
# CTF Transformer AI System

## Overview

A complete evolutionary training system for CTF (Capture The Flag) game AI using Transformer-based policy networks, genetic algorithms, and adversarial training.

**Status**: ✅ Production Ready (Commit e255747)

---

## Architecture

```
Population Manager (8 individuals)
    ↓
Adversarial Training Engine (self-play)
    ↓
Genetic Evolution (selection, crossover, mutation)
    ↓
Transformer Policy Network (3-player actions)
```

---

## Core Components

### 1. **transformer_model.py**
- Lightweight Vision Transformer for CTF policy
- ~130K parameters
- Multi-agent output heads (3 players)
- Runtime feature dimension validation

### 2. **population.py**
- Individual: Encapsulates model + statistics
- Population: Manages 8 AI agents
- Unified constructor API with optional auto-initialization

### 3. **genetic_ops.py**
- Tournament selection with temperature control
- Weight averaging crossover
- Gaussian mutation with annealing
- Complete evolution pipeline

### 4. **reward_system.py**
- Sparse rewards: Win/loss, flag capture
- Dense rewards: Movement, positioning, teamwork
- Curriculum learning: Dense→Sparse transition
- **RewardShaping**: A+B+D options with decay
- Per-player breakdown (decays over training)

### 5. **game_interface.py**
- GameInterface: Episode execution
- PolicyAgent: Abstract agent interface
- TransformerAgent: Transformer-based agent
- RandomAgent: Baseline for testing

### 6. **adversarial_trainer.py**
- Matchup strategies: RoundRobin, Tournament, Adaptive
- Parallel game execution (4 workers)
- Fitness updates from game results
- Type-safe result tracking

### 7. **train.py**
- TrainingConfig: Hyperparameter management
- CheckpointManager: Model persistence
- TrainingLogger: CSV + text logging
- EvolutionaryTrainer: Main training loop

### 8. **encoding.py & sim_env.py**
- State encoding to token sequences
- Complete CTF game simulation

---

## Recent Fixes (Commit e255747)

| # | Issue | Fix | Impact |
|---|-------|-----|--------|
| 1 | Type mismatch | GameResult IDs to str | Type safety |
| 2 | Missing attrs | Added epoch_* fields | Training stability |
| 3 | Runtime imports | Module-level imports | Error detection |
| 4 | API friction | Unified constructor | Developer experience |
| 5 | Missing feature | RewardShaping A+B+D | Training guidance |
| 6 | No validation | Feature dim check | Debugging |
| 7 | No breakdown | Per-player rewards | Analysis |

---

## Quick Start

### Installation
```bash
cd /mnt/c/Users/Earmer/CTF/backend/transformer
```

### Quick Test (5 min)
```bash
python train.py --quick-test
```
- 4 individuals, 5 generations, 100 steps/game

### Full Training
```bash
python train.py
```
- 8 individuals, 50 generations, 1000 steps/game

### Custom Config
```bash
python train.py --population-size 16 --num-generations 100 --num-workers 8
```

---

## Output Structure

```
checkpoints/
├── ctf_evolution/
│   ├── best_gen_0.pth
│   ├── best_gen_5.pth
│   └── checkpoint_gen_10.pth
logs/
├── ctf_evolution/
│   ├── training_log.csv
│   └── training.log
```

---

## Key Features

✅ **Type Safety**: All type annotations verified
✅ **Runtime Validation**: Feature dimension checks
✅ **Per-Player Tracking**: Individual reward breakdown
✅ **Reward Shaping**: A+B+D options with decay
✅ **Curriculum Learning**: Dense→Sparse transition
✅ **Parallel Execution**: 4-worker game execution
✅ **Checkpoint Management**: Auto-cleanup old models
✅ **Comprehensive Logging**: CSV + text logs

---

## Documentation

- **IMPLEMENTATION_COMPLETE.md**: Full component status
- **MISSING_COMPONENTS.md**: Previously missing items (now complete)
- **IMPLEMENTATION_ROADMAP.md**: Architecture overview
- **Token_Convert.md**: State encoding details
- **VISUALIZATION_DESIGN.md**: UI/visualization plans

---

## Next Steps

1. Run quick test to verify setup
2. Tune hyperparameters for your hardware
3. Monitor training logs for convergence
4. Analyze per-player breakdowns for strategy insights
5. Deploy best models for evaluation

---

**Last Updated**: 2025-12-28
**Commit**: e255747
**Status**: Production Ready

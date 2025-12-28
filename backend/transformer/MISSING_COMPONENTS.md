# Missing Components Analysis

## Overview
The transformer folder has code that references modules that don't exist. This document lists what needs to be created.

---

## 1. Missing Module: `encoding.py`

### Location
Should be created at: `backend/transformer/encoding.py`

### Required By
- `tokenizer.py` (line 12)
- `game_interface.py` (line 269)

### Required Functions

#### `encode_status_for_team(status_req, geometry, max_tokens)`
**Purpose:** Convert game state to token sequences for the Transformer model

**Parameters:**
- `status_req`: Dict - Game status from `sim.status(team)` call
- `geometry`: Geometry object - Static map information
- `max_tokens`: int - Maximum token sequence length

**Returns:** Tuple of:
- `type_ids`: List[int] - Token type IDs
- `features`: List[List[float]] - Feature vectors for each token
- `padding_mask`: List[bool] - Mask for padding tokens
- `my_player_indices`: Tuple[int, ...] - Indices of player tokens

#### `to_torch_batch(encoded_list)`
**Purpose:** Convert list of encoded states to PyTorch batch

**Parameters:**
- `encoded_list`: List of tuples from `encode_status_for_team()`

**Returns:** `EncodedBatch` object with batched tensors

#### `EncodedBatch` (dataclass)
**Purpose:** Container for batched encoded data

**Fields:**
- `type_ids`: torch.Tensor - Shape (batch, max_tokens)
- `features`: torch.Tensor - Shape (batch, max_tokens, feature_dim)
- `padding_mask`: torch.Tensor - Shape (batch, max_tokens)
- `my_player_token_indices`: torch.Tensor - Shape (batch, num_players)

---

## 2. Missing Module: `sim_env.py`

### Location
Should be created at: `backend/transformer/sim_env.py`

### Required By
- `adversarial_trainer.py` (line 204, currently commented)
- `game_interface.py` (lines 519, 588)

### Required Class: `CTFSim`

**Purpose:** Game simulation environment for training (no WebSocket needed)

**Constructor:**
```python
CTFSim(width=20, height=20, num_players=3, seed=None)
```

**Required Methods:**
- `reset()` - Reset game to initial state
- `init_payload(team)` - Get initialization data for a team
- `status(team)` - Get current game state for a team
- `step(l_actions, r_actions)` - Execute one game step with actions from both teams

**Required Attributes:**
- `l_score`: int - Left team score
- `r_score`: int - Right team score
- `done`: bool - Whether game is finished
- `step_count`: int - Current step number
- `dt_ms`: float - Time step in milliseconds

**Expected Behavior:**
- Should simulate the CTF game logic
- Track player positions, flags, prisons
- Handle collisions, tagging, flag capture
- Update scores when flags reach target zones

---


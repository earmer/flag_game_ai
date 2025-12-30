# Backend/Transformer Implementation Issues

## 🔴 Critical Issues (Must Fix)

### Issue #1: Survival Rate Calculation Bug
**File:** `game_interface.py:464-476`
**Problem:** Loop recalculates same final state instead of tracking historical survival
**Impact:** Incorrect fitness metrics, wrong evolutionary guidance
**Fix:** Track survival during main game loop, not after

### Issue #2: Prison Position Not Passed to Reward System
**File:** `game_interface.py:421-426`
**Problem:** `calculate_reward()` called without `my_prison_pos` parameter, uses wrong default (0,0)
**Impact:** Reward shaping for rescue behavior calculated incorrectly
**Fix:** Extract prison positions and pass to `calculate_reward()` calls

### Issue #3: Enemy Tagging Statistics Not Tracked
**File:** `game_interface.py:491-492`
**Problem:** `l_enemies_tagged` and `r_enemies_tagged` hardcoded to 0
**Impact:** Missing defensive metrics in fitness calculation
**Fix:** Track tagging events during game loop by detecting prison state changes

---

## 🟡 Medium Priority Issues (Should Fix)

### Issue #4: Target Position Logic Ambiguity
**File:** `game_interface.py:338-347`
**Problem:** `_get_target_pos()` uses `my_side_is_left` which may not correctly handle both teams
**Impact:** Potential incorrect spatial rewards for team R
**Fix:** Investigate geometry initialization, consider using absolute position references

### Issue #5: Feature Dimension Hardcoded
**File:** `encoding.py:175`
**Problem:** Padding features hardcoded to `[-1.0] * 8` instead of using config constant
**Impact:** Fragile code that breaks if `CTFTransformerConfig.feature_dim` changes
**Fix:** Replace with `CTFTransformerConfig.feature_dim`

---

## 🟢 Low Priority Issues (Code Quality)

### Issue #6: Reward Shaping Formula Clarity
**File:** `reward_system.py:328`
**Problem:** Formula `0.3 - generation / 100` is correct but non-intuitive
**Impact:** Code readability
**Fix:** Add detailed comment explaining decay schedule

### Issue #7: Curriculum Transition Documentation
**File:** `reward_system.py:295-307`
**Problem:** Linear interpolation logic lacks inline documentation
**Impact:** Code readability
**Fix:** Add comments explaining transition endpoints

---

## Implementation Priority

1. **Phase 1 (Critical):** Fix issues #1, #2, #3
2. **Phase 2 (Medium):** Fix issues #4, #5
3. **Phase 3 (Polish):** Fix issues #6, #7

## Testing Required

- Unit tests for survival rate tracking
- Verify prison position parameter passing
- Validate enemy tagging counts
- Integration test: Run quick training and verify all metrics are reasonable

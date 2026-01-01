"""
test_c_vs_python.py

Compare C and Python implementations for consistency and performance.
"""

import time
import random
from typing import Dict, List

# Import both implementations
from sim_env import CTFSim as CTFSimPy
from sim_env_wrapper import CTFSimC


def test_basic_functionality():
    """Test basic operations work on both implementations."""
    print("\n=== Test 1: Basic Functionality ===")

    for impl_name, SimClass in [("Python", CTFSimPy), ("C", CTFSimC)]:
        sim = SimClass(seed=42)
        sim.reset()

        # Check initial state
        status = sim.status("L")
        assert status["myteamScore"] == 0
        assert status["opponentScore"] == 0
        assert len(status["myteamPlayer"]) == 3

        # Do a few steps
        for _ in range(10):
            sim.step({"L0": "right", "L1": "right", "L2": "right"},
                    {"R0": "left", "R1": "left", "R2": "left"})

        assert sim.step_count == 10
        print(f"  {impl_name}: OK")

    print("  PASSED")


def test_determinism():
    """Test that same seed produces same results."""
    print("\n=== Test 2: Determinism (same seed) ===")

    for impl_name, SimClass in [("Python", CTFSimPy), ("C", CTFSimC)]:
        results = []
        for _ in range(3):
            sim = SimClass(seed=12345)
            sim.reset()

            # Run 50 steps with random actions
            rng = random.Random(999)
            actions = ["", "up", "down", "left", "right"]

            for _ in range(50):
                l_acts = {f"L{i}": rng.choice(actions) for i in range(3)}
                r_acts = {f"R{i}": rng.choice(actions) for i in range(3)}
                sim.step(l_acts, r_acts)

            status = sim.status("L")
            results.append((sim.l_score, sim.r_score, sim.step_count))

        # All runs should be identical
        assert results[0] == results[1] == results[2], f"{impl_name} not deterministic"
        print(f"  {impl_name}: {results[0]} (consistent)")

    print("  PASSED")


def test_performance():
    """Benchmark performance of both implementations."""
    print("\n=== Test 3: Performance Benchmark ===")

    num_games = 5
    steps_per_game = 500
    results = {}

    for impl_name, SimClass in [("Python", CTFSimPy), ("C", CTFSimC)]:
        rng = random.Random(42)
        actions = ["", "up", "down", "left", "right"]

        start = time.perf_counter()

        for game in range(num_games):
            sim = SimClass(seed=game)
            sim.reset()

            for _ in range(steps_per_game):
                l_acts = {f"L{i}": rng.choice(actions) for i in range(3)}
                r_acts = {f"R{i}": rng.choice(actions) for i in range(3)}
                sim.step(l_acts, r_acts)

        elapsed = time.perf_counter() - start
        total_steps = num_games * steps_per_game
        steps_per_sec = total_steps / elapsed

        results[impl_name] = {
            "elapsed": elapsed,
            "steps_per_sec": steps_per_sec
        }

        print(f"  {impl_name}: {elapsed:.3f}s ({steps_per_sec:.0f} steps/sec)")

    # Calculate speedup
    if "Python" in results and "C" in results:
        speedup = results["Python"]["elapsed"] / results["C"]["elapsed"]
        print(f"\n  Speedup: {speedup:.1f}x")

    print("  PASSED")


def test_binary_performance():
    """Benchmark binary protocol vs JSON."""
    print("\n=== Test 4: Binary Protocol Benchmark ===")

    num_games = 5
    steps_per_game = 500
    results = {}

    for mode in ["C (JSON)", "C (Binary)"]:
        rng = random.Random(42)
        actions = ["", "up", "down", "left", "right"]

        start = time.perf_counter()

        for game in range(num_games):
            sim = CTFSimC(seed=game)
            sim.reset()

            for _ in range(steps_per_game):
                l_acts = {f"L{i}": rng.choice(actions) for i in range(3)}
                r_acts = {f"R{i}": rng.choice(actions) for i in range(3)}
                if mode == "C (Binary)":
                    sim.step_fast(l_acts, r_acts)
                else:
                    sim.step(l_acts, r_acts)

        elapsed = time.perf_counter() - start
        total_steps = num_games * steps_per_game
        steps_per_sec = total_steps / elapsed

        results[mode] = {
            "elapsed": elapsed,
            "steps_per_sec": steps_per_sec
        }

        print(f"  {mode}: {elapsed:.3f}s ({steps_per_sec:.0f} steps/sec)")

    # Calculate speedup
    speedup = results["C (JSON)"]["elapsed"] / results["C (Binary)"]["elapsed"]
    print(f"\n  Binary speedup: {speedup:.1f}x")

    print("  PASSED")


def main():
    print("=" * 60)
    print("CTF Simulator: C vs Python Comparison")
    print("=" * 60)

    test_basic_functionality()
    test_determinism()
    test_performance()
    test_binary_performance()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

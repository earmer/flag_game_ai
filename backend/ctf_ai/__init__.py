"""CTF AI: lightweight self-play + small Transformer scaffolding.

This package is intentionally dependency-light. PyTorch is optional at runtime:
- If `torch` is installed, you can use Transformer inference/training.
- If not, the backend can still run with heuristic policies.
"""


"""Experiment scripts live flat in experiments/code and import each other by name."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

"""Lets the tests import the scripts that live in the pipeline/ folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
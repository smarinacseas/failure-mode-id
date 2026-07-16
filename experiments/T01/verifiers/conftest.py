"""Put this self-contained verifier package on sys.path so the flat modules
(base, coverage_pool, precision_pool, reward) import each other and the tests
import them, without needing experiments/ to be an installed package."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

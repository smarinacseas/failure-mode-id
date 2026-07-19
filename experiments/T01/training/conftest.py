"""Put the training package and the sibling flat `verifiers/` package on
sys.path so the training modules (common, datasets_t01, reward_adapter, …) and
the verifier modules (base, reward, coverage_pool, precision_pool) import each
other and the tests import them, without needing experiments/ to be an installed
package (mirrors verifiers/conftest.py)."""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # training/ (common, datasets_t01, …)
sys.path.insert(0, str(_HERE.parent / "verifiers"))  # base, reward, coverage_pool, precision_pool

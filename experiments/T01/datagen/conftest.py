"""Put datagen/ (constraints, compose) AND the sibling verifiers/ package on
sys.path, so datagen modules import the verifier registry (base, archetypes,
coverage_pool, precision_pool) and the tests import datagen modules."""
import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
for p in (_here, _here.parent / "verifiers"):
    sys.path.insert(0, str(p))

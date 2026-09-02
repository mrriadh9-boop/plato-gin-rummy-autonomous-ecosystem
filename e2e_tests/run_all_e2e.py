"""
E2E Test Runner entrypoint.
Executes the Plato Gin Rummy E2E test suite.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2e_tests.harness.runner import run_e2e_suite

if __name__ == "__main__":
    suite_dir = str(Path(__file__).resolve().parent)
    exit_code = run_e2e_suite(suite_dir=suite_dir, verbose=True)
    sys.exit(exit_code)

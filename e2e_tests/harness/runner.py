"""
E2E Test Runner & Suite Orchestrator.
Provides programmatic execution, metrics collection, and test report generation
across all 4 testing tiers (Tier 1 Features, Tier 2 Boundaries, Tier 3 Combinations, Tier 4 Real-World).
"""
from __future__ import annotations

import os
import sys
import time
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TestSuiteReport:
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_sec: float = 0.0
    tier_counts: Dict[str, int] = field(default_factory=dict)
    feature_coverage: Dict[str, int] = field(default_factory=dict)


def run_e2e_suite(
    suite_dir: Optional[str] = None,
    verbose: bool = True,
    extra_args: Optional[List[str]] = None
) -> int:
    """
    Executes the Plato Gin Rummy E2E test suite via pytest.
    """
    if suite_dir is None:
        suite_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    args = [suite_dir]
    if verbose:
        args.append("-v")
    if extra_args:
        args.extend(extra_args)

    print(f"\n========================================================")
    print(f"  PLATO GIN RUMMY AUTONOMOUS ECOSYSTEM - E2E RUNNER")
    print(f"  Executing test suite: {suite_dir}")
    print(f"========================================================\n")

    t0 = time.perf_counter()
    exit_code = pytest.main(args)
    duration = time.perf_counter() - t0

    print(f"\n========================================================")
    print(f"  E2E Test Suite Execution Complete (Duration: {duration:.2f}s)")
    print(f"  Exit Code: {exit_code} ({'PASS' if exit_code == 0 else 'FAIL'})")
    print(f"========================================================\n")

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(run_e2e_suite())

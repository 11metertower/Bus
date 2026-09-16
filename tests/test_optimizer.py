"""Unit tests for the CP-SAT optimizer.

The solver requires OR-Tools, and the end-to-end solve additionally requires the
processed CSV inputs under data/processed/.  Both are guarded so the suite still
runs (skipping these) in environments without those dependencies.
"""

import unittest
from pathlib import Path

try:
    from src import optimizer as opt

    HAS_ORTOOLS = True
except Exception:  # pragma: no cover - environment without ortools
    HAS_ORTOOLS = False

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
HAS_PROCESSED = PROCESSED_DIR.exists() and (PROCESSED_DIR / "requests.csv").exists()


@unittest.skipUnless(HAS_ORTOOLS, "OR-Tools not installed")
class TestDefaultProfiles(unittest.TestCase):
    def test_five_profiles_with_expected_ids(self):
        ids = [p.candidate_id for p in opt.DEFAULT_PROFILES]
        self.assertEqual(
            ids,
            ["candidate_01", "candidate_02", "candidate_03", "candidate_04", "candidate_05"],
        )

    def test_only_expert_style_profile_has_expert_bonus(self):
        for profile in opt.DEFAULT_PROFILES[:-1]:
            self.assertEqual(profile.expert_vehicle_bonus, 0)
            self.assertEqual(profile.expert_driver_bonus, 0)
        last = opt.DEFAULT_PROFILES[-1]
        self.assertGreater(last.expert_vehicle_bonus, 0)
        self.assertGreater(last.expert_driver_bonus, 0)

    def test_diversity_penalty_defaults_to_zero(self):
        for profile in opt.DEFAULT_PROFILES:
            self.assertEqual(profile.diversity_penalty_weight, 0)


@unittest.skipUnless(HAS_ORTOOLS and HAS_PROCESSED, "OR-Tools and processed data required")
class TestSolveIntegration(unittest.TestCase):
    def test_baseline_solution_has_no_hard_violations(self):
        tables = opt.load_processed_inputs(PROCESSED_DIR)
        solution = opt.solve_candidate(tables, opt.DEFAULT_PROFILES[0])
        self.assertFalse(solution.schedule.empty)
        # The independent checker must agree the solver produced a legal schedule.
        self.assertEqual(opt.hard_violation_total(solution.hard_constraint_check), 0)

    def test_solve_all_returns_five_candidates(self):
        tables = opt.load_processed_inputs(PROCESSED_DIR)
        solutions = opt.solve_all_default_profiles(tables)
        self.assertEqual(len(solutions), 5)
        for solution in solutions:
            self.assertEqual(opt.hard_violation_total(solution.hard_constraint_check), 0)


if __name__ == "__main__":
    unittest.main()

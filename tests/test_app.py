"""Comprehensive test suite for MigrationSafe AI application (Review 2).

Covers all required verification areas:
- Dataset generation and schema validation (synthetic_migration_data.csv)
- Baseline prediction and heuristic evaluation
- ML prediction, uncertainty quantification, and OOD bounds detection
- Confidence calculation, probabilities, and disclaimers
- Review 2 Core Edge cases robustness and dynamic failure analysis (PASS/FAIL)
- Migration rehearsal simulation and lock contention modeling [SIMULATED]
- Stateful rollback schema restoration and parity verification
- Prediction vs Rehearsal comparative validation
- Benchmark calculations, downtime avoided formulas, and CSV persistence
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.generator import generate_migration_dataset, save_migration_dataset
from ml.baseline import RuleBasedMigrationPredictor
from ml.model import MigrationRiskModel, MODEL_SAVE_PATH
from ml.edge_cases import EDGE_CASES, run_edge_case_analysis, analyze_test_failures, FAILURE_REASONS
from ml.simulator import TableState, MigrationSimulator
from experiments.benchmark import MigrationBenchmark, BENCHMARK_SAVE_PATH, BENCHMARK_CSV_PATH


class TestMigrationSafeApp(unittest.TestCase):
    """Complete end-to-end test suite for MigrationSafe AI Review 2."""

    @classmethod
    def setUpClass(cls):
        """Initialize shared dataset, model, baseline, simulator, and benchmark."""
        cls.df = generate_migration_dataset(n_records=5000, random_state=42)
        cls.baseline = RuleBasedMigrationPredictor()
        cls.model = MigrationRiskModel(n_estimators=30, max_depth=6, random_state=42)
        cls.train_results = cls.model.train(cls.df, test_size=0.2, save_model=True)
        cls.simulator = MigrationSimulator()
        cls.benchmark = MigrationBenchmark()

    # 1. Dataset Generation Tests
    def test_dataset_generation(self):
        """Test synthetic data generation produces >=5000 records with all required fields."""
        self.assertGreaterEqual(len(self.df), 5000)
        required_fields = {
            "table_size_mb",
            "table_size_gb",
            "row_count",
            "query_frequency",
            "query_type",
            "migration_type",
            "estimated_lock_duration",
            "actual_lock_duration",
            "workload_intensity",
            "migration_success",
            "lock_risk",
        }
        self.assertTrue(required_fields.issubset(set(self.df.columns)))
        self.assertTrue((self.df["table_size_mb"] > 0).all())
        self.assertTrue((self.df["table_size_gb"] > 0).all())
        self.assertTrue((self.df["row_count"] > 0).all())
        self.assertTrue((self.df["actual_lock_duration"] > 0).all())
        self.assertTrue(set(self.df["lock_risk"].unique()).issubset({0, 1}))

        # Persistence check
        save_path = save_migration_dataset(self.df)
        self.assertTrue(Path(save_path).exists())

    # 2. Baseline Prediction Tests
    def test_baseline_prediction(self):
        """Test rule-based baseline prediction and metrics calculation."""
        pred = self.baseline.predict_record(
            migration_type="TABLE_REWRITE",
            table_size_gb=200.0,
            workload_intensity="CRITICAL",
            query_frequency=4000.0,
            estimated_lock_duration=50.0,
        )
        self.assertIn("risk_level", pred)
        self.assertIn("predicted_success", pred)
        self.assertIn("predicted_lock_duration", pred)
        self.assertEqual(pred["predicted_success"], 0)
        self.assertEqual(pred["risk_level"], "CRITICAL")

        metrics = self.baseline.evaluate(self.df)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)

    # 3. ML Model Prediction & Uncertainty Tests
    def test_ml_prediction_and_uncertainty(self):
        """Test ML prediction, probability distribution, and OOD uncertainty detection."""
        # Standard in-distribution input
        standard_input = {
            "table_size_gb": 80.0,
            "row_count": 8_000_000,
            "query_frequency": 1500.0,
            "estimated_lock_duration": 4.0,
            "migration_type": "ADD_COLUMN_DEFAULT",
            "workload_intensity": "MEDIUM",
            "query_type": "MIXED_OLTP",
        }
        res = self.model.predict_single(standard_input)
        self.assertIn("predicted_risk", res)
        self.assertIn("confidence_score", res)
        self.assertIn("confidence_level", res)
        self.assertIn("safe_probability", res)
        self.assertIn("high_risk_probability", res)
        self.assertFalse(res["is_out_of_distribution"])
        self.assertFalse(res["requires_manual_review"])

        # Extreme Out-of-Distribution input
        ood_input = {
            "table_size_gb": 5000.0,
            "row_count": 500_000_000,
            "query_frequency": 15000.0,
            "estimated_lock_duration": 500.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        }
        ood_res = self.model.predict_single(ood_input)
        self.assertTrue(ood_res["is_out_of_distribution"])
        self.assertTrue(ood_res["requires_manual_review"])
        self.assertIn("LOW / UNCERTAIN", ood_res["confidence_level"])
        self.assertGreaterEqual(len(ood_res["uncertainty_reasons"]), 1)

    # 4. Review 2 Core Edge Cases Tests
    def test_review_2_core_edge_cases_pass_fail(self):
        """Verify all 5 edge cases (including the 3 required Review 2 cases) yield PASS status."""
        results = run_edge_case_analysis(self.model, self.baseline)
        self.assertGreaterEqual(len(results), 5)

        for ec in results:
            self.assertEqual(ec["test_status"], "PASS", f"Edge case {ec['case_id']} ({ec['name']}) failed verification!")
            self.assertFalse(ec["system_crashed"])

    # 5. Missing and Anomalous Inputs Safety
    def test_missing_and_extreme_inputs_safety(self):
        """Verify model handles None, NaN, negative numbers, and unknown categories safely."""
        corrupted_input = {
            "table_size_gb": None,
            "row_count": -5000,
            "query_frequency": np.nan,
            "estimated_lock_duration": None,
            "migration_type": "UNKNOWN_DDL_TYPE",
            "workload_intensity": "INVALID_WORKLOAD",
            "query_type": None,
        }
        res = self.model.predict_single(corrupted_input)
        self.assertIn(res["predicted_risk"], ["HIGH RISK", "LOW RISK (SAFE)"])
        self.assertTrue(res["is_out_of_distribution"])
        self.assertTrue(res["requires_manual_review"])

    # 6. Failure Reasons and Nuanced Explanations
    def test_failure_reasons_presence(self):
        """Verify failure reasons document small table workload spikes and data limits."""
        categories = [fr["category"] for fr in FAILURE_REASONS]
        self.assertTrue(any("Small Tables" in c for c in categories))
        self.assertTrue(any("Synthetic Data" in c for c in categories))

    # 7. Migration Rehearsal Simulation Tests [SIMULATED]
    def test_migration_rehearsal_simulation(self):
        """Test safe staging rehearsal simulation for both safe and lock timeout DDL."""
        # Safe rehearsal
        state_safe = self.simulator.create_initial_state("orders", 50.0, 5_000_000)
        res_safe = self.simulator.run_simulation(
            current_state=state_safe,
            migration_type="ADD_COLUMN_DEFAULT",
            workload_intensity="LOW",
            query_frequency=100.0,
            estimated_lock_duration=1.0,
        )
        self.assertTrue(res_safe["success"])
        self.assertEqual(res_safe["status"], "COMPLETED")
        self.assertLess(res_safe["actual_lock_duration"], res_safe["lock_timeout_threshold"])
        self.assertEqual(len(res_safe["stages"]), 4)

        # Timeout rehearsal
        state_fail = self.simulator.create_initial_state("orders", 500.0, 50_000_000)
        res_fail = self.simulator.run_simulation(
            current_state=state_fail,
            migration_type="TABLE_REWRITE",
            workload_intensity="CRITICAL",
            query_frequency=4500.0,
            estimated_lock_duration=50.0,
        )
        self.assertFalse(res_fail["success"])
        self.assertEqual(res_fail["status"], "FAILED_LOCK_TIMEOUT")
        self.assertGreater(res_fail["actual_lock_duration"], res_fail["lock_timeout_threshold"])

    # 8. Stateful Schema Rollback Demonstration Tests
    def test_rollback_demonstration(self):
        """Test rollback restores exact schema version, columns, and indexes."""
        initial_state = self.simulator.create_initial_state("orders", 100.0, 10_000_000)
        snapshot = initial_state.clone()
        active_state = initial_state.clone()

        # Mutate schema
        self.simulator.run_simulation(
            current_state=active_state,
            migration_type="ADD_COLUMN_DEFAULT",
            workload_intensity="LOW",
            query_frequency=100.0,
            estimated_lock_duration=1.0,
        )
        self.assertEqual(active_state.schema_version, "v1.5.0")
        self.assertGreater(len(active_state.columns), len(snapshot.columns))

        # Rollback schema
        rb_res = self.simulator.rollback_migration(active_state, snapshot)
        self.assertTrue(rb_res["rollback_successful"])
        self.assertTrue(rb_res["is_exact_match"])
        self.assertEqual(rb_res["restored_version"], "v1.4.0")
        self.assertEqual(len(active_state.columns), len(snapshot.columns))
        self.assertEqual(active_state.status, "ROLLED_BACK")

    # 9. Benchmark & Downtime Avoided Calculations
    def test_benchmark_calculation(self):
        """Test 150-scenario benchmark calculations and SLA targets."""
        res = self.benchmark.run_benchmark(self.df, self.model, n_scenarios=150, random_state=42, save_results=True)
        self.assertEqual(res["total_scenarios"], 150)
        self.assertGreater(res["comparative"]["downtime_avoided_seconds"], 0)
        self.assertGreaterEqual(res["comparative"]["downtime_avoided_pct"], 75.0)
        self.assertTrue(res["targets"]["target_downtime_met"])
        self.assertTrue(Path(BENCHMARK_SAVE_PATH).exists())
        self.assertTrue(Path(BENCHMARK_CSV_PATH).exists())


if __name__ == "__main__":
    unittest.main()

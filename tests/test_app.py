"""Comprehensive test suite for MigrationSafe AI application.

Covers all required verification areas:
- Dataset generation and schema validation (synthetic_migration_data.csv)
- Baseline prediction and heuristic evaluation
- ML prediction and unseen test split evaluation
- Confidence calculation, probabilities, and disclaimers
- Edge cases robustness and dynamic failure analysis
- Migration simulation and lock contention modeling
- Stateful rollback schema restoration
- Benchmark calculations, downtime avoided formulas, and CSV persistence
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.generator import generate_migration_dataset, save_migration_dataset
from ml.baseline import RuleBasedMigrationPredictor
from ml.model import MigrationRiskModel, MODEL_SAVE_PATH
from ml.edge_cases import EDGE_CASES, run_edge_case_analysis, analyze_test_failures
from ml.simulator import TableState, MigrationSimulator
from experiments.benchmark import MigrationBenchmark, BENCHMARK_SAVE_PATH, BENCHMARK_CSV_PATH


class TestMigrationSafeApp(unittest.TestCase):
    """Complete end-to-end test suite for MigrationSafe AI."""

    @classmethod
    def setUpClass(cls):
        """Initialize shared dataset, model, baseline, and simulator."""
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
        self.assertTrue((Path(save_path).parent / "synthetic_migration_data.csv").exists())

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

    # 3. ML Model Prediction Tests
    def test_ml_prediction(self):
        """Test Scikit-learn classification pipeline trained on train and evaluated on test."""
        self.assertTrue(self.model.is_trained)
        ml_metrics = self.train_results["ml_metrics"]
        self.assertIn("accuracy", ml_metrics)
        self.assertIn("precision", ml_metrics)
        self.assertIn("recall", ml_metrics)
        self.assertIn("f1_score", ml_metrics)
        self.assertTrue(0.0 <= ml_metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= ml_metrics["f1_score"] <= 1.0)

        # Baseline vs ML comparison DataFrame check
        comp_df = self.train_results["comparison_df"]
        self.assertIsInstance(comp_df, pd.DataFrame)
        self.assertEqual(len(comp_df), 4)
        self.assertIn("Rule-Based Baseline", comp_df.columns)
        self.assertIn("Random Forest ML", comp_df.columns)

    # 4. Confidence and Uncertainty Calculation
    def test_confidence_calculation(self):
        """Test prediction interface returns risk, confidence probability, and disclaimer."""
        sample_input = {
            "table_size_mb": 150000.0,
            "table_size_gb": 146.48,
            "row_count": 18000000,
            "query_frequency": 2500.0,
            "estimated_lock_duration": 25.0,
            "migration_type": "ALTER_COLUMN_TYPE",
            "workload_intensity": "HIGH",
            "query_type": "WRITE_HEAVY",
        }
        res = self.model.predict_single(sample_input)
        self.assertIn("predicted_risk", res)
        self.assertIn("confidence_score", res)
        self.assertIn("high_risk_probability", res)
        self.assertIn("safe_probability", res)
        self.assertIn("disclaimer", res)
        self.assertTrue(50.0 <= res["confidence_score"] <= 100.0)
        self.assertIn("guarantee", res["disclaimer"].lower())

    # 5. Edge Cases & Failure Analysis
    def test_edge_cases_and_failure_analysis(self):
        """Test all edge cases execute safely and failure statistics are computed dynamically."""
        results = run_edge_case_analysis(self.model, self.baseline)
        self.assertEqual(len(results), len(EDGE_CASES))
        for item in results:
            self.assertFalse(item["system_crashed"])
            self.assertIn("expected_result", item)
            self.assertIn("actual_result", item)
            self.assertIn("status", item)

        # Dynamic failure statistics
        failure_stats = analyze_test_failures(self.model, self.df)
        self.assertEqual(failure_stats["total_test_samples"], 1000)
        self.assertEqual(
            failure_stats["correct_predictions"] + failure_stats["misclassified_count"],
            failure_stats["total_test_samples"],
        )
        self.assertTrue(0.0 <= failure_stats["error_rate_pct"] <= 100.0)

    # 6. Migration Simulation Tests
    def test_migration_simulation(self):
        """Test local simulated migration workflow for both success and lock timeout."""
        init_state = self.simulator.create_initial_state("orders", 120.0, 15000000)

        # Safe migration simulation
        state_safe = init_state.clone()
        res_safe = self.simulator.run_simulation(
            current_state=state_safe,
            migration_type="ADD_INDEX_CONCURRENTLY",
            workload_intensity="LOW",
            query_frequency=200.0,
            estimated_lock_duration=0.5,
        )
        self.assertTrue(res_safe["success"])
        self.assertEqual(res_safe["status"], "COMPLETED")
        self.assertEqual(len(res_safe["stages"]), 4)

        # Timeout migration simulation
        state_timeout = init_state.clone()
        res_timeout = self.simulator.run_simulation(
            current_state=state_timeout,
            migration_type="TABLE_REWRITE",
            workload_intensity="CRITICAL",
            query_frequency=4500.0,
            estimated_lock_duration=60.0,
        )
        self.assertFalse(res_timeout["success"])
        self.assertEqual(res_timeout["status"], "FAILED_LOCK_TIMEOUT")
        self.assertGreater(res_timeout["blocked_queries"], 0)

    # 7. Rollback Demonstration Tests
    def test_rollback_demonstration(self):
        """Test simulated rollback restores exact original state."""
        init_state = self.simulator.create_initial_state("orders", 100.0, 10000000)
        snapshot = init_state.clone()
        active_state = init_state.clone()

        # Apply schema change
        self.simulator.run_simulation(
            current_state=active_state,
            migration_type="ADD_COLUMN_DEFAULT",
            workload_intensity="LOW",
            query_frequency=150.0,
            estimated_lock_duration=1.0,
        )
        self.assertNotEqual(len(active_state.columns), len(snapshot.columns))

        # Rollback
        rollback_res = self.simulator.rollback_migration(active_state, snapshot)
        self.assertTrue(rollback_res["rollback_successful"])
        self.assertTrue(rollback_res["is_exact_match"])
        self.assertEqual(active_state.schema_version, snapshot.schema_version)
        self.assertEqual(len(active_state.columns), len(snapshot.columns))
        self.assertEqual(len(active_state.indexes), len(snapshot.indexes))

    # 8. Benchmark Calculation Tests
    def test_benchmark_calculation(self):
        """Test benchmark execution, downtime avoided calculation, and local CSV persistence."""
        res = self.benchmark.run_benchmark(
            df=self.df,
            model=self.model,
            n_scenarios=120,
            random_state=42,
            save_results=True,
        )
        self.assertGreaterEqual(res["total_scenarios"], 100)

        base_dt = res["baseline"]["total_downtime_seconds"]
        ai_dt = res["migrationsafe_ai"]["total_downtime_seconds"]
        avoided_dt = res["comparative"]["downtime_avoided_seconds"]
        avoided_pct = res["comparative"]["downtime_avoided_pct"]

        # Formula check: downtime avoided = baseline downtime - migrationsafe downtime
        self.assertAlmostEqual(avoided_dt, round(base_dt - ai_dt, 2), places=1)
        self.assertTrue(0.0 <= avoided_pct <= 100.0)
        self.assertGreaterEqual(
            res["migrationsafe_ai"]["success_rate_pct"],
            res["baseline"]["success_rate_pct"],
        )

        # File persistence checks
        self.assertTrue(BENCHMARK_SAVE_PATH.exists())
        self.assertTrue(BENCHMARK_CSV_PATH.exists())

    # 9. Extreme Inputs & Robustness Tests
    def test_missing_and_extreme_inputs_safety(self):
        """Verify model handles None, NaN, negative numbers, and unknown categories safely."""
        extreme_cases = [
            {"table_size_gb": None, "query_frequency": None, "migration_type": "UNKNOWN_OP"},
            {"table_size_gb": -50.0, "row_count": -1000, "query_frequency": -200.0},
            {"table_size_gb": 10000.0, "query_frequency": 50000.0, "estimated_lock_duration": 1000.0},
        ]
        for case in extreme_cases:
            res = self.model.predict_single(case)
            self.assertIn(res["predicted_risk"], ["HIGH RISK", "LOW RISK (SAFE)"])
            self.assertTrue(0.0 <= res["confidence_score"] <= 100.0)

    # 10. Arbitrary Simulator Input Safety
    def test_arbitrary_simulator_input_safety(self):
        """Verify simulator handles arbitrary or unknown migration types without crashing."""
        init_state = self.simulator.create_initial_state("orders", 100.0, 10000000)
        res = self.simulator.run_simulation(
            current_state=init_state,
            migration_type="CUSTOM_NON_EXISTENT_OP",
            workload_intensity="UNKNOWN_WORKLOAD",
            query_frequency=100.0,
            estimated_lock_duration=5.0,
            random_state=42,
        )
        self.assertIn(res["status"], ["COMPLETED", "FAILED_LOCK_TIMEOUT"])
        self.assertIsInstance(res["actual_lock_duration"], float)

    # 11. Target SLA Metrics Validation
    def test_benchmark_target_metrics_presence(self):
        """Verify target SLA comparisons exist and have valid boolean outcomes."""
        res = self.benchmark.run_benchmark(df=self.df, model=self.model, n_scenarios=100, random_state=42, save_results=False)
        targets = res["targets"]
        self.assertIn("target_downtime_avoided_pct", targets)
        self.assertIn("measured_downtime_avoided_pct", targets)
        self.assertIn("target_downtime_met", targets)
        self.assertIn("target_success_rate_pct", targets)
        self.assertIn("measured_success_rate_pct", targets)
        self.assertIn("target_success_met", targets)
        self.assertIsInstance(targets["target_downtime_met"], bool)
        self.assertIsInstance(targets["target_success_met"], bool)

    # 12. Dynamic Estimated Lock Fallback Tests
    def test_dynamic_estimated_lock_fallback(self):
        """Verify model dynamically computes realistic lock estimates when omitted."""
        input_without_est = {
            "table_size_gb": 120.0,
            "row_count": 15000000,
            "query_frequency": 1200.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "HIGH",
            "query_type": "WRITE_HEAVY",
            # estimated_lock_duration intentionally omitted
        }
        res = self.model.predict_single(input_without_est)
        self.assertIn("predicted_risk", res)
        self.assertTrue(50.0 <= res["confidence_score"] <= 100.0)

    # 13. End-to-End Simulation Workflow Integration Test
    def test_simulation_workflow_integration(self):
        """Verify complete sequence: Prediction -> Simulation -> Result -> Rollback."""
        # 1. Prediction
        pred_payload = {
            "table_size_gb": 100.0,
            "row_count": 10000000,
            "query_frequency": 3500.0,
            "estimated_lock_duration": 50.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        }
        pred = self.model.predict_single(pred_payload)
        self.assertEqual(pred["predicted_risk"], "HIGH RISK")

        # 2. Simulation
        initial_state = self.simulator.create_initial_state("orders", 100.0, 10000000)
        snapshot = initial_state.clone()
        active_state = initial_state.clone()

        sim_res = self.simulator.run_simulation(
            current_state=active_state,
            migration_type="TABLE_REWRITE",
            workload_intensity="CRITICAL",
            query_frequency=3500.0,
            estimated_lock_duration=50.0,
        )

        # 3. Result
        self.assertFalse(sim_res["success"])
        self.assertEqual(sim_res["status"], "FAILED_LOCK_TIMEOUT")
        self.assertGreater(sim_res["blocked_queries"], 0)

        # 4. Rollback
        rb_res = self.simulator.rollback_migration(active_state, snapshot)
        self.assertTrue(rb_res["rollback_successful"])
        self.assertTrue(rb_res["is_exact_match"])
        self.assertEqual(active_state.schema_version, snapshot.schema_version)


if __name__ == "__main__":
    unittest.main()


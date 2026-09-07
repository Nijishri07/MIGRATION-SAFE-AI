"""Edge-case testing and prediction failure analysis for MigrationSafe AI."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .baseline import RuleBasedMigrationPredictor
    from .model import MigrationRiskModel
except ImportError:
    from ml.baseline import RuleBasedMigrationPredictor
    from ml.model import MigrationRiskModel


EDGE_CASES: List[Dict[str, Any]] = [
    {
        "id": "EDGE-1",
        "name": "Massive Table with Low Concurrency",
        "description": "Very large table (850 GB) undergoing data rewrite, but executed off-peak with minimal concurrent traffic (60 QPS).",
        "inputs": {
            "table_size_gb": 850.0,
            "row_count": 80_000_000,
            "query_frequency": 60.0,
            "estimated_lock_duration": 65.0,
            "migration_type": "ALTER_COLUMN_TYPE",
            "workload_intensity": "LOW",
            "query_type": "READ_HEAVY",
        },
        "expected_behavior": "Low Lock Contention (Safe if scheduled in maintenance window)",
        "domain_notes": "Static rule heuristics flag high risk due to massive table size, whereas ML recognizes that low active QPS eliminates lock queue pileup.",
    },
    {
        "id": "EDGE-2",
        "name": "Micro Table under Flash-Crowd Traffic",
        "description": "Small table (3.5 GB) undergoing index lock during peak flash-sale query burst (4,900 QPS).",
        "inputs": {
            "table_size_gb": 3.5,
            "row_count": 350_000,
            "query_frequency": 4900.0,
            "estimated_lock_duration": 2.5,
            "migration_type": "ADD_INDEX_LOCKING",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        },
        "expected_behavior": "High Risk (Catastrophic lock contention despite small size)",
        "domain_notes": "Even momentary exclusive locks create connection pool exhaustion under 4,900 QPS, triggering transaction queue timeout.",
    },
    {
        "id": "EDGE-3",
        "name": "Heavy Workload Combined with Risky DDL",
        "description": "Full table rewrite on primary transactional orders table (380 GB) during active critical business operations.",
        "inputs": {
            "table_size_gb": 380.0,
            "row_count": 42_000_000,
            "query_frequency": 3800.0,
            "estimated_lock_duration": 110.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        },
        "expected_behavior": "High Risk / Guaranteed Timeout",
        "domain_notes": "Both Baseline and ML model should achieve strong consensus that this migration must be aborted or rescheduled.",
    },
    {
        "id": "EDGE-4",
        "name": "Missing & Anomalous Input Values",
        "description": "Form input contains missing/NaN fields, empty migration type, and negative numeric values.",
        "inputs": {
            "table_size_gb": None,
            "row_count": -500,
            "query_frequency": None,
            "estimated_lock_duration": np.nan,
            "migration_type": "",
            "workload_intensity": "UNKNOWN_LEVEL",
            "query_type": "INVALID_QUERY_TYPE",
        },
        "expected_behavior": "Defensive Fallback (Safe handling without crashing)",
        "domain_notes": "Tests system robustness. Missing values are imputed with median/mode defaults to maintain uninterrupted operational availability.",
    },
    {
        "id": "EDGE-5",
        "name": "Extremely High Numeric Values (5 TB Scale)",
        "description": "Hyper-scale enterprise database partition (5,000 GB, 500M rows, 15,000 QPS) outside standard training bounds.",
        "inputs": {
            "table_size_gb": 5000.0,
            "row_count": 500_000_000,
            "query_frequency": 15000.0,
            "estimated_lock_duration": 500.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        },
        "expected_behavior": "Extreme High Risk (Proper extrapolation without arithmetic overflow)",
        "domain_notes": "Verifies that extreme out-of-distribution values correctly produce maximum risk alerts without pipeline errors.",
    },
]


def run_edge_case_analysis(
    ml_model: MigrationRiskModel,
    baseline_predictor: RuleBasedMigrationPredictor,
) -> List[Dict[str, Any]]:
    """Execute all edge cases through both Baseline and ML systems and return structured outcomes."""
    results = []

    for case in EDGE_CASES:
        inp = case["inputs"]

        # 1. Run through ML Model
        try:
            ml_out = ml_model.predict_single(inp)
            ml_risk = ml_out["predicted_risk"]
            ml_conf = f"{ml_out['confidence_score']:.1f}%"
            ml_high_prob = ml_out["high_risk_probability"]
            crashed = False
        except Exception as e:
            ml_risk = "ERROR"
            ml_conf = "0.0%"
            ml_high_prob = 0.0
            crashed = True

        # 2. Run through Baseline Predictor (with defensive safe extraction)
        try:
            clean_table_size = float(inp.get("table_size_gb") or 100.0)
            clean_qps = float(inp.get("query_frequency") or 1000.0)
            clean_est = float(inp.get("estimated_lock_duration") or 10.0)
            clean_m_type = str(inp.get("migration_type") or "ADD_COLUMN_DEFAULT")
            clean_w_int = str(inp.get("workload_intensity") or "MEDIUM")

            base_out = baseline_predictor.predict_record(
                migration_type=clean_m_type,
                table_size_gb=clean_table_size,
                workload_intensity=clean_w_int,
                query_frequency=clean_qps,
                estimated_lock_duration=clean_est,
            )
            base_risk = f"{base_out['risk_level']} RISK"
            base_outcome = "SAFE" if base_out["predicted_success"] == 1 else "HIGH RISK"
        except Exception:
            base_risk = "UNKNOWN"
            base_outcome = "ERROR"

        # Check agreement and correct status
        ml_is_high = "HIGH" in ml_risk
        base_is_high = "HIGH" in base_outcome or "HIGH" in base_risk or "CRITICAL" in base_risk
        agreement = "Consensus (Both High)" if (ml_is_high and base_is_high) else (
            "Consensus (Both Safe)" if (not ml_is_high and not base_is_high) else "Divergent"
        )

        expected_val = case["expected_behavior"]
        if case["id"] == "EDGE-4":
            correct_status = "Correct (Handled Safely)" if not crashed else "Incorrect (Crashed)"
            actual_res = "Handled Safely (Defaulted)" if not crashed else "Crashed"
        elif "Safe" in expected_val or "Low" in expected_val:
            correct_status = "Correct" if not ml_is_high else "Incorrect (Conservative False Alarm)"
            actual_res = ml_risk
        else:
            correct_status = "Correct" if ml_is_high else "Incorrect (Missed High Risk)"
            actual_res = ml_risk

        results.append(
            {
                "case_id": case["id"],
                "name": case["name"],
                "description": case["description"],
                "inputs": inp,
                "input_summary": f"{inp.get('table_size_gb', 'N/A')}GB | {inp.get('migration_type', 'N/A')} | {inp.get('workload_intensity', 'N/A')} | {inp.get('query_frequency', 'N/A')} QPS",
                "baseline_prediction": base_outcome,
                "baseline_risk_level": base_risk,
                "ml_prediction": ml_risk,
                "ml_confidence": ml_conf,
                "ml_high_risk_prob": ml_high_prob,
                "expected_result": expected_val,
                "expected_behavior": expected_val,
                "actual_result": actual_res,
                "status": correct_status,
                "agreement": agreement,
                "domain_notes": case["domain_notes"],
                "system_crashed": crashed,
            }
        )

    return results


FAILURE_REASONS = [
    {
        "category": "Unusual Workload Patterns",
        "description": "Flash-sale bursts (e.g. 4,900+ QPS) on lightweight tables can cause connection pool starvation even if table size is minimal. Models trained on typical linear volume scaling can misjudge queue wait spikes under sudden concurrency surges.",
    },
    {
        "category": "Synthetic Data Limitations",
        "description": "Synthetic generation uses parametric distributions (e.g. lognormal/uniform). Real enterprise database workloads exhibit long-tail micro-bursts, dirty buffer contention, and storage I/O throttling that may not be fully represented.",
    },
    {
        "category": "Unseen Feature Combinations",
        "description": "Novel permutations (e.g. 850 GB table rewrite running at an off-peak 60 QPS during a designated maintenance window) exist near the boundary of the feature space where heuristic rules and ML decision surfaces may diverge.",
    },
    {
        "category": "Noisy Data & Contention Jitter",
        "description": "Dynamic database locking involves non-deterministic factors including OS scheduler latency, concurrent background autovacuum processes, and lock queue ordering, introducing intrinsic irreducible variance near timeout thresholds.",
    },
]


def analyze_test_failures(
    ml_model: MigrationRiskModel,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Dynamically analyze misclassifications on the holdout test set without fake numbers."""
    if not ml_model.is_trained:
        ml_model.train(df, test_size=0.2, save_model=True)

    test_df = ml_model.test_df.copy()
    y_true = ml_model.test_y_true
    y_pred = ml_model.test_y_pred
    y_prob = ml_model.test_y_prob

    total_test = len(y_true)
    correct_mask = (y_true == y_pred)
    correct_count = int(np.sum(correct_mask))
    misclassified_count = int(total_test - correct_count)

    # Breakdown of classifications:
    # 0 = Safe, 1 = High Risk
    tn_mask = (y_true == 0) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)  # Safe predicted as High Risk (Type I)
    fn_mask = (y_true == 1) & (y_pred == 0)  # High Risk predicted as Safe (Type II)

    tn = int(np.sum(tn_mask))
    tp = int(np.sum(tp_mask))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))

    fp_rate = float(fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fn_rate = float(fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    # Build detailed misclassified records dataframe
    failed_indices = np.where(~correct_mask)[0]
    failed_records = []

    for idx in failed_indices:
        row = test_df.iloc[idx]
        actual_val = int(y_true[idx])
        pred_val = int(y_pred[idx])
        probs = y_prob[idx]

        err_type = "False Positive (Safe Flagged High Risk)" if (actual_val == 0 and pred_val == 1) else "False Negative (High Risk Missed)"
        conf = float(np.max(probs) * 100.0)

        failed_records.append(
            {
                "migration_id": row.get("migration_id", f"TEST-{idx}"),
                "table_name": row.get("table_name", "orders"),
                "table_size_gb": float(row["table_size_gb"]),
                "migration_type": str(row["migration_type"]),
                "workload_intensity": str(row["workload_intensity"]),
                "query_frequency": float(row["query_frequency"]),
                "actual_lock_duration": float(row["actual_lock_duration"]),
                "true_outcome": "Safe" if actual_val == 0 else "High Risk",
                "predicted_outcome": "High Risk" if pred_val == 1 else "Safe",
                "error_type": err_type,
                "confidence_score": round(conf, 1),
            }
        )

    failed_df = pd.DataFrame(failed_records)

    # Failure distribution by migration type
    if not failed_df.empty:
        failures_by_migration = failed_df["migration_type"].value_counts().to_dict()
        failures_by_workload = failed_df["workload_intensity"].value_counts().to_dict()
    else:
        failures_by_migration = {}
        failures_by_workload = {}

    return {
        "total_test_samples": total_test,
        "correct_predictions": correct_count,
        "misclassified_count": misclassified_count,
        "true_negatives": tn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy_pct": round((correct_count / total_test) * 100.0, 2),
        "error_rate_pct": round((misclassified_count / total_test) * 100.0, 2),
        "false_positive_rate_pct": round(fp_rate, 2),
        "false_negative_rate_pct": round(fn_rate, 2),
        "failed_records_df": failed_df,
        "failures_by_migration": failures_by_migration,
        "failures_by_workload": failures_by_workload,
    }


if __name__ == "__main__":
    from data.generator import generate_migration_dataset
    csv_path = PROJECT_ROOT / "data" / "synthetic_migration_data.csv"
    if csv_path.exists():
        data_ec = pd.read_csv(csv_path)
    else:
        data_ec = generate_migration_dataset(n_records=5000, random_state=42)

    model_ec = MigrationRiskModel(n_estimators=50, max_depth=8, random_state=42)
    model_ec.train(data_ec, test_size=0.2, save_model=True)
    baseline_ec = RuleBasedMigrationPredictor()

    print("[MigrationSafe AI] Running Edge Case Scenarios...")
    ec_results = run_edge_case_analysis(model_ec, baseline_ec)
    for r in ec_results:
        print(f"  {r['case_id']}: {r['name']}")
        print(f"     Status: {r['status']} | ML: {r['ml_prediction']} ({r['ml_confidence']}) | Base: {r['baseline_prediction']}")

    print("\n[MigrationSafe AI] Running Test Set Failure Analysis...")
    fa = analyze_test_failures(model_ec, data_ec)
    print(f"  Accuracy: {fa['accuracy_pct']:.2f}% | Error Rate: {fa['error_rate_pct']:.2f}% | Misclassified: {fa['misclassified_count']}")
    print(f"  False Positives: {fa['false_positives']} ({fa['false_positive_rate_pct']:.2f}%) | False Negatives: {fa['false_negatives']} ({fa['false_negative_rate_pct']:.2f}%)")


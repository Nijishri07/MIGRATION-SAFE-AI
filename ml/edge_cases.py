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
        "name": "Very Large Table + High Workload",
        "category": "Review 2 Core Edge Case",
        "description": "Heavy full table rewrite on high-volume 380 GB transactional table under 3,800 QPS active production workload.",
        "inputs": {
            "table_size_gb": 380.0,
            "row_count": 42_000_000,
            "query_frequency": 3800.0,
            "estimated_lock_duration": 110.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        },
        "expected_behavior": "HIGH RISK (Severe Lock Queue Contention & Statement Timeout)",
        "domain_notes": "Both Baseline heuristic and ML classifier must intercept this operation. Exclusive table locks will back up thousands of queries within seconds.",
    },
    {
        "id": "EDGE-2",
        "name": "Small Table + Extremely High Workload (Flash Crowd)",
        "category": "Review 2 Core Edge Case",
        "description": "Small table (3.5 GB) undergoing index locking during an intense flash-sale traffic spike (4,900 QPS).",
        "inputs": {
            "table_size_gb": 3.5,
            "row_count": 350_000,
            "query_frequency": 4900.0,
            "estimated_lock_duration": 2.5,
            "migration_type": "ADD_INDEX_LOCKING",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        },
        "expected_behavior": "HIGH RISK (Catastrophic connection pool starvation despite small table size)",
        "domain_notes": "Even a 2.5s exclusive lock under 4,900 QPS queues > 12,000 queries, instantly exhausting connection pool limits and causing cascading timeouts.",
    },
    {
        "id": "EDGE-3",
        "name": "Missing / Invalid Migration Information",
        "category": "Review 2 Core Edge Case",
        "description": "Migration request contains missing/NaN fields, negative values, and an invalid migration type.",
        "inputs": {
            "table_size_gb": None,
            "row_count": -500,
            "query_frequency": None,
            "estimated_lock_duration": np.nan,
            "migration_type": "",
            "workload_intensity": "UNKNOWN_LEVEL",
            "query_type": "INVALID_QUERY_TYPE",
        },
        "expected_behavior": "SAFE DEFENSIVE HANDLING & MANUAL REVIEW FLAG (Zero Crashes + Low Confidence Alert)",
        "domain_notes": "System must sanitize inputs defensively with median fallbacks, avoid throwing uncaught exceptions, and alert the DBA that manual review is required.",
    },
    {
        "id": "EDGE-4",
        "name": "Massive Table with Low Concurrency (Off-Peak Maintenance)",
        "category": "Advanced Maintenance Scheduling",
        "description": "Very large table (850 GB) undergoing non-blocking index creation during a quiet off-peak maintenance window (60 QPS).",
        "inputs": {
            "table_size_gb": 850.0,
            "row_count": 80_000_000,
            "query_frequency": 60.0,
            "estimated_lock_duration": 0.5,
            "migration_type": "ADD_INDEX_CONCURRENTLY",
            "workload_intensity": "LOW",
            "query_type": "READ_HEAVY",
        },
        "expected_behavior": "LOW RISK (Safe non-blocking maintenance execution)",
        "domain_notes": "Non-blocking concurrent DDL under quiet 60 QPS completes safely without holding exclusive table locks.",
    },
    {
        "id": "EDGE-5",
        "name": "Hyper-Scale Out-of-Distribution Scale (5 TB)",
        "category": "Stress Test & Bounds Check",
        "description": "Massive enterprise partition (5,000 GB, 500M rows, 15,000 QPS) far exceeding normal training bounds.",
        "inputs": {
            "table_size_gb": 5000.0,
            "row_count": 500_000_000,
            "query_frequency": 15000.0,
            "estimated_lock_duration": 500.0,
            "migration_type": "TABLE_REWRITE",
            "workload_intensity": "CRITICAL",
            "query_type": "WRITE_HEAVY",
        },
        "expected_behavior": "HIGH RISK & OUT-OF-DISTRIBUTION WARNING",
        "domain_notes": "Extrapolates safely to High Risk while alerting DBAs that inputs exceed the reliable training envelope.",
    },
]

def run_edge_case_analysis(
    ml_model: MigrationRiskModel,
    baseline_predictor: RuleBasedMigrationPredictor,
) -> List[Dict[str, Any]]:
    """Execute all edge cases through both Baseline and ML systems and return structured PASS/FAIL outcomes."""
    results = []

    for case in EDGE_CASES:
        inp = case["inputs"]

        try:
            ml_out = ml_model.predict_single(inp)
            ml_risk = ml_out["predicted_risk"]
            ml_conf = f"{ml_out['confidence_score']:.1f}%"
            ml_conf_level = ml_out.get("confidence_level", "STANDARD")
            ml_high_prob = ml_out["high_risk_probability"]
            is_ood = ml_out.get("is_out_of_distribution", False)
            crashed = False
        except Exception:
            ml_risk = "ERROR (CRASHED)"
            ml_conf = "0.0%"
            ml_conf_level = "ERROR"
            ml_high_prob = 0.0
            is_ood = True
            crashed = True

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
            base_risk = base_out["risk_level"]
            base_outcome = "HIGH RISK" if base_out["predicted_high_risk"] == 1 else "SAFE"
        except Exception:
            base_risk = "UNKNOWN"
            base_outcome = "ERROR"

        expected_val = case["expected_behavior"]
        if case["id"] == "EDGE-3":
            test_passed = (not crashed) and is_ood
            actual_res = "Handled Safely (Imputed & OOD Flagged)" if test_passed else "Failed Defensive Check"
            pass_status = "PASS" if test_passed else "FAIL"
        elif "HIGH RISK" in expected_val or "High Risk" in expected_val:
            test_passed = ("HIGH" in ml_risk) and (not crashed)
            actual_res = ml_risk
            pass_status = "PASS" if test_passed else "FAIL"
        else:
            test_passed = ("SAFE" in ml_risk or "LOW" in ml_risk) and (not crashed)
            actual_res = ml_risk
            pass_status = "PASS" if test_passed else "FAIL"

        s_size = str(inp.get("table_size_gb", "None"))
        s_qps = str(inp.get("query_frequency", "None"))
        s_type = str(inp.get("migration_type", "None"))
        s_load = str(inp.get("workload_intensity", "None"))

        results.append(
            {
                "case_id": case["id"],
                "name": case["name"],
                "category": case.get("category", "General Edge Case"),
                "description": case["description"],
                "inputs": inp,
                "input_summary": f"Size: {s_size} GB | QPS: {s_qps} | Type: {s_type} | Load: {s_load}",
                "baseline_prediction": base_outcome,
                "baseline_risk_level": base_risk,
                "ml_prediction": ml_risk,
                "ml_confidence": ml_conf,
                "ml_confidence_level": ml_conf_level,
                "ml_high_risk_prob": ml_high_prob,
                "is_out_of_distribution": is_ood,
                "expected_behavior": expected_val,
                "actual_result": actual_res,
                "test_status": pass_status,
                "domain_notes": case["domain_notes"],
                "system_crashed": crashed,
            }
        )

    return results

FAILURE_REASONS = [
    {
        "category": "Workload Spikes on Small Tables (Flash Crowd Nuance)",
        "description": "Standard DBA intuition relies on table size as a proxy for safety. However, during flash sales or traffic surges (e.g. 4,900+ QPS), even a lightweight 3.5 GB table acquiring an exclusive lock blocks incoming transactions at a rate of 5,000 queries per second. Within 3 seconds, the connection pool hits 100% capacity, triggering application-wide cascading timeouts.",
    },
    {
        "category": "Heuristic Misjudgments on Off-Peak Operations",
        "description": "Static size rules automatically reject massive table rewrites (e.g. 850 GB) even if executed during dedicated 60 QPS maintenance windows where zero lock queuing occurs. Machine learning avoids these costly false alarms by jointly weighing concurrency volume and table size.",
    },
    {
        "category": "Synthetic Data Distribution Boundaries",
        "description": "Synthetic generation employs parameterized distributions. Real enterprise production systems exhibit stochastic I/O saturation, dirty page writeback throttling, and autovacuum lock competition that synthetic distributions approximate.",
    },
    {
        "category": "Dynamic Lock Queue Jitter & Scheduling Variance",
        "description": "Relational lock acquisition is subject to operating system thread scheduling, client transaction hold times, and lock queue ordering, resulting in irreducible physical variance around statement timeout thresholds.",
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

    tn_mask = (y_true == 0) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    tn = int(np.sum(tn_mask))
    tp = int(np.sum(tp_mask))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))

    fp_rate = float(fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fn_rate = float(fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

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
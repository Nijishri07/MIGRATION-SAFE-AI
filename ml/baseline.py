"""Rule-based baseline predictor for database migration lock risks."""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class RuleBasedMigrationPredictor:
    """Domain heuristic baseline method to assess migration lock risk and predict outcomes.

    Uses transparent operational DBA rules without training machine learning models:
    - High-impact DDL types ('TABLE_REWRITE', 'ALTER_COLUMN_TYPE', 'ADD_INDEX_LOCKING')
      on large tables (> 60 GB) with HIGH/CRITICAL workloads trigger HIGH risk (predicted failure).
    - Long estimated lock durations (> 20 seconds) in OLTP environments trigger HIGH risk.
    - Low-impact DDL ('ADD_INDEX_CONCURRENTLY', 'DROP_COLUMN') under LOW/MEDIUM workloads
      trigger LOW risk (predicted success).
    """

    HEAVY_MIGRATIONS = {"TABLE_REWRITE", "ALTER_COLUMN_TYPE", "ADD_INDEX_LOCKING"}
    HIGH_INTENSITY = {"HIGH", "CRITICAL"}

    BASE_LOCK_LOOKUP = {
        "ADD_INDEX_CONCURRENTLY": 0.5,
        "ADD_COLUMN_DEFAULT": 1.2,
        "DROP_COLUMN": 0.8,
        "ADD_INDEX_LOCKING": 14.0,
        "ALTER_COLUMN_TYPE": 35.0,
        "TABLE_REWRITE": 60.0,
    }

    WORKLOAD_FACTOR = {
        "LOW": 1.0,
        "MEDIUM": 1.5,
        "HIGH": 2.5,
        "CRITICAL": 4.0,
    }

    def predict_record(
        self,
        migration_type: str,
        table_size_gb: Optional[float] = None,
        workload_intensity: str = "MEDIUM",
        query_frequency: float = 1200.0,
        estimated_lock_duration: float = 10.0,
        table_size_mb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict lock risk and migration success for a single migration event."""
        if table_size_gb is None:
            if table_size_mb is not None:
                table_size_gb = float(table_size_mb) / 1024.0
            else:
                table_size_gb = 100.0
        else:
            table_size_gb = float(table_size_gb)

        is_heavy = migration_type in self.HEAVY_MIGRATIONS
        is_high_load = workload_intensity in self.HIGH_INTENSITY or query_frequency >= 2000
        is_large_table = table_size_gb >= 60.0
        is_high_est_lock = estimated_lock_duration >= 20.0

        # Rule evaluation
        reasons = []
        if is_heavy and is_large_table and is_high_load:
            risk_level = "CRITICAL"
            predicted_success = 0
            predicted_high_risk = 1
            reasons.append(f"Heavy DDL '{migration_type}' on large table ({table_size_gb:.1f}GB) under heavy traffic causes lock queue starvation.")
        elif is_heavy and is_large_table:
            risk_level = "HIGH"
            predicted_success = 0
            predicted_high_risk = 1
            reasons.append(f"Exclusive lock on large table ({table_size_gb:.1f}GB) risks exceeding transaction timeout.")
        elif is_high_est_lock and is_high_load:
            risk_level = "HIGH"
            predicted_success = 0
            predicted_high_risk = 1
            reasons.append(f"Estimated lock ({estimated_lock_duration:.1f}s) combined with high QPS ({query_frequency}) causes connection pool exhaustion.")
        elif is_heavy:
            risk_level = "MEDIUM"
            predicted_success = 1
            predicted_high_risk = 0
            reasons.append("Heavy DDL operation, but table size and workload are within tolerable limits.")
        else:
            risk_level = "LOW"
            predicted_success = 1
            predicted_high_risk = 0
            reasons.append(f"Safe non-blocking migration type '{migration_type}'.")

        # Baseline predicted lock duration using rule-based formula
        base = self.BASE_LOCK_LOOKUP.get(migration_type, 2.0)
        size_multiplier = max(0.2, (table_size_gb / 100.0) ** 0.7)
        workload_mult = self.WORKLOAD_FACTOR.get(workload_intensity, 1.2)
        qps_penalty = 1.0 + (query_frequency / 3000.0)
        predicted_lock = round(base * size_multiplier * workload_mult * qps_penalty, 2)

        return {
            "risk_level": risk_level,
            "predicted_success": predicted_success,
            "predicted_high_risk": predicted_high_risk,
            "predicted_lock_duration": predicted_lock,
            "reason": " ".join(reasons),
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rule-based predictions across an entire DataFrame."""
        results = []
        for _, row in df.iterrows():
            size_gb = row.get("table_size_gb")
            if size_gb is None or pd.isna(size_gb):
                size_mb = row.get("table_size_mb", 102400.0)
                size_gb = float(size_mb) / 1024.0
            else:
                size_gb = float(size_gb)

            pred = self.predict_record(
                migration_type=str(row.get("migration_type", "ADD_COLUMN_DEFAULT")),
                table_size_gb=size_gb,
                workload_intensity=str(row.get("workload_intensity", "MEDIUM")),
                query_frequency=float(row.get("query_frequency", 1200.0)),
                estimated_lock_duration=float(row.get("estimated_lock_duration", 10.0)),
            )
            results.append(pred)

        pred_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute real, dynamic evaluation and error metrics from the dataset."""
        eval_df = self.predict_dataframe(df)

        y_true = eval_df["migration_success"].to_numpy().astype(int)
        y_pred = eval_df["predicted_success"].to_numpy().astype(int)

        actual_duration = eval_df["actual_lock_duration"].to_numpy().astype(float)
        predicted_duration = eval_df["predicted_lock_duration"].to_numpy().astype(float)
        estimated_duration = eval_df["estimated_lock_duration"].to_numpy().astype(float)

        # Classification metrics
        total = len(y_true)
        correct = np.sum(y_true == y_pred)
        accuracy = float(correct / total) if total > 0 else 0.0

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rule_mae = float(np.mean(np.abs(actual_duration - predicted_duration)))
        rule_rmse = float(np.sqrt(np.mean((actual_duration - predicted_duration) ** 2)))

        est_mae = float(np.mean(np.abs(actual_duration - estimated_duration)))
        est_rmse = float(np.sqrt(np.mean((actual_duration - estimated_duration) ** 2)))

        return {
            "total_records": total,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            },
            "rule_duration_mae": round(rule_mae, 2),
            "rule_duration_rmse": round(rule_rmse, 2),
            "est_duration_mae": round(est_mae, 2),
            "est_duration_rmse": round(est_rmse, 2),
        }

    def evaluate_high_risk_classification(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate rule-based classification against target is_high_risk (1=high risk/fail, 0=safe).

        Used for side-by-side comparison on identical test sets with the ML model.
        """
        eval_df = self.predict_dataframe(df)
        y_true = (1 - eval_df["migration_success"]).to_numpy().astype(int)
        y_pred = eval_df["predicted_high_risk"].to_numpy().astype(int)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

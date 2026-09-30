"""Machine learning classification model for migration lock risk prediction."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .baseline import RuleBasedMigrationPredictor
except ImportError:
    from ml.baseline import RuleBasedMigrationPredictor

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "saved_model.joblib"

class MigrationRiskModel:
    """End-to-end Machine Learning pipeline to predict database migration lock risk."""

    NUMERIC_FEATURES = [
        "table_size_gb",
        "row_count",
        "query_frequency",
        "estimated_lock_duration",
    ]

    CATEGORICAL_FEATURES = [
        "migration_type",
        "workload_intensity",
        "query_type",
    ]

    FEATURE_DEFAULTS = {
        "table_size_gb": 100.0,
        "row_count": 10_000_000,
        "query_frequency": 1200.0,
        "estimated_lock_duration": 10.0,
        "migration_type": "ADD_COLUMN_DEFAULT",
        "workload_intensity": "MEDIUM",
        "query_type": "MIXED_OLTP",
    }

    VALID_MIGRATION_TYPES = {
        "ADD_INDEX_CONCURRENTLY",
        "ADD_COLUMN_DEFAULT",
        "DROP_COLUMN",
        "ADD_INDEX_LOCKING",
        "ALTER_COLUMN_TYPE",
        "TABLE_REWRITE",
    }

    VALID_WORKLOADS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    VALID_QUERY_TYPES = {"WRITE_HEAVY", "READ_HEAVY", "MIXED_OLTP", "ANALYTICAL_BATCH"}

    TRAINING_BOUNDS = {
        "table_size_gb_min": 0.5,
        "table_size_gb_max": 1000.0,
        "row_count_min": 50_000,
        "row_count_max": 120_000_000,
        "query_frequency_min": 20.0,
        "query_frequency_max": 6000.0,
        "estimated_lock_duration_max": 180.0,
    }

    def __init__(self, n_estimators: int = 50, max_depth: int = 8, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.training_results: Dict[str, Any] = {}
        self.test_df: Optional[pd.DataFrame] = None
        self.test_y_true: Optional[np.ndarray] = None
        self.test_y_pred: Optional[np.ndarray] = None
        self.test_y_prob: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
            ]
        )

        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """Train the Random Forest classifier and evaluate performance on unseen test data."""
        required_cols = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES + ["lock_risk"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in training dataframe: {missing}")

        X = df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        y = df["lock_risk"].astype(int)

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)

        self.test_df = df.iloc[test_idx].reset_index(drop=True)
        self.test_y_true = y_test.to_numpy()
        self.test_y_pred = y_pred
        self.test_y_prob = y_prob

        ml_metrics = {
            "total_samples": len(y_test),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        ml_metrics["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        test_df = df.loc[test_idx].copy()
        baseline_predictor = RuleBasedMigrationPredictor()
        baseline_metrics = baseline_predictor.evaluate_high_risk_classification(test_df)

        comparison_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Rule-Based Baseline": [
                    round(baseline_metrics["accuracy"] * 100, 2),
                    round(baseline_metrics["precision"] * 100, 2),
                    round(baseline_metrics["recall"] * 100, 2),
                    round(baseline_metrics["f1_score"] * 100, 2),
                ],
                "Random Forest ML": [
                    round(ml_metrics["accuracy"] * 100, 2),
                    round(ml_metrics["precision"] * 100, 2),
                    round(ml_metrics["recall"] * 100, 2),
                    round(ml_metrics["f1_score"] * 100, 2),
                ],
            }
        )

        results = {
            "ml_metrics": ml_metrics,
            "baseline_metrics": baseline_metrics,
            "comparison_df": comparison_df,
        }
        self.training_results = results

        if save_model:
            self.save(str(MODEL_SAVE_PATH))

        return results

    def _check_uncertainty_and_bounds(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review 2 Feature: Check for missing data, anomalous values, and out-of-distribution conditions."""
        reasons = []
        is_ood = False

        raw_size = input_data.get("table_size_gb", input_data.get("table_size_mb"))
        if raw_size is None or pd.isna(raw_size):
            reasons.append("Table size was omitted or NaN; imputed from defaults.")
            is_ood = True
        else:
            try:
                val_size = float(raw_size)
                if val_size <= 0:
                    reasons.append(f"Negative or zero table size ({val_size} GB) is physically anomalous.")
                    is_ood = True
                elif val_size > self.TRAINING_BOUNDS["table_size_gb_max"]:
                    reasons.append(f"Table size ({val_size:.1f} GB) exceeds reliable training distribution bound (max {self.TRAINING_BOUNDS['table_size_gb_max']} GB).")
                    is_ood = True
            except (ValueError, TypeError):
                reasons.append("Invalid non-numeric table size format.")
                is_ood = True

        raw_qps = input_data.get("query_frequency")
        if raw_qps is None or pd.isna(raw_qps):
            reasons.append("Query frequency (QPS) was omitted or NaN; imputed from defaults.")
            is_ood = True
        else:
            try:
                val_qps = float(raw_qps)
                if val_qps < 0:
                    reasons.append(f"Negative query frequency ({val_qps} QPS) is anomalous.")
                    is_ood = True
                elif val_qps > self.TRAINING_BOUNDS["query_frequency_max"]:
                    reasons.append(f"Query frequency ({val_qps:.0f} QPS) exceeds reliable training distribution bound (max {self.TRAINING_BOUNDS['query_frequency_max']} QPS).")
                    is_ood = True
            except (ValueError, TypeError):
                reasons.append("Invalid non-numeric query frequency format.")
                is_ood = True

        raw_rows = input_data.get("row_count")
        if raw_rows is None or pd.isna(raw_rows):
            reasons.append("Row count was omitted; defaulted.")
            is_ood = True
        else:
            try:
                val_rows = int(raw_rows)
                if val_rows < 0:
                    reasons.append(f"Negative row count ({val_rows:,}) is invalid.")
                    is_ood = True
            except (ValueError, TypeError):
                reasons.append("Invalid non-integer row count.")
                is_ood = True

        raw_mtype = str(input_data.get("migration_type", "")).strip().upper()
        if not raw_mtype or raw_mtype not in self.VALID_MIGRATION_TYPES:
            reasons.append(f"Migration type '{raw_mtype}' is missing or unrecognised; defaulted.")
            is_ood = True

        raw_wint = str(input_data.get("workload_intensity", "")).strip().upper()
        if raw_wint and raw_wint not in self.VALID_WORKLOADS:
            reasons.append(f"Workload intensity '{raw_wint}' is non-standard.")
            is_ood = True

        requires_manual_review = is_ood
        return {
            "is_out_of_distribution": is_ood,
            "requires_manual_review": requires_manual_review,
            "reasons": reasons,
        }

    def _sanitize_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Defensively clean and validate input values to prevent crashes on edge cases."""
        sanitized = {}

        raw_size_gb = input_data.get("table_size_gb", None)
        if (raw_size_gb is None or pd.isna(raw_size_gb)) and "table_size_mb" in input_data:
            mb_val = input_data.get("table_size_mb")
            if mb_val is not None and not pd.isna(mb_val):
                try:
                    raw_size_gb = float(mb_val) / 1024.0
                except (ValueError, TypeError):
                    raw_size_gb = self.FEATURE_DEFAULTS["table_size_gb"]

        for feat in self.NUMERIC_FEATURES:
            if feat == "table_size_gb" and raw_size_gb is not None:
                val = raw_size_gb
            else:
                val = input_data.get(feat, None)

            if val is None or pd.isna(val):
                val = self.FEATURE_DEFAULTS[feat]
            else:
                try:
                    val = float(val)
                    if np.isnan(val) or np.isinf(val):
                        val = self.FEATURE_DEFAULTS[feat]
                    else:
                        val = max(0.01, val)
                except (ValueError, TypeError):
                    val = self.FEATURE_DEFAULTS[feat]
            sanitized[feat] = val

        sanitized["row_count"] = int(sanitized["row_count"])

        m_type = str(input_data.get("migration_type", "")).strip().upper()
        sanitized["migration_type"] = m_type if m_type in self.VALID_MIGRATION_TYPES else self.FEATURE_DEFAULTS["migration_type"]

        raw_est = input_data.get("estimated_lock_duration", None)
        if raw_est is None or (isinstance(raw_est, float) and (np.isnan(raw_est) or np.isinf(raw_est))):
            base_ddl_times = {
                "ADD_INDEX_CONCURRENTLY": 0.5,
                "ADD_COLUMN_DEFAULT": 1.2,
                "DROP_COLUMN": 0.8,
                "ADD_INDEX_LOCKING": 14.0,
                "ALTER_COLUMN_TYPE": 35.0,
                "TABLE_REWRITE": 60.0,
            }
            base_t = base_ddl_times.get(sanitized["migration_type"], 5.0)
            size_f = max(0.1, (sanitized["table_size_gb"] / 100.0) ** 0.7)
            sanitized["estimated_lock_duration"] = round(base_t * size_f, 2)

        w_int = str(input_data.get("workload_intensity", "")).strip().upper()
        sanitized["workload_intensity"] = w_int if w_int in self.VALID_WORKLOADS else self.FEATURE_DEFAULTS["workload_intensity"]

        q_type = str(input_data.get("query_type", "")).strip().upper()
        sanitized["query_type"] = q_type if q_type in self.VALID_QUERY_TYPES else self.FEATURE_DEFAULTS["query_type"]

        return pd.DataFrame([sanitized])

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict lock risk, confidence score, and evaluate model uncertainty / OOD bounds."""
        if not self.is_trained:
            if MODEL_SAVE_PATH.exists():
                self.load(str(MODEL_SAVE_PATH))
            else:
                raise RuntimeError("Model must be trained before calling predict_single.")

        ood_check = self._check_uncertainty_and_bounds(input_data)
        row_df = self._sanitize_input(input_data)
        pred = self.pipeline.predict(row_df)[0]
        probs = self.pipeline.predict_proba(row_df)[0]

        classes = list(self.pipeline.classes_)
        safe_idx = classes.index(0) if 0 in classes else 0
        high_risk_idx = classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)

        safe_prob = float(probs[safe_idx])
        high_risk_prob = float(probs[high_risk_idx])

        is_high_risk = bool(pred == 1)
        predicted_risk = "HIGH RISK" if is_high_risk else "LOW RISK (SAFE)"
        raw_confidence = float(max(safe_prob, high_risk_prob) * 100.0)

        if ood_check["is_out_of_distribution"]:
            confidence_level = "LOW / UNCERTAIN (OOD or Missing Inputs)"
            confidence_status = "MANUAL_REVIEW_REQUIRED"
            confidence_score = round(min(raw_confidence, 55.0), 2)
        elif raw_confidence >= 85.0:
            confidence_level = "HIGH CONFIDENCE"
            confidence_status = "RELIABLE"
            confidence_score = round(raw_confidence, 2)
        elif raw_confidence >= 65.0:
            confidence_level = "MODERATE CONFIDENCE"
            confidence_status = "ACCEPTABLE"
            confidence_score = round(raw_confidence, 2)
        else:
            confidence_level = "LOW CONFIDENCE"
            confidence_status = "BORDERLINE"
            confidence_score = round(raw_confidence, 2)

        return {
            "predicted_risk": predicted_risk,
            "is_high_risk": is_high_risk,
            "confidence_score": confidence_score,
            "raw_model_confidence": round(raw_confidence, 2),
            "confidence_level": confidence_level,
            "confidence_status": confidence_status,
            "is_out_of_distribution": ood_check["is_out_of_distribution"],
            "requires_manual_review": ood_check["requires_manual_review"],
            "uncertainty_reasons": ood_check["reasons"],
            "high_risk_probability": round(high_risk_prob * 100.0, 2),
            "safe_probability": round(safe_prob * 100.0, 2),
            "disclaimer": "The confidence score is a statistical model probability and NOT an operational guarantee. Staging rehearsal is required prior to production execution.",
        }

    def save(self, filepath: Optional[str] = None) -> str:
        """Persist trained pipeline to disk."""
        target_path = Path(filepath) if filepath else MODEL_SAVE_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target_path)
        return str(target_path)

    def load(self, filepath: Optional[str] = None) -> None:
        """Load trained pipeline from disk."""
        target_path = Path(filepath) if filepath else MODEL_SAVE_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at {target_path}")
        self.pipeline = joblib.load(target_path)
        self.is_trained = True

if __name__ == "__main__":
    from data.generator import generate_migration_dataset, save_migration_dataset
    csv_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_migration_data.csv"
    if csv_path.exists():
        print(f"[MigrationSafe AI] Loading dataset from {csv_path}...")
        data = pd.read_csv(csv_path)
    else:
        print("[MigrationSafe AI] Generating fresh dataset...")
        data = generate_migration_dataset(n_records=5000, random_state=42)
        save_migration_dataset(data, str(csv_path))

    model = MigrationRiskModel(n_estimators=50, max_depth=8, random_state=42)
    print(f"[MigrationSafe AI] Training Random Forest model on {len(data):,} records...")
    res = model.train(data, test_size=0.2, save_model=True)
    print("\n=== Model Evaluation on 1,000 Unseen Test Records ===")
    print(res["comparison_df"].to_string(index=False))
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
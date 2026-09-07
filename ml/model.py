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

    # Defaults for defensive input sanitization
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

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Build clean scikit-learn pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL_FEATURES),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.is_trained = False
        self.training_results: Optional[Dict[str, Any]] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.test_y_true: Optional[np.ndarray] = None
        self.test_y_pred: Optional[np.ndarray] = None
        self.test_y_prob: Optional[np.ndarray] = None

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """Train Random Forest classifier exclusively on train split and evaluate on unseen test data."""
        if "migration_success" not in df.columns:
            raise ValueError("Dataframe must contain 'migration_success' column.")

        # Target: 1 = High Lock Risk / Failure, 0 = Low Lock Risk / Safe
        y = (1 - df["migration_success"]).to_numpy().astype(int)
        X = df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES].copy()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X,
            y,
            df.index,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        # Train ONLY on training set
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on unseen test data
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)

        self.test_df = df.loc[test_idx].copy()
        self.test_y_true = y_test
        self.test_y_pred = y_pred
        self.test_y_prob = y_prob

        ml_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
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

        # Evaluate Phase 2 baseline on the exact same unseen test subset for fair comparison
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

    def _sanitize_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Defensively clean and validate input values to prevent crashes on edge cases."""
        sanitized = {}

        # If table_size_gb not provided or None, check table_size_mb
        raw_size_gb = input_data.get("table_size_gb", None)
        if (raw_size_gb is None or pd.isna(raw_size_gb)) and "table_size_mb" in input_data:
            mb_val = input_data.get("table_size_mb")
            if mb_val is not None and not pd.isna(mb_val):
                try:
                    raw_size_gb = float(mb_val) / 1024.0
                except (ValueError, TypeError):
                    raw_size_gb = self.FEATURE_DEFAULTS["table_size_gb"]

        # Numeric features with bounds clipping and NaN replacement
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
                        # Non-negative numbers
                        val = max(0.01, val)
                except (ValueError, TypeError):
                    val = self.FEATURE_DEFAULTS[feat]
            sanitized[feat] = val

        # Ensure row_count is int
        sanitized["row_count"] = int(sanitized["row_count"])

        # Categorical features with fallback for unknown categories
        m_type = str(input_data.get("migration_type", "")).strip().upper()
        sanitized["migration_type"] = m_type if m_type in self.VALID_MIGRATION_TYPES else self.FEATURE_DEFAULTS["migration_type"]

        # If estimated_lock_duration was missing or NaN, dynamically compute realistic heuristic fallback
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
        """Predict lock risk and confidence probability score with defensive input sanitation."""
        if not self.is_trained:
            if MODEL_SAVE_PATH.exists():
                self.load(str(MODEL_SAVE_PATH))
            else:
                raise RuntimeError("Model must be trained before calling predict_single.")

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
        confidence_score = float(max(safe_prob, high_risk_prob) * 100.0)

        return {
            "predicted_risk": predicted_risk,
            "is_high_risk": is_high_risk,
            "confidence_score": round(confidence_score, 2),
            "high_risk_probability": round(high_risk_prob * 100.0, 2),
            "safe_probability": round(safe_prob * 100.0, 2),
            "disclaimer": "The confidence score is a model probability and NOT an operational guarantee.",
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


"""MigrationSafe AI - Complete End-to-End Capstone Dashboard.

Unified 9-Tab Streamlit Dashboard for High-Volume Order Database Migration Safety.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.generator import generate_migration_dataset, save_migration_dataset
from ml.baseline import RuleBasedMigrationPredictor
from ml.model import MigrationRiskModel, MODEL_SAVE_PATH
from ml.edge_cases import (
    EDGE_CASES,
    run_edge_case_analysis,
    analyze_test_failures,
    FAILURE_REASONS,
)
from ml.simulator import TableState, MigrationSimulator
from experiments.benchmark import (
    MigrationBenchmark,
    BENCHMARK_SAVE_PATH,
    BENCHMARK_CSV_PATH,
)


DATA_PRIMARY_PATH = Path(__file__).resolve().parent / "data" / "synthetic_migration_data.csv"
DATA_LEGACY_PATH = Path(__file__).resolve().parent / "data" / "migration_records.csv"


@st.cache_data
def load_or_create_data(n_records: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Load synthetic migration dataset from CSV or generate deterministically if absent."""
    target_path = DATA_PRIMARY_PATH if DATA_PRIMARY_PATH.exists() else DATA_LEGACY_PATH
    if target_path.exists():
        try:
            df = pd.read_csv(target_path)
            if len(df) >= n_records and "lock_risk" in df.columns and "table_size_mb" in df.columns:
                return df
        except Exception:
            pass

    df = generate_migration_dataset(n_records=n_records, random_state=random_state)
    save_migration_dataset(df, str(DATA_PRIMARY_PATH))
    return df


@st.cache_resource
def load_or_train_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[MigrationRiskModel, Dict[str, Any]]:
    """Initialize, train, or load the Random Forest ML pipeline."""
    model = MigrationRiskModel(n_estimators=50, max_depth=8, random_state=random_state)
    train_results = model.train(df, test_size=0.2, save_model=True)
    return model, train_results


def main():
    st.set_page_config(
        page_title="MigrationSafe AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -------------------------------------------------------------
    # Sidebar: System Controls & Configuration
    # -------------------------------------------------------------
    st.sidebar.title("🛡️ MigrationSafe AI")
    st.sidebar.caption("High-Volume Order Database Migration Safety")
    st.sidebar.divider()

    st.sidebar.subheader("Dataset Configuration")
    dataset_size = st.sidebar.slider(
        "Synthetic Records Count",
        min_value=5000,
        max_value=15000,
        value=5000,
        step=1000,
        help="Generate at least 5,000 synthetic migration records for training and evaluation.",
    )
    seed = st.sidebar.number_input("Deterministic Random Seed", value=42, step=1)

    if st.sidebar.button("Regenerate Dataset & Retrain", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        with st.spinner("Generating fresh synthetic dataset and retraining ML pipeline..."):
            fresh_df = generate_migration_dataset(n_records=dataset_size, random_state=int(seed))
            save_migration_dataset(fresh_df, str(DATA_PRIMARY_PATH))
        st.sidebar.success(f"Generated & saved {len(fresh_df):,} records!")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Local Execution Principles**:\n"
        "- 100% Local (Zero External APIs/DBs)\n"
        "- Synthetic Order DB Data Only\n"
        "- Scikit-learn + Domain Heuristics\n"
        "- Verified Automated Rollback"
    )

    # Load core data, baseline predictor, and ML model
    df = load_or_create_data(n_records=dataset_size, random_state=int(seed))
    baseline_predictor = RuleBasedMigrationPredictor()
    ml_model, training_results = load_or_train_model(df, random_state=int(seed))

    # -------------------------------------------------------------
    # Main Header
    # -------------------------------------------------------------
    st.title("🛡️ MigrationSafe AI")
    st.markdown(
        "**AI-Powered Pre-Flight Risk Prediction, Concurrency Lock Simulation, and Rollback Protection "
        "for Mission-Critical Order Processing Databases.**"
    )

    # Navigation Tabs (All 9 Required Sections)
    tabs = st.tabs([
        "📋 Overview",
        "🔮 Risk Prediction",
        "⚖️ Baseline vs ML",
        "⚠️ Edge Cases & Failure Analysis",
        "🚀 Migration Simulation",
        "🔄 Rollback Demonstration",
        "📊 Benchmark Results",
        "🛡️ Ethics & Privacy",
        "📖 Deployment Guide",
    ])

    # =============================================================
    # TAB 1: OVERVIEW
    # =============================================================
    with tabs[0]:
        st.header("📋 Project Overview & Architecture")
        st.markdown(
            "In high-throughput e-commerce systems, database migrations against active order tables "
            "(`orders`, `order_items`, `order_payments`) can trigger catastrophic production outages. "
            "Exclusive DDL locks block incoming transactional queries, exhaust database connection pools, "
            "and result in client timeouts. **MigrationSafe AI** introduces pre-flight risk evaluation, "
            "workload-aware lock modeling, and guaranteed stateful rollback."
        )

        st.info(
            "💡 **The Problem:** Traditional rule-based heuristics only look at static table size or operation type. "
            "They fail to predict lock queue starvation caused by high concurrency bursts, or falsely alarm on large tables "
            "executing during quiet maintenance windows. MigrationSafe AI bridges this gap with machine learning."
        )

        # High-level KPIs
        st.subheader("Synthetic Order Database Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_records = len(df)
        success_count = int(df["migration_success"].sum())
        safe_rate = (success_count / total_records) * 100.0 if total_records > 0 else 0.0
        avg_table_size_gb = float(df["table_size_gb"].mean())
        avg_lock_s = float(df["actual_lock_duration"].mean())

        kpi1.metric("Total Records", f"{total_records:,}")
        kpi2.metric("Safe Migrations", f"{safe_rate:.1f}%")
        kpi3.metric("Avg Table Size", f"{avg_table_size_gb:.1f} GB")
        kpi4.metric("Avg Actual Lock", f"{avg_lock_s:.2f} s")

        st.divider()

        # Interactive Data Explorer
        st.subheader("Filter & Explore Synthetic Records")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            sel_tables = st.multiselect(
                "Tables",
                options=sorted(df["table_name"].unique()),
                default=sorted(df["table_name"].unique()),
            )
        with fcol2:
            sel_types = st.multiselect(
                "Migration Types",
                options=sorted(df["migration_type"].unique()),
                default=sorted(df["migration_type"].unique()),
            )
        with fcol3:
            sel_loads = st.multiselect(
                "Workload Intensity",
                options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                default=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            )

        filtered = df[
            df["table_name"].isin(sel_tables)
            & df["migration_type"].isin(sel_types)
            & df["workload_intensity"].isin(sel_loads)
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            column_config={
                "table_size_mb": st.column_config.NumberColumn("Size (MB)", format="%.1f MB"),
                "table_size_gb": st.column_config.NumberColumn("Size (GB)", format="%.1f GB"),
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
                "query_frequency": st.column_config.NumberColumn("Active QPS", format="%d"),
                "estimated_lock_duration": st.column_config.NumberColumn("Est Lock (s)", format="%.2f s"),
                "actual_lock_duration": st.column_config.NumberColumn("Actual Lock (s)", format="%.2f s"),
                "migration_success": st.column_config.CheckboxColumn("Success (Safe)"),
                "lock_risk": st.column_config.NumberColumn("Lock Risk (1=High)"),
            },
        )
        st.caption(f"Displaying {len(filtered):,} matching records from `{DATA_PRIMARY_PATH.name}`.")

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=csv_bytes,
            file_name="synthetic_migration_data.csv",
            mime="text/csv",
        )

    # =============================================================
    # TAB 2: RISK PREDICTION & CONFIDENCE
    # =============================================================
    with tabs[1]:
        st.header("🔮 Pre-Flight Migration Risk & Confidence")
        st.markdown(
            "Enter proposed migration parameters below to evaluate lock risk and model confidence probability."
        )

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            in_table = st.selectbox(
                "Target Order Table",
                options=["orders", "order_items", "order_payments", "order_shipments", "customer_sessions"],
                index=0,
            )
            in_mtype = st.selectbox(
                "Migration DDL Type",
                options=[
                    "ADD_INDEX_CONCURRENTLY",
                    "ADD_COLUMN_DEFAULT",
                    "DROP_COLUMN",
                    "ADD_INDEX_LOCKING",
                    "ALTER_COLUMN_TYPE",
                    "TABLE_REWRITE",
                ],
                index=1,
            )
            size_unit = st.radio("Size Unit", ["GB", "MB"], horizontal=True)
            if size_unit == "GB":
                in_size_gb = st.number_input("Table Size (GB)", min_value=0.1, max_value=2000.0, value=120.0, step=10.0)
                in_size_mb = in_size_gb * 1024.0
            else:
                in_size_mb = st.number_input("Table Size (MB)", min_value=100.0, max_value=2000000.0, value=122880.0, step=1000.0)
                in_size_gb = in_size_mb / 1024.0

            in_rows = st.number_input(
                "Row Count",
                min_value=1000,
                max_value=200_000_000,
                value=15_000_000,
                step=500_000,
            )

        with p_col2:
            in_workload = st.selectbox(
                "Live Workload Intensity",
                options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                index=1,
            )
            in_qps = st.slider(
                "Active Query Frequency (QPS)",
                min_value=50,
                max_value=6000,
                value=1200,
                step=50,
            )
            in_qtype = st.selectbox(
                "Dominant Query Workload Type",
                options=["MIXED_OLTP", "WRITE_HEAVY", "READ_HEAVY", "ANALYTICAL_BATCH"],
                index=0,
            )
            est_default_lookup = {
                "ADD_INDEX_CONCURRENTLY": 0.5,
                "ADD_COLUMN_DEFAULT": 1.2,
                "DROP_COLUMN": 0.8,
                "ADD_INDEX_LOCKING": 14.0,
                "ALTER_COLUMN_TYPE": 35.0,
                "TABLE_REWRITE": 60.0,
            }
            suggested_lock = round(est_default_lookup.get(in_mtype, 5.0) * max(0.1, (in_size_gb / 100.0) ** 0.7), 1)
            in_est_lock = st.number_input(
                "Developer Estimated Lock Duration (seconds)",
                min_value=0.1,
                max_value=300.0,
                value=float(suggested_lock),
                step=0.5,
                help=f"Baseline formula heuristic for {in_mtype} on {in_size_gb:.1f} GB is ~{suggested_lock}s.",
            )
            st.caption(f"💡 *Baseline heuristic estimate for {in_mtype} on {in_size_gb:.1f} GB: ~{suggested_lock}s.*")

        if st.button("🚀 Evaluate Migration Risk", type="primary", use_container_width=True):

            input_payload = {
                "table_size_gb": in_size_gb,
                "table_size_mb": in_size_mb,
                "row_count": in_rows,
                "query_frequency": in_qps,
                "estimated_lock_duration": in_est_lock,
                "migration_type": in_mtype,
                "workload_intensity": in_workload,
                "query_type": in_qtype,
            }

            # Machine Learning Prediction
            ml_pred = ml_model.predict_single(input_payload)

            # Baseline Prediction for Comparison
            base_pred = baseline_predictor.predict_record(
                migration_type=in_mtype,
                table_size_gb=in_size_gb,
                workload_intensity=in_workload,
                query_frequency=in_qps,
                estimated_lock_duration=in_est_lock,
            )

            st.divider()
            st.subheader("Prediction Results")

            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                if ml_pred["is_high_risk"]:
                    st.error(f"### ML Assessment: {ml_pred['predicted_risk']}")
                else:
                    st.success(f"### ML Assessment: {ml_pred['predicted_risk']}")
                st.caption("Evaluated by trained Random Forest classification model.")

            with r_col2:
                conf = ml_pred["confidence_score"]
                st.metric("Model Confidence Score", f"{conf:.1f}%")
                st.progress(conf / 100.0)

            with r_col3:
                base_outcome = "SAFE" if base_pred["predicted_success"] == 1 else "HIGH RISK"
                st.metric(
                    "Rule-Based Baseline",
                    f"{base_pred['risk_level']} ({base_outcome})",
                )
                st.caption(f"Rule Predicted Lock: {base_pred['predicted_lock_duration']:.2f}s")

            # Probability Breakdown
            st.markdown("#### Probability Distribution")
            p_df = pd.DataFrame(
                {
                    "Class": ["Safe Migration Probability", "High Lock Risk Probability"],
                    "Probability (%)": [ml_pred["safe_probability"], ml_pred["high_risk_probability"]],
                }
            )
            st.bar_chart(p_df.set_index("Class"), horizontal=True)

            # Operational Disclaimer
            st.warning(
                "⚠️ **Operational Disclaimer**: "
                f"{ml_pred['disclaimer']} "
                "Confidence scores represent model class posterior probabilities computed on synthetic data features. "
                "Always perform staging rehearsal with active production-like traffic before modifying schema."
            )

            with st.expander("View Heuristic Baseline Rationale"):
                st.write(f"**Baseline Rule Explanation:** {base_pred['reason']}")

    # =============================================================
    # TAB 3: BASELINE VS ML COMPARISON
    # =============================================================
    with tabs[2]:
        st.header("⚖️ Rule-Based Baseline vs. Machine Learning Model")
        st.markdown(
            "To evaluate the predictive capability of MigrationSafe AI, we compare the Scikit-learn **Random Forest Classifier** "
            "against the transparent **Rule-Based Baseline Predictor** on identical, unseen test split data (1,000 holdout records)."
        )

        ml_m = training_results["ml_metrics"]
        base_m = training_results["baseline_metrics"]
        comp_df = training_results["comparison_df"]

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ML Accuracy", f"{ml_m['accuracy'] * 100:.2f}%", f"{(ml_m['accuracy'] - base_m['accuracy']) * 100:+.2f}% vs Base")
        m2.metric("ML Precision", f"{ml_m['precision'] * 100:.2f}%", f"{(ml_m['precision'] - base_m['precision']) * 100:+.2f}% vs Base")
        m3.metric("ML Recall", f"{ml_m['recall'] * 100:.2f}%", f"{(ml_m['recall'] - base_m['recall']) * 100:+.2f}% vs Base")
        m4.metric("ML F1-Score", f"{ml_m['f1_score']:.4f}", f"{(ml_m['f1_score'] - base_m['f1_score']):+.4f} vs Base")

        st.divider()

        # Side-by-side Table & Chart
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.subheader("Performance Comparison Table")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.caption("Evaluated on 1,000 unseen test records with fixed random seed 42.")

        with c_right:
            st.subheader("Metric Comparison Chart")
            chart_df = comp_df.set_index("Metric")
            st.bar_chart(chart_df, use_container_width=True)

        st.divider()

        # Confusion Matrices
        st.subheader("Confusion Matrices (Holdout Test Set)")
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("#### Random Forest Classifier")
            ml_cm = ml_m["confusion_matrix"]
            ml_cm_df = pd.DataFrame(
                {
                    "Pred Safe": [ml_cm["true_negative"], ml_cm["false_negative"]],
                    "Pred High Risk": [ml_cm["false_positive"], ml_cm["true_positive"]],
                },
                index=["Actual Safe", "Actual High Risk"],
            )
            st.dataframe(ml_cm_df, use_container_width=True)

        with cm_col2:
            st.markdown("#### Rule-Based Baseline")
            base_cm_full = baseline_predictor.evaluate(df)["confusion_matrix"]
            base_cm_df = pd.DataFrame(
                {
                    "Pred Safe": [base_cm_full["true_positive"], base_cm_full["false_negative"]],
                    "Pred High Risk": [base_cm_full["false_positive"], base_cm_full["true_negative"]],
                },
                index=["Actual Safe", "Actual High Risk"],
            )
            st.dataframe(base_cm_df, use_container_width=True)

        st.markdown(
            "**Key Insight:** While the Rule-Based heuristic provides an intuitive starting point, it struggles with non-linear "
            "interactions—such as high concurrency on smaller tables, where queue starvation occurs rapidly, or large tables "
            "executing during off-peak hours where lock duration is high but contention is non-existent."
        )

    # =============================================================
    # TAB 4: EDGE CASES & FAILURE ANALYSIS
    # =============================================================
    with tabs[3]:
        st.header("⚠️ Edge Cases & Prediction Failure Analysis")
        st.markdown(
            "A production-grade migration safety system must maintain stability under extreme operational edge cases and "
            "provide transparent analysis of prediction failures without masking errors."
        )

        st.subheader("1. Realistic Edge-Case Test Scenarios")
        edge_results = run_edge_case_analysis(ml_model, baseline_predictor)

        for case in edge_results:
            with st.expander(f"📌 {case['case_id']}: {case['name']} — Status: {case['status']}", expanded=True):
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown(f"**Description:** {case['description']}")
                    st.markdown(f"**Inputs:** `{case['input_summary']}`")
                with ec2:
                    st.markdown(f"**Baseline Prediction:** `{case['baseline_prediction']}`")
                    st.markdown(f"**ML Prediction:** `{case['ml_prediction']}` ({case['ml_confidence']})")
                    st.markdown(f"**Agreement:** `{case['agreement']}`")
                with ec3:
                    st.markdown(f"**Expected Result:** {case['expected_result']}")
                    st.markdown(f"**Actual Result:** {case['actual_result']}")
                    st.markdown(f"**Domain Rationale:** {case['domain_notes']}")

        st.divider()

        # Dynamic Holdout Test Failure Analysis
        st.subheader("2. Dynamic Holdout Test Failure Statistics")
        failure_stats = analyze_test_failures(ml_model, df)

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Holdout Test Records", f"{failure_stats['total_test_samples']:,}")
        f2.metric("Overall Accuracy", f"{failure_stats['accuracy_pct']:.2f}%")
        f3.metric("Error / Failure Rate", f"{failure_stats['error_rate_pct']:.2f}%")
        f4.metric("Misclassified Records", f"{failure_stats['misclassified_count']}")

        f5, f6 = st.columns(2)
        f5.metric("False Positive Rate (Type I)", f"{failure_stats['false_positive_rate_pct']:.2f}%", help="Safe migrations flagged as High Risk (overly cautious)")
        f6.metric("False Negative Rate (Type II)", f"{failure_stats['false_negative_rate_pct']:.2f}%", help="High risk migrations predicted as Safe (dangerous)")

        if not failure_stats["failed_records_df"].empty:
            st.markdown("#### Sample Misclassified Test Records")
            st.dataframe(
                failure_stats["failed_records_df"].head(10),
                use_container_width=True,
            )
            st.caption("True vs Predicted outcomes for actual test instances misclassified by the model.")

        st.divider()

        # Root Cause Analysis
        st.subheader("3. Root Cause Analysis of Prediction Failures")
        for reason in FAILURE_REASONS:
            st.markdown(f"**• {reason['category']}:** {reason['description']}")

    # =============================================================
    # TAB 5: MIGRATION SIMULATION
    # =============================================================
    with tabs[4]:
        st.header("🚀 Simulated Schema Migration Workflow")
        st.markdown(
            "Execute a realistic local simulation of a database schema migration following the end-to-end safety lifecycle:\n\n"
            "$$\\mathbf{Prediction} \\longrightarrow \\mathbf{Migration\\ Simulation} \\longrightarrow \\mathbf{Migration\\ Result} \\longrightarrow \\mathbf{Rollback\\ if\\ required}$$"
        )

        st.info("ℹ️ **Local Simulation Only**: This engine executes entirely in memory. It does NOT connect to external or production databases.")

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_table = st.selectbox(
                "Table to Migrate",
                ["orders", "order_items", "order_payments", "order_shipments"],
                key="sim_table",
            )
            sim_mtype = st.selectbox(
                "Migration Operation",
                [
                    "ADD_INDEX_CONCURRENTLY",
                    "ADD_COLUMN_DEFAULT",
                    "DROP_COLUMN",
                    "ADD_INDEX_LOCKING",
                    "ALTER_COLUMN_TYPE",
                    "TABLE_REWRITE",
                ],
                index=1,
                key="sim_mtype",
            )
            sim_size = st.number_input("Table Size (GB)", min_value=1.0, max_value=1000.0, value=120.0, key="sim_size")

        with sim_col2:
            sim_workload = st.selectbox("Active Workload", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], index=1, key="sim_workload")
            sim_qps = st.slider("Active QPS", 100, 5000, 1500, key="sim_qps")
            est_default_lookup = {
                "ADD_INDEX_CONCURRENTLY": 0.5,
                "ADD_COLUMN_DEFAULT": 1.2,
                "DROP_COLUMN": 0.8,
                "ADD_INDEX_LOCKING": 14.0,
                "ALTER_COLUMN_TYPE": 35.0,
                "TABLE_REWRITE": 60.0,
            }
            sim_est_default = round(est_default_lookup.get(sim_mtype, 5.0) * max(0.1, (sim_size / 100.0) ** 0.7), 1)
            sim_est_lock = st.number_input(
                "Estimated Lock (seconds)",
                min_value=0.1,
                max_value=100.0,
                value=float(sim_est_default),
                key="sim_est_lock",
            )

        # Initialize workflow session state
        if "wf_snapshot" not in st.session_state:
            st.session_state.wf_snapshot = None
            st.session_state.wf_active_state = None
            st.session_state.wf_pred = None
            st.session_state.wf_sim_res = None
            st.session_state.wf_rollback_res = None

        if st.button("▶️ Execute Full Workflow (Prediction → Simulation → Result)", type="primary", use_container_width=True):
            simulator = MigrationSimulator()
            initial_state = simulator.create_initial_state(
                table_name=sim_table,
                table_size_gb=sim_size,
                row_count=int(sim_size * 125_000),
            )
            st.session_state.wf_snapshot = initial_state.clone()
            st.session_state.wf_active_state = initial_state.clone()
            st.session_state.wf_rollback_res = None

            # Step 1: Pre-Flight ML Prediction
            pred_payload = {
                "table_size_gb": sim_size,
                "table_size_mb": sim_size * 1024.0,
                "row_count": int(sim_size * 125_000),
                "query_frequency": sim_qps,
                "estimated_lock_duration": sim_est_lock,
                "migration_type": sim_mtype,
                "workload_intensity": sim_workload,
                "query_type": "WRITE_HEAVY" if sim_workload in ["HIGH", "CRITICAL"] else "MIXED_OLTP",
            }
            st.session_state.wf_pred = ml_model.predict_single(pred_payload)

            # Step 2: Run local simulation
            st.session_state.wf_sim_res = simulator.run_simulation(
                current_state=st.session_state.wf_active_state,
                migration_type=sim_mtype,
                workload_intensity=sim_workload,
                query_frequency=sim_qps,
                estimated_lock_duration=sim_est_lock,
            )

        # Render Workflow Stages if executed
        if st.session_state.wf_sim_res is not None:
            wf_pred = st.session_state.wf_pred
            sim_res = st.session_state.wf_sim_res

            st.divider()

            # --- STEP 1: PREDICTION ---
            st.subheader("1️⃣ Pre-Flight Risk Prediction")
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                if wf_pred["is_high_risk"]:
                    st.error(f"**ML Assessment:** {wf_pred['predicted_risk']}")
                else:
                    st.success(f"**ML Assessment:** {wf_pred['predicted_risk']}")
            with p_c2:
                st.metric("Model Confidence", f"{wf_pred['confidence_score']:.1f}%")
            with p_c3:
                st.metric("High Risk Probability", f"{wf_pred['high_risk_probability']:.1f}%")

            # --- STEP 2: MIGRATION SIMULATION ---
            st.subheader("2️⃣ Migration Simulation (Execution Progress)")
            st.caption(f"🏁 **Migration Started** on table `{sim_table}` with {sim_size:.1f} GB under {sim_qps:,} active QPS...")
            prog_bar = st.progress(100)
            for stage in sim_res["stages"]:
                st.markdown(f"**{stage['stage']}** ({stage['progress']}%) — {stage['detail']}")

            # --- STEP 3: MIGRATION RESULT ---
            st.subheader("3️⃣ Migration Result & Lock Contention Impact")
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            with r_c1:
                if sim_res["success"]:
                    st.success(f"### Status: {sim_res['status']}")
                else:
                    st.error(f"### Status: {sim_res['status']}")
            with r_c2:
                st.metric("Estimated Lock", f"{sim_res['estimated_lock_duration']:.2f} s")
            with r_c3:
                st.metric("Actual Lock Duration", f"{sim_res['actual_lock_duration']:.2f} s", f"Timeout: {sim_res['lock_timeout_threshold']:.1f}s")
            with r_c4:
                st.metric("Blocked Transactions", f"{sim_res['blocked_queries']:,}")

            if not sim_res["success"]:
                st.error(f"🚨 **Lock Watchdog Alert:** {sim_res['error_message']}")

            with st.expander("View Applied / Attempted DDL Statement", expanded=True):
                st.code(sim_res["applied_ddl"], language="sql")

            # --- STEP 4: ROLLBACK IF REQUIRED ---
            st.subheader("4️⃣ Rollback Action (If Required)")
            st.markdown(
                "If the migration triggered unacceptable lock starvation, timed out, or requires operational reversal, "
                "MigrationSafe AI can immediately restore the original schema state."
            )

            rb_col1, rb_col2 = st.columns([1, 1])
            with rb_col1:
                if st.button("⏪ Trigger Immediate Rollback", use_container_width=True):
                    simulator = MigrationSimulator()
                    st.session_state.wf_rollback_res = simulator.rollback_migration(
                        current_state=st.session_state.wf_active_state,
                        original_snapshot=st.session_state.wf_snapshot,
                    )

            if st.session_state.wf_rollback_res is not None:
                rb_res = st.session_state.wf_rollback_res
                with rb_col2:
                    st.success("✅ **Rollback Executed Successfully!**")
                    st.markdown(f"- **Restored Schema:** `{rb_res['restored_version']}`")
                    st.markdown(f"- **Exact Deep Match:** `{rb_res['is_exact_match']}`")
                    st.markdown(f"- **Status:** `{st.session_state.wf_active_state.status}`")
                st.markdown("**Generated Rollback DDL:**")
                st.code(rb_res["rollback_ddl"], language="sql")


    # =============================================================
    # TAB 6: ROLLBACK DEMONSTRATION
    # =============================================================
    with tabs[5]:
        st.header("🔄 Safe Stateful Rollback Demonstration")
        st.markdown(
            "When migrations fail or cause unacceptable contention, MigrationSafe AI supports stateful schema rollback. "
            "The system captures deep snapshots of schema version, columns, and index catalog state, and executes inverse DDL "
            "to restore the original state with verified exact equality."
        )

        simulator = MigrationSimulator()

        # Session state for rollback demonstration
        if "demo_initial_state" not in st.session_state:
            st.session_state.demo_initial_state = simulator.create_initial_state("orders", 120.0, 15_000_000)
            st.session_state.demo_active_state = st.session_state.demo_initial_state.clone()
            st.session_state.demo_rollback_res = None

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Step 1: Baseline Pre-Migration Snapshot")
            init_s = st.session_state.demo_initial_state
            st.markdown(f"**Table:** `{init_s.table_name}` | **Schema Version:** `{init_s.schema_version}`")
            st.markdown(f"**Status:** `{init_s.status}` | **Columns Count:** `{len(init_s.columns)}` | **Indexes:** `{len(init_s.indexes)}`")
            st.dataframe(pd.DataFrame(init_s.columns), use_container_width=True)

            st.divider()

            st.subheader("Step 2: Apply Migration Action")
            demo_action = st.selectbox(
                "Choose Migration Action to Test",
                [
                    "ADD_COLUMN_DEFAULT (Adds is_loyalty_order column)",
                    "ADD_INDEX_CONCURRENTLY (Adds idx_orders_status_amount)",
                    "ALTER_COLUMN_TYPE (Changes amount precision)",
                ],
            )
            action_key = demo_action.split(" ")[0]

            if st.button("Apply Migration to Table", use_container_width=True):
                active_s = st.session_state.demo_initial_state.clone()
                sim_out = simulator.run_simulation(
                    current_state=active_s,
                    migration_type=action_key,
                    workload_intensity="LOW",
                    query_frequency=200.0,
                    estimated_lock_duration=1.0,
                )
                st.session_state.demo_active_state = active_s
                st.session_state.demo_rollback_res = None
                st.success(f"Applied {action_key}! Schema updated to {active_s.schema_version}.")

        with col_right:
            st.subheader("Step 3: Current Live Schema State")
            cur_s = st.session_state.demo_active_state
            st.markdown(f"**Table:** `{cur_s.table_name}` | **Schema Version:** `{cur_s.schema_version}`")
            st.markdown(f"**Status:** `{cur_s.status}` | **Columns Count:** `{len(cur_s.columns)}` | **Indexes:** `{len(cur_s.indexes)}`")
            st.dataframe(pd.DataFrame(cur_s.columns), use_container_width=True)

            st.divider()

            st.subheader("Step 4 & 5: Trigger Safe Rollback")
            if st.button("⏪ Execute Automated Rollback", type="secondary", use_container_width=True):
                r_out = simulator.rollback_migration(
                    current_state=st.session_state.demo_active_state,
                    original_snapshot=st.session_state.demo_initial_state,
                )
                st.session_state.demo_rollback_res = r_out

            if st.session_state.demo_rollback_res:
                r_out = st.session_state.demo_rollback_res
                st.success("✅ **Rollback Completed Successfully!**")
                st.markdown(f"**Restored Version:** `{r_out['restored_version']}`")
                st.markdown(f"**Exact Deep Schema Match:** `{r_out['is_exact_match']}`")
                st.markdown("**Generated Rollback DDL:**")
                st.code(r_out["rollback_ddl"], language="sql")

    # =============================================================
    # TAB 7: BENCHMARK RESULTS
    # =============================================================
    with tabs[6]:
        st.header("📊 Benchmark Experiments & Downtime Avoided")
        st.markdown(
            "To quantitatively measure MigrationSafe AI's effectiveness, we run an automated benchmark across **150 synthetic migration scenarios**. "
            "We compare **Approach A (Simple Baseline - Unprotected Direct Execution)** against **Approach B (MigrationSafe AI - Pre-flight Interception & Off-peak Mitigation)**."
        )

        benchmark = MigrationBenchmark()
        loaded_res = benchmark.load_results(str(BENCHMARK_SAVE_PATH))

        if loaded_res is None or st.button("🔄 Rerun Benchmark Experiment (150 Scenarios)", use_container_width=True):
            with st.spinner("Running 150 scenario benchmark evaluation..."):
                loaded_res = benchmark.run_benchmark(
                    df=df,
                    model=ml_model,
                    n_scenarios=150,
                    random_state=int(seed),
                    save_results=True,
                )
            st.success("Benchmark completed and saved locally!")

        # Key Comparative Metrics
        st.subheader("Measurable Impact & Downtime Avoided")
        b_kpi1, b_kpi2, b_kpi3, b_kpi4 = st.columns(4)

        base_dt = loaded_res["baseline"]["total_downtime_seconds"]
        ai_dt = loaded_res["migrationsafe_ai"]["total_downtime_seconds"]
        avoided_s = loaded_res["comparative"]["downtime_avoided_seconds"]
        avoided_pct = loaded_res["comparative"]["downtime_avoided_pct"]

        b_kpi1.metric("Baseline Total Downtime", f"{base_dt:,.1f} s")
        b_kpi2.metric("MigrationSafe AI Downtime", f"{ai_dt:,.1f} s")
        b_kpi3.metric("Downtime Avoided", f"{avoided_s:,.1f} s")
        b_kpi4.metric("Downtime Avoided %", f"{avoided_pct:.1f}%")

        b_kpi5, b_kpi6, b_kpi7, b_kpi8 = st.columns(4)
        base_sr = loaded_res["baseline"]["success_rate_pct"]
        ai_sr = loaded_res["migrationsafe_ai"]["success_rate_pct"]
        risky_det = loaded_res["migrationsafe_ai"]["risky_migrations_detected"]
        total_scenarios = loaded_res["total_scenarios"]

        b_kpi5.metric("Baseline Success Rate", f"{base_sr:.1f}%")
        b_kpi6.metric("MigrationSafe Success Rate", f"{ai_sr:.1f}%", f"+{ai_sr - base_sr:.1f}% Gain")
        b_kpi7.metric("Risky Migrations Intercepted", f"{risky_det} / {total_scenarios}")
        b_kpi8.metric("SLA Targets Met", "✅ PASSED" if loaded_res["targets"]["target_downtime_met"] else "❌ FAILED")

        st.divider()

        # Comparison Charts using Matplotlib
        st.subheader("Visual Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Chart 1: Total Downtime
        approaches = ["Simple Baseline", "MigrationSafe AI"]
        downtimes = [base_dt, ai_dt]
        colors = ["#e74c3c", "#2ecc71"]
        ax1.bar(approaches, downtimes, color=colors, width=0.5)
        ax1.set_ylabel("Total Downtime (Seconds)")
        ax1.set_title(f"Cumulative Downtime Comparison\n({avoided_pct:.1f}% Downtime Avoided)")
        for i, v in enumerate(downtimes):
            ax1.text(i, v + max(downtimes) * 0.02, f"{v:,.1f}s", ha="center", fontweight="bold")

        # Chart 2: Success Rate
        success_rates = [base_sr, ai_sr]
        ax2.bar(approaches, success_rates, color=["#f39c12", "#3498db"], width=0.5)
        ax2.set_ylabel("Migration Success Rate (%)")
        ax2.set_ylim(0, 110)
        ax2.set_title("Success Rate Improvement")
        for i, v in enumerate(success_rates):
            ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # Scenarios Summary Table
        st.subheader("Scenario Execution Log")
        scenarios_df = pd.DataFrame(loaded_res["scenarios_summary"])
        st.dataframe(scenarios_df, use_container_width=True)

        # Download CSV button
        if BENCHMARK_CSV_PATH.exists():
            with open(BENCHMARK_CSV_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Benchmark Results (CSV)",
                    data=f.read(),
                    file_name="benchmark_results.csv",
                    mime="text/csv",
                )

    # =============================================================
    # TAB 8: ETHICS & PRIVACY
    # =============================================================
    with tabs[7]:
        st.header("🛡️ Ethics, Privacy, and Model Limitations")
        st.markdown(
            "Deploying artificial intelligence systems in production infrastructure introduces ethical, "
            "reliability, and governance considerations."
        )

        st.subheader("1. Synthetic Data & Zero Real Customer PII")
        st.markdown(
            "- **No Real Customer Data**: All order transactions, customer identifiers, amounts, and database parameters "
            "are synthetically generated using deterministic pseudo-random generators.\n"
            "- **Zero PII Storage**: The system contains no personally identifiable information (PII), payment credentials, "
            "or proprietary commercial data.\n"
            "- **Compliance**: Fully compatible with strict data governance policies (GDPR, CCPA, PCI-DSS) by design."
        )

        st.subheader("2. AI as a Decision-Support Advisory Tool")
        st.markdown(
            "- **Not an Autonomous Executor**: MigrationSafe AI provides risk advisories and lock estimates to support human Database "
            "Administrators (DBAs) and Site Reliability Engineers (SREs). It does not autonomously execute uncontrolled DDL.\n"
            "- **Probabilistic Nature**: Model confidence scores are statistical class probabilities based on synthetic distributions, "
            "NOT an operational guarantee. Unforeseen infrastructure conditions (I/O saturation, network split-brain) can impact lock times."
        )

        st.subheader("3. Model Limitations & Real Production Differences")
        st.markdown(
            "- **Distributional Shift**: Real production workloads feature long-tail traffic spikes, multi-region replication lags, "
            "and dirty cache evictions that simplified synthetic datasets cannot completely capture.\n"
            "- **Mandatory Staging Rehearsal**: Pre-flight risk scores must be augmented with pre-production dry-runs on read-only clones "
            "or isolated staging clusters prior to major physical schema changes."
        )

    # =============================================================
    # TAB 9: DEPLOYMENT GUIDE
    # =============================================================
    with tabs[8]:
        st.header("📖 Local Deployment & Architecture Guide")
        st.markdown(
            "This guide provides quick-start commands and architectural references for running MigrationSafe AI locally."
        )

        st.subheader("1. Prerequisites & Installation")
        st.code(
            "# 1. Clone or navigate to the project directory\n"
            "cd MigrationSafeAI\n\n"
            "# 2. Install required Python packages (Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib)\n"
            "pip install -r requirements.txt\n\n"
            "# 3. Launch the Streamlit application\n"
            "streamlit run app.py\n\n"
            "# 4. Open web browser\n"
            "# Navigate to: http://localhost:8501",
            language="bash",
        )

        st.subheader("2. Automated Test Suite Execution")
        st.code(
            "# Run all 32 comprehensive tests across all modules:\n"
            "python -m unittest discover -s tests -p \"test_*.py\"\n\n"
            "# Run the primary unified app test suite:\n"
            "python -m unittest tests/test_app.py",
            language="bash",
        )

        st.subheader("3. Production Best Practices for Zero-Downtime Schema Changes")
        st.markdown(
            "1. **Use Non-Blocking DDL:** Always prefer `ADD INDEX CONCURRENTLY` over standard index creation.\n"
            "2. **Separate Constraint Validation:** Add foreign keys with `NOT VALID`, then validate in a separate background phase.\n"
            "3. **Short Statement Timeouts:** Always set explicit statement lock timeouts (e.g. `SET lock_timeout = '3s';`) to prevent queue piles.\n"
            "4. **Shadow Table Patterns:** For column type alterations on multi-gigabyte tables, use shadow column / shadow table replication."
        )


if __name__ == "__main__":
    main()

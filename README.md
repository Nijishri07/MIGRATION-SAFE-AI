# MigrationSafe AI

**Pre-Flight Risk Prediction, Concurrency Lock Simulation, and Rollback Protection for High-Volume Order Database Migrations**

---

## 1. Project Title & Overview
**MigrationSafe AI** is an intelligent database operations safety framework engineered to prevent catastrophic production downtime during schema migrations on high-volume transactional databases (e.g., e-commerce order management systems).

---

## 2. Problem Statement
In mission-critical order databases handling thousands of concurrent queries per second (QPS) and tens of millions of records, standard Data Definition Language (DDL) migrations—such as physical table rewrites, column alterations, and index creations—acquire exclusive table-level locks (`AccessExclusiveLock`). 

When an exclusive lock request enters the lock queue:
1. Incoming read and write transactions back up behind the DDL request.
2. The database connection pool is rapidly saturated.
3. Client applications experience cascading timeouts, latency spikes, and severe operational outages.

Traditional DBA rule heuristics (e.g., assessing only table disk size or migration type) fail to model dynamic concurrency surges or generate false alarms for quiet maintenance periods. MigrationSafe AI solves this by combining machine learning risk scoring, transparent rule heuristics, simulated migration workflows, and verified stateful rollback.

---

## 3. Objectives
- **Accurate Lock Risk Prediction:** Predict whether a prospective DDL migration will succeed or trip statement lock timeouts under concurrent query loads.
- **Probabilistic Confidence Assessment:** Output calibrated confidence probability scores for every prediction with explicit operational disclaimers.
- **Transparent Heuristic Baseline Comparison:** Benchmark machine learning predictions against clear DBA domain rules on identical holdout test data.
- **Extreme Edge-Case Robustness:** Ensure defensive handling and zero crashes across missing values, flash-crowd bursts, and multi-terabyte tables.
- **Dynamic Failure Analysis:** Quantify real test set error rates, false positives, and false negatives without fabricated metrics.
- **Safe Simulated Migration & Rollback:** Provide realistic in-memory simulation of lock impact and stateful restoration of database schemas.
- **Quantifiable Downtime Avoidance:** Measure downtime avoided and success rate gains over >= 100 benchmark scenarios.

---

## 4. Architecture & Workflow
The system operates as an end-to-end safety gatekeeper:

```
[ Prospective Migration Request ]
                │
                ▼
┌───────────────────────────────────────┐
│       1. Input Data Sanitization      │
│  (Defensive bounds & NaN handling)    │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│     2. Dual-Engine Risk Scoring       │
│  ┌─────────────────┐ ┌──────────────┐ │
│  │ ML Random Forest│ │ Rule Baseline│ │
│  └────────┬────────┘ └──────┬───────┘ │
└───────────┼─────────────────┼─────────┘
            │                 │
            ▼                 ▼
┌───────────────────────────────────────┐
│    3. Pre-Flight Risk Assessment      │
│  • Predicted Risk: HIGH / SAFE        │
│  • Confidence Probability Score (%)   │
│  • Estimated Contention & Lock Window │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│     4. Local Migration Simulation     │
│  • Lock Acquisition -> DDL Mod ->     │
│    Validation -> Lock Release         │
└──────────────────┬────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 [ Lock Timeout ]     [ Safe Completion ]
        │                     │
        ▼                     ▼
┌───────────────┐     ┌───────────────┐
│   Rollback    │     │  Live Commit  │
│  Protection   │     │ Schema v1.5.0 │
│ Schema v1.4.0 │     └───────────────┘
└───────────────┘
```

---

## 5. Dataset Description
The system utilizes a realistic, deterministically reproducible synthetic dataset of **5,000+ migration events** stored at `data/synthetic_migration_data.csv`.

### Key Features:
- `migration_id`: Unique migration identifier (`MIG-10000`...)
- `table_name`: Primary order schema tables (`orders`, `order_items`, `order_payments`, `order_shipments`, `customer_sessions`)
- `table_size_mb` / `table_size_gb`: Physical table footprint on disk
- `row_count`: Table volume (ranging up to 100M+ rows)
- `query_frequency`: Active concurrent transactions per second (QPS)
- `query_type`: Workload profile (`WRITE_HEAVY`, `READ_HEAVY`, `MIXED_OLTP`, `ANALYTICAL_BATCH`)
- `migration_type`: DDL operation (`ADD_INDEX_CONCURRENTLY`, `ADD_COLUMN_DEFAULT`, `DROP_COLUMN`, `ADD_INDEX_LOCKING`, `ALTER_COLUMN_TYPE`, `TABLE_REWRITE`)
- `workload_intensity`: Concurrency category (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- `estimated_lock_duration`: Developer / query planner baseline estimation (seconds)
- `actual_lock_duration`: Realistic actual lock duration with concurrency contention spikes (seconds)
- `migration_success`: Binary target (1 = Safe completion within 25.0s timeout, 0 = Lock timeout)
- `lock_risk`: Binary risk classification target (1 = High Risk / Timeout, 0 = Safe)

---

## 6. Baseline Method
The rule-based baseline (`ml/baseline.py`) models domain DBA decision heuristics:
- Flags **CRITICAL RISK** when heavy DDL (`TABLE_REWRITE`, `ALTER_COLUMN_TYPE`, `ADD_INDEX_LOCKING`) is applied to tables >= 60 GB under `HIGH` or `CRITICAL` workloads.
- Flags **HIGH RISK** when estimated lock duration exceeds 20.0s under active QPS >= 2,000.
- Predicts **LOW RISK** for non-blocking concurrent DDL (`ADD_INDEX_CONCURRENTLY`, `DROP_COLUMN`) during low-to-medium traffic.
- Evaluated on holdout data to produce true Accuracy, Precision, Recall, and F1 metrics.

---

## 7. Machine Learning Method
The machine learning pipeline (`ml/model.py`) implements a Scikit-learn classification architecture:
- **Preprocessing:** `ColumnTransformer` with `StandardScaler` for numeric features (`table_size_gb`, `row_count`, `query_frequency`, `estimated_lock_duration`) and `OneHotEncoder(handle_unknown='ignore')` for categorical dimensions.
- **Classifier:** `RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)`.
- **Validation Protocol:** Strict 80/20 train/test split. The model is fitted exclusively on training data (4,000 samples) and evaluated on unseen holdout test data (1,000 samples).
- **Metric Computation:** Dynamically computes true Accuracy, Precision, Recall, F1-score, and full Confusion Matrix.

---

## 8. Prediction Workflow & Confidence Scoring
1. User provides or inputs prospective migration parameters in the dashboard.
2. Input values are defensively cleaned and sanitized (handling None, negative values, and out-of-bounds metrics).
3. The ML model outputs:
   - **Predicted Risk:** `HIGH RISK` or `LOW RISK (SAFE)`
   - **Confidence Score:** Model class posterior probability (`max(P(safe), P(high_risk)) * 100%`)
   - **Probability Distribution:** Exact probability percentages for both outcomes.
4. **Operational Disclaimer:** The system prominently alerts users that confidence scores represent statistical model probabilities and do not replace staging rehearsals.

---

## 9. Edge Cases & Failure Analysis
Five rigorous operational edge cases are validated in `ml/edge_cases.py`:
1. **Massive Table with Low Concurrency (`EDGE-1`):** 850 GB table rewrite running at 60 QPS during off-peak maintenance. ML correctly distinguishes zero queue pileup where static size heuristics fail.
2. **Micro Table under Flash-Crowd Traffic (`EDGE-2`):** 3.5 GB table locking under 4,900 QPS flash sale. Momentary exclusive locks cause severe queue pileup.
3. **Heavy Workload with Risky DDL (`EDGE-3`):** 380 GB table rewrite under 3,800 QPS critical traffic. Unanimous consensus to abort.
4. **Missing & Anomalous Values (`EDGE-4`):** Inputs with NaN, negative row counts, and empty categories are safely imputed with median/mode fallbacks without crashing.
5. **Hyper-Scale Scale Test (`EDGE-5`):** 5,000 GB, 500M rows, 15,000 QPS evaluated cleanly without numerical overflow.

### Dynamic Failure Root Causes:
- **Unusual Workload Patterns:** Flash sales on small tables.
- **Synthetic Data Limitations:** Bounded parametric variances vs long-tail real traffic.
- **Unseen Feature Combinations:** Rare intersections of massive tables and low concurrency.
- **Noisy Contention Jitter:** Non-deterministic lock queue scheduling.

---

## 10. Migration Simulation
The in-memory simulator (`ml/simulator.py`) executes a 4-stage lifecycle:
1. **Phase 1: Lock Acquisition** — Simulates waiting for active in-flight transactions.
2. **Phase 2: DDL Modification & Table Scan** — Simulates physical heap modifications.
3. **Phase 3: Integrity Validation** — Verifies constraints, foreign keys, and catalog metadata.
4. **Phase 4: Lock Release & Commit** — Releases exclusive locks and commits catalog updates.

Calculates actual lock duration, queues blocked queries (`actual_lock_duration * QPS`), and trips at `25.0s` statement timeout.

---

## 11. Rollback Demonstration
When a migration fails or causes excessive lock starvation, stateful rollback restores the exact schema:
1. Captures pre-migration `TableState` snapshot (version `v1.4.0`, columns, indexes).
2. Applies simulated migration mutations.
3. On failure or rollback trigger, generates inverse DDL (`DROP COLUMN`, `DROP INDEX`).
4. Reinstates deep snapshot and performs verification assertion confirming exact match.

---

## 12. Benchmark Methodology & Downtime Avoided
The benchmark engine (`experiments/benchmark.py`) evaluates **150 representative migration scenarios**:
- **Approach A (Simple Baseline):** Unprotected execution. High-risk migrations encounter lock timeouts (25.0s+ downtime) and client query blockage.
- **Approach B (MigrationSafe AI):** Pre-flight AI detection intercepts high-risk DDL and reschedules them to off-peak low-QPS windows.
- **Measured Metrics:**
  - `Baseline Total Downtime` (seconds)
  - `MigrationSafe AI Total Downtime` (seconds)
  - `Downtime Avoided` = `Baseline Downtime - MigrationSafe AI Downtime`
  - `Downtime Avoided Percentage` = `(Downtime Avoided / Baseline Downtime) * 100%`
  - `Migration Success Rate` improvement.
- Results are saved to `experiments/benchmark_results.csv` and `experiments/benchmark_results.json`.

---

## 13. Actual Measured Results

| Metric | Rule-Based Baseline | Random Forest ML / MigrationSafe AI | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 92.90% | **98.80%** | **+5.90%** |
| **Precision** | 84.70% | **96.35%** | **+11.65%** |
| **Recall** | 89.47% | **99.25%** | **+9.78%** |
| **F1-Score** | 0.8702 | **0.9778** | **+0.1076** |
| **Benchmark Total Downtime** | 12,663.6 s | **1,303.2 s** | **11,360.4 s Avoided** |
| **Downtime Avoided %** | — | **89.7%** | **Target >= 75.0% Met** |
| **Migration Success Rate** | 70.7% | **100.0%** | **+29.3% Gain** |
| **Risky Migrations Intercepted** | — | **44 / 150** | **100% Mitigated** |

*(Metrics computed on holdout test splits and 150-scenario benchmark runs with fixed random seed 42).*

---

## 14. Limitations
1. **Synthetic Data Boundaries:** Real workloads feature hardware-level cache thrashing and disk I/O bottlenecks that synthetic distributions approximate.
2. **Stateless Prediction:** Does not account for concurrent external database batch jobs unless reflected in active QPS.
3. **Engine-Specific DDL:** Lock timeouts and catalog behavior are modeled around standard enterprise relational databases (e.g. PostgreSQL / MySQL InnoDB conventions).

---

## 15. Ethics & Privacy
- **100% Synthetic Data:** Zero customer PII, real credit card details, addresses, or commercial records.
- **Human-in-the-Loop:** Designed strictly as a decision-support advisory system for engineering teams, not an autonomous unsupervised executor.
- **Transparency:** All failure statistics, misclassifications, and baseline comparisons are visible without obfuscation.

---

## 16. Installation & Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

---

## 17. How to Run & Use the Dashboard

### 1. Launch Dashboard
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

### 2. Run Automated Test Suite
```bash
# Run all 32 tests:
python -m unittest discover -s tests -p "test_*.py"

# Run unified app tests:
python -m unittest tests/test_app.py
```

### 3. Dashboard Navigation
- **📋 Overview:** Explore synthetic order dataset with interactive filters and metrics.
- **🔮 Risk Prediction:** Input table parameters and inspect ML risk, confidence probability, and probability distributions.
- **⚖️ Baseline vs ML:** Inspect side-by-side performance metrics, comparison charts, and confusion matrices.
- **⚠️ Edge Cases:** Review realistic edge-case responses and examine holdout test set misclassification records.
- **🚀 Migration Simulation:** Run simulated DDL migrations and monitor 4-stage lock progression.
- **🔄 Rollback Demonstration:** Apply schema modifications and trigger exact snapshot state rollback.
- **📊 Benchmark Results:** Review 150-scenario benchmark results, download CSV, and inspect downtime avoided.
- **🛡️ Ethics & Privacy:** Review governance principles, PII guarantees, and limitations.
- **📖 Deployment Guide:** Access production best practices and deployment guidelines.
#   M I G R A T I O N - S A F E - A I  
 
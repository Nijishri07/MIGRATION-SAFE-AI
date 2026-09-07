"""Synthetic data generation for high-volume order database migrations."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def generate_migration_dataset(
    n_records: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic records for high-volume order database migrations.

    Parameters
    ----------
    n_records : int, default=5000
        Number of migration event records (must be at least 5000).
    random_state : int, default=42
        Seed for reproducible random generation.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with all required migration and database workload features.
    """
    if n_records < 5000:
        n_records = 5000

    rng = np.random.default_rng(random_state)

    # 1. Database table schemas in high-volume order processing
    table_pool = [
        {"name": "orders", "weight": 0.30, "base_gb": 120.0, "base_rows": 15_000_000},
        {"name": "order_items", "weight": 0.35, "base_gb": 280.0, "base_rows": 45_000_000},
        {"name": "order_payments", "weight": 0.15, "base_gb": 65.0, "base_rows": 16_000_000},
        {"name": "order_shipments", "weight": 0.12, "base_gb": 45.0, "base_rows": 12_000_000},
        {"name": "customer_sessions", "weight": 0.08, "base_gb": 35.0, "base_rows": 8_000_000},
    ]

    table_choices = rng.choice(
        [t["name"] for t in table_pool],
        size=n_records,
        p=[t["weight"] for t in table_pool],
    )

    table_lookup = {t["name"]: t for t in table_pool}

    # Generate table size and row count scaled with realistic variance
    table_sizes_gb = []
    row_counts = []
    for t_name in table_choices:
        base = table_lookup[t_name]
        scale_factor = rng.uniform(0.4, 2.2)
        size_gb = np.round(base["base_gb"] * scale_factor, 2)
        rows = int(base["base_rows"] * scale_factor)
        table_sizes_gb.append(size_gb)
        row_counts.append(rows)

    table_sizes_gb = np.array(table_sizes_gb)
    row_counts = np.array(row_counts)

    # 2. Query workload features
    query_types = rng.choice(
        ["WRITE_HEAVY", "READ_HEAVY", "MIXED_OLTP", "ANALYTICAL_BATCH"],
        size=n_records,
        p=[0.35, 0.30, 0.25, 0.10],
    )

    workload_intensities = rng.choice(
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        size=n_records,
        p=[0.20, 0.40, 0.25, 0.15],
    )

    workload_qps_map = {
        "LOW": (100, 500),
        "MEDIUM": (500, 1500),
        "HIGH": (1500, 3000),
        "CRITICAL": (3000, 5500),
    }

    query_frequencies = np.array(
        [
            rng.integers(workload_qps_map[w][0], workload_qps_map[w][1])
            for w in workload_intensities
        ]
    )

    # 3. Migration types
    migration_types = rng.choice(
        [
            "ADD_INDEX_CONCURRENTLY",
            "ADD_COLUMN_DEFAULT",
            "DROP_COLUMN",
            "ADD_INDEX_LOCKING",
            "ALTER_COLUMN_TYPE",
            "TABLE_REWRITE",
        ],
        size=n_records,
        p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08],
    )

    # 4. Realistic lock durations modeling
    # Base estimated lock duration (developer / query planner rule-of-thumb in seconds)
    migration_base_lock_seconds = {
        "ADD_INDEX_CONCURRENTLY": 0.5,    # Short share update lock
        "ADD_COLUMN_DEFAULT": 1.2,        # Fast metadata lock in modern DBs, slightly higher on large tables
        "DROP_COLUMN": 0.8,               # Quick schema catalog update
        "ADD_INDEX_LOCKING": 14.0,        # Full shared table lock
        "ALTER_COLUMN_TYPE": 35.0,        # AccessExclusive table lock, rewrites data
        "TABLE_REWRITE": 60.0,            # Full AccessExclusive lock for table rebuild
    }

    workload_multiplier = {
        "LOW": 1.0,
        "MEDIUM": 1.35,
        "HIGH": 2.1,
        "CRITICAL": 3.4,
    }

    query_type_lock_bias = {
        "READ_HEAVY": 1.05,
        "WRITE_HEAVY": 1.45,
        "MIXED_OLTP": 1.25,
        "ANALYTICAL_BATCH": 1.60,
    }

    estimated_locks = []
    actual_locks = []
    migration_success = []

    for i in range(n_records):
        m_type = migration_types[i]
        t_size = table_sizes_gb[i]
        w_int = workload_intensities[i]
        q_type = query_types[i]
        qps = query_frequencies[i]

        base_time = migration_base_lock_seconds[m_type]
        size_factor = (t_size / 100.0) ** 0.7

        # Developer estimation: typically accounts for table size and base operation type
        est_lock = np.round(base_time * size_factor * rng.uniform(0.85, 1.25), 2)
        # Minimum baseline lock is 0.1s
        est_lock = max(0.1, est_lock)
        estimated_locks.append(est_lock)

        # Actual lock duration: heavily impacted by active concurrency, lock queues, and table size
        concurrency_penalty = 1.0 + (qps / 2500.0) * workload_multiplier[w_int] * query_type_lock_bias[q_type]
        lock_noise = rng.normal(loc=1.0, scale=0.20)
        lock_noise = max(0.6, lock_noise)

        # If operation requires AccessExclusiveLock under heavy load, queue saturation can occur
        is_heavy_lock = m_type in ["TABLE_REWRITE", "ALTER_COLUMN_TYPE", "ADD_INDEX_LOCKING"]
        if is_heavy_lock and w_int in ["HIGH", "CRITICAL"]:
            contention_spike = rng.uniform(1.8, 3.8)
        else:
            contention_spike = 1.0

        act_lock = np.round(base_time * size_factor * concurrency_penalty * lock_noise * contention_spike, 2)
        act_lock = max(0.1, act_lock)
        actual_locks.append(act_lock)

        # Migration success threshold:
        # Migration fails if lock exceeds safe lock_timeout (e.g. 30 seconds for OLTP, or 60s for batch)
        # or if critical contention causes lock acquisition timeout
        lock_timeout_threshold = 45.0 if q_type == "ANALYTICAL_BATCH" else 25.0
        if act_lock > lock_timeout_threshold:
            success = 0
        elif is_heavy_lock and w_int == "CRITICAL" and rng.random() < 0.35:
            # Deadlock or client connection starvation failure
            success = 0
        else:
            success = 1
        migration_success.append(success)

    lock_risks = [1 - s for s in migration_success]
    table_sizes_mb = np.round(table_sizes_gb * 1024.0, 2)

    df = pd.DataFrame(
        {
            "migration_id": [f"MIG-{10000 + i}" for i in range(n_records)],
            "table_name": table_choices,
            "table_size_mb": table_sizes_mb,
            "table_size_gb": table_sizes_gb,
            "row_count": row_counts,
            "query_frequency": query_frequencies,
            "query_type": query_types,
            "migration_type": migration_types,
            "workload_intensity": workload_intensities,
            "estimated_lock_duration": estimated_locks,
            "actual_lock_duration": actual_locks,
            "migration_success": migration_success,
            "lock_risk": lock_risks,
        }
    )

    return df


def save_migration_dataset(
    df: pd.DataFrame,
    filepath: Optional[str] = None,
) -> str:
    """Save migration records dataframe to CSV format in the data folder."""
    data_dir = Path(__file__).resolve().parent
    primary_path = data_dir / "synthetic_migration_data.csv"
    legacy_path = data_dir / "migration_records.csv"

    if filepath is None:
        target_path = primary_path
    else:
        target_path = Path(filepath)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)

    # Also keep synthetic_migration_data.csv and migration_records.csv in sync
    if target_path != primary_path and not primary_path.exists():
        df.to_csv(primary_path, index=False)
    if target_path != legacy_path:
        df.to_csv(legacy_path, index=False)

    return str(target_path)


# Backward compatibility for Phase 1 skeleton tests
def generate_synthetic_orders(
    n_samples: int = 500,
    corruption_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Retained for Phase 1 backwards compatibility."""
    rng = np.random.default_rng(random_state)
    order_ids = [f"ORD-{100000 + i}" for i in range(n_samples)]
    customer_ids = [f"CUST-{rng.integers(1000, 9999)}" for _ in range(n_samples)]
    amounts = np.round(rng.lognormal(mean=3.5, sigma=0.8, size=n_samples), 2)
    statuses = rng.choice(
        ["COMPLETED", "PENDING", "PROCESSING", "CANCELLED"],
        size=n_samples,
        p=[0.70, 0.15, 0.10, 0.05],
    )
    batch_ids = rng.integers(1, 11, size=n_samples)
    latencies_ms = np.round(rng.normal(loc=45.0, scale=12.0, size=n_samples).clip(5.0, 500.0), 2)

    n_corrupted = int(n_samples * corruption_rate)
    corrupted_indices = set(rng.choice(n_samples, size=n_corrupted, replace=False))
    source_checksums = [f"CHK-{rng.integers(100000, 999999)}" for _ in range(n_samples)]
    target_checksums = []
    is_corrupted = []

    for idx, s_chk in enumerate(source_checksums):
        if idx in corrupted_indices:
            target_checksums.append(f"CHK-MISMATCH-{rng.integers(10000, 99999)}")
            is_corrupted.append(1)
        else:
            target_checksums.append(s_chk)
            is_corrupted.append(0)

    return pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "amount": amounts,
            "order_status": statuses,
            "batch_id": batch_ids,
            "latency_ms": latencies_ms,
            "source_checksum": source_checksums,
            "target_checksum": target_checksums,
            "is_corrupted": is_corrupted,
        }
    )


if __name__ == "__main__":
    print("[MigrationSafe AI] Generating 5,000 synthetic migration records...")
    dataset = generate_migration_dataset(n_records=5000, random_state=42)
    saved_path = save_migration_dataset(dataset)
    print(f"[MigrationSafe AI] Successfully saved {len(dataset):,} records to: {saved_path}")
    print(f"  - Table footprint: {dataset['table_size_gb'].mean():.1f} GB avg")
    print(f"  - Safe migrations: {dataset['migration_success'].mean() * 100:.1f}%")
    print(f"  - High risk migrations: {dataset['lock_risk'].mean() * 100:.1f}%")


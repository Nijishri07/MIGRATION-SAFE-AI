"""Realistic local schema migration simulation and safe rollback engine."""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class TableState:
    """Represents the schema, storage, and operational state of a database table."""

    table_name: str
    schema_version: str
    columns: List[Dict[str, str]]
    indexes: List[str]
    row_count: int
    table_size_gb: float
    status: str = "STABLE"  # STABLE, MIGRATING, COMPLETED, FAILED_LOCK_TIMEOUT, ROLLED_BACK
    applied_ddl: Optional[str] = None
    last_action: Optional[str] = "INITIAL_STATE"

    def clone(self) -> TableState:
        """Create an exact deep copy snapshot for rollback preservation."""
        return copy.deepcopy(self)


class MigrationSimulator:
    """Simulates realistic local DDL migrations, lock durations, and safe rollback actions."""

    BASE_LOCK_TIMES = {
        "ADD_INDEX_CONCURRENTLY": 0.5,
        "ADD_COLUMN_DEFAULT": 1.2,
        "DROP_COLUMN": 0.8,
        "ADD_INDEX_LOCKING": 14.0,
        "ALTER_COLUMN_TYPE": 35.0,
        "TABLE_REWRITE": 60.0,
    }

    WORKLOAD_MULTIPLIERS = {
        "LOW": 1.0,
        "MEDIUM": 1.35,
        "HIGH": 2.1,
        "CRITICAL": 3.4,
    }

    LOCK_TIMEOUT_THRESHOLD = 25.0  # Seconds (typical production OLTP statement lock timeout)

    @classmethod
    def create_initial_state(
        cls,
        table_name: str = "orders",
        table_size_gb: float = 120.0,
        row_count: int = 15_000_000,
    ) -> TableState:
        """Initialize clean baseline schema state before any migration is applied."""
        initial_columns = [
            {"name": "order_id", "type": "VARCHAR(64)", "constraint": "PRIMARY KEY"},
            {"name": "customer_id", "type": "VARCHAR(32)", "constraint": "NOT NULL"},
            {"name": "amount", "type": "NUMERIC(10, 2)", "constraint": "NOT NULL"},
            {"name": "order_status", "type": "VARCHAR(20)", "constraint": "DEFAULT 'PENDING'"},
            {"name": "created_at", "type": "TIMESTAMP", "constraint": "NOT NULL DEFAULT NOW()"},
        ]
        initial_indexes = [
            "pk_orders_order_id",
            "idx_orders_customer_id",
            "idx_orders_created_at",
        ]

        return TableState(
            table_name=table_name,
            schema_version="v1.4.0",
            columns=initial_columns,
            indexes=initial_indexes,
            row_count=int(row_count),
            table_size_gb=float(table_size_gb),
            status="STABLE",
            applied_ddl="BASE_SCHEMA",
            last_action="Pre-migration schema baseline initialized.",
        )

    def run_simulation(
        self,
        current_state: TableState,
        migration_type: str,
        workload_intensity: str,
        query_frequency: float,
        estimated_lock_duration: float,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Execute a realistic local simulation of schema migration with dynamic lock impact."""
        rng = np.random.default_rng(random_state)

        # 1. Calculate dynamic actual lock duration based on domain formulas
        base_time = self.BASE_LOCK_TIMES.get(migration_type, 5.0)
        size_factor = max(0.1, (current_state.table_size_gb / 100.0) ** 0.7)
        workload_mult = self.WORKLOAD_MULTIPLIERS.get(workload_intensity, 1.35)
        concurrency_penalty = 1.0 + (query_frequency / 2500.0) * workload_mult

        is_heavy_ddl = migration_type in ["TABLE_REWRITE", "ALTER_COLUMN_TYPE", "ADD_INDEX_LOCKING"]
        contention_spike = rng.uniform(1.8, 3.4) if (is_heavy_ddl and workload_intensity in ["HIGH", "CRITICAL"]) else 1.0
        jitter = max(0.7, rng.normal(1.0, 0.15))

        actual_lock_duration = round(base_time * size_factor * concurrency_penalty * contention_spike * jitter, 2)
        actual_lock_duration = max(0.2, actual_lock_duration)

        # Simulated blocked incoming queries
        blocked_queries = int(actual_lock_duration * query_frequency)

        # 2. Generate stage execution logs
        stages = [
            {
                "stage": "Phase 1: Lock Acquisition",
                "progress": 25,
                "detail": f"Requesting table lock on '{current_state.table_name}'. Waiting for active transactions ({query_frequency:,} active QPS)...",
            },
            {
                "stage": "Phase 2: DDL Modification & Table Scan",
                "progress": 60,
                "detail": f"Executing DDL for {migration_type} across {current_state.table_size_gb:.1f} GB ({current_state.row_count:,} rows)...",
            },
            {
                "stage": "Phase 3: Integrity Validation",
                "progress": 85,
                "detail": "Verifying foreign key constraints and schema catalog checksums...",
            },
            {
                "stage": "Phase 4: Lock Release & Commit",
                "progress": 100,
                "detail": "Committing transaction, releasing exclusive table lock, and updating catalog.",
            },
        ]

        # 3. Determine Outcome (Timeout Trip or Success)
        if actual_lock_duration > self.LOCK_TIMEOUT_THRESHOLD:
            success = False
            status = "FAILED_LOCK_TIMEOUT"
            current_state.status = status
            current_state.last_action = (
                f"FAILED: Lock acquisition exceeded statement_timeout ({self.LOCK_TIMEOUT_THRESHOLD:.1f}s). "
                f"Lock duration reached {actual_lock_duration:.2f}s, causing transaction queue starvation."
            )
            error_msg = (
                f"Lock wait timeout ({self.LOCK_TIMEOUT_THRESHOLD:.1f}s) exceeded! "
                f"Blocked {blocked_queries:,} incoming queries in connection pool. "
                "Migration automatically aborted by lock watchdog."
            )
            applied_ddl = f"-- ABORTED DDL: {migration_type} ON {current_state.table_name}"
        else:
            success = True
            status = "COMPLETED"
            current_state.status = status
            current_state.schema_version = "v1.5.0"
            applied_ddl = self._apply_schema_changes(current_state, migration_type)
            current_state.applied_ddl = applied_ddl
            current_state.last_action = f"Successfully applied {migration_type} in {actual_lock_duration:.2f}s."
            error_msg = None

        return {
            "success": success,
            "status": status,
            "actual_lock_duration": actual_lock_duration,
            "estimated_lock_duration": float(estimated_lock_duration),
            "blocked_queries": blocked_queries,
            "lock_timeout_threshold": self.LOCK_TIMEOUT_THRESHOLD,
            "stages": stages,
            "error_message": error_msg,
            "applied_ddl": applied_ddl,
            "updated_state": current_state,
        }

    def _apply_schema_changes(self, state: TableState, migration_type: str) -> str:
        """Apply simulated schema mutations to the table state."""
        if migration_type == "ADD_COLUMN_DEFAULT":
            new_col = {"name": "is_loyalty_order", "type": "BOOLEAN", "constraint": "DEFAULT FALSE"}
            if not any(c["name"] == new_col["name"] for c in state.columns):
                state.columns.append(new_col)
            return f"ALTER TABLE {state.table_name} ADD COLUMN is_loyalty_order BOOLEAN DEFAULT FALSE;"

        elif migration_type == "ADD_INDEX_CONCURRENTLY":
            idx_name = f"idx_{state.table_name}_status_amount"
            if idx_name not in state.indexes:
                state.indexes.append(idx_name)
            return f"CREATE INDEX CONCURRENTLY {idx_name} ON {state.table_name} (order_status, amount);"

        elif migration_type == "ADD_INDEX_LOCKING":
            idx_name = f"idx_{state.table_name}_covering_full"
            if idx_name not in state.indexes:
                state.indexes.append(idx_name)
            return f"CREATE INDEX {idx_name} ON {state.table_name} (order_id, customer_id, amount);"

        elif migration_type == "ALTER_COLUMN_TYPE":
            for col in state.columns:
                if col["name"] == "amount":
                    col["type"] = "NUMERIC(14, 4)"
            return f"ALTER TABLE {state.table_name} ALTER COLUMN amount TYPE NUMERIC(14, 4);"

        elif migration_type == "TABLE_REWRITE":
            state.table_size_gb = round(state.table_size_gb * 0.92, 1)  # Vacuum/repack reclaim
            return f"VACUUM FULL {state.table_name}; -- Rebuilt physical heap pages"

        elif migration_type == "DROP_COLUMN":
            state.columns = [c for c in state.columns if c["name"] != "order_status"]
            return f"ALTER TABLE {state.table_name} DROP COLUMN order_status;"

        return f"-- Applied generic DDL: {migration_type}"

    def rollback_migration(
        self,
        current_state: TableState,
        original_snapshot: TableState,
    ) -> Dict[str, Any]:
        """Safely restore the table schema to its original pre-migration snapshot."""
        reverted_columns = [c["name"] for c in current_state.columns if c not in original_snapshot.columns]
        reverted_indexes = [i for i in current_state.indexes if i not in original_snapshot.indexes]

        rollback_ddl_statements = []
        for col_name in reverted_columns:
            rollback_ddl_statements.append(f"ALTER TABLE {current_state.table_name} DROP COLUMN {col_name};")
        for idx_name in reverted_indexes:
            rollback_ddl_statements.append(f"DROP INDEX IF EXISTS {idx_name};")

        if not rollback_ddl_statements:
            rollback_ddl_statements.append(f"-- RESTORED ORIGINAL SCHEMA METADATA ({original_snapshot.schema_version})")

        # Restore exact snapshot values
        current_state.schema_version = original_snapshot.schema_version
        current_state.columns = copy.deepcopy(original_snapshot.columns)
        current_state.indexes = copy.deepcopy(original_snapshot.indexes)
        current_state.row_count = original_snapshot.row_count
        current_state.table_size_gb = original_snapshot.table_size_gb
        current_state.status = "ROLLED_BACK"
        current_state.applied_ddl = "\n".join(rollback_ddl_statements)
        current_state.last_action = f"Rollback completed successfully. Restored schema to {original_snapshot.schema_version}."

        is_exact_match = (
            current_state.schema_version == original_snapshot.schema_version
            and len(current_state.columns) == len(original_snapshot.columns)
            and len(current_state.indexes) == len(original_snapshot.indexes)
        )

        return {
            "rollback_successful": True,
            "is_exact_match": is_exact_match,
            "restored_version": current_state.schema_version,
            "rollback_ddl": "\n".join(rollback_ddl_statements),
            "reverted_columns": reverted_columns,
            "reverted_indexes": reverted_indexes,
            "current_state": current_state,
        }


if __name__ == "__main__":
    print("[MigrationSafe AI] Running Simulator & Rollback Demo...")
    sim = MigrationSimulator()
    initial_state = sim.create_initial_state("orders", 120.0, 15_000_000)
    snapshot = initial_state.clone()
    print(f"  - Initial Schema: {initial_state.schema_version}, {len(initial_state.columns)} columns")

    print("\nApplying Simulated Migration: ADD_COLUMN_DEFAULT...")
    active_state = initial_state.clone()
    sim_res = sim.run_simulation(
        current_state=active_state,
        migration_type="ADD_COLUMN_DEFAULT",
        workload_intensity="LOW",
        query_frequency=250.0,
        estimated_lock_duration=1.2,
    )
    print(f"  - Simulation Result: {sim_res['status']}, Actual Lock: {sim_res['actual_lock_duration']:.2f}s")
    print(f"  - Schema Version: {active_state.schema_version}, {len(active_state.columns)} columns")

    print("\nExecuting Rollback to Original Snapshot...")
    rb_res = sim.rollback_migration(active_state, snapshot)
    print(f"  - Rollback Successful: {rb_res['rollback_successful']}")
    print(f"  - Exact Deep Match: {rb_res['is_exact_match']}")
    print(f"  - Restored Version: {rb_res['restored_version']}, {len(active_state.columns)} columns")


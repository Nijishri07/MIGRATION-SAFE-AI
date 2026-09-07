"""Data generation and management module for MigrationSafe AI."""

from .generator import (
    generate_synthetic_orders,
    generate_migration_dataset,
    save_migration_dataset,
)

__all__ = [
    "generate_synthetic_orders",
    "generate_migration_dataset",
    "save_migration_dataset",
]

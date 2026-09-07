# MigrationSafe AI - Experiments Protocol

This directory tracks real experimentation logs and benchmarking runs for migration safety models and data validation algorithms.

## Experimentation Principles

1. **No Fake / Hardcoded Results**: All metrics, benchmark numbers, and validation scores must be generated dynamically from live executions on synthetic datasets.
2. **Reproducibility**: All experiments must specify fixed random seeds (`random_state`) and reproducible hyperparameter configurations.
3. **Tracking Schema**: Future experiment runs should record:
   - `run_id`: Unique run identifier
   - `timestamp`: UTC execution timestamp
   - `n_samples`: Size of synthetic transaction batch tested
   - `corruption_rate`: Injected anomaly/drift percentage
   - `model_parameters`: Classifier/anomaly detection hyperparameters
   - `actual_metrics`: True validation metrics computed dynamically on test splits (Accuracy, Precision, Recall, F1)

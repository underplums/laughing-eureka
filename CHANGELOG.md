## [0.1.0] - 2026-06-19
### Added
- MVP train preprocessing pipeline with DAC audience extraction, target labeling, feature loading, train/holdout split and temporary S3 storage.
- CatBoost train pipeline with Optuna tuning, StratifiedKFold validation, isotonic calibration, MLflow metrics, plots, artifacts and model registration.
- Inference preprocessing pipeline with DAC scoring audience extraction, feature loading, validation and temporary S3 storage.
- Inference pipeline with MLflow model loading by train event timestamp, S3 prediction output and data statistics for drift monitoring.
- README with project description, run commands, date logic and S3 storage layout.

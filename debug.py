from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import logging
import os
import shutil
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import cvm_ml_metrics.classification as mc
import cvm_model.functions as func
import cvm_model.optuna_tuning as ot
import cvm_model.utils as utils
from cvm_model.io import State
from cvm_model.parameters import (
    CALIBRATION,
    CALIBRATION_SIZE,
    DEFAULT_PARAMS,
    N_TRIALS,
    OPTUNA_TUNING,
    RANDOM_STATE,
    artifacts_dir,
    features,
    input_suffix,
    jira,
    model_type,
    project_name,
    score,
    target,
    template,
    threshold,
    train_data_stat_suffix,
)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _metric_name(prefix: str, name: str) -> str:
    return (
        f'{prefix}_{name}'
        .replace(' ', '_')
        .replace('-', '_')
        .replace('(', '')
        .replace(')', '')
    )


def evaluate_metrics(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    threshold_value: float,
    log: bool,
    prefix: str,
) -> Dict[str, Any]:
    """Calculate binary classification metrics with selected threshold."""
    y_true = df[target].to_numpy()
    y_score = df[score].to_numpy()
    y_pred = (y_score >= threshold_value).astype(int)
    ece, mce = mc.ece_mce_fast(y_true, y_score)
    sensitivity, specificity = mc.sensitivity_specificity(y_true, y_pred)

    metrics = {
        'ROC_AUC': roc_auc_score(y_true, y_score),
        'PR_AUC': mc.precision_recall_auc(y_true, y_score),
        'Log_Loss': log_loss(y_true, y_score),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Balanced_Accuracy': mc.balanced_accuracy(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'F05': fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        'F2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'Brier_Score': brier_score_loss(y_true, y_score),
        'Youden_J': mc.youden_j(y_true, y_pred),
        'Markedness': mc.markedness(y_true, y_score),
        'Lift': mc.lift(y_true, y_score),
        'Gini': mc.gini(y_true, y_score),
        'Kolmogorov_Smirnov_Statistic': mc.ks_stat_bin_class(y_true, y_score),
        'Expected_Calibration_Error_ECE': ece,
        'Maximum_Calibration_Error_MCE': mce,
        'Matthews_Corrcoef_MCC': matthews_corrcoef(y_true, y_pred),
        'Cohen_Kappa_Score': cohen_kappa_score(y_true, y_pred),
    }
    metrics = {_metric_name(prefix, name): float(value) for name, value in metrics.items()}

    if log:
        logging.info(f'Threshold: {threshold_value}')
        for name, value in metrics.items():
            logging.info(f'{name}: {value:.4f}')

    return {'model': model, 'threshold': threshold_value, 'metrics': metrics}


def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: List[str],
    plot: bool,
    model_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit CatBoost model and evaluate on validation dataset."""
    counter = Counter(df_train[target].to_numpy())
    class_weights = [counter[1] / counter[0] if counter[0] else 1.0, 1.0]
    plot_file = str(Path(artifacts_dir) / 'training_plot.html')

    model = CatBoostClassifier(
        **model_params,
        use_best_model=True,
        verbose=100,
        random_seed=RANDOM_STATE,
        custom_metric='BalancedAccuracy',
        eval_metric='PRAUC',
        class_weights=class_weights,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    model.fit(
        df_train[features],
        df_train[target],
        eval_set=(df_val[features], df_val[target]),
        use_best_model=True,
        plot=plot,
        plot_file=plot_file,
    )

    df_val = df_val.copy()
    df_val[score] = model.predict_proba(df_val[features])[:, 1]
    result = evaluate_metrics(model, df_val, threshold, plot, 'CV')
    result['best_iteration'] = model.get_best_iteration()
    return result


def train(event_timestamp: datetime) -> None:
    """Train CatBoost churn-from-DAC model and log results to MLflow."""
    logging.info('Start train')
    t_train = time.time()
    best_optuna_score = None

    # Set connections.
    state = State.from_env()
    session = state.spark.session
    event_date = event_timestamp.date().isoformat()
    os.environ.update(state.credentials.mlflow.environment)
    if state.credentials.mlflow.tracking_uri:
        mlflow.set_tracking_uri(state.credentials.mlflow.tracking_uri)
    client = MlflowClient()

    # Clean artifact directory.
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    for file in artifacts_path.rglob('*'):
        if file.is_file():
            file.unlink()

    try:
        # Resolve S3 input paths.
        input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix
        input_bucket = input_prefix.split('//')[1].split('/')[0]
        input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])
        input_path = template.format(bucket=input_bucket, prefix=input_prefix)

        # Load train and holdout datasets.
        try:
            df_train = session.read.parquet(f'{input_path}/train').toPandas()
            df_holdout = session.read.parquet(f'{input_path}/holdout').toPandas()
        except Exception:
            logging.exception(f'Failed to load datasets from {input_path}')
            raise

        logging.info(f'Train target rate: {df_train[target].mean():.6f}')
        logging.info(f'Holdout target rate: {df_holdout[target].mean():.6f}')

        # Tune CatBoost hyperparameters using Optuna.
        if OPTUNA_TUNING:
            logging.info('Start Optuna hyperparameters tuning')
            model_params, best_optuna_score = ot.tune_catboost_params(
                df=df_train,
                features=features,
                target=target,
                random_state=RANDOM_STATE,
                n_folds=5,
                n_trials=N_TRIALS,
            )
        else:
            model_params = DEFAULT_PARAMS

        # Run SKF CV with selected parameters.
        logging.info('Start CV with selected parameters')
        skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

        train_results = []
        for i, (train_idx, val_idx) in enumerate(
            skf.split(df_train[features], df_train[target]),
            start=1,
        ):
            logging.info(f'Training fold {i}...')
            fold_result = train_model(
                df_train.iloc[train_idx],
                df_train.iloc[val_idx],
                features,
                False,
                model_params,
            )
            train_results.append(fold_result)

        best_iteration_cv = float(np.round(np.mean([res['best_iteration'] for res in train_results]), 2))
        cv_metrics = pd.DataFrame([res['metrics'] for res in train_results]).mean().to_dict()
        logging.info(f'StratifiedKFold metrics: {cv_metrics}')

        # Keep holdout untouched: final early stopping/calibration uses only train data.
        if CALIBRATION:
            df_fit, df_val = train_test_split(
                df_train,
                test_size=CALIBRATION_SIZE,
                random_state=RANDOM_STATE,
                stratify=df_train[target],
            )
            df_calib = df_val
        else:
            df_fit, df_val = train_test_split(
                df_train,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=df_train[target],
            )
            df_calib = None

        # Train final model without using holdout as eval_set.
        logging.info('Start training final model')
        train_result_final = train_model(df_fit, df_val, features, True, model_params)
        base_model = train_result_final['model']
        model = base_model
        best_iteration_train = train_result_final['best_iteration']

        # Calibrate model on validation slice if enabled.
        if CALIBRATION:
            logging.info('Start model calibration')
            model = CalibratedClassifierCV(
                estimator=base_model,
                method='isotonic',
                cv='prefit',
            )
            model.fit(df_calib[features], df_calib[target])

        # Score train and holdout datasets for metrics.
        df_train = df_train.copy()
        df_holdout = df_holdout.copy()
        df_train[score] = model.predict_proba(df_train[features])[:, 1]
        df_holdout[score] = model.predict_proba(df_holdout[features])[:, 1]

        train_metrics = evaluate_metrics(model, df_train, threshold, True, 'Train')
        holdout_metrics = evaluate_metrics(model, df_holdout, threshold, True, 'Holdout')
        logging.info(f'Train metrics: {train_metrics["metrics"]}')
        logging.info(f'Holdout metrics: {holdout_metrics["metrics"]}')

        # Plot PR-AUC, ROC-AUC curves and Confusion Matrix.
        y_true = df_holdout[target]
        y_score = df_holdout[score]
        y_pred = y_score >= threshold

        func.plot_precision_recall_curve(y_true, y_score, artifacts_dir=artifacts_dir)
        func.plot_roc(y_true, y_score, artifacts_dir=artifacts_dir)
        func.plot_confusion_matrix(y_true, y_pred, artifacts_dir=artifacts_dir)

        # Save feature importances from raw CatBoost model.
        feature_importance = pd.DataFrame(
            {
                'feature': features,
                'importance': base_model.get_feature_importance(),
            }
        ).sort_values('importance', ascending=False)
        feature_importance.to_csv(artifacts_path / 'feature_importance.csv', index=False)

        # Save SHAP values plot from raw CatBoost model.
        sample = df_train[features].sample(min(50_000, len(df_train)), random_state=RANDOM_STATE)
        shap.summary_plot(shap.TreeExplainer(base_model).shap_values(sample), sample, show=False)
        plt.title('SHAP Summary Plot')
        plt.savefig(artifacts_path / 'SHAP Summary Plot.png', bbox_inches='tight')
        plt.close()

        (artifacts_path / 'features.txt').write_text('\n'.join(features), encoding='utf-8')

        # Log metrics, artifacts and model to MLflow.
        model_name = f'{project_name}_{model_type}'
        experiment_name = f'{model_name}_{jira}_train'
        mlflow_params = {
            'event_timestamp': event_date,
            'threshold': threshold,
            'best_iteration_cv': best_iteration_cv,
            'best_iteration_train': best_iteration_train,
            'n_folds': 5,
            'train_rows': len(df_train),
            'holdout_rows': len(df_holdout),
            'train_target_rate': df_train[target].mean(),
            'holdout_target_rate': df_holdout[target].mean(),
            'optuna_best_pr_auc': best_optuna_score,
            **{f'catboost_{key}': value for key, value in model_params.items()},
        }

        mlflow.set_experiment(experiment_name)
        signature = infer_signature(df_train.head(1)[features], df_train.head(1)[score])
        run_name = f'train_{event_date}_auto'

        with mlflow.start_run(run_name=run_name, description=run_name):
            mlflow.log_params(mlflow_params)
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics(train_metrics['metrics'])
            mlflow.log_metrics(holdout_metrics['metrics'])

            for root, dirs, files in os.walk(artifacts_dir):
                dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
                for file in files:
                    mlflow.log_artifact(os.path.join(root, file))

            mlflow.sklearn.log_model(
                model,
                name=model_name,
                signature=signature,
                registered_model_name=model_name,
                tags=mlflow_params,
            )

        last_version = client.search_model_versions(
            f'name = \'{model_name}\'',
            order_by=['version_number DESC'],
        )[0].version
        for key, value in mlflow_params.items():
            client.set_model_version_tag(model_name, last_version, key, value)

        # Save data statistics.
        train_data_stat_prefix = state.settings.preprocess_prefix(event_timestamp) / train_data_stat_suffix
        train_data_stat_bucket = train_data_stat_prefix.split('//')[1].split('/')[0]
        train_data_stat_prefix = '/'.join(train_data_stat_prefix.split('//')[1].split('/')[1:])
        data_stat_prefix = f'{train_data_stat_prefix}/{event_date}/'
        report = func.get_data_report(df_train, features + [target, score])
        utils.save_df_to_s3(report, state.credentials.cvm_s3, train_data_stat_bucket, data_stat_prefix)

        logging.info(f'End train at {round(time.time() - t_train)} sec')

    finally:
        # Clean temporary files and directories.
        for path in [Path(artifacts_dir), Path('catboost_info'), Path('mlruns')]:
            shutil.rmtree(path, ignore_errors=True)
        for path in Path('.').rglob('__pycache__'):
            shutil.rmtree(path, ignore_errors=True)
        logging.info('Local training artifacts cleaned')

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import logging
import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

import cvm_ml_metrics.classification as mc
import magpie.sql_utils as su
from cvm_model.io import State
import cvm_model.functions as func
import cvm_model.utils as utils
from cvm_model.parameters import (
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
pd.set_option('display.max_columns', None)

metrics_lib: Dict[str, Any] = {
    'log_loss': log_loss,
    'sensitivity': lambda x, y: mc.sensitivity_specificity(x, y)[0],
    'specificity': lambda x, y: mc.sensitivity_specificity(x, y)[1],
    'balanced_accuracy_lib': mc.balanced_accuracy,
    'youden_j': mc.youden_j,
    'matthews_corrcoef': matthews_corrcoef,
    'cohen_kappa_score': mc.cohen_kappa_score,
}

metrics_lib_proba: Dict[str, Any] = {
    'precision_recall_auc': mc.precision_recall_auc,
    'brier_score_loss': brier_score_loss,
    'markedness': mc.markedness,
    'lift': mc.lift,
    'gini': mc.gini,
    'ks_stat_bin_class': mc.ks_stat_bin_class,
    'ece': lambda x, y: mc.ece_mce_fast(x, y)[0],
    'mce': lambda x, y: mc.ece_mce_fast(x, y)[1],
}


# Calculate the same metric set as the reference model, using fixed threshold from parameters.py.
def evaluate_metrics(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    threshold_value: float,
    log: bool,
    prefix: str,
) -> Dict[str, Any]:
    y_true = df[target].to_numpy()
    y_pred_proba = df[score].to_numpy()
    y_pred = (y_pred_proba >= threshold_value).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    pr_auc = mc.precision_recall_auc(y_true, y_pred_proba)

    if log:
        logging.info(f'Threshold: {threshold_value}')
        logging.info(f'Precision: {precision:.3f}')
        logging.info(f'Recall: {recall:.3f}')
        logging.info(f'F1-score: {f1:.3f}')
        logging.info(f'Balanced accuracy: {balanced_accuracy:.3f}')
        logging.info(f'PR-AUC: {pr_auc:.3f}')

    metrics = {
        'precision': np.round(precision, 4),
        'recall': np.round(recall, 4),
        'f1': np.round(f1, 4),
        'balanced_accuracy': np.round(balanced_accuracy, 4),
    }

    for metric_name, method in metrics_lib_proba.items():
        metrics[metric_name] = method(y_true, y_pred_proba)

    for metric_name, method in metrics_lib.items():
        metric_input = y_pred_proba if metric_name == 'log_loss' else y_pred
        metrics[metric_name] = method(y_true, metric_input)

    metrics = {f'threshold {prefix} {k}': v for k, v in metrics.items()}
    return {'model': model, 'threshold': threshold_value, 'metrics': metrics}


# Train one CatBoost model on a train fold and evaluate it on the validation fold.
def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: List[str],
    plot: bool,
) -> Dict[str, Any]:
    x_train = df_train[features]
    y_train = df_train[target].to_numpy()
    x_val = df_val[features]
    y_val = df_val[target].to_numpy()

    counter = Counter(y_train)
    minority_to_majority_ratio = counter[1] / counter[0] if counter[0] else 1.0
    class_weights = [minority_to_majority_ratio, 1.0]

    model = CatBoostClassifier(
        use_best_model=True,
        verbose=False,
        random_seed=RANDOM_STATE,
        custom_metric='BalancedAccuracy',
        eval_metric='PRAUC',
        class_weights=class_weights,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=(x_val, y_val),
        use_best_model=True,
        plot=plot,
        plot_file='training_plot.html',
    )

    df_val = df_val.copy()
    df_val[score] = model.predict_proba(x_val)[:, 1]

    result = evaluate_metrics(model, df_val, threshold, plot, 'cv')
    result['best_iteration'] = model.get_best_iteration()
    return result


# Main entry point used by the training pipeline.
def train(event_timestamp: datetime) -> None:
    logging.info('Start train')
    t_train = time.time()

    state = State.from_env()
    client = MlflowClient()
    session = state.spark.session
    event_date = event_timestamp.date().isoformat()

    # Clean local artifacts before a new MLflow run.
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    for file in Path(artifacts_dir).rglob('*'):
        if file.is_file():
            file.unlink()

    # Load train and holdout datasets prepared by preprocess.py.
    input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])
    input_path = template.format(bucket=input_bucket, prefix=input_prefix)

    df_train = session.read.parquet(f'{input_path}/train').toPandas()
    df_holdout = session.read.parquet(f'{input_path}/holdout').toPandas()
    
    missing_features = sorted(set(features) - set(df_train.columns))
    assert not missing_features, f'Missing features in train dataset: {missing_features}'
    assert target in df_train.columns, f'{target} is missing in train dataset'
    assert target in df_holdout.columns, f'{target} is missing in holdout dataset'
    assert not (set(df_train['contact_id']) & set(df_holdout['contact_id'])), 'Train and holdout contact_id overlap'

    logging.info(f'Train target rate: {df_train[target].mean():.6f}')
    logging.info(f'Holdout target rate: {df_holdout[target].mean():.6f}')

    train_results = []
    skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(df_train[features], df_train[target])):
        logging.info(f'Fold {i}')
        result = train_model(
            df_train.iloc[train_index],
            df_train.iloc[val_index],
            features,
            False,
        )
        train_results.append(result)

    best_iteration_cv = np.round(np.mean([x['best_iteration'] for x in train_results]), 2)
    logging.info(f'Cross-validation fixed_threshold: {threshold}; best_iteration: {best_iteration_cv}')

    cv_metrics_values = np.array([list(x['metrics'].values()) for x in train_results])
    cv_metrics_values = cv_metrics_values.mean(axis=0)
    cv_metrics = {
        f'cv mean {k}': v for k, v in zip(train_results[0]['metrics'], cv_metrics_values)
    }
    logging.info(f'StratifiedKFold metrics: {cv_metrics}')

    train_result_final = train_model(df_train, df_holdout, features, True)
    model = train_result_final['model']
    best_iteration_train = train_result_final['best_iteration']
    logging.info(f'Final model threshold: {threshold}; best_iteration: {best_iteration_train}')

    df_train = df_train.copy()
    df_holdout = df_holdout.copy()
    df_train[score] = model.predict_proba(df_train[features])[:, 1]
    df_holdout[score] = model.predict_proba(df_holdout[features])[:, 1]

    holdout_metrics = evaluate_metrics(model, df_holdout, threshold, True, 'holdout')
    train_metrics = evaluate_metrics(model, df_train, threshold, True, 'train')
    logging.info(f'Holdout metrics: {holdout_metrics["metrics"]}')
    logging.info(f'Train metrics: {train_metrics["metrics"]}')

    # Save standard diagnostics plots for MLflow artifacts.
    t = df_holdout[target]
    s = df_holdout[score]
    c = s >= threshold

    func.plot_precision_recall_curve(t, s, artifacts_dir=artifacts_dir)
    func.plot_roc(t, s, artifacts_dir=artifacts_dir)

    plt.figure(figsize=(8, 6))
    sns.histplot(s[t == 0], label='not churn', kde=True, stat='density', color='blue')
    sns.histplot(s[t == 1], label='churn', kde=True, stat='density', alpha=0.2, color='orange')
    title = 'Labeled Score Histogram'
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(artifacts_dir, title), bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(t, c)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not churn', 'churn'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    title = 'Confusion Matrix'
    plt.title(title)
    plt.savefig(os.path.join(artifacts_dir, title), bbox_inches='tight')
    plt.close()

    import shap

    sample_size = min(50_000, len(df_train))
    sample = df_train[features].sample(sample_size, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False)
    title = 'SHAP Summary Plot'
    plt.title(title)
    plt.savefig(os.path.join(artifacts_dir, title), bbox_inches='tight')
    plt.close()

    Path(os.path.join(artifacts_dir, 'features.txt')).write_text('\n'.join(features), encoding='utf-8')

    model_name = f'{project_name}_{model_type}'
    experiment_name = f'{model_name}_{jira}_train'
    description = f'train_{event_date}_auto'
    params = {
        'event_timestamp': event_date,
        'threshold': threshold,
        'best_iteration_cv': best_iteration_cv,
        'best_iteration_train': best_iteration_train,
        'n_folds': 5,
        'train_rows': len(df_train),
        'holdout_rows': len(df_holdout),
        'train_target_rate': df_train[target].mean(),
        'holdout_target_rate': df_holdout[target].mean(),
    }

    mlflow.set_experiment(experiment_name)
    signature = infer_signature(df_train.head(1)[features], df_train.head(1)[score])

    with mlflow.start_run(run_name=description, description=description):
        mlflow.log_params(params)
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
            tags=params,
        )

    last_version = client.search_model_versions(
        f'name = \'{model_name}\'',
        order_by=['version_number DESC'],
    )[0].version
    for key, value in params.items():
        client.set_model_version_tag(name=model_name, version=last_version, key=key, value=value)

    train_data_stat_prefix = state.settings.get_prefix(temp=False, suffix=train_data_stat_suffix)
    train_data_stat_bucket = train_data_stat_prefix.split('//')[1].split('/')[0]
    train_data_stat_prefix = '/'.join(train_data_stat_prefix.split('//')[1].split('/')[1:])

    report_cols = features + [target, score]
    logging.info('Saving train dataset stats...')
    data_stat_prefix = f'{train_data_stat_prefix}/{event_date}/'
    report = func.get_data_report(df_train, report_cols)
    su.save_to_s3(report, data_stat_prefix, input_type='df', bucket=train_data_stat_bucket)

    logging.info(f'End train at {round(time.time() - t_train)} sec')

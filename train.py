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
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import cvm_ml_metrics.classification as mc
import cvm_model.functions as func
import cvm_model.utils as utils
from cvm_model.io import State
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
    'sensitivity': lambda y, p: mc.sensitivity_specificity(y, p)[0],
    'specificity': lambda y, p: mc.sensitivity_specificity(y, p)[1],
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
    'ece': lambda y, s: mc.ece_mce_fast(y, s)[0],
    'mce': lambda y, s: mc.ece_mce_fast(y, s)[1],
}


def evaluate_metrics(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    threshold_value: float,
    log: bool,
    prefix: str,
) -> Dict[str, Any]:
    y_true = df[target].to_numpy()
    y_score = df[score].to_numpy()
    y_pred = (y_score >= threshold_value).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y_true, y_score),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f05': fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }
    metrics.update({name: method(y_true, y_score) for name, method in metrics_lib_proba.items()})
    metrics.update({name: method(y_true, y_score if name == 'log_loss' else y_pred) for name, method in metrics_lib.items()})
    metrics = {f'threshold {prefix} {name}': value for name, value in metrics.items()}

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
    counter = Counter(df_train[target].to_numpy())
    class_weights = [counter[1] / counter[0] if counter[0] else 1.0, 1.0]

    model = CatBoostClassifier(
        **model_params,
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
        df_train[features],
        df_train[target],
        eval_set=(df_val[features], df_val[target]),
        use_best_model=True,
        plot=plot,
        plot_file='training_plot.html',
    )

    df_val = df_val.copy()
    df_val[score] = model.predict_proba(df_val[features])[:, 1]
    result = evaluate_metrics(model, df_val, threshold, plot, 'cv')
    result['best_iteration'] = model.get_best_iteration()
    return result


def train(event_timestamp: datetime) -> None:
    logging.info('Start train')
    t_train = time.time()

    state = State.from_env()
    client = MlflowClient()
    session = state.spark.session
    event_date = event_timestamp.date().isoformat()

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    for file in artifacts_path.rglob('*'):
        if file.is_file():
            file.unlink()

    input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])
    input_path = template.format(bucket=input_bucket, prefix=input_prefix)

    try:
        df_train = utils.load_dataset(session, f'{input_path}/train')
        df_holdout = utils.load_dataset(session, f'{input_path}/holdout')
    except Exception:
        logging.exception(f'Failed to load datasets from {input_path}')
        raw_input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix
        legacy_input_path = template.format(bucket=input_bucket, prefix=str(raw_input_prefix).strip('/'))
        logging.info(f'Trying legacy preprocess path: {legacy_input_path}')
        df_train = utils.load_dataset(session, f'{legacy_input_path}/train')
        df_holdout = utils.load_dataset(session, f'{legacy_input_path}/holdout')

    missing_features = sorted(set(features) - set(df_train.columns))
    assert not missing_features, f'Missing features in train dataset: {missing_features}'
    assert target in df_train.columns, f'{target} is missing in train dataset'
    assert target in df_holdout.columns, f'{target} is missing in holdout dataset'
    assert not (set(df_train['contact_id']) & set(df_holdout['contact_id'])), 'Train and holdout contact_id overlap'
    logging.info(f'Train target rate: {df_train[target].mean():.6f}')
    logging.info(f'Holdout target rate: {df_holdout[target].mean():.6f}')

    from cvm_model.optuna_tuning import tune_catboost_params

    best_params, best_optuna_score = tune_catboost_params(
        df=df_train,
        features=features,
        target=target,
        random_state=RANDOM_STATE,
        n_folds=5,
        n_trials=30,
    )

    skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    train_results = [
        train_model(df_train.iloc[train_idx], df_train.iloc[val_idx], features, False, best_params)
        for train_idx, val_idx in skf.split(df_train[features], df_train[target])
    ]
    best_iteration_cv = float(np.round(np.mean([res['best_iteration'] for res in train_results]), 2))
    cv_metrics = {
        f'cv mean {key}': value
        for key, value in zip(
            train_results[0]['metrics'],
            np.array([list(res['metrics'].values()) for res in train_results]).mean(axis=0),
        )
    }
    logging.info(f'StratifiedKFold metrics: {cv_metrics}')

    train_result_final = train_model(df_train, df_holdout, features, True, best_params)
    model = train_result_final['model']
    best_iteration_train = train_result_final['best_iteration']

    df_train = df_train.copy()
    df_holdout = df_holdout.copy()
    df_train[score] = model.predict_proba(df_train[features])[:, 1]
    df_holdout[score] = model.predict_proba(df_holdout[features])[:, 1]

    train_metrics = evaluate_metrics(model, df_train, threshold, True, 'train')
    holdout_metrics = evaluate_metrics(model, df_holdout, threshold, True, 'holdout')
    logging.info(f'Train metrics: {train_metrics["metrics"]}')
    logging.info(f'Holdout metrics: {holdout_metrics["metrics"]}')

    y_true = df_holdout[target]
    y_score = df_holdout[score]
    y_pred = y_score >= threshold

    func.plot_precision_recall_curve(y_true, y_score, artifacts_dir=artifacts_dir)
    func.plot_roc(y_true, y_score, artifacts_dir=artifacts_dir)

    plt.figure(figsize=(8, 6))
    sns.histplot(y_score[y_true == 0], label='not churn', kde=True, stat='density', color='blue')
    sns.histplot(y_score[y_true == 1], label='churn', kde=True, stat='density', alpha=0.2, color='orange')
    plt.title('Labeled Score Histogram')
    plt.legend()
    plt.savefig(artifacts_path / 'Labeled Score Histogram', bbox_inches='tight')
    plt.close()

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=['not churn', 'churn'],
    ).plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(artifacts_path / 'Confusion Matrix', bbox_inches='tight')
    plt.close()

    import shap

    sample = df_train[features].sample(min(50_000, len(df_train)), random_state=RANDOM_STATE)
    shap.summary_plot(shap.TreeExplainer(model).shap_values(sample), sample, show=False)
    plt.title('SHAP Summary Plot')
    plt.savefig(artifacts_path / 'SHAP Summary Plot', bbox_inches='tight')
    plt.close()

    (artifacts_path / 'features.txt').write_text('\n'.join(features), encoding='utf-8')

    model_name = f'{project_name}_{model_type}'
    experiment_name = f'{model_name}_{jira}_train'
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
        'optuna_best_pr_auc': best_optuna_score,
        **{f'catboost_{key}': value for key, value in best_params.items()},
    }

    mlflow.set_experiment(experiment_name)
    signature = infer_signature(df_train.head(1)[features], df_train.head(1)[score])
    run_name = f'train_{event_date}_auto'

    with mlflow.start_run(run_name=run_name, description=run_name):
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
        client.set_model_version_tag(model_name, last_version, key, value)

    train_data_stat_prefix = state.settings.get_prefix(temp=False, suffix=train_data_stat_suffix)
    train_data_stat_bucket = train_data_stat_prefix.split('//')[1].split('/')[0]
    train_data_stat_prefix = '/'.join(train_data_stat_prefix.split('//')[1].split('/')[1:])
    data_stat_prefix = f'{train_data_stat_prefix}/{event_date}/'
    report = func.get_data_report(df_train, features + [target, score])
    utils.save_df_to_s3(report, state.credentials.cvm_s3, train_data_stat_bucket, data_stat_prefix)

    logging.info(f'End train at {round(time.time() - t_train)} sec')

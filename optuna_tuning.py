from collections import Counter
from typing import List, Tuple

import logging

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

import cvm_ml_metrics.classification as mc


def tune_catboost_params(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    random_state: int,
    n_folds: int = 5,
    n_trials: int = 30,
) -> Tuple[dict, float]:
    """Tune CatBoost hyperparameters by mean StratifiedKFold PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            'border_count': trial.suggest_int('border_count', 64, 254),
        }

        fold_scores = []
        skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

        for train_index, val_index in skf.split(df[features], df[target]):
            df_train = df.iloc[train_index]
            df_val = df.iloc[val_index]

            counter = Counter(df_train[target].to_numpy())
            minority_to_majority_ratio = counter[1] / counter[0] if counter[0] else 1.0

            model = CatBoostClassifier(
                **params,
                use_best_model=True,
                verbose=False,
                random_seed=random_state,
                custom_metric='BalancedAccuracy',
                eval_metric='PRAUC',
                class_weights=[minority_to_majority_ratio, 1.0],
                early_stopping_rounds=50,
                thread_count=-1,
            )
            model.fit(
                df_train[features],
                df_train[target],
                eval_set=(df_val[features], df_val[target]),
                use_best_model=True,
            )

            pred = model.predict_proba(df_val[features])[:, 1]
            fold_scores.append(mc.precision_recall_auc(df_val[target], pred))

        mean_score = float(np.mean(fold_scores))
        logging.info(f'Optuna trial {trial.number}: mean PR-AUC = {mean_score:.6f}; params = {params}')
        return mean_score

    study = optuna.create_study(direction='maximize', study_name='catboost_pr_auc')
    study.optimize(objective, n_trials=n_trials)

    logging.info(f'Best Optuna PR-AUC: {study.best_value:.6f}')
    logging.info(f'Best Optuna params: {study.best_params}')

    return study.best_params, float(study.best_value)

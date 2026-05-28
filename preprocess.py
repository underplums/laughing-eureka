from datetime import datetime
from pathlib import Path

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cvm_model.io import State
import cvm_model.sql as sql
import cvm_model.utils as utils
from cvm_model.parameters import (
    RANDOM_STATE,
    aud_table,
    fav_omni_features_table,
    features,
    features_for_outliers,
    input_suffix,
    preperiod_months,
    target,
)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
pd.set_option('display.max_columns', None)

HOLDOUT_TEST_SIZE = 0.2
RECENCY_COLS = ['cheque_recency', 'login_recency', 'omni_qr_recency', 'omni_features_recency', 'perf_recency']


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    calc_period = preperiod_months[0]

    # Normalize technical columns and remove infinite values before modeling.
    df['contact_id'] = df['contact_id'].astype('int64')
    df = df.replace([np.inf, -np.inf], np.nan)

    # Build recency-to-average-lag ratios used as activity tendency features.
    df['transaction_tendency'] = df['cheque_recency'] / np.maximum(1, df[f'trans_lag_avg_{calc_period}'])
    df['login_tendency'] = df['login_recency'] / np.maximum(1, df[f'login_lag_avg_{calc_period}'])
    df['omni_qr_tendency'] = df['omni_qr_recency'] / np.maximum(1, df[f'omni_qr_lag_avg_{calc_period}'])
    df['omni_features_tendency'] = df['omni_features_recency'] / np.maximum(
        1, df[f'omni_features_lag_avg_{calc_period}']
    )

    for col in ['dac_months_count', 'dac_months_last_3', 'dac_months_last_6', 'dac_months_last_12']:
        df[col] = df[col].fillna(0)

    # Convert raw DAC history counters into compact ratio and segment flags.
    df['dac_age_months'] = df['dac_age_months'].fillna(0)
    df.loc[df['dac_age_months'] == 0, 'dac_age_months'] = 1
    df['dac_months_per_dac_age_ratio'] = df['dac_months_count'] / df['dac_age_months']
    df['dac_share_last_12'] = df['dac_months_last_12'] / 12
    df['is_stable_dac'] = (df['dac_months_last_12'] >= 10).astype(np.int32)
    df['is_regular_dac'] = df['dac_months_last_12'].between(6, 9).astype(np.int32)
    df['is_unstable_dac'] = df['dac_months_last_12'].between(2, 5).astype(np.int32)
    df['is_new_dac'] = (df['dac_months_last_12'] == 1).astype(np.int32)

    for col in RECENCY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    # Keep a readable DAC segment for stratification, reports, and model diagnostics.
    conditions = [
        df['is_stable_dac'] == 1,
        df['is_regular_dac'] == 1,
        df['is_unstable_dac'] == 1,
        df['is_new_dac'] == 1,
    ]
    choices = ['stable_10_12m', 'regular_6_9m', 'unstable_2_5m', 'new_1m']
    df['dac_segment_12m'] = np.select(conditions, choices, default='no_history')
    df['segment'] = df['dac_segment_12m'].astype(str)
    df['strat'] = df['segment'].astype(str) + '_' + df[target].astype(str)

    if (df['strat'].value_counts() < 5).any():
        logging.info('Small strata found, fallback to target-only stratification')
        df['strat'] = df[target].astype(str)

    return df


def validate_dataset(df: pd.DataFrame, name: str) -> None:
    # Fail fast before saving data that cannot be used by train.py.
    missing_features = set(features) - set(df.columns)
    leakage_cols = {'contact_id', target, 'target_churn_from_dac', 'is_dac_next_month', 'score', 'class_0', 'strat'}
    bad_features = sorted(set(features) & leakage_cols)

    assert len(df) > 0, f'{name} is empty'
    assert df['contact_id'].nunique() == len(df), f'{name}: contact_id is not unique'
    assert target in df.columns, f'{name}: target is missing'
    assert set(df[target].dropna().unique()).issubset({0, 1}), f'{name}: target must be binary'
    assert not missing_features, f'{name}: features are missing: {missing_features}'
    assert not bad_features, f'{name}: service or target columns are present in features: {bad_features}'


def preprocess_train(event_timestamp: datetime):
    # Runtime state provides GP, S3, and path settings from environment variables.
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    s3 = state.credentials.cvm_s3

    input_prefix = state.settings.get_prefix(temp=True, suffix=input_suffix)
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])

    base_month = event_timestamp.date().replace(day=1).isoformat()
    target_month = (pd.Timestamp(base_month) + pd.DateOffset(months=1)).date().isoformat()
    feature_date = target_month

    logging.info(f'Base DAC month = {base_month}')
    logging.info(f'Target month = {target_month}')
    logging.info(f'Feature date = {feature_date}')

    try:
        # Start from a clean S3 prefix because parquet readers will read every file under it.
        utils.remove_s3_prefix(s3, input_bucket, input_prefix)
        assert not utils.list_s3_objects(s3, input_bucket, input_prefix), 'S3 input prefix was not cleaned'

        # Base audience: clients who are DAC in the base month.
        df = utils.get_df(engine, sql.aud_query.format(base_month=base_month)).fillna(0)
        df = df.astype({col: np.int32 for col in {'contact_id'} & set(df.columns)})
        assert len(df) > 0, 'No DAC audience contacts were loaded'
        assert df['contact_id'].is_unique, 'DAC audience contains duplicate contact_id values'
        logging.info(f'Audience shape: {df.shape}')

        # Upload audience to GP so all feature SQL queries can join the same client set.
        utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table)
        aud_query = f'select distinct contact_id :: int as contact_id from {aud_table}'

        # Target: churn from DAC in the next month.
        df_part = utils.get_df(engine, sql.target_query.format(aud=aud_query, target_month=target_month))
        df_part = df_part.astype({col: np.int32 for col in {target} & set(df_part.columns)})
        assert df_part['contact_id'].is_unique, 'target_query returned duplicate contact_id values'

        df = df.merge(df_part, on='contact_id', how='inner')
        assert len(df) > 0, 'Dataset is empty after target labeling'
        logging.info(f'After target: {df.shape}; target rate = {df[target].mean():.6f}')

        # Re-upload the target-labeled population before feature extraction.
        utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table)
        aud_query = f'select distinct contact_id :: int as contact_id from {aud_table}'

        # Load SQL feature blocks; pandas-only derived features are added below.
        df = utils.load_features(
            engine=engine,
            df=df,
            aud_query=aud_query,
            date=feature_date,
            fav_omni_features_table=fav_omni_features_table,
            preperiod_months=preperiod_months,
        )

        # Add deterministic derived features and run basic dataset checks.
        df = prepare_dataset(df)
        validate_dataset(df, 'full_dataset')

        # Remove only extreme outliers in selected high-variance numeric features.
        df['outlier'] = 0
        for feature in features_for_outliers:
            if feature in df.columns and df[feature].notna().sum() > 0:
                df.loc[df[feature] > np.nanquantile(df[feature], 0.999), 'outlier'] = 1

        outlier_share = df['outlier'].mean()
        assert outlier_share <= 0.01, 'More than 1% of the audience was removed as outliers'
        df = df[df['outlier'] == 0].drop(columns=['outlier']).reset_index(drop=True)
        validate_dataset(df, 'full_dataset_without_outliers')
        logging.info(f'Final dataset shape: {df.shape}; target rate = {df[target].mean():.6f}')

        # Holdout is kept unseen by train.py for final validation metrics.
        df, holdout_df = train_test_split(
            df,
            test_size=HOLDOUT_TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df[target],
        )
        df = df.reset_index(drop=True)
        holdout_df = holdout_df.reset_index(drop=True)

        assert not (set(df['contact_id']) & set(holdout_df['contact_id'])), 'Train and holdout contact_id overlap'
        validate_dataset(df, 'train')
        validate_dataset(holdout_df, 'holdout')

        logging.info(f'Train shape: {df.shape}; target rate = {df[target].mean():.6f}')
        logging.info(f'Holdout shape: {holdout_df.shape}; target rate = {holdout_df[target].mean():.6f}')

        # Store train and holdout separately; train.py should read both prefixes.
        utils.save_df_to_s3(df, s3, input_bucket, f'{input_prefix}/train')
        utils.save_df_to_s3(holdout_df, s3, input_bucket, f'{input_prefix}/holdout')

    finally:
        # Temporary GP tables and local cache should not survive failed runs.
        for table in [aud_table, fav_omni_features_table]:
            try:
                utils.execute_query(engine, f'drop table if exists {table}')
            except Exception:
                logging.exception(f'Failed to drop temporary table {table}')

        Path('df_cache.parquet').unlink(missing_ok=True)

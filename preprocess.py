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


RECENCY_COLS = [
    'cheque_recency',
    'login_recency',
    'omni_qr_recency',
    'omni_features_recency',
    'perf_recency',
]

HOLDOUT_TEST_SIZE = 0.2


def _add_dac_segment(df: pd.DataFrame) -> pd.DataFrame:
    '''Add DAC-history segment used for train stratification and reporting.'''

    df = df.copy()
    zero_flag = pd.Series(0, index=df.index)

    conditions = [
        df.get('is_stable_dac', zero_flag) == 1,
        df.get('is_regular_dac', zero_flag) == 1,
        df.get('is_unstable_dac', zero_flag) == 1,
        df.get('is_new_dac', zero_flag) == 1,
    ]
    choices = ['stable_10_12m', 'regular_6_9m', 'unstable_2_5m', 'new_1m']

    df['dac_segment_12m'] = np.select(conditions, choices, default='no_history')
    df['segment'] = df['dac_segment_12m'].astype(str)

    return df


def _add_tendency_features(df: pd.DataFrame) -> pd.DataFrame:
    '''Add recency-to-average-lag ratios after all feature blocks are loaded.'''

    df = df.copy()
    calc_period = preperiod_months[0]

    df['transaction_tendency'] = df['cheque_recency'] / np.maximum(
        1, df[f'trans_lag_avg_{calc_period}']
    )
    df['login_tendency'] = df['login_recency'] / np.maximum(
        1, df[f'login_lag_avg_{calc_period}']
    )
    df['omni_qr_tendency'] = df['omni_qr_recency'] / np.maximum(
        1, df[f'omni_qr_lag_avg_{calc_period}']
    )
    df['omni_features_tendency'] = df['omni_features_recency'] / np.maximum(
        1, df[f'omni_features_lag_avg_{calc_period}']
    )

    return df


def _add_dac_history_features(df: pd.DataFrame) -> pd.DataFrame:
    '''Add derived DAC-history ratios and segment flags.'''

    df = df.copy()

    dac_history_cols = [
        'dac_months_count',
        'dac_months_last_3',
        'dac_months_last_6',
        'dac_months_last_12',
    ]
    for col in dac_history_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'dac_age_months' in df.columns:
        df['dac_age_months'] = df['dac_age_months'].fillna(0)
        df.loc[df['dac_age_months'] == 0, 'dac_age_months'] = 1
        df['dac_months_per_dac_age_ratio'] = df['dac_months_count'] / df['dac_age_months']

    df['dac_share_last_12'] = df['dac_months_last_12'] / 12
    df['is_stable_dac'] = (df['dac_months_last_12'] >= 10).astype(np.int32)
    df['is_regular_dac'] = df['dac_months_last_12'].between(6, 9).astype(np.int32)
    df['is_unstable_dac'] = df['dac_months_last_12'].between(2, 5).astype(np.int32)
    df['is_new_dac'] = (df['dac_months_last_12'] == 1).astype(np.int32)

    return df


def _prepare_train_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''Apply deterministic preprocessing that must be shared by train runs.'''

    df = df.copy()

    df['contact_id'] = df['contact_id'].astype('int64')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = _add_tendency_features(df)
    df = _add_dac_history_features(df)

    for col in RECENCY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    df = _add_dac_segment(df)
    df['strat'] = df['segment'].astype(str) + '_' + df[target].astype(str)

    strat_counts = df['strat'].value_counts()
    if (strat_counts < 5).any():
        logging.info('Small strata found, fallback to target-only stratification.')
        df['strat'] = df[target].astype(str)

    return df


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove extreme outliers by selected numeric feature quantiles.'''

    df = df.copy()
    q = 0.999
    df['outlier'] = 0

    for feature in features_for_outliers:
        if feature not in df.columns or df[feature].notna().sum() == 0:
            continue
        threshold_value = np.nanquantile(df[feature], q)
        df.loc[df[feature] > threshold_value, 'outlier'] = 1

    outlier_share = len(df[df['outlier'] == 1]) / len(df)
    assert outlier_share <= 0.01, 'More than 1% of the audience was removed as outliers.'

    logging.info(f'Outlier share: {outlier_share:.6f}')
    return df[df['outlier'] == 0].drop(columns=['outlier']).reset_index(drop=True)


def _validate_train_dataset(df: pd.DataFrame) -> None:
    '''Fail fast if the training dataset is malformed.'''

    missing_required_cols = {'contact_id', target} - set(df.columns)
    assert not missing_required_cols, f'Required columns are missing from the dataset: {missing_required_cols}'

    missing_features = set(features) - set(df.columns)
    assert not missing_features, f'Features are missing from the dataset: {missing_features}'

    leakage_cols = {
        'contact_id',
        target,
        'target_churn_from_dac',
        'is_dac_next_month',
        'score',
        'class_0',
        'strat',
    }
    bad_features = sorted(set(features) & leakage_cols)
    assert not bad_features, f'Service or target columns are present in features: {bad_features}'

    assert len(df) > 0, 'The train dataset is empty.'
    assert df['contact_id'].nunique() == len(df), 'contact_id is not unique.'
    assert set(df[target].dropna().unique()).issubset({0, 1}), f'{target} must be binary.'


def _split_train_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Split the prepared dataset into train and holdout-test parts.'''

    df = df.copy()
    stratify = df[target]

    df, holdout_df = train_test_split(
        df,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=42,
        stratify=stratify,
    )

    df = df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    overlap = set(df['contact_id']) & set(holdout_df['contact_id'])
    assert len(overlap) == 0, 'Train and holdout datasets contain overlapping contact_id values.'

    logging.info(f'Train dataset shape: {df.shape}')
    logging.info(f'Holdout dataset shape: {holdout_df.shape}')
    logging.info(f'Train target rate: {df[target].mean():.6f}')
    logging.info(f'Holdout target rate: {holdout_df[target].mean():.6f}')

    return df, holdout_df


def preprocess_train(event_timestamp: datetime):
    # Initialize runtime state and external connections.
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    s3 = state.credentials.cvm_s3

    # Resolve the S3 temporary input path used by the train step.
    input_prefix = state.settings.get_prefix(temp=True, suffix=input_suffix)
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])

    # Define the monthly train setup: base DAC month, target month, and feature cutoff.
    base_month = event_timestamp.date().replace(day=1).isoformat()
    target_month = (pd.Timestamp(base_month) + pd.DateOffset(months=1)).date().isoformat()
    feature_date = target_month

    logging.info(f'Base DAC month = {base_month}')
    logging.info(f'Target month = {target_month}')
    logging.info(f'Feature date = {feature_date}')

    try:
        # Clean the previous input dataset to avoid mixing old and new parquet parts.
        logging.info(f'Cleaning S3 input prefix: s3://{input_bucket}/{input_prefix}')
        utils.remove_s3_prefix(s3, input_bucket, input_prefix)
        files = utils.list_s3_objects(s3, input_bucket, input_prefix)
        assert len(files) == 0, 'The dataset S3 prefix was not cleaned.'

        # Load the base-month DAC audience.
        logging.info('Loading DAC audience...')
        aud_query_raw = sql.aud_query.format(base_month=base_month)
        df = utils.get_df(engine, aud_query_raw).fillna(0)
        df = df.astype({col: np.int32 for col in {'contact_id'} & set(df.columns)})

        assert len(df) > 0, 'No DAC audience contacts were loaded.'
        assert df['contact_id'].nunique() == len(df), 'The DAC audience contains duplicate contact_id values.'

        logging.info(f'Audience shape: {df.shape}')

        # Upload the audience to a GP table so all feature SQL blocks can join it efficiently.
        logging.info(f'Uploading audience to GP temp table: {aud_table}')
        utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table)
        aud_query = f'select distinct contact_id :: int as contact_id from {aud_table}'
        df.to_parquet('df_cache.parquet')

        # Mark the next-month DAC churn target.
        logging.info('Loading target...')
        query_kwargs = {
            'aud': aud_query,
            'target_month': target_month,
        }
        df_part = utils.get_df(engine, sql.target_query.format(**query_kwargs))
        cast_cols = {target} & set(df_part.columns)
        df_part = df_part.astype({col: np.int32 for col in cast_cols})

        assert df_part['contact_id'].is_unique, 'target_query returned duplicate contact_id values.'

        df = df.merge(df_part, on='contact_id', how='inner')
        assert len(df) > 0, 'The dataset is empty after target labeling.'
        logging.info(f'After target: {df.shape}; target rate = {df[target].mean():.6f}')

        # Restrict feature SQL blocks to the final train population.
        utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table)
        aud_query = f'select distinct contact_id :: int as contact_id from {aud_table}'

        # Load all feature blocks selected for the model.
        logging.info('Loading feature blocks...')
        df = utils.load_features(
            engine=engine,
            df=df,
            aud_query=aud_query,
            date=feature_date,
            fav_omni_features_table=fav_omni_features_table,
            preperiod_months=preperiod_months,
        )
        df.to_parquet('df_cache.parquet')

        # Apply deterministic preprocessing shared by all train runs.
        df = _prepare_train_dataset(df)
        _validate_train_dataset(df)

        # Remove extreme outliers from selected monetary/frequency features.
        df = _remove_outliers(df)
        _validate_train_dataset(df)

        logging.info(f'Final train dataset shape: {df.shape}')
        logging.info(f'Final target rate: {df[target].mean():.6f}')

        # Split the prepared dataset into model train and final holdout-test parts.
        df, holdout_df = _split_train_holdout(df)
        _validate_train_dataset(df)
        _validate_train_dataset(holdout_df)

        # Save train and holdout-test datasets to S3 for the train step.
        train_input_prefix = f'{input_prefix}/train'
        holdout_input_prefix = f'{input_prefix}/holdout'
        logging.info(f'Saving train dataset to s3://{input_bucket}/{train_input_prefix}')
        utils.save_df_to_s3(df, s3, input_bucket, train_input_prefix)
        logging.info(f'Saving holdout dataset to s3://{input_bucket}/{holdout_input_prefix}')
        utils.save_df_to_s3(holdout_df, s3, input_bucket, holdout_input_prefix)

    finally:
        # Clean temporary GP tables and local cache even if preprocessing fails.
        logging.info('Cleaning temporary local and GP objects...')
        for table in [aud_table, fav_omni_features_table]:
            try:
                utils.execute_query(engine, f'drop table if exists {table}')
            except Exception:
                logging.exception(f'Failed to drop temporary table {table}')

        Path('df_cache.parquet').unlink(missing_ok=True)

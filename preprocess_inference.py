from datetime import datetime
from pathlib import Path

import logging

import numpy as np
import pandas as pd

from cvm_model.io import State
import cvm_model.sql as sql
import cvm_model.utils as utils
from cvm_model.parameters import (
    aud_table,
    fav_omni_features_table,
    features,
    recency_cols,
    features_for_outliers,
    input_suffix,
    preperiod_months,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column preprocessing and add derived features to inference dataset"""

    df = df.copy()

    # Remove infinite values
    df['contact_id'] = df['contact_id'].astype('int64')
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill recency values before tendency feature calculation
    for col in recency_cols:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    calc_period = preperiod_months[0]

    # Build tendency features
    df['transaction_tendency'] = df['cheque_recency'] / np.maximum(1, df[f'trans_lag_avg_{calc_period}'])
    df['login_tendency'] = df['login_recency'] / np.maximum(1, df[f'login_lag_avg_{calc_period}'])
    df['omni_qr_tendency'] = df['omni_qr_recency'] / np.maximum(1, df[f'omni_qr_lag_avg_{calc_period}'])
    df['omni_features_tendency'] = df['omni_features_recency'] / np.maximum(
        1, df[f'omni_features_lag_avg_{calc_period}']
    )

    for col in ['dac_months_count', 'dac_months_last_3', 'dac_months_last_6', 'dac_months_last_12']:
        df[col] = df[col].fillna(0)

    # Convert raw DAC history counters into segment flags
    df['dac_age_months'] = df['dac_age_months'].fillna(0)
    df.loc[df['dac_age_months'] == 0, 'dac_age_months'] = 1
    df['dac_months_per_dac_age_ratio'] = df['dac_months_count'] / df['dac_age_months']
    df['dac_share_last_12'] = df['dac_months_last_12'] / 12
    df['is_stable_dac'] = (df['dac_months_last_12'] >= 10).astype(np.int32)
    df['is_regular_dac'] = df['dac_months_last_12'].between(6, 9).astype(np.int32)
    df['is_unstable_dac'] = df['dac_months_last_12'].between(2, 5).astype(np.int32)
    df['is_new_dac'] = (df['dac_months_last_12'] == 1).astype(np.int32)

    # Build DAC segment feature
    conditions = [
        df['is_stable_dac'] == 1,
        df['is_regular_dac'] == 1,
        df['is_unstable_dac'] == 1,
        df['is_new_dac'] == 1,
    ]
    choices = ['stable_10_12m', 'regular_6_9m', 'unstable_2_5m', 'new_1m']

    df['dac_segment_12m'] = np.select(conditions, choices, default='no_history')
    df['segment'] = df['dac_segment_12m'].astype(str)

    return df


def _validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Validate inference dataset before saving"""

    missing_features = set(features) - set(df.columns)
    leakage_cols = {
        'target',
        'target_churn_from_dac',
        'is_dac_next_month',
        'score',
        'class_0',
        'strat',
    }
    leakages = sorted(set(features) & leakage_cols)

    assert len(df) > 0, f'{name} is empty'
    assert df['contact_id'].nunique() == len(df), f'{name}: contact_id is not unique'
    assert not missing_features, f'{name}: features are missing: {missing_features}'
    assert not leakages, f'{name}: service or target columns are present in features: {leakages}'


def preprocess_inference(event_timestamp: datetime):
    # Set connections
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    s3 = state.credentials.cvm_s3

    # Resolve S3 input path. Inference input must not overwrite train/holdout.
    input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix / 'inference'
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])

    # Set inference dates
    run_month = pd.Timestamp(event_timestamp.date().replace(day=1).isoformat())

    base_month = (run_month - pd.DateOffset(months=1)).date().isoformat()
    feature_date = run_month.date().isoformat()
    base_month_suffix = base_month.replace('-', '_')
    aud_table_month = f'{aud_table}_{base_month_suffix}_inference'
    fav_omni_features_table_month = f'{fav_omni_features_table}_{base_month_suffix}_inference'

    logging.info(f'Base DAC month = {base_month}')
    logging.info(f'Feature date = {feature_date}')

    try:
        # Clean only inference S3 prefix, not train/holdout input prefix.
        utils.remove_s3_prefix(s3, input_bucket, input_prefix)
        assert not utils.list_s3_objects(s3, input_bucket, input_prefix), 'S3 inference prefix was not cleaned'

        # Load full actual DAC audience for scoring.
        aud_sql = sql.aud_query.format(base_month=base_month)
        df = utils.get_df(engine, aud_sql).fillna(0)

        df = df.astype({col: np.int32 for col in {'contact_id'} & set(df.columns)})
        assert len(df) > 0, 'No DAC audience contacts were loaded'
        assert df['contact_id'].is_unique, 'DAC audience contains duplicate contact_id values'

        logging.info(f'Inference audience shape: {df.shape}')

        # Upload inference audience to GP once for all feature SQL joins.
        utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table_month)
        aud_query = f'select distinct contact_id :: int as contact_id from {aud_table_month}'

        # Load SQL feature blocks.
        df = utils.load_features(
            engine=engine,
            df=df,
            aud_query=aud_query,
            date=feature_date,
            fav_omni_features_table=fav_omni_features_table_month,
            preperiod_months=preperiod_months,
        )

        # Add derived features and validate the scoring dataset.
        df = _prepare_dataset(df)
        _validate_dataset(df, 'inference_dataset')

        # Remove extreme outliers using the same rule as train preprocess.
        df['outlier'] = 0
        for feature in features_for_outliers:
            if feature in df.columns and df[feature].notna().sum() > 0:
                df.loc[df[feature] > np.nanquantile(df[feature], 0.999), 'outlier'] = 1

        outlier_share = df['outlier'].mean()
        assert outlier_share <= 0.01, 'More than 1% of the audience was removed as outliers'

        df = df[df['outlier'] == 0].drop(columns=['outlier']).reset_index(drop=True)
        _validate_dataset(df, 'inference_dataset_without_outliers')

        logging.info(f'Final inference dataset shape: {df.shape}')

        utils.save_df_to_s3(df, s3, input_bucket, input_prefix)
        logging.info('Inference dataset saved to S3')

    finally:
        # Delete temporary files
        for table in [aud_table_month, fav_omni_features_table_month]:
            try:
                utils.execute_query(engine, f'drop table if exists {table}')
            except Exception:
                logging.exception(f'Failed to drop temporary table {table}')

        Path('df_cache.parquet').unlink(missing_ok=True)

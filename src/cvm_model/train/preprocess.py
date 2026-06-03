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
    SAMPLE_SIZE,
    RANDOM_STATE,
    HOLDOUT_SIZE,
    aud_table,
    fav_omni_features_table,
    features,
    recency_cols,
    features_for_outliers,
    input_suffix,
    preperiod_months,
    target,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column preprocessing and add derived features to selected dataset"""

    df = df.copy()
    
    # Remove infinite values
    df['contact_id'] = df['contact_id'].astype('int64')
    df = df.replace([np.inf, -np.inf], np.nan)

    # Build tendency features
    for col in recency_cols:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    calc_period = preperiod_months[0]

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

    # Build DAC segments features
    conditions = [
        df['is_stable_dac'] == 1,
        df['is_regular_dac'] == 1,
        df['is_unstable_dac'] == 1,
        df['is_new_dac'] == 1,
    ]

    choices = ['stable_10_12m', 'regular_6_9m', 'unstable_2_5m', 'new_1m']

    # Add stratification column
    df['dac_segment_12m'] = np.select(conditions, choices, default='no_history')
    df['segment'] = df['dac_segment_12m'].astype(str)
    df['strat'] = df['segment'].astype(str) + '_' + df[target].astype(str)

    if (df['strat'].value_counts() < 5).any():
        df['strat'] = df[target].astype(str)

    return df


def _validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Validate dataset before saving"""

    missing_features = set(features) - set(df.columns)

    leakage_cols = {'contact_id', target, 'target_churn_from_dac', 'is_dac_next_month', 'score', 'class_0', 'strat'}
    leakages = sorted(set(features) & leakage_cols)

    assert len(df) > 0, f'{name} is empty'
    assert df['contact_id'].nunique() == len(df), f'{name}: contact_id is not unique'
    assert target in df.columns, f'{name}: target is missing'
    assert not missing_features, f'{name}: features are missing: {missing_features}'
    assert not leakages, f'{name}: service or target columns are present in features: {leakages}'


def preprocess_train(event_timestamp: datetime):
    #Set connections
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    s3 = state.credentials.cvm_s3

    #Resolve S3 input paths
    input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])

    #Set training dates
    run_month = pd.Timestamp(event_timestamp.date().replace(day=1).isoformat())
    base_month = (run_month - pd.DateOffset(months=2)).date().isoformat()
    target_month = (run_month - pd.DateOffset(months=1)).date().isoformat()
    feature_date = target_month
    base_month_suffix = base_month.replace('-', '_')

    #Set table paths
    aud_table_month = f'{aud_table}_{base_month_suffix}'
    fav_omni_features_table_month = f'{fav_omni_features_table}_{base_month_suffix}'

    logging.info(f'Base DAC month = {base_month}')
    logging.info(f'Target month = {target_month}')
    logging.info(f'Feature date = {feature_date}')

    try:
        # Clean train and holdout prefixes without touching inference input
        train_prefix = f'{input_prefix}/train'
        holdout_prefix = f'{input_prefix}/holdout'
        utils.remove_s3_prefix(s3, input_bucket, train_prefix)
        utils.remove_s3_prefix(s3, input_bucket, holdout_prefix)
        assert not utils.list_s3_objects(s3, input_bucket, train_prefix), 'S3 train prefix was not cleaned'
        assert not utils.list_s3_objects(s3, input_bucket, holdout_prefix), 'S3 holdout prefix was not cleaned'

        # Load full DAC audience
        full_aud_query = sql.aud_query.format(base_month=base_month)
        df = utils.get_df(engine, full_aud_query).fillna(0)
        df = df.astype({col: np.int64 for col in {'contact_id'} & set(df.columns)})

        assert len(df) > 0, 'No DAC audience contacts were loaded'
        assert df['contact_id'].is_unique, 'DAC audience contains duplicate contact_id values'

        logging.info(f'Full audience shape: {df.shape}')

        # Load target
        df_part = utils.get_df(engine, sql.target_query.format(aud=full_aud_query, target_month=target_month))
        df_part = df_part.astype({col: np.int32 for col in {target} & set(df_part.columns)})

        assert df_part['contact_id'].is_unique, 'target_query returned duplicate contact_id values'

        df = df.merge(df_part, on='contact_id', how='inner')
        assert len(df) > 0, 'Dataset is empty after target labeling'

        logging.info(f'After target: {df.shape}; target rate = {df[target].mean():.6f}')

        # Sample audience with SAMPLE_SIZE with stratification by target
        df, _ = train_test_split(
            df,
            train_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            stratify=df[target],
            )
        
        df = df.reset_index(drop=True)
        logging.info(f'After deterministic sample: {df.shape}; target rate = {df[target].mean():.6f}')

        # Upload sampled audience to GP
        utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(np.int64), aud_table_month)
        aud_query = f'select distinct contact_id ::bigint as contact_id from {aud_table_month}'

        # Load SQL feature blocks
        logging.info(f'Load features...')
        df = utils.load_features(
            engine=engine,
            df=df,
            aud_query=aud_query,
            date=feature_date,
            fav_omni_features_table=fav_omni_features_table_month,
            preperiod_months=preperiod_months,
        )

        # Add derived features and run dataset validation
        df = _prepare_dataset(df)
        _validate_dataset(df, 'full_dataset')

        # Remove extreme outliers
        df['outlier'] = 0
        for feature in features_for_outliers:
            if feature in df.columns and df[feature].notna().sum() > 0:
                df.loc[df[feature] > np.nanquantile(df[feature], 0.999), 'outlier'] = 1

        outlier_share = df['outlier'].mean()

        assert outlier_share <= 0.01, 'More than 1% of the audience was removed as outliers'

        df = df[df['outlier'] == 0].drop(columns=['outlier']).reset_index(drop=True)
        _validate_dataset(df, 'full_dataset_without_outliers')

        logging.info(f'Final dataset shape: {df.shape}; target rate = {df[target].mean():.6f}')

        # Split dataset into train and holdout parts
        df, holdout_df = train_test_split(
            df,
            test_size=HOLDOUT_SIZE,
            random_state=RANDOM_STATE,
            stratify=df[target],
        )
        
        df = df.reset_index(drop=True)
        holdout_df = holdout_df.reset_index(drop=True)

        assert not (set(df['contact_id']) & set(holdout_df['contact_id'])), 'Train and holdout contact_id overlap'

        _validate_dataset(df, 'train')
        _validate_dataset(holdout_df, 'holdout')

        logging.info(f'Train shape: {df.shape}; target rate = {df[target].mean():.6f}')
        logging.info(f'Holdout shape: {holdout_df.shape}; target rate = {holdout_df[target].mean():.6f}')

        utils.save_df_to_s3(df, s3, input_bucket, f'{input_prefix}/train')
        utils.save_df_to_s3(holdout_df, s3, input_bucket, f'{input_prefix}/holdout')

        logging.info('Saved to S3')

    finally:
        # Delete temporary files
        for table in [aud_table_month, fav_omni_features_table_month]:
            try:
                utils.execute_query(engine, f'drop table if exists {table}')
            except Exception:
                logging.exception(f'Failed to drop temporary table {table}')

        Path('df_cache.parquet').unlink(missing_ok=True)

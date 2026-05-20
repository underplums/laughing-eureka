from datetime import datetime
from pathlib import Path

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import magpie.sql_utils as su

from cvm_model.io import State
import cvm_model.functions as func
import cvm_model.utils as utils
import cvm_model.sql_my as sql
from cvm_model.parameters import (
    features_for_outliers,
    aud_table,
    fav_omni_features_table,
    preperiod_months,
    uplift_rate_prefix,
    aud_suffix,
    features,
    input_suffix,
)


def preprocess_train(event_timestamp: datetime):
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    session = state.spark.session
    s3 = su.get_s3_client()

    input_prefix = state.settings.get_prefix(temp=True, suffix=input_suffix)
    input_bucket, input_prefix = input_prefix.split('//')[1].split('/')[0], '/'.join(input_prefix.split('//')[1].split('/')[1:])
    su.remove_from_s3(input_prefix, dry_run=False)
    files = su.list_all_s3_objects(s3, input_prefix)
    assert len(files) == 0, "Не очистилась папка для датасета!"

    base_month = event_timestamp.date().replace(day=1).isoformat()
    target_month = (pd.Timestamp(base_month) + pd.DateOffset(months=1)).date().isoformat()
    feature_date = target_month
    logging.info(f"Base DAC month = {base_month}")
    logging.info(f"Target DAC month = {target_month}")
    logging.info(f"Feature date = {feature_date}")

    # Выгружаем аудиторию
    aud_query = sql.aud_query.format(base_month=base_month)
    
    logging.info("Loading audience...")
    cast_cols = ['contact_id']
    df = utils.get_df(engine, aud_query).fillna(0)
    df = df.astype({col: np.int32 for col in set(cast_cols) & set(df.columns)})

    assert all([
        len(df) > 0,
        all(col in df.columns for col in ['contact_id']),
        df['contact_id'].nunique() == len(df)
    ])

    # Загружаем аудиторию во временную таблицу для ускорения селектов
    utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table)
    aud_query = f"select * from {aud_table}"

    # Загружаем recency
    logging.info("Loading recency...")
    query_kwargs = dict(
        aud=aud_query,
        date=feature_date,
        month=preperiod_months[0],
    )

    df_part = utils.get_df(engine, sql.recency_query.format(**query_kwargs))
    assert len(df_part) / len(df) >= 0.99, "Не выгрузилась recency-таблица более чем для 1% аудитории."

    df = df.merge(df_part, on='contact_id', how='left')
    df.to_parquet("df_cache.parquet")

    # Загружаем отфильтрованную аудиторию
    utils.upload_df(engine, pd.DataFrame(df['contact_id']).astype(int), aud_table)

    
    # Грузим таргет
    query_kwargs = dict(
        aud=aud_query,
        target_month=target_month,
    )

    logging.info("Loading targets...")
    cast_cols = ['target_trns', 'target_login']
    df_part = utils.get_df(engine, sql.target_query.format(**query_kwargs)).fillna(0)
    df_part = df_part.astype({col: np.int32 for col in set(cast_cols) & set(df_part.columns)})
    df_part['target_dac'] = df_part['target_trns'] * df_part['target_login']

    df = df.merge(df_part, on='contact_id')
    

    # Грузим фичи
    aud_prefix = state.settings.get_prefix(temp=True, suffix=aud_suffix)
    df = \
    utils.load_features(
        engine=engine,
        session=session,
        df=df,
        aud_query=aud_query, 
        date=feature_date,
        fav_omni_features_table=fav_omni_features_table,
        preperiod_months=preperiod_months,
        aud_prefix=aud_prefix,
        uplift_rate_prefix=uplift_rate_prefix,
    )

    df.to_parquet("df_cache.parquet")

    missing_features = set(features) - set(df.columns)
    assert len(missing_features) == 0, f"В датасете не хватает фичей: {missing_features}."

    
    # Удаляем выбросы
    q = 0.999
    df['outlier'] = 0

    for f in features_for_outliers:
        if df[f].notna().sum() == 0:
            continue
        t = np.nanquantile(df[f], q)
        df.loc[df[f] > t, 'outlier'] = 1
        
    assert len(df[df['outlier'] == 1]) / len(df) <= 0.01, "Удалено как выбросы более 1% аудитории."

    df = df[df['outlier'] == 0].reset_index(drop=True)
    df = df.drop(columns=['outlier'])


    # Загружаем в S3
    logging.info(f"Сохраняем датасет в {input_prefix}...")
    su.save_to_s3(df, input_prefix, input_type='df', bucket=input_bucket)

    # Удаляем временные файлы
    su.remove_from_s3(aud_prefix, dry_run=False)
    for table in [aud_table, fav_omni_features_table]:
        utils.execute_query(engine, f"drop table if exists {table}")
    Path("df_cache.parquet").unlink(missing_ok=True)

from datetime import datetime
from pathlib import Path

import logging

import numpy as np
import pandas as pd
import magpie.sql_utils as su

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
pd.set_option("display.max_columns", None)


RECENCY_COLS = [
    "cheque_recency",
    "login_recency",
    "omni_qr_recency",
    "omni_features_recency",
    "perf_recency",
]


def add_dac_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Add DAC-history segment used for train stratification and reporting."""

    df = df.copy()
    zero_flag = pd.Series(0, index=df.index)

    conditions = [
        df.get("is_stable_dac", zero_flag) == 1,
        df.get("is_regular_dac", zero_flag) == 1,
        df.get("is_unstable_dac", zero_flag) == 1,
        df.get("is_new_dac", zero_flag) == 1,
    ]
    choices = ["stable_10_12m", "regular_6_9m", "unstable_2_5m", "new_1m"]

    df["dac_segment_12m"] = np.select(conditions, choices, default="no_history")
    df["segment"] = df["dac_segment_12m"].astype(str)

    return df


def prepare_train_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic preprocessing that must be shared by train runs."""

    df = df.copy()

    df["contact_id"] = df["contact_id"].astype("int64")
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in RECENCY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    df = add_dac_segment(df)
    df["strat"] = df["segment"].astype(str) + "_" + df[target].astype(str)

    strat_counts = df["strat"].value_counts()
    if (strat_counts < 5).any():
        logging.info("Small strata found, fallback to target-only stratification.")
        df["strat"] = df[target].astype(str)

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers by selected numeric feature quantiles."""

    df = df.copy()
    q = 0.999
    df["outlier"] = 0

    for feature in features_for_outliers:
        if feature not in df.columns or df[feature].notna().sum() == 0:
            continue
        threshold_value = np.nanquantile(df[feature], q)
        df.loc[df[feature] > threshold_value, "outlier"] = 1

    outlier_share = len(df[df["outlier"] == 1]) / len(df)
    assert outlier_share <= 0.01, "Удалено как выбросы более 1% аудитории."

    logging.info(f"Outlier share: {outlier_share:.6f}")
    return df[df["outlier"] == 0].drop(columns=["outlier"]).reset_index(drop=True)


def validate_train_dataset(df: pd.DataFrame) -> None:
    """Fail fast if the training dataset is malformed."""

    missing_required_cols = {"contact_id", target} - set(df.columns)
    assert not missing_required_cols, f"В датасете нет обязательных колонок: {missing_required_cols}"

    missing_features = set(features) - set(df.columns)
    assert not missing_features, f"В датасете не хватает фичей: {missing_features}"

    leakage_cols = {
        "contact_id",
        target,
        "target_churn_from_dac",
        "is_dac_next_month",
        "score",
        "class_0",
        "strat",
    }
    bad_features = sorted(set(features) & leakage_cols)
    assert not bad_features, f"Служебные/таргетные колонки попали в features: {bad_features}"

    assert len(df) > 0, "Не выгружено ни одной строки train dataset."
    assert df["contact_id"].nunique() == len(df), "contact_id не уникален."
    assert set(df[target].dropna().unique()).issubset({0, 1}), f"{target} должен быть бинарным."


def preprocess_train(event_timestamp: datetime):
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    s3 = su.get_s3_client()

    input_prefix = state.settings.get_prefix(temp=True, suffix=input_suffix)
    input_bucket = input_prefix.split("//")[1].split("/")[0]
    input_prefix = "/".join(input_prefix.split("//")[1].split("/")[1:])

    base_month = event_timestamp.date().replace(day=1).isoformat()
    target_month = (pd.Timestamp(base_month) + pd.DateOffset(months=1)).date().isoformat()
    feature_date = target_month

    logging.info(f"Base DAC month = {base_month}")
    logging.info(f"Target month = {target_month}")
    logging.info(f"Feature date = {feature_date}")

    try:
        logging.info(f"Cleaning S3 input prefix: s3://{input_bucket}/{input_prefix}")
        su.remove_from_s3(input_prefix, dry_run=False)
        files = su.list_all_s3_objects(s3, input_prefix)
        assert len(files) == 0, "Не очистилась папка для датасета!"

        logging.info("Loading DAC audience...")
        aud_query_raw = sql.aud_query.format(base_month=base_month)
        df = utils.get_df(engine, aud_query_raw).fillna(0)
        df = df.astype({col: np.int32 for col in {"contact_id"} & set(df.columns)})

        assert len(df) > 0, "Не выгружено ни одного контакта DAC-аудитории."
        assert df["contact_id"].nunique() == len(df), "DAC-аудитория содержит дубли contact_id."

        logging.info(f"Audience shape: {df.shape}")

        logging.info(f"Uploading audience to GP temp table: {aud_table}")
        utils.upload_df(engine, pd.DataFrame(df["contact_id"]).astype(int), aud_table)
        aud_query = f"select distinct contact_id :: int as contact_id from {aud_table}"
        df.to_parquet("df_cache.parquet")

        logging.info("Loading target...")
        query_kwargs = {
            "aud": aud_query,
            "target_month": target_month,
        }
        df_part = utils.get_df(engine, sql.target_query.format(**query_kwargs))
        cast_cols = {target} & set(df_part.columns)
        df_part = df_part.astype({col: np.int32 for col in cast_cols})

        assert df_part["contact_id"].is_unique, "target_query вернул дубли contact_id."

        df = df.merge(df_part, on="contact_id", how="inner")
        assert len(df) > 0, "После разметки target датасет пуст."
        logging.info(f"After target: {df.shape}; target rate = {df[target].mean():.6f}")

        # Restrict feature SQL blocks to the final train population.
        utils.upload_df(engine, pd.DataFrame(df["contact_id"]).astype(int), aud_table)
        aud_query = f"select distinct contact_id :: int as contact_id from {aud_table}"

        logging.info("Loading feature blocks...")
        df = utils.load_features(
            engine=engine,
            df=df,
            aud_query=aud_query,
            date=feature_date,
            fav_omni_features_table=fav_omni_features_table,
            preperiod_months=preperiod_months,
        )
        df.to_parquet("df_cache.parquet")

        df = prepare_train_dataset(df)
        validate_train_dataset(df)

        df = remove_outliers(df)
        validate_train_dataset(df)

        logging.info(f"Final train dataset shape: {df.shape}")
        logging.info(f"Final target rate: {df[target].mean():.6f}")

        logging.info(f"Saving train dataset to s3://{input_bucket}/{input_prefix}")
        su.save_to_s3(
            df,
            input_prefix,
            input_type="df",
            bucket=input_bucket,
        )

    finally:
        logging.info("Cleaning temporary local and GP objects...")
        for table in [aud_table, fav_omni_features_table]:
            try:
                utils.execute_query(engine, f"drop table if exists {table}")
            except Exception:
                logging.exception(f"Failed to drop temporary table {table}")

        Path("df_cache.parquet").unlink(missing_ok=True)

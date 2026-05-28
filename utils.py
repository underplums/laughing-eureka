import io
import logging
import time
from typing import Any, Callable, List, Sequence, TypeVar, cast

import numpy as np
import pandas as pd
import psycopg
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from tqdm import tqdm

import cvm_model.sql as sql


T = TypeVar("T")


def _is_transient_db_error(error: Exception) -> bool:
    """Return True for connection-level errors worth retrying."""

    error_text = str(error).lower()
    transient_markers = [
        "eof detected",
        "ssl syscall error",
        "remote server read/write error",
        "terminating connection",
        "server closed the connection",
        "connection not open",
        "connection already closed",
        "odyssey",
    ]

    return isinstance(error, (OperationalError, DBAPIError, psycopg.OperationalError)) or any(
        marker in error_text for marker in transient_markers
    )


def _run_with_retries(
    engine: Engine,
    operation: Callable[[], T],
    operation_name: str,
    retries: int = 3,
    sleep_seconds: int = 15,
) -> T:
    """Run a DB operation with engine reset between transient failures."""

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            engine.dispose()

            if not _is_transient_db_error(error) or attempt == retries:
                logging.exception(f"{operation_name} failed on attempt {attempt}/{retries}")
                raise

            wait_seconds = sleep_seconds * attempt
            logging.warning(
                f"{operation_name} failed on attempt {attempt}/{retries}: {error}. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


def get_df(engine: Engine, query: str, disable_broadcast: bool = False) -> pd.DataFrame:
    """Load query result from GP into pandas with fresh connections and retries."""

    def operation() -> pd.DataFrame:
        with engine.connect() as conn:
            if disable_broadcast:
                conn.execute(text("set optimizer_enable_motion_broadcast = off"))
            conn.execute(text("set optimizer = on"))
            df = pd.read_sql_query(text(query), conn)

        engine.dispose()
        return df

    return _run_with_retries(engine, operation, "get_df")


def get_df_stream(engine: Engine, query: str, disable_broadcast: bool = False) -> pd.DataFrame:
    """Load a large query result from GP into pandas by chunks."""

    chunk_size = 250_000

    def operation() -> pd.DataFrame:
        with engine.connect() as conn:
            if disable_broadcast:
                conn.execute(text("set optimizer_enable_motion_broadcast = off"))
            conn.execute(text("set optimizer = on"))

            stream_conn = conn.execution_options(
                stream_results=True,
                max_row_buffer=chunk_size,
            )
            chunks = pd.read_sql_query(text(query), stream_conn, chunksize=chunk_size)

            all_chunks = []
            with tqdm(desc="Загрузка данных", unit=" chunk") as pbar:
                for chunk in chunks:
                    all_chunks.append(chunk)
                    pbar.update(1)
                    pbar.set_postfix({"rows": f"{sum(len(c) for c in all_chunks):,}"})

        engine.dispose()
        if not all_chunks:
            return pd.DataFrame()

        return pd.concat(all_chunks, ignore_index=True)

    return _run_with_retries(engine, operation, "get_df_stream")


def execute_query(
    engine: Engine,
    query: str,
    disable_broadcast: bool = False,
    enable_optimizer: bool = True,
) -> None:
    """Execute arbitrary SQL query in GP with connection reset on transient failures."""

    def operation() -> None:
        with engine.begin() as conn:
            if disable_broadcast:
                conn.execute(text("set optimizer_enable_motion_broadcast = off"))
            if enable_optimizer:
                conn.execute(text("set optimizer = on"))
            conn.execute(text(query))

        engine.dispose()

    _run_with_retries(engine, operation, "execute_query")


def copy_dataframe_csv(
    engine: Engine,
    df: pd.DataFrame,
    table_name: str,
    batch_size_mb: float = 100,
) -> None:
    """Upload pandas DataFrame into GP table through COPY."""

    total_size = float(df.memory_usage(deep=True).sum())
    total_rows = len(df)
    bytes_per_row = total_size / max(total_rows, 1)
    rows_per_batch = max(int((batch_size_mb * 1024 * 1024) // max(bytes_per_row, 1)), 1)

    logging.info(
        f"trying to load using csv: table_name: {table_name}; "
        f"total_size_mb: {total_size / 1024 / 1024:_.2f}; "
        f"total_rows: {total_rows:_}; bytes_per_row {bytes_per_row:_.2f}; "
        f"rows_per_batch: {rows_per_batch:_}..."
    )

    copy_sql = f"COPY {table_name} ({', '.join(df.columns)}) FROM STDIN WITH CSV".encode()

    def operation() -> None:
        conn = cast(psycopg.Connection[Any], engine.raw_connection())
        try:
            with conn.cursor() as cursor:
                with cursor.copy(copy_sql) as copy:
                    for batch_n, start in enumerate(range(0, len(df), rows_per_batch)):
                        logging.info(f"loading batch: {batch_n:05d}...")
                        end = start + rows_per_batch
                        batch = df.iloc[start:end]

                        csv_buffer = io.StringIO()
                        batch.to_csv(csv_buffer, index=False, header=False)
                        csv_buffer.seek(0)

                        copy.write(csv_buffer.getvalue())

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
            engine.dispose()

    _run_with_retries(engine, operation, f"copy_dataframe_csv({table_name})")
    logging.info(f"done loading into: {table_name}")


def upload_df(
    engine: Engine,
    df: pd.DataFrame,
    table: str,
    rewrite: bool = True,
    distributed_by_contact: bool = True,
) -> None:
    """Upload pandas DataFrame into a GP table."""

    if rewrite:
        execute_query(engine, f"drop table if exists {table}")

        format_mapper = {
            "object": "varchar",
            "int32": "int",
            "int64": "int",
            "Int32": "int",
            "Int64": "int",
            "float64": "real",
            "float32": "real",
            "bool": "bool",
            "datetime64[ns]": "date",
        }

        columns_str = ""
        for i, column in enumerate(df.columns):
            columns_str += ", " if i > 0 else ""
            columns_str += f"{column} {format_mapper[str(df[column].dtype)]}"
            columns_str += "\n" if i + 1 < len(df.columns) else ""

        create_table_query = sql.create_table_from_cols_query.format(
            table=table,
            cols=columns_str,
        )

        if distributed_by_contact is False:
            create_table_query = create_table_query.replace(
                "distributed by(contact_id)",
                "distributed randomly",
            )

        execute_query(engine, create_table_query)
        execute_query(engine, f"grant select on {table} to cvm_sbx")
        execute_query(engine, f"grant select on {table} to lipchanskiy_k_v")
        execute_query(engine, f"grant select on {table} to kirilinaea")

    copy_dataframe_csv(engine, df, table)


def merge_features(df: pd.DataFrame, df_part: pd.DataFrame, block_name: str) -> pd.DataFrame:
    """Merge a feature block and fail fast if it duplicates audience rows."""

    assert "contact_id" in df_part.columns, f"{block_name}: нет contact_id в df_part"
    assert df_part["contact_id"].is_unique, f"{block_name}: df_part содержит дубли contact_id"

    rows_before = len(df)
    cols_before = df.shape[1]
    df = df.merge(df_part, on="contact_id", how="left")

    assert len(df) == rows_before, f"{block_name}: после merge изменилось число строк"
    logging.info(
        f"{block_name}: added {df.shape[1] - cols_before} columns. "
        f"Dataset shape: {df.shape}"
    )

    return df


def _feature_query_plan(preperiod_months: Sequence[int]) -> list[tuple[str, str, Sequence[int], bool]]:
    """Describe feature SQL blocks: name, query, periods, use_stream."""

    return [
        ("Чеки", sql.cheque_query, [preperiod_months[0]], True),
        ("Чеки (сокращ.)", sql.cheque_query_short, preperiod_months[1:], True),
        ("Логины", sql.app_query, preperiod_months, True),
        ("QR", sql.omni_qr_query, preperiod_months, False),
        ("ОМНИ фичи", sql.omni_features_query, preperiod_months, False),
        ("Любимые ОМНИ фичи", sql.fav_omni_features_select_query, preperiod_months, False),
        ("Уникальные ОМНИ фичи", sql.omni_unique_features_count_query, preperiod_months, False),
        ("Миссии", sql.omni_goals_query, preperiod_months[1:], True),
        ("Акцепты", sql.accept_query, preperiod_months, True),
        ("Бонусы", sql.bonus_query, preperiod_months, False),
        ("Уровни", sql.level_query, preperiod_months, False),
    ]


def _load_sql_block(
    engine: Engine,
    query_template: str,
    query_kwargs: dict[str, Any],
    block_name: str,
    use_stream: bool = False,
) -> pd.DataFrame:
    query = query_template.format(**query_kwargs)
    if use_stream:
        return get_df_stream(engine, query)
    return get_df(engine, query)


def _add_tendency_features(df: pd.DataFrame, calc_period: int) -> pd.DataFrame:
    df = df.copy()
    df["transaction_tendency"] = df["cheque_recency"] / np.maximum(
        1, df[f"trans_lag_avg_{calc_period}"]
    )
    df["login_tendency"] = df["login_recency"] / np.maximum(
        1, df[f"login_lag_avg_{calc_period}"]
    )
    df["omni_qr_tendency"] = df["omni_qr_recency"] / np.maximum(
        1, df[f"omni_qr_lag_avg_{calc_period}"]
    )
    df["omni_features_tendency"] = df["omni_features_recency"] / np.maximum(
        1, df[f"omni_features_lag_avg_{calc_period}"]
    )
    return df


def _add_dac_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dac_months_count"] = df["dac_months_count"].fillna(0)
    df["dac_months_last_3"] = df["dac_months_last_3"].fillna(0)
    df["dac_months_last_6"] = df["dac_months_last_6"].fillna(0)
    df["dac_months_last_12"] = df["dac_months_last_12"].fillna(0)

    if "dac_age_months" in df.columns:
        df["dac_age_months"] = df["dac_age_months"].fillna(0)
        df.loc[df["dac_age_months"] == 0, "dac_age_months"] = 1
        df["dac_months_per_dac_age_ratio"] = df["dac_months_count"] / df["dac_age_months"]

    df["dac_share_last_12"] = df["dac_months_last_12"] / 12
    df["is_stable_dac"] = (df["dac_months_last_12"] >= 10).astype(np.int32)
    df["is_regular_dac"] = df["dac_months_last_12"].between(6, 9).astype(np.int32)
    df["is_unstable_dac"] = df["dac_months_last_12"].between(2, 5).astype(np.int32)
    df["is_new_dac"] = (df["dac_months_last_12"] == 1).astype(np.int32)

    return df


def _fill_default_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    null_cols = [
        col
        for col in df.columns
        if any(marker in col for marker in ["count", "sum", "rto", "aov"])
    ]
    df[null_cols] = df[null_cols].fillna(0)

    return df


def load_features(
    engine: Engine,
    df: pd.DataFrame,
    aud_query: str,
    date: str,
    fav_omni_features_table: str,
    preperiod_months: List[int],
) -> pd.DataFrame:
    """Load model feature blocks from GP and merge them into the base dataset."""

    query_kwargs: dict[str, Any] = {
        "aud": aud_query,
        "date": date,
        "month": preperiod_months[0],
    }

    logging.info("Loading recency features...")
    df_part = get_df(
        engine,
        sql.recency_query.format(
            aud=aud_query,
            date=date,
            month=preperiod_months[2],
        ),
    )
    df = merge_features(df, df_part, "recency")

    logging.info("Loading perf recency features...")
    df_part = get_df(
        engine,
        sql.perf_recency_query.format(
            aud=aud_query,
            date=date,
            month=preperiod_months[2],
        ),
    )
    df = merge_features(df, df_part, "perf_recency")
    if "perf_recency" in df.columns:
        df["perf_recency"] = df["perf_recency"].fillna(999)

    logging.info("Creating favourite OMNI features temp table...")
    fav_omni_query = sql.fav_omni_features_create_query.format(**query_kwargs)
    create_query = sql.create_table_from_select_query.format(
        table=fav_omni_features_table,
        query=fav_omni_query,
        distribution_col="contact_id",
    )
    execute_query(engine, f"drop table if exists {fav_omni_features_table}")
    execute_query(engine, create_query)

    query_kwargs["fav_omni_features_table"] = fav_omni_features_table

    for block_name, query_template, periods, use_stream in _feature_query_plan(preperiod_months):
        for month in periods:
            logging.info(f"Выгружаем {block_name} за {month} мес...")
            query_kwargs["month"] = month

            df_part = _load_sql_block(
                engine=engine,
                query_template=query_template,
                query_kwargs=query_kwargs,
                block_name=f"{block_name}_{month}m",
                use_stream=use_stream,
            )
            df = merge_features(df, df_part, f"{block_name}_{month}m")
            df.to_parquet("df_cache.parquet")

    calc_period = preperiod_months[0]
    df = _add_tendency_features(df, calc_period)

    logging.info("Выгружаем статичные признаки...")
    df_part = get_df(engine, sql.static_features_query.format(**query_kwargs))
    df = merge_features(df, df_part, "static_features")

    logging.info("Создание признаков сегментации DAC...")
    df_part = get_df(engine, sql.dac_months_count_query.format(**query_kwargs))
    df = merge_features(df, df_part, "dac_history")
    df = _add_dac_history_features(df)

    df = _fill_default_nulls(df)
    return df

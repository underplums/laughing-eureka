import io
import logging
import time
from typing import Any, Callable, List, TypeVar, cast

import pandas as pd
import psycopg
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from tqdm import tqdm

import cvm_model.sql as sql


T = TypeVar('T')


def _retry(engine: Engine, fn: Callable[[], T], name: str, retries: int = 3) -> T:
    # Retry only connection-level failures; SQL/data errors should fail immediately.
    markers = [
        'eof detected',
        'ssl syscall error',
        'remote server read/write error',
        'terminating connection',
        'server closed the connection',
        'connection not open',
        'connection already closed',
        'odyssey',
    ]

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as error:
            engine.dispose()
            is_transient = isinstance(error, (OperationalError, DBAPIError, psycopg.OperationalError))
            is_transient = is_transient or any(marker in str(error).lower() for marker in markers)

            if not is_transient or attempt == retries:
                logging.exception(f'{name} failed on attempt {attempt}/{retries}')
                raise

            wait_seconds = 15 * attempt
            logging.warning(f'{name} failed on attempt {attempt}/{retries}. Retrying in {wait_seconds} seconds...')
            time.sleep(wait_seconds)

    raise RuntimeError(f'{name} failed')


def get_df(engine: Engine, query: str, disable_broadcast: bool = False) -> pd.DataFrame:
    # Use a fresh connection per query to avoid stale Odyssey/GP connections.
    def run() -> pd.DataFrame:
        with engine.connect() as conn:
            if disable_broadcast:
                conn.execute(text('set optimizer_enable_motion_broadcast = off'))
            conn.execute(text('set optimizer = on'))
            df = pd.read_sql_query(text(query), conn)
        engine.dispose()
        return df

    return _retry(engine, run, 'get_df')


def get_df_stream(engine: Engine, query: str, disable_broadcast: bool = False) -> pd.DataFrame:
    # Chunked loading is useful for wide or heavy feature blocks.
    chunk_size = 250_000

    def run() -> pd.DataFrame:
        chunks = []
        with engine.connect() as conn:
            if disable_broadcast:
                conn.execute(text('set optimizer_enable_motion_broadcast = off'))
            conn.execute(text('set optimizer = on'))
            stream_conn = conn.execution_options(stream_results=True, max_row_buffer=chunk_size)
            reader = pd.read_sql_query(text(query), stream_conn, chunksize=chunk_size)

            with tqdm(desc='Loading data', unit=' chunk') as pbar:
                for chunk in reader:
                    chunks.append(chunk)
                    pbar.update(1)
                    pbar.set_postfix({'rows': f'{sum(len(x) for x in chunks):,}'})

        engine.dispose()
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    return _retry(engine, run, 'get_df_stream')


def execute_query(engine: Engine, query: str, disable_broadcast: bool = False) -> None:
    def run() -> None:
        with engine.begin() as conn:
            if disable_broadcast:
                conn.execute(text('set optimizer_enable_motion_broadcast = off'))
            conn.execute(text('set optimizer = on'))
            conn.execute(text(query))
        engine.dispose()

    _retry(engine, run, 'execute_query')


def upload_df(engine: Engine, df: pd.DataFrame, table: str, rewrite: bool = True) -> None:
    # Create a GP table with simple dtype mapping, then load data through COPY.
    if rewrite:
        mapper = {
            'object': 'varchar',
            'int32': 'int',
            'int64': 'int',
            'Int32': 'int',
            'Int64': 'int',
            'float64': 'real',
            'float32': 'real',
            'bool': 'bool',
            'datetime64[ns]': 'date',
        }
        cols = ',\n'.join(f'{col} {mapper[str(df[col].dtype)]}' for col in df.columns)

        execute_query(engine, f'drop table if exists {table}')
        execute_query(engine, sql.create_table_from_cols_query.format(table=table, cols=cols))
        execute_query(engine, f'grant select on {table} to cvm_sbx')
        execute_query(engine, f'grant select on {table} to lipchanskiy_k_v')
        execute_query(engine, f'grant select on {table} to kirilinaea')

    total_size = float(df.memory_usage(deep=True).sum())
    bytes_per_row = total_size / max(len(df), 1)
    rows_per_batch = max(int((100 * 1024 * 1024) // max(bytes_per_row, 1)), 1)
    columns = ', '.join(df.columns)
    copy_sql = f'COPY {table} ({columns}) FROM STDIN WITH CSV'.encode()

    def run() -> None:
        conn = cast(psycopg.Connection[Any], engine.raw_connection())
        try:
            with conn.cursor() as cursor:
                with cursor.copy(copy_sql) as copy:
                    for batch_n, start in enumerate(range(0, len(df), rows_per_batch)):
                        logging.info(f'Loading {table} batch {batch_n:05d}')
                        buffer = io.StringIO()
                        df.iloc[start:start + rows_per_batch].to_csv(buffer, index=False, header=False)
                        buffer.seek(0)
                        copy.write(buffer.getvalue())
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
            engine.dispose()

    _retry(engine, run, f'upload_df({table})')


def remove_s3_prefix(s3_credentials: Any, bucket: str, prefix: str) -> None:
    clean_prefix = prefix.strip('/')
    path = f'{bucket}/{clean_prefix}'
    if s3_credentials.s3fs.exists(path):
        s3_credentials.s3fs.rm(path, recursive=True)


def list_s3_objects(s3_credentials: Any, bucket: str, prefix: str) -> List[str]:
    clean_prefix = prefix.strip('/')
    path = f'{bucket}/{clean_prefix}'
    return list(s3_credentials.s3fs.find(path)) if s3_credentials.s3fs.exists(path) else []


def save_df_to_s3(df: pd.DataFrame, s3_credentials: Any, bucket: str, prefix: str) -> None:
    clean_prefix = prefix.strip('/')
    path = f'{bucket}/{clean_prefix}'
    s3_credentials.s3fs.makedirs(path, exist_ok=True)
    with s3_credentials.s3fs.open(f'{path}/part-00000.parquet', 'wb') as file:
        df.to_parquet(file, index=False)


def load_features(
    engine: Engine,
    df: pd.DataFrame,
    aud_query: str,
    date: str,
    fav_omni_features_table: str,
    preperiod_months: List[int],
) -> pd.DataFrame:
    query_kwargs = {'aud': aud_query, 'date': date, 'month': preperiod_months[0]}

    # Load recency blocks first because later preprocessing uses them for tendency features.
    for name, query in [
        ('recency', sql.recency_query.format(aud=aud_query, date=date, month=preperiod_months[2])),
        ('perf_recency', sql.perf_recency_query.format(aud=aud_query, date=date, month=preperiod_months[2])),
    ]:
        df_part = get_df(engine, query)
        assert 'contact_id' in df_part.columns, f'{name}: contact_id is missing'
        assert df_part['contact_id'].is_unique, f'{name}: duplicate contact_id values'

        rows_before = len(df)
        cols_before = df.shape[1]
        df = df.merge(df_part, on='contact_id', how='left')
        assert len(df) == rows_before, f'{name}: row count changed after merge'
        logging.info(f'{name}: added {df.shape[1] - cols_before} columns; shape={df.shape}')

    df['perf_recency'] = df['perf_recency'].fillna(999)

    # Materialize favorite OMNI features once; several later queries reuse this table.
    fav_query = sql.fav_omni_features_create_query.format(**query_kwargs)
    create_query = sql.create_table_from_select_query.format(
        table=fav_omni_features_table,
        query=fav_query,
        distribution_col='contact_id',
    )
    execute_query(engine, f'drop table if exists {fav_omni_features_table}')
    execute_query(engine, create_query)
    query_kwargs['fav_omni_features_table'] = fav_omni_features_table

    # Each tuple contains: feature block name, SQL template, periods, and loading mode.
    blocks = [
        ('cheques', sql.cheque_query, [preperiod_months[0]], True),
        ('cheques_short', sql.cheque_query_short, preperiod_months[1:], True),
        ('logins', sql.app_query, preperiod_months, True),
        ('qr', sql.omni_qr_query, preperiod_months, False),
        ('omni_features', sql.omni_features_query, preperiod_months, False),
        ('favorite_omni_features', sql.fav_omni_features_select_query, preperiod_months, False),
        ('unique_omni_features', sql.omni_unique_features_count_query, preperiod_months, False),
        ('missions', sql.omni_goals_query, preperiod_months[1:], True),
        ('accepts', sql.accept_query, preperiod_months, True),
        ('bonuses', sql.bonus_query, preperiod_months, False),
        ('levels', sql.level_query, preperiod_months, False),
        ('static_features', sql.static_features_query, [None], False),
        ('dac_history', sql.dac_months_count_query, [None], False),
    ]

    for name, query, months, stream in blocks:
        for month in months:
            if month is None:
                logging.info(f'Loading {name} features')
                block_name = name
            else:
                logging.info(f'Loading {name} features for {month} months')
                query_kwargs['month'] = month
                block_name = f'{name}_{month}m'

            loader = get_df_stream if stream else get_df

            df_part = loader(engine, query.format(**query_kwargs))
            assert 'contact_id' in df_part.columns, f'{block_name}: contact_id is missing'
            assert df_part['contact_id'].is_unique, f'{block_name}: duplicate contact_id values'

            rows_before = len(df)
            cols_before = df.shape[1]
            df = df.merge(df_part, on='contact_id', how='left')
            assert len(df) == rows_before, f'{block_name}: row count changed after merge'
            logging.info(f'{block_name}: added {df.shape[1] - cols_before} columns; shape={df.shape}')
            df.to_parquet('df_cache.parquet')

    # Missing counters and sums mean no matching activity, so zero is the natural value.
    null_cols = [col for col in df.columns if any(marker in col for marker in ['count', 'sum', 'rto', 'aov'])]
    df[null_cols] = df[null_cols].fillna(0)

    return df

from sqlalchemy import text
import io
import logging
import numpy as np
import pandas as pd
import psycopg
import cvm_model.sql as sql
from tqdm import tqdm

# For type hints:
from typing import Any, List, cast
from sqlalchemy.engine import Engine


def get_df(engine: Engine, query: str, disable_broadcast: bool = False) -> pd.DataFrame:
    """Загружает данные из БД в Pandas

    Args:
        query: SQL скрипт
        disable_broadcast: отключать или нет бродкаст в запросе

    Returns:
        df: pandas.DataFrame с данными
    """

    try:
        with engine.connect() as conn:
            if disable_broadcast:
                conn.execute(text("set optimizer_enable_motion_broadcast = off"))
            conn.execute(text("set optimizer = on"))
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as ex:
        print(ex)
        raise
        
    return df


def get_df_stream(engine: Engine, query: str, disable_broadcast: bool = False) -> pd.DataFrame:
    """Загружает данные из БД в Pandas

    Args:
        query: SQL скрипт
        disable_broadcast: отключать или нет бродкаст в запросе

    Returns:
        df: pandas.DataFrame с данными
    """

    chunk_size = 250_000

    try:
        with engine.connect() as conn:
            if disable_broadcast:
                conn.execute(text("set optimizer_enable_motion_broadcast = off"))
            conn.execute(text("set optimizer = on"))
            
            stream_conn = conn.execution_options(
                stream_results=True, 
                max_row_buffer=chunk_size
                )
            query = text(query)
            chunks = pd.read_sql(query, stream_conn, chunksize=chunk_size)
            all_chunks = []
            with tqdm(desc="Загрузка данных", unit=" chunk") as pbar:
                for chunk in chunks:
                    all_chunks.append(chunk)
                    pbar.update(1)
                    pbar.set_postfix({"rows": f"{len(all_chunks) * chunk_size:,}"})
            df = pd.concat(all_chunks, ignore_index=True)
    except Exception as ex:
        print(ex)
        raise
        
    return df


def execute_query(
        engine: Engine,
        query: str, 
        disable_broadcast: bool = False,
        enable_optimizer: bool = True
) -> None:
    """Выполняет произвольный запрос к БД

    Args:
        query: SQL скрипт
        disable_broadcast: отключать или нет бродкаст в запросе
        enable_optimizer: включать или нет оптимизатор

    Returns:
        None
    """

    with engine.begin() as conn:
        if disable_broadcast:
            conn.execute(text("set optimizer_enable_motion_broadcast = off"))
        if enable_optimizer:
            conn.execute(text("set optimizer = on"))
        conn.execute(text(query))
    engine.dispose()


def copy_dataframe_csv(
    engine: Engine, df: pd.DataFrame, table_name: str, batch_size_mb: float = 100
):
    # Прикидываем сколько весит одна строка
    total_size = float(df.memory_usage(deep=True).sum())  # pyright: ignore[reportUnknownArgumentType] пандас крутая либа
    total_rows = len(df)
    bytes_per_row = total_size / total_rows
    rows_per_batch = max(int((batch_size_mb * 1024 * 1024) // bytes_per_row), 1)

    logging.info(
        f"trying to load using csv: table_name: {table_name}; total_size_mb: {total_size / 1024 / 1024:_.2f}; total_rows: {total_rows:_}; bytes_per_row {bytes_per_row:_.2f}; rows_per_batch: {rows_per_batch:_}..."
    )
    copy_sql = (
        f"COPY {table_name} ({', '.join(df.columns)}) FROM STDIN WITH CSV".encode()
    )

    conn = cast(psycopg.Connection[Any], engine.raw_connection())
    try:
        with conn.cursor() as cursor:
            with cursor.copy(copy_sql) as copy:
                # Батчево конвертируем в csv и загружаем в базу
                for batch_n, start in enumerate(range(0, len(df), rows_per_batch)):
                    logging.info(f"loading batch: {batch_n:05d}...")
                    end = start + rows_per_batch
                    batch = df.iloc[start:end]

                    csv_buffer = io.StringIO()
                    batch.to_csv(csv_buffer, index=False, header=False)
                    csv_buffer.seek(0)

                    copy.write(csv_buffer.getvalue())

        conn.commit()
    finally:
        conn.close()

    logging.info(f"done loading into: {table_name}")


def upload_df(
    engine: Engine,
    df: pd.DataFrame,
    table: str,
    rewrite: bool = True,
    distributed_by_contact: bool = True,
) -> None:
    """Загружает датафрейм в БД
    
    Args:
        df: датафрейм
        table: таблица в БД
        force_null: заполнять или нет пропуски нулами
        rewrite: перезаписывать таблицу или дополнять
        
    Returns:
        None
    """
        
    if rewrite:
        # Стираем таблицу в БД, если есть
        execute_query(engine, f"drop table if exists {table}")

        # Создаём новую таблицу в БД
        format_mapper = {
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

        columns_str = ''
        for i, c in enumerate(df.columns):
            columns_str += ', ' if i > 0 else ''
            columns_str += f'{c} {format_mapper[str(df[c].dtype)]}' 
            columns_str += '\n' if i + 1 < len(df.columns) else ''

        create_table_query = sql.create_table_from_cols_query.format(
            table=table,
            cols=columns_str,
        )
        
        if distributed_by_contact == False:
            create_table_query = create_table_query.replace('distributed by(contact_id)', 'distributed randomly')

        execute_query(engine, create_table_query)

        # Даём грант на селект песочнице
        execute_query(engine, f"grant select on {table} to cvm_sbx")
        execute_query(engine, f"grant select on {table} to lipchanskiy_k_v")
        execute_query(engine, f"grant select on {table} to kirilinaea")

    copy_dataframe_csv(engine, df, table)
    
    
def load_features(
    engine: Engine,
    df: pd.DataFrame,
    aud_query: str, 
    date: str,
    fav_omni_features_table: str,
    preperiod_months: List[int],
    sql_module: Any = None,
) -> pd.DataFrame:
    """"""
    sql_used = sql_module or sql

    query_kwargs = dict(
        aud=aud_query,
        date=date,
        month=preperiod_months[0],
    )
    
    # Создаём временную таблицу с любимыми ОМНИ фичами
    logging.info("Creating favourite OMNI features temp table...")
    query = sql_used.fav_omni_features_create_query.format(**query_kwargs)

    create_query_kwargs = dict(
        table=fav_omni_features_table,
        query=query,
        distribution_col='contact_id'
    )
    create_query = sql_used.create_table_from_select_query.format(**create_query_kwargs)

    execute_query(engine, f"drop table if exists {fav_omni_features_table}")
    execute_query(engine, create_query)
    

    # Грузим признаки по предпериодам
    queries = {
        'Чеки': sql_used.cheque_query, 
        'Чеки (сокращ.)': sql_used.cheque_query_short,
        'Логины': sql_used.app_query,
        'QR': sql_used.omni_qr_query, 
        'ОМНИ фичи': sql_used.omni_features_query, 
        'Любимые ОМНИ фичи': sql_used.fav_omni_features_select_query,
        'Уникальные ОМНИ фичи': sql_used.omni_unique_features_count_query,
        'Миссии': sql_used.omni_goals_query,
        'Акцепты': sql_used.accept_query, 
        'Бонусы': sql_used.bonus_query,
        'Уровни': sql_used.level_query
    }
    # Разный набор периодов для разных запросов
    periods = [[preperiod_months[0]] if query == 'Чеки' else preperiod_months[1:] if query in ['Чеки (сокращ.)', 'Миссии'] else preperiod_months for query in queries]

    query_kwargs['fav_omni_features_table'] = fav_omni_features_table

    for i, query in enumerate(queries):
        for month in preperiod_months:
            if month in periods[i]:
                logging.info(f"Выгружаем {query} за {month} мес...")
                query_kwargs['month'] = month

                if query in ['QR', 'ОМНИ фичи', 'Любимые ОМНИ фичи', 'Уникальные ОМНИ фичи', 'Бонусы', 'Уровни']:
                    df_part = get_df(engine, queries[query].format(**query_kwargs))
                else:
                    df_part = get_df_stream(engine, queries[query].format(**query_kwargs))

                df = df.merge(df_part, on='contact_id', how='left')
                logging.info(f"Adding {df_part.shape[1] - 1} columns to dataset. New total columns: {df.shape[1]}")
                df.to_parquet("df_cache.parquet")
    
    # Рассчитываем tendency
    calc_period = preperiod_months[0]
    df['transaction_tendency']   = df['cheque_recency']        / np.max((np.ones(len(df)), df[f'trans_lag_avg_{calc_period}']), axis=0)
    df['login_tendency']         = df['login_recency']         / np.max((np.ones(len(df)), df[f'login_lag_avg_{calc_period}']), axis=0)
    df['omni_qr_tendency']       = df['omni_qr_recency']       / np.max((np.ones(len(df)), df[f'omni_qr_lag_avg_{calc_period}']), axis=0)
    df['omni_features_tendency'] = df['omni_features_recency'] / np.max((np.ones(len(df)), df[f'omni_features_lag_avg_{calc_period}']), axis=0)

    # Выгружаем статичные признаки
    logging.info(f"Выгружаем статичные признаки...")
    df_part = get_df(engine, sql_used.static_features_query.format(**query_kwargs))
    df = df.merge(df_part, on='contact_id', how='left')

    # Выгружаем признаки месяцев DAC
    df_part = get_df(engine, sql_used.dac_months_count_query.format(**query_kwargs))
    df = df.merge(df_part, on='contact_id', how='left')
    df.loc[df['dac_age_months'] == 0, 'dac_age_months'] = 1
    df['dac_months_per_dac_age_ratio'] = df['dac_months_count'] / df['dac_age_months']
    df['dac_share_last_12'] = df['dac_months_last_12'] / 12
    df['is_stable_dac'] = (df['dac_months_last_12'] >= 10).astype(np.int32)
    df['is_regular_dac'] = df['dac_months_last_12'].between(6, 9).astype(np.int32)
    df['is_unstable_dac'] = df['dac_months_last_12'].between(2, 5).astype(np.int32)
    df['is_new_dac'] = (df['dac_months_last_12'] == 1).astype(np.int32)

    # Заполняем нулы
    null_cols = [c for c in df.columns if any([s in c for s in ['count', 'sum', 'rto', 'aov']])]
    df[null_cols] = df[null_cols].fillna(0)
    
    return df

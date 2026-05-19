# MVP dataset for `cvm_churn-from-dac_binary-class_mvp`

Цель файла: собрать минимальный набор SQL и фичей, чтобы быстро обкатать preprocess-пайплайн для модели ухода клиента из DAC.

Файл не является production-кодом. Его удобно использовать как заготовку: блоки ниже надо перенести в `src/cvm_model/parameters.py`, `src/cvm_model/sql.py` и `src/cvm_model/train/preprocess.py` нового проекта.

## Постановка

Аудитория модели: клиенты, которые являются DAC в базовом месяце.

Таргет: уход из DAC в следующем месяце.

```text
target_churn_from_dac = 1, если клиент был DAC в base_month, но не является DAC в target_month.
target_churn_from_dac = 0, если клиент был DAC в base_month и остается DAC в target_month.
```

Актуальное определение DAC:

```sql
has_transaction_activity = 1
and (
    has_mobapp_activity = 1
    or has_pwa_activity = 1
    or vcoff_trn_cnt > 0
)
```

Рекомендуемая интерпретация дат:

```text
base_month = date_trunc('month', event_timestamp)
target_month = base_month + interval '1 month'
features_end_date = target_month
```

То есть для обучения на `base_month = 2025-11-01`:

```text
Аудитория: DAC в ноябре 2025.
Таргет: DAC / not DAC в декабре 2025.
Фичи: история до 2025-12-01, без заглядывания в декабрь.
```

## MVP-блок для `parameters.py`

```python
RANDOM_STATE = 42

project_name = "cvm_churn-from-dac_binary-class_mvp"
model_type = "catboost"
jira = "TODO"

template = "s3a://{bucket}/{prefix}"

target = "target_churn_from_dac"
score = "score"

sample_n = 10_000
preperiod_months = [6, 3, 1]

model_predictions_suffix = "model_predictions"
train_data_stat_suffix = "data_stat/train"
inference_data_stat_suffix = "data_stat/inference"

input_suffix = "input"

features = [
    "cheque_recency",
    "login_recency",
    "rto_6",
    "trans_count_6",
    "aov_6",
    "trans_lag_avg_6",
    "rto_3",
    "trans_count_3",
    "aov_3",
    "trans_lag_avg_3",
    "rto_1",
    "trans_count_1",
    "aov_1",
    "trans_lag_avg_1",
    "login_count_6",
    "login_lag_avg_6",
    "login_count_3",
    "login_lag_avg_3",
    "login_count_1",
    "login_lag_avg_1",
    "dac_months_count",
    "dac_months_per_dac_age_ratio",
]
```

Почему именно эти фичи:

- это небольшая часть признаков из текущего репозитория;
- они быстро объяснимы и удобны для первого теста;
- покрывают покупки, приложение, recency и историю DAC;
- почти все уже есть в логике текущего `sql.py` / `utils.load_features()`;
- не требуют сразу тащить сложные OMNI, бонусы, uplift и shap-артефакты.

## MVP-блок для `sql.py`

### Аудитория: DAC в базовом месяце

```python
aud_query = """
select distinct contact_id :: int
from dm.prvdr_dac_monthly
where date_month = date'{base_month}'
  and has_transaction_activity = 1
  and (
      has_mobapp_activity = 1
      or has_pwa_activity = 1
      or vcoff_trn_cnt > 0
  )
  and contact_id is not null
"""
```

### Таргет: ушел из DAC в следующем месяце

```python
target_query = """
with aud as
    (
        {aud}
    )
, dac_next_month as
    (
        select distinct contact_id :: int
            , 1 :: int is_dac_next_month
        from dm.prvdr_dac_monthly
        where date_month = date'{target_month}'
          and has_transaction_activity = 1
          and (
              has_mobapp_activity = 1
              or has_pwa_activity = 1
              or vcoff_trn_cnt > 0
          )
          and contact_id is not null
    )
select aud.contact_id
    , case
        when coalesce(d.is_dac_next_month, 0) = 1 then 0
        else 1
      end :: int as target_churn_from_dac
from aud
left join dac_next_month d using(contact_id)
"""
```

### Recency покупок и логинов

Для MVP берем `login_recency` из той же таблицы, что использует текущий репозиторий: `dm.prvrd_dau_main_metrics`.

Важно: по новой DAC-логике цифровая активность включает `mobapp`, `pwa` и `vcoff_trn_cnt`. В MVP `login_recency` можно оставить как быстрый старт, но позже лучше расширить digital-recency под PWA и offline virtual card.

```python
recency_query = """
with aud as
    (
        {aud}
    )
, cheques as
    (
        select contact_id :: int
            , date'{features_end_date}' - max(datetime::date) as cheque_recency
        from cvm_sbx.v_cheque_filtered
        where datetime :: date >= date'{features_end_date}' - interval '{month} month'
          and datetime :: date < date'{features_end_date}'
          and contact_id is not null
          and sale_channel_id in (0, 1, 2)
        group by contact_id
    )
, logins as
    (
        select contact_id :: int
            , date'{features_end_date}' - max(date(ptn_dt)) as login_recency
        from dm.prvrd_dau_main_metrics
        where ptn_dt >= date'{features_end_date}' - interval '{month} month'
          and ptn_dt < date'{features_end_date}'
          and contact_id is not null
        group by contact_id
    )
select aud.contact_id
    , cheques.cheque_recency
    , logins.login_recency
from aud
left join cheques using(contact_id)
left join logins using(contact_id)
"""
```

### Покупочные фичи

Это упрощенная версия `cheque_query_short` из текущего репозитория.

```python
cheque_features_query = """
with aud as
    (
        {aud}
    )
, cheques as
    (
        select contact_id :: int
            , summ_discounted :: real as rto
            , date(datetime) - lag(date(datetime)) over (
                partition by contact_id
                order by date(datetime)
              ) as trans_lag
            , datetime :: date as datetime
        from cvm_sbx.v_cheque_filtered
        where datetime :: date >= date'{features_end_date}' - interval '{month} month'
          and datetime :: date < date'{features_end_date}'
          and contact_id is not null
          and sale_channel_id in (0, 1, 2)
    )
select contact_id
    , sum(rto) :: real as rto_{month}
    , count(*) :: int as trans_count_{month}
    , (sum(rto) / nullif(count(*), 0)) :: real as aov_{month}
    , avg(trans_lag) :: real as trans_lag_avg_{month}
from aud
join cheques using(contact_id)
group by contact_id
"""
```

### App-фичи

Это упрощенная версия `app_query` из текущего репозитория.

```python
app_features_query = """
with logins as
    (
        select c.contact_id :: int
            , date(ptn_dt) - lag(date(ptn_dt)) over (
                partition by c.contact_id
                order by date(ptn_dt)
              ) as login_lag
        from dm.prvrd_dau_main_metrics d
        join ({aud}) c
            on c.contact_id = d.contact_id :: int
        where ptn_dt >= date'{features_end_date}' - interval '{month} month'
          and ptn_dt < date'{features_end_date}'
    )
select contact_id
    , count(*) :: int as login_count_{month}
    , avg(login_lag) :: real as login_lag_avg_{month}
from logins
group by contact_id
"""
```

### История DAC

Фичи по истории DAC удобно взять из текущего `dac_months_count_query`, но адаптировать под новое определение DAC.

```python
dac_history_query = """
with dac_history as
    (
        select contact_id :: int
            , date_month
            , date_month - (
                row_number() over (
                    partition by contact_id
                    order by date_month
                ) * interval '1 month'
              ) as grp
        from dm.prvdr_dac_monthly
        join ({aud}) aud using(contact_id)
        where date_month < date'{features_end_date}'
          and has_transaction_activity = 1
          and (
              has_mobapp_activity = 1
              or has_pwa_activity = 1
              or vcoff_trn_cnt > 0
          )
    )
, islands as
    (
        select contact_id
            , count(*) as streak_length
        from dac_history
        group by contact_id, grp
    )
select contact_id
    , sum(streak_length) :: int as dac_months_count
    , max(streak_length) :: int as max_consecutive_dac_months
    , avg(streak_length) :: real as avg_consecutive_dac_months
from islands
group by contact_id
"""
```

## MVP-логика для `train/preprocess.py`

Ниже не полный файл, а порядок действий, который надо встроить в функцию `preprocess_train(event_timestamp)`.

```python
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import magpie.sql_utils as su

from cvm_model.io import State
import cvm_model.sql as sql
import cvm_model.utils as utils
from cvm_model.parameters import (
    RANDOM_STATE,
    features,
    input_suffix,
    preperiod_months,
    sample_n,
    target,
)


def preprocess_train(event_timestamp: datetime):
    state = State.from_env()
    engine = state.credentials.loyalty_gp.sa_engine
    session = state.spark.session
    s3 = su.get_s3_client()

    input_prefix = state.settings.get_prefix(temp=True, suffix=input_suffix)
    input_bucket = input_prefix.split("//")[1].split("/")[0]
    input_prefix = "/".join(input_prefix.split("//")[1].split("/")[1:])

    su.remove_from_s3(input_prefix, dry_run=False)
    files = su.list_all_s3_objects(s3, input_prefix)
    assert len(files) == 0, "Не очистилась папка для датасета!"

    base_month = event_timestamp.date().replace(day=1).isoformat()
    target_month = pd.Timestamp(base_month) + pd.DateOffset(months=1)
    target_month = target_month.date().isoformat()
    features_end_date = target_month

    logging.info(f"Base month = {base_month}")
    logging.info(f"Target month = {target_month}")
    logging.info(f"Features end date = {features_end_date}")

    aud_query = sql.aud_query.format(base_month=base_month)

    logging.info("Loading DAC audience...")
    df = utils.get_df(engine, aud_query).fillna(0)
    df = df.astype({"contact_id": np.int32})

    assert len(df) > 0
    assert df["contact_id"].nunique() == len(df)

    logging.info("Loading target...")
    target_part = utils.get_df(
        engine,
        sql.target_query.format(
            aud=aud_query,
            target_month=target_month,
        ),
    ).fillna(0)
    target_part = target_part.astype({"contact_id": np.int32, target: np.int32})
    df = df.merge(target_part, on="contact_id", how="inner")

    query_kwargs = {
        "aud": aud_query,
        "features_end_date": features_end_date,
        "month": max(preperiod_months),
    }

    logging.info("Loading recency features...")
    df_part = utils.get_df(engine, sql.recency_query.format(**query_kwargs))
    df = df.merge(df_part, on="contact_id", how="left")

    for month in preperiod_months:
        query_kwargs["month"] = month

        logging.info(f"Loading cheque features for {month} months...")
        df_part = utils.get_df_stream(engine, sql.cheque_features_query.format(**query_kwargs))
        df = df.merge(df_part, on="contact_id", how="left")

        logging.info(f"Loading app features for {month} months...")
        df_part = utils.get_df(engine, sql.app_features_query.format(**query_kwargs))
        df = df.merge(df_part, on="contact_id", how="left")

    logging.info("Loading DAC history features...")
    df_part = utils.get_df(
        engine,
        sql.dac_history_query.format(
            aud=aud_query,
            features_end_date=features_end_date,
        ),
    )
    df = df.merge(df_part, on="contact_id", how="left")

    # Упрощенная производная фича.
    df["dac_months_count"] = df["dac_months_count"].fillna(0)
    df["dac_months_per_dac_age_ratio"] = df["dac_months_count"] / np.maximum(
        1,
        max(preperiod_months),
    )

    null_cols = [
        c for c in df.columns
        if any(s in c for s in ["count", "sum", "rto", "aov"])
    ]
    df[null_cols] = df[null_cols].fillna(0)

    missing_features = set(features) - set(df.columns)
    assert len(missing_features) == 0, f"В датасете не хватает фичей: {missing_features}"

    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=RANDOM_STATE).reset_index(drop=True)

    logging.info(f"Saving dataset with shape = {df.shape} to {input_prefix}...")
    su.save_to_s3(df, input_prefix, input_type="df", bucket=input_bucket)
```

## Что делать дальше

1. В новом проекте перенести блок `parameters.py`.
2. В новом проекте перенести SQL-блоки в `src/cvm_model/sql.py`.
3. В `src/cvm_model/train/preprocess.py` собрать функцию по MVP-логике выше.
4. Оставить `train/train.py` почти как в текущем репозитории: CatBoost уже сможет обучиться, если есть `features` и `target`.
5. Сначала запустить только `train preprocess` на одной исторической дате.
6. Проверить, что в датасете есть:
   - `contact_id`;
   - `target_churn_from_dac`;
   - все колонки из `features`;
   - не больше 10_000 строк;
   - target не состоит только из одного класса.
7. После этого запускать `train run`.

## Важные TODO перед production

- Расширить app/digital-фичи под полное новое определение DAC: mobapp + PWA + offline virtual card.
- Уточнить, какие события в `dm.prvdr_dac_monthly` считаются покупкой по программе лояльности.
- Добавить inference-preprocess без будущего таргета.
- Подумать над разметкой нескольких исторических месяцев, а не одного `event_timestamp`.
- Исправить выбор модели в inference: использовать `train_event_timestamp`, а не просто последнюю версию из MLflow.

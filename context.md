````md
# CVMB_24118 - Анализ механики "Упущенная выгода"

## Бизнес-задача

Период анализа:
- 15.12.2025 - 29.04.2026

Кампании:
- ZTAN41
- ZTAN42
- ZTAN43
- ZTAN45
- ZTAN47
- ZTAN51
- ZTAN57
- ZTAN60
- ZTAN61
- ZTANTT63

Цель:
Понять, не "устает" ли аудитория от механики УВ:
- сколько людей участвуют в большом числе кампаний,
- как меняется их поведение,
- есть ли признаки saturation/fatigue.

---

# Используемые таблицы

## 1. Кампании

```sql
cvm_sbx.mt_cvm_t_marketing_campaign
````

Поля:

* mrc_id
* mrc_desc
* mrc_com_id
* mrc_start_date

---

## 2. Коммуникации

```sql
cvm_sbx.mt_cvm_t_communication
```

Поля:

* com_id
* com_cus_gr_id
* com_cus_sgr_id
* com_start_date

---

## 3. Группы клиентов

```sql
cvm_sbx.mt_cvm_t_cust_group
```

Поля:

* cus_gr_id
* cus_sgr_id
* cus_gr_contact_id
* cus_gr_target
* mrc_start_date

---

## 4. Чеки / покупки

```sql
dm.cheque
```

Ключевые поля:

* contact_id
* datetime
* summ_discounted
* operation_type_id

Фильтры:

```sql
operation_type_id = 1
summ_discounted > 0
```

---

# Построение аудитории кампаний

Итоговая таблица:

```text
client_id | campaign_name
```

SQL логика:

* campaigns
* communications
* cust_groups
* join между ними

Ключевые join:

```sql
comm.com_id = c.mrc_com_id

cg.cus_gr_id = comm.com_cus_gr_id
cg.cus_sgr_id = comm.com_cus_sgr_id
cg.mrc_start_date = c.mrc_start_date
```

---

# Таблица campaigns_audience

Содержит:

```text
client_id
campaign_name
```

Используется:

* для построения когорт,
* для будущих analyses,
* для номиналов,
* для ML-score.

Ее нужно хранить.

---

# Когорты

Определение:

Когорта = группа клиентов, попавших в одинаковое число кампаний УВ за период.

Пример:

* campaigns_cnt = 1
* campaigns_cnt = 2
* ...
* campaigns_cnt = 8

---

# Построение когорт

DuckDB:

```python
import duckdb

con = duckdb.connect()

con.execute("""
create or replace table campaigns_audience as
select distinct
    cast(client_id as bigint) as client_id,
    campaign_name
from read_csv_auto('data/campaigns_audience.csv')
where client_id is not null
  and campaign_name is not null
""")

con.execute("""
create or replace table client_cohorts as
select
    client_id,
    count(distinct campaign_name) as campaigns_cnt
from campaigns_audience
group by client_id
""")
```

---

# Почему когорт меньше, чем audience

Потому что:

```text
один клиент может участвовать сразу в нескольких кампаниях
```

В cohorts:

```text
1 строка = 1 клиент
```

В campaigns_audience:

```text
1 строка = 1 клиент + 1 кампания
```

---

# Ответ на вопрос №1

## Что считалось

Для каждой когорты:

* число клиентов,
* доля клиентов.

---

# Итоговая таблица

```text
campaigns_cnt
client_cnt
client_share_pct
```

---

# SQL / pandas логика

```python
df_base = (
    df[['client_id', 'campaign_name']]
    .drop_duplicates()
)

cohorts = (
    df_base
    .groupby('client_id')
    .agg(
        campaigns_cnt=('campaign_name', 'nunique')
    )
    .reset_index()
)

result_1 = (
    cohorts
    .groupby('campaigns_cnt')
    .agg(
        client_cnt=('client_id', 'nunique')
    )
    .reset_index()
)

result_1['client_share_pct'] = round(
    result_1['client_cnt'] * 100.0 /
    result_1['client_cnt'].sum(),
    2
)
```

---

# Визуализация вопроса №1

2 графика:

* абсолютное число клиентов,
* доля клиентов.

На одной figure.

---

# Таблица cohort в GP

Создана:

```sql
cvm_sbx.cvmb_23662_client_cohorts
```

Содержит:

```text
client_id
campaigns_cnt
```

---

# Загрузка CSV в GP

Через DBeaver import.

Проблемы:

* очень медленно,
* commit каждые 10k строк,
* 22 млн строк.

---

# Важное замечание

Лучше хранить ОБЕ таблицы:

* campaigns_audience
* client_cohorts

Почему:

* cohorts нужны для агрегатов,
* audience нужна для:

  * номиналов,
  * ML-score,
  * CTR,
  * переходов,
  * trajectory analyses.

---

# RTO / траты

Изначально RTO считался как:

```text
доля клиентов с покупками
```

Но значения были почти 99%.

Методология признана неудачной.

---

# Новая метрика

Используется:

```text
траты клиентов
```

На основе:

```sql
dm.cheque.summ_discounted
```

---

# Baseline методология

Проблема:

* кампании стартуют 15 декабря,
* декабрь получался "половинчатым".

Исправление:

* baseline = полный ноябрь,
* анализ = декабрь-апрель.

---

# Итоговая методология

## baseline

```text
01.11.2025 - 30.11.2025
```

## анализ

```text
01.12.2025 - 30.04.2026
```

---

# Delta spend

Основная метрика:

```text
delta_spend =
month_spend - baseline_spend
```

---

# Финальный SQL

Используются CTE:

* client_baseline
* client_month_spend
* client_delta
* cohort_delta

Метрики:

* avg_delta_spend
* avg_spend_per_client
* total_spend

---

# Интерпретация графиков

Основные выводы:

## 1.

Клиенты в большем числе кампаний:

```text
тратят больше
```

---

## 2.

Нет устойчивого падения:

```text
fatigue hypothesis не подтверждается
```

---

## 3.

Линии когорт почти параллельны:

```text
динамика синхронна
```

Вероятно:

* сезонность,
* общие consumer trends.

---

## 4.

Скорее видна:

```text
сегментация клиентов
```

А не деградация механики.

---

# Пункт про номиналы

ТЗ:

```text
какая доля попадает из месяца в месяц в один и тот же номинал
```

Упрощение:

* 1 = 200
* 2 = 300
* 3 = 400
* 4 = 500

---

# Бизнес-логика номиналов

Клиент:

* попадает в кампанию,
* получает номинал,
* далее попадает в новые кампании.

Нужно проверить:

```text
меняется ли номинал между флайтами
```

---

# Что потребуется

Таблица:

```text
client_id
month_dt
nominal
```

Далее:

```sql
lag(nominal)
```

Метрика:

```text
same_nominal_flag
```

---

# Возможные analyses

## 1.

Доля:

```text
nominal_t == nominal_t-1
```

---

## 2.

Матрица переходов:

```text
200 -> 200
200 -> 300
300 -> 400
...
```

---

# Пункт про ML-score

Это отдельный пункт.

Нужно:

* найти таблицу score,
* определить response metric,
* анализировать последние 3 флайта.

Пока не реализовано.

---

# Технические проблемы

## Greenplum spill

Причина:

* слишком большие join/groupby,
* нехватка RAM,
* spill на disk.

---

# Как уменьшать spill

* фильтровать раньше,
* уменьшать window,
* считать по месяцам,
* materialize intermediate tables.

---

# DBeaver hanging query

Был зависший запрос:

```sql
pg_total_relation_size
```

Вероятно:

* metadata/statistics query DBeaver.

---

# Как избегать

* закрывать соединения,
* останавливать running statements,
* не держать активные cursors,
* осторожнее с preview huge tables.

---

```
```

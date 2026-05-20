# ------------------------------------------
# Utils

# Запрос для проверки уникальности клиентов
check_unique_query = """
with aud as
    (
        {aud}
    )
select sum(1) all_contacts_count
    , count(distinct contact_id) unique_contacts_count
from aud
"""

# Запрос на создание пустой таблицы
create_table_from_cols_query = """
create table {table} (
    {cols}
)
with (
    appendonly=true,
    blocksize=32768,
    orientation=column,
    compresstype=zstd,
    compresslevel=4
)
distributed by(contact_id)
"""

# Запрос на создание таблицы из запроса
create_table_from_select_query = """
create table {table} with 
    (
        appendonly = true
        , blocksize = 32768
        , orientation = column
        , compresstype = zstd
        , compresslevel = 4
    )
as 
    (
        {query}
    )
distributed by({distribution_col})
"""

# ------------------------------------------
# Audience
#
# MVP-аудитория для новой задачи churn-from-DAC:
# берем клиентов, которые являются DAC в базовом месяце.
#
# Актуальное определение DAC:
#   has_transaction_activity = 1
#   and (
#       has_mobapp_activity = 1
#       or has_pwa_activity = 1
#       or vcoff_trn_cnt > 0
#   )
#
# Ожидаемые параметры:
#   base_month: первый день базового месяца, например "2025-11-01".
#
# Важно:
#   Этот запрос уже НЕ возвращает старые поля churn_range/churn_type.
#   Старый preprocess.py из organic-return-dac завязан на эти поля и
#   фильтрует churn-сегменты, поэтому для новой задачи его надо минимально
#   адаптировать: убрать старую churn-сегментацию и использовать DAC-аудиторию.

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

gcg_query = """
select distinct cus_gr_contact_id :: int as contact_id
from 
(
select distinct  mrc_id,com_blk_lvl_id, mrc_prj_id, mrc_owner, mrc_desc, mrc_start_date,
       com_cus_gr_id , com_cus_sgr_id , com_t_id, com_start_date , com_finish_date
from cvm_sbx.mt_cvm_t_marketing_campaign
join cvm_sbx.mt_cvm_t_communication
    on com_id = mrc_com_id
join cvm_sbx.mt_cvm_t_message 
    on com_mes_id = mes_id
where  1=1
    and mrc_is_approved
    and mrc_prj_id=11  
    and com_start_date <= '{start_date}'
    and com_finish_date >= '{finish_date}'
) a1
inner join cvm_sbx.MT_CVM_T_CUST_GROUP a2
    on a1.com_cus_gr_id=a2.cus_gr_id
    and  a1.com_cus_sgr_id=a2.cus_sgr_id
    and  a1.mrc_start_date=a2.mrc_start_date
where 1=1
    and a2.cus_gr_target = false
"""

# ------------------------------------------
# Target
#
# Таргет новой задачи:
#   target_churn_from_dac = 1, если клиент был DAC в base_month,
#   но НЕ является DAC в target_month.
#
#   target_churn_from_dac = 0, если клиент был DAC в base_month
#   и остается DAC в target_month.
#
# Ожидаемые параметры:
#   aud: SQL аудитории базового месяца.
#   target_month: первый день следующего месяца, например "2025-12-01".
#
# Поля target_trns и target_login оставлены как compatibility shim:
#   target_trns = target_churn_from_dac
#   target_login = 1
# Это позволяет старому коду, который делает
#   target_dac = target_trns * target_login
# получить тот же бинарный таргет. В новом проекте лучше явно использовать
# target_churn_from_dac.

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
, marked as
    (
        select aud.contact_id
            , coalesce(d.is_dac_next_month, 0) :: int as is_dac_next_month
            , case
                when coalesce(d.is_dac_next_month, 0) = 1 then 0
                else 1
              end :: int as target_churn_from_dac
        from aud
        left join dac_next_month d using(contact_id)
    )
select contact_id
    , is_dac_next_month
    , target_churn_from_dac
    , target_churn_from_dac :: int as target_trns
    , 1 :: int as target_login
from marked
"""

# ------------------------------------------
# Features

recency_query = """
with aud as
    (
        {aud}
    )
, cheques as
    (
        SELECT 
            contact_id :: int
            , date'{date}' - max(datetime::date) cheque_recency
        FROM cvm_sbx.v_cheque_filtered 
        WHERE true
            AND datetime :: date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            AND contact_id IS NOT NULL
            and sale_channel_id in (0, 1, 2)
        group by contact_id
    )
, logins as
    (
        select contact_id :: int
            , date'{date}' - max(date(ptn_dt)) login_recency
        FROM dm.prvrd_dau_main_metrics d
        WHERE 1=1
            AND ptn_dt BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
        group by contact_id
    )
, omni_qr_and_features as
    (
        select customer_id :: int contact_id
            , date'{date}' - max(case when feature = 'QR' then date(event_date) end) omni_qr_recency
            , date'{date}' - max(case when feature != 'QR' then date(event_date) end) omni_features_recency
        FROM dm.prvrd_omni_features_usage u
        WHERE 1=1
            AND event_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            and target_cnt > 0
        group by customer_id
    )
select *
from aud
left join cheques using(contact_id)
left join logins using(contact_id)
left join omni_qr_and_features using(contact_id)
"""

perf_recency_query = """
with aud as
    (
        {aud}
    )
, perf as
    (
        select t2.contact_id :: int
            , date'{date}' - max(install_date) perf_recency
        from dm.pvdr_appsflyer_installs_gr t1
        join dm.contact t2 on t1.magnit_id = t2.magnit_id
        where true
            and install_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            and o_type != 'Органика'
        group by 1
    )
select *
from aud
join perf using(contact_id)
"""

cheque_query = """
with aud as
    (
        {aud}
    )
, cheques as
    (
        SELECT 
            contact_id :: int
            , orgunit_id
            , summ_discounted :: real rto
            , date(datetime) - lag(date(datetime)) over (partition by contact_id order by date(datetime)) trans_lag
            , datetime :: date
        FROM cvm_sbx.v_cheque_filtered 
        WHERE true
            AND datetime :: date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            AND contact_id IS NOT NULL
            and sale_channel_id in (0, 1, 2)
    )
, whs as 
    (
        select orgunit_id
            , frmt :: varchar(10)
            , (('{date}' - date(open_dt)) / 365.0) :: real whs_age
            , case when date(close_dt) < '{date}' then 1 else 0 end :: int whs_is_closed
            , size_of_population :: int population
            , square_trade :: real
            , competitor_grp_name :: varchar(11)
            , case when competitor_open_dt < '{date}'
                then ('{date}' - date(competitor_open_dt)) / 365.0 else null end :: real competitor_age
            , case when competitor_open_dt < '{date}'
                then main_competitor_dist else null end :: real competitor_dist
            , case 
                  when fed_region in ('Москва', 'Московская область', 'Санкт-Петербург', 'Ленинградская область') 
                  then 1 else 0 end :: int whs_in_capital
            , case when city_type = 'Город' then 1 else 0 end :: int whs_in_city
        from dm.whs
    )
select 
    contact_id
    , sum(rto) :: real rto_{month}
    , sum(1) :: int trans_count_{month}
    , sum(rto) / sum(1) :: real aov_{month}
    , avg(trans_lag) :: real trans_lag_avg_{month}
    ---------------------------------------------------------
    , count(distinct orgunit_id) :: int whs_count_{month}
    , count(distinct frmt) :: int format_count_{month}
    , count(distinct population) :: int location_count_{month}
    ---------------------------------------------------------
    , avg(whs_age) :: real whs_age_avg_{month}
    , avg(whs_is_closed) :: real whs_is_closed_avg_{month}
    , avg(whs_in_capital) :: real whs_in_capital_avg_{month}
    , avg(whs_in_city) :: real whs_in_city_avg_{month}
    , avg(population) :: real population_avg_{month}
    , avg(square_trade) :: real square_trade_avg_{month}
    ---------------------------------------------------------
    , avg(case 
        when competitor_grp_name is null or competitor_grp_name = '' or competitor_age < 0 then 0
        when competitor_grp_name = '1-ый эшелон' then 2
        else 1 end) :: real competitor_echelon_avg_{month}
    , avg(case when competitor_age <= 0 then 0 else competitor_age end) :: real competitor_age_avg_{month}
    , avg(case when competitor_age < 0 then 0 else competitor_dist end) :: real competitor_dist_avg_{month}
from aud
join cheques using(contact_id)
left join whs using(orgunit_id)
group by contact_id
"""

cheque_query_short = """
with aud as
    (
        {aud}
    )
, cheques as
    (
        SELECT 
            contact_id :: int
            , summ_discounted :: real rto
            , date(datetime) - lag(date(datetime)) over (partition by contact_id order by date(datetime)) trans_lag
            , sale_channel_id
            , datetime :: date
        FROM cvm_sbx.v_cheque_filtered 
        WHERE true
            AND datetime :: date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            AND contact_id IS NOT NULL
            and sale_channel_id in (0, 1, 2)
    )
select 
    contact_id
    , sum(rto) :: real rto_{month}
    , sum(1) :: int trans_count_{month}
    , sum(rto) / sum(1) :: real aov_{month}
    , avg(trans_lag) :: real trans_lag_avg_{month}
from aud
join cheques using(contact_id)
group by contact_id
"""

app_query = '''
with logins as
    (
        select c.contact_id :: int
            , date(ptn_dt) - lag(date(ptn_dt)) over (partition by c.contact_id order by date(ptn_dt)) login_lag
        FROM dm.prvrd_dau_main_metrics d
        JOIN ({aud}) c
            ON c.contact_id = d.contact_id :: int
        WHERE 1=1
            AND ptn_dt BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
    )
select contact_id
    , sum(1) :: int login_count_{month}
    , avg(login_lag) :: real login_lag_avg_{month}
from logins
GROUP BY contact_id
'''

omni_qr_query = """
with omni_qr as
    (
        select customer_id :: int contact_id
            , date(event_date) - lag(date(event_date)) over (partition by customer_id order by date(event_date)) omni_qr_lag
        FROM dm.prvrd_omni_features_usage u
        JOIN ({aud}) c
            ON c.contact_id = u.customer_id::int
        WHERE 1=1
            AND event_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            and target_cnt > 0
            and feature = 'QR'
    )   
select contact_id
    , sum(1) :: int omni_qr_days_count_{month}
    , avg(omni_qr_lag) :: real omni_qr_lag_avg_{month}
from omni_qr
GROUP BY contact_id
"""

omni_features_query = """
with omni_features as
    (
        select customer_id :: int contact_id
            , date(event_date) - lag(date(event_date)) over (partition by customer_id order by date(event_date)) omni_features_lag
        FROM dm.prvrd_omni_features_usage u
        JOIN ({aud}) c
            ON c.contact_id = u.customer_id::int
        WHERE 1=1
            AND event_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            and target_cnt > 0
            and feature != 'QR'
    )   
select contact_id
    , sum(1) :: int omni_features_days_count_{month}
    , avg(omni_features_lag) :: real omni_features_lag_avg_{month}
from omni_features
GROUP BY contact_id
"""

fav_omni_features_create_query = """
with f_sum as
    (
        select customer_id :: int contact_id
            , feature
            , sum(1) :: int feature_days_count  -- equals to count(distinct event_date) - tested
        from dm.prvrd_omni_features_usage u
        JOIN ({aud}) c
            ON c.contact_id = u.customer_id::int
        WHERE 1=1
            AND event_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            and target_cnt > 0
            and feature != 'QR'
        GROUP BY customer_id, feature
    )
, f_max as
    (
        select contact_id
            , feature
            , feature_days_count
            , max(feature_days_count) over (partition by contact_id) :: int feature_days_count_max
        from f_sum
    )
select contact_id
    , feature
from f_max
where feature_days_count = feature_days_count_max
"""

fav_omni_features_select_query = """
with f_sum as
    (
        select customer_id :: int contact_id
            , feature
            , sum(1) :: int feature_days_count
        from dm.prvrd_omni_features_usage u
        JOIN ({aud}) c
            ON c.contact_id = u.customer_id::int
        WHERE 1=1
            AND event_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
            and target_cnt > 0
        GROUP BY customer_id, feature
    )
select u.contact_id :: int
    , sum(u.feature_days_count) :: int omni_fav_features_days_count_{month}
from f_sum u
JOIN {fav_omni_features_table} f
    on true
        and f.contact_id = u.contact_id
        and f.feature = u.feature
group by u.contact_id
"""

omni_unique_features_count_query = """
SELECT 
    customer_id :: int contact_id
    , count(distinct feature) :: int omni_unique_features_count_{month}
FROM dm.prvrd_omni_features_usage u
JOIN ({aud}) c
    ON c.contact_id = u.customer_id::int
WHERE 1=1
    AND event_date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
    and target_cnt > 0
    and feature != 'QR'
GROUP BY customer_id
"""

omni_goals_query = """
select contact_id :: int contact_id
    , sum(case when event_name = 'goal_activated' then 1 end) :: int omni_goals_activated_count_{month}
    , count(distinct case when event_name = 'goal_updated' then goal_id end) :: int omni_goals_updated_count_{month}
    , sum(case when event_name = 'goal_finished' then 1 end) :: int omni_goals_finished_count_{month}
from dm.contact_goal
join ({aud}) aud using(contact_id)
where true
    and event_datetime between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
group by contact_id
"""

accept_query = """
with aud as
    (
        {aud}
    )
, accept as
    (
        select contact_id
            , count(distinct offer_pk) accept_count
        from dm.offer_contact oc
        join dm.offer o using(offer_pk)
        join aud using(contact_id)
        where oc.created_on :: date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
        group by contact_id
        ---------
        union all
        ---------
        select contact_id
            , count(distinct rule_pk) accept_count
        from dm.offer_perspromo
        join aud using(contact_id)
        where true
            and is_accept = 1
            and datetime_accept :: date BETWEEN date'{date}' - interval '{month} month' AND date'{date}' - interval '1 day'
        group by contact_id
    )
select contact_id :: int
    , sum(accept_count) :: int accept_count_{month}
from accept
group by contact_id
"""

bonus_query = """
with aud as
    (
        {aud}
    )
, card as
    (
        select contact_id
            , card_number
        from dm.card 
    )
select contact_id :: int
    , sum(case when bonus_type = 'addition' then value / 100 end :: real)                                 bonus_accrued_sum_{month}
    , sum(case when parent_type_id is not null and bonus_type = 'addition' then value / 100 end :: real)  bonus_accrued_offer_sum_{month}
    , sum(case when parent_type_id = 6 and bonus_type = 'write_off' then value / -100 end :: real)        bonus_expired_sum_{month}
    , sum(case when parent_type_id != 6 and bonus_type = 'write_off' then value / -100 end :: real)       bonus_redeemed_sum_{month}
from dm.bonus_all
join card using(card_number)
join aud using(contact_id)
where created_on :: date between date'{date}' - interval '{month} month' and date'{date}' - interval '5 day'
group by contact_id
"""

level_query = """
select contact_id :: int
    , avg(level_id) :: real avg_level_{month}
from dm.contact_loyality_lvl_month
join ({aud}) aud using(contact_id)
where true
    and month_start_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
group by contact_id
"""

static_features_query = """
with aud as
    (
        {aud}
    )
, cohort as
    (
        select contact_id :: int  -- uniques positive
            , extract(year from age(date'{date}', first_dac_month)) * 12 + extract(month from age(date'{date}', first_dac_month)) :: int dac_age_months
        from dm.prvdr_dac_cohorts
    )
, demo as
    (
        select contact_id :: int  -- uniques negative
            , max(extract(year from age(date'{date}', birth_date))) :: int cust_age_years
            , max(case when gendercalc = 'M' then 1 else 0 end) :: int is_male
        from dm.contact
        group by contact_id
    )
select *
from aud
left join cohort using(contact_id)
left join demo using(contact_id)
"""

dac_months_count_query = """
WITH aud AS (
    {aud}
), monthly_activity AS (
    SELECT
        d.contact_id :: int AS contact_id,
        d.date_month,
        d.has_transaction_activity :: int AS has_transaction_activity,
        d.has_mobapp_activity :: int AS has_mobapp_activity,
        d.has_pwa_activity :: int AS has_pwa_activity,
        CASE WHEN d.vcoff_trn_cnt > 0 THEN 1 ELSE 0 END :: int AS has_vcoff_activity,
        d.vcoff_trn_cnt :: int AS vcoff_trn_cnt,
        CASE
            WHEN d.has_transaction_activity = 1
             AND (
                  d.has_mobapp_activity = 1
                  OR d.has_pwa_activity = 1
                  OR d.vcoff_trn_cnt > 0
             )
            THEN 1 ELSE 0
        END :: int AS is_dac_month
    FROM dm.prvdr_dac_monthly d
    JOIN aud USING(contact_id)
    WHERE d.date_month < date_trunc('month', date'{date}')
), dac_history AS (
    SELECT
        contact_id,
        date_month,
        date_month - (
            ROW_NUMBER() OVER(PARTITION BY contact_id ORDER BY date_month) * INTERVAL '1 month'
        ) AS grp
    FROM monthly_activity
    WHERE is_dac_month = 1
), islands AS (
    SELECT
        contact_id,
        grp,
        COUNT(*) AS streak_length
    FROM dac_history
    GROUP BY contact_id, grp
), last_dac_month AS (
    SELECT
        contact_id,
        MAX(date_month) AS last_dac_month
    FROM dac_history
    GROUP BY contact_id
), current_streak AS (
    SELECT
        h.contact_id,
        i.streak_length :: int AS current_dac_streak
    FROM dac_history h
    JOIN last_dac_month l
        ON h.contact_id = l.contact_id
       AND h.date_month = l.last_dac_month
    JOIN islands i
        ON h.contact_id = i.contact_id
       AND h.grp = i.grp
), history_agg AS (
    SELECT
        contact_id,
        SUM(streak_length) :: int AS dac_months_count,
        MAX(streak_length) :: int AS max_consecutive_dac_months,
        AVG(streak_length) :: real AS avg_consecutive_dac_months
    FROM islands
    GROUP BY contact_id
), window_agg AS (
    SELECT
        contact_id,
        SUM(CASE WHEN is_dac_month = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '3 month'  THEN 1 ELSE 0 END) :: int AS dac_months_last_3,
        SUM(CASE WHEN is_dac_month = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '6 month'  THEN 1 ELSE 0 END) :: int AS dac_months_last_6,
        SUM(CASE WHEN is_dac_month = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '12 month' THEN 1 ELSE 0 END) :: int AS dac_months_last_12,
        (SUM(CASE WHEN is_dac_month = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '3 month'  THEN 1 ELSE 0 END) / 3.0) :: real AS dac_share_last_3,
        (SUM(CASE WHEN is_dac_month = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '6 month'  THEN 1 ELSE 0 END) / 6.0) :: real AS dac_share_last_6,
        (SUM(CASE WHEN is_dac_month = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '12 month' THEN 1 ELSE 0 END) / 12.0) :: real AS dac_share_last_12,
        SUM(CASE WHEN has_transaction_activity = 1 AND date_month >= date_trunc('month', date'{date}') - INTERVAL '12 month' THEN 1 ELSE 0 END) :: int AS transaction_active_months_last_12,
        SUM(CASE WHEN has_mobapp_activity = 1      AND date_month >= date_trunc('month', date'{date}') - INTERVAL '12 month' THEN 1 ELSE 0 END) :: int AS mobapp_active_months_last_12,
        SUM(CASE WHEN has_pwa_activity = 1         AND date_month >= date_trunc('month', date'{date}') - INTERVAL '12 month' THEN 1 ELSE 0 END) :: int AS pwa_active_months_last_12,
        SUM(CASE WHEN has_vcoff_activity = 1       AND date_month >= date_trunc('month', date'{date}') - INTERVAL '12 month' THEN 1 ELSE 0 END) :: int AS vcoff_active_months_last_12
    FROM monthly_activity
    GROUP BY contact_id
), base_month_flags AS (
    SELECT
        contact_id,
        MAX(has_transaction_activity) :: int AS base_has_transaction_activity,
        MAX(has_mobapp_activity) :: int AS base_has_mobapp_activity,
        MAX(has_pwa_activity) :: int AS base_has_pwa_activity,
        MAX(has_vcoff_activity) :: int AS base_has_vcoff_activity,
        SUM(vcoff_trn_cnt) :: int AS base_vcoff_trn_cnt,
        (
            MAX(has_mobapp_activity)
            + MAX(has_pwa_activity)
            + MAX(has_vcoff_activity)
        ) :: int AS base_digital_mechanism_count
    FROM monthly_activity
    WHERE date_month = date_trunc('month', date'{date}') - INTERVAL '1 month'
    GROUP BY contact_id
)
SELECT
    aud.contact_id :: int,
    COALESCE(h.dac_months_count, 0) :: int AS dac_months_count,
    COALESCE(h.max_consecutive_dac_months, 0) :: int AS max_consecutive_dac_months,
    COALESCE(h.avg_consecutive_dac_months, 0) :: real AS avg_consecutive_dac_months,
    COALESCE(cs.current_dac_streak, 0) :: int AS current_dac_streak,
    COALESCE(w.dac_months_last_3, 0) :: int AS dac_months_last_3,
    COALESCE(w.dac_months_last_6, 0) :: int AS dac_months_last_6,
    COALESCE(w.dac_months_last_12, 0) :: int AS dac_months_last_12,
    COALESCE(w.dac_share_last_3, 0) :: real AS dac_share_last_3,
    COALESCE(w.dac_share_last_6, 0) :: real AS dac_share_last_6,
    COALESCE(w.dac_share_last_12, 0) :: real AS dac_share_last_12,
    CASE WHEN COALESCE(w.dac_months_last_12, 0) >= 10 THEN 1 ELSE 0 END :: int AS is_stable_dac,
    CASE WHEN COALESCE(w.dac_months_last_12, 0) BETWEEN 6 AND 9 THEN 1 ELSE 0 END :: int AS is_regular_dac,
    CASE WHEN COALESCE(w.dac_months_last_12, 0) BETWEEN 2 AND 5 THEN 1 ELSE 0 END :: int AS is_unstable_dac,
    CASE WHEN COALESCE(w.dac_months_last_12, 0) = 1 THEN 1 ELSE 0 END :: int AS is_new_dac,
    COALESCE(w.transaction_active_months_last_12, 0) :: int AS transaction_active_months_last_12,
    COALESCE(w.mobapp_active_months_last_12, 0) :: int AS mobapp_active_months_last_12,
    COALESCE(w.pwa_active_months_last_12, 0) :: int AS pwa_active_months_last_12,
    COALESCE(w.vcoff_active_months_last_12, 0) :: int AS vcoff_active_months_last_12,
    COALESCE(b.base_has_transaction_activity, 0) :: int AS base_has_transaction_activity,
    COALESCE(b.base_has_mobapp_activity, 0) :: int AS base_has_mobapp_activity,
    COALESCE(b.base_has_pwa_activity, 0) :: int AS base_has_pwa_activity,
    COALESCE(b.base_has_vcoff_activity, 0) :: int AS base_has_vcoff_activity,
    COALESCE(b.base_vcoff_trn_cnt, 0) :: int AS base_vcoff_trn_cnt,
    COALESCE(b.base_digital_mechanism_count, 0) :: int AS base_digital_mechanism_count
FROM aud
LEFT JOIN history_agg h USING(contact_id)
LEFT JOIN current_streak cs USING(contact_id)
LEFT JOIN window_agg w USING(contact_id)
LEFT JOIN base_month_flags b USING(contact_id)
"""
uplift_rate_aud_query = """
select 
    cg.cus_gr_contact_id :: int contact_id
    , mc.mrc_desc
    , mc.mrc_start_date start_date
    , mc.mrc_finish_date finish_date
    , cg.cus_gr_target :: int treatment
from cvm_sbx.mt_cvm_t_marketing_campaign mc
join cvm_sbx.mt_cvm_t_communication c
    on mc.mrc_com_id = c.com_id
join cvm_sbx.mt_cvm_t_cust_group cg
    on c.com_cus_gr_id = cg.cus_gr_id 
    and c.com_cus_sgr_id = cg.cus_sgr_id
where true 
    and mc.mrc_desc = '{mrc_desc}'
    and mc.mrc_start_date = '{start_date}'
    and mc.mrc_finish_date = '{finish_date}'
    and cg.mrc_start_date = mc.mrc_start_date
    and cg.cus_gr_contact_id > 0
    and mc.mrc_is_approved = true
    and com_t_id = 2
group by 
    mc.mrc_desc
    , mc.mrc_start_date
    , mc.mrc_finish_date
    , cg.cus_gr_target 
    , cg.cus_gr_contact_id
"""

check_unique_query = """
with aud as (
        {aud}
)
select sum(1) all_contacts_count
    , count(distinct contact_id) unique_contacts_count
from aud
"""

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

create_table_from_select_query = """
create table {table} with (
        appendonly = true
        , blocksize = 32768
        , orientation = column
        , compresstype = zstd
        , compresslevel = 4
)
as (
    {query}
)
distributed by({distribution_col})
"""

aud_query = """
select distinct 
    contact_id::bigint
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

target_query = """
with aud as (
    {aud}
), 

dac_next_month as (
    select distinct 
        contact_id :: bigint
        , 1 :: int is_dac_next_month
    from dm.prvdr_dac_monthly
    where date_month = date '{target_month}'
    and has_transaction_activity = 1
    and (
        has_mobapp_activity = 1
        or has_pwa_activity = 1
        or vcoff_trn_cnt > 0
        )
    and contact_id is not null
)

select distinct
    aud.contact_id
    , case
        when coalesce(d.is_dac_next_month, 0) = 1 then 0
        else 1
        end :: int as target_churn_from_dac
from aud
left join dac_next_month d using(contact_id)
"""

recency_query = """
with aud as (
    {aud}
), 

cheques as (
    select
        c.contact_id :: bigint
        , date'{date}' - max(c.datetime::date) cheque_recency
    from cvm_sbx.v_cheque_filtered c
    join aud a
        on a.contact_id = c.contact_id :: bigint
    where true
    and c.datetime :: date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and c.contact_id is not null
    and c.sale_channel_id in (0, 1, 2)
    group by c.contact_id :: bigint
), 

logins as (
    select 
        d.contact_id :: bigint
        , date'{date}' - max(date(d.ptn_dt)) login_recency
    from dm.prvrd_dau_main_metrics d
    join aud a
        on a.contact_id = d.contact_id :: bigint
    where 1=1
    and d.ptn_dt between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    group by d.contact_id :: bigint
), 

omni_qr_and_features as (
    select 
        u.customer_id :: bigint contact_id
        , date'{date}' - max(case when u.feature = 'QR' then date(u.event_date) end) omni_qr_recency
        , date'{date}' - max(case when u.feature != 'QR' then date(u.event_date) end) omni_features_recency
    from dm.prvrd_omni_features_usage u
    join aud a
        on a.contact_id = u.customer_id :: bigint
    where 1=1
    and u.event_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and u.target_cnt > 0
    group by u.customer_id :: bigint
)

select *
from aud
left join cheques using(contact_id)
left join logins using(contact_id)
left join omni_qr_and_features using(contact_id)
"""

perf_recency_query = """
with aud as (
    {aud}
), 

perf as (
    select 
        t2.contact_id :: bigint
        , date'{date}' - max(install_date) perf_recency
    from dm.pvdr_appsflyer_installs_gr t1
    join dm.contact t2 on t1.magnit_id = t2.magnit_id
    join aud a on a.contact_id = t2.contact_id :: bigint
    where true
    and install_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    group by 1
)

select *
from aud
join perf using(contact_id)
"""

cheque_query = """
with aud as (
    {aud}), 

cheques as (
    select
        c.contact_id :: bigint
        , c.orgunit_id
        , c.summ_discounted :: real rto
        , date(c.datetime) - lag(date(c.datetime)) over (partition by c.contact_id order by date(c.datetime)) trans_lag
        , c.datetime :: date
    from cvm_sbx.v_cheque_filtered c
    join aud a
        on a.contact_id = c.contact_id :: bigint
    where true
    and c.datetime :: date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and c.contact_id is not null
    and c.sale_channel_id in (0, 1, 2)
), 

whs as (
    select orgunit_id
        , frmt :: varchar(10)
        , (('{date}' - date(open_dt)) / 365.0) :: real whs_age
        , case when date(close_dt) < '{date}' then 1 else 0 end :: int whs_is_closed
        , size_of_population :: int population
        , square_trade :: real
        , competitor_grp_name :: varchar(11)
        , case 
                when competitor_open_dt < '{date}'
                then ('{date}' - date(competitor_open_dt)) / 365.0 else null 
                end :: real competitor_age
        , case 
            when competitor_open_dt < '{date}'
            then main_competitor_dist else null 
            end :: real competitor_dist
        , case 
            when fed_region in ('Москва', 'Московская область', 'Санкт-Петербург', 'Ленинградская область') 
            then 1 else 0 
            end :: int whs_in_capital
        , case 
            when city_type = 'Город' 
            then 1 else 0 
            end :: int whs_in_city
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
with aud as (
    {aud}
), 

cheques as (
    select 
        c.contact_id :: bigint
        , c.summ_discounted :: real rto
        , date(c.datetime) - lag(date(c.datetime)) over (partition by c.contact_id order by date(c.datetime)) trans_lag
        , c.sale_channel_id
        , c.datetime :: date
    from cvm_sbx.v_cheque_filtered c
    join aud a
        on a.contact_id = c.contact_id :: bigint
    where true
    and c.datetime :: date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and c.contact_id is not null
    and c.sale_channel_id in (0, 1, 2)
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
with logins as (
    select 
        c.contact_id :: bigint
        , date(ptn_dt) - lag(date(ptn_dt)) over (partition by c.contact_id order by date(ptn_dt)) login_lag
    from dm.prvrd_dau_main_metrics d
    join ({aud}) c
        on c.contact_id = d.contact_id :: bigint
    where 1=1
    and ptn_dt between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
)

select 
    contact_id
    , sum(1) :: int login_count_{month}
    , avg(login_lag) :: real login_lag_avg_{month}
from logins
group by contact_id
'''

omni_qr_query = """
with omni_qr as (
    select 
        customer_id :: bigint contact_id
        , date(event_date) - lag(date(event_date)) over (partition by customer_id order by date(event_date)) omni_qr_lag
    from dm.prvrd_omni_features_usage u
    join ({aud}) c
        on c.contact_id = u.customer_id::bigint
    where 1=1
    and event_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and target_cnt > 0
    and feature = 'QR'
)  

select contact_id
    , sum(1) :: int omni_qr_days_count_{month}
    , avg(omni_qr_lag) :: real omni_qr_lag_avg_{month}
from omni_qr
group by contact_id
"""

omni_features_query = """
with omni_features as (
    select 
        customer_id :: bigint contact_id
        , date(event_date) - lag(date(event_date)) over (partition by customer_id order by date(event_date)) omni_features_lag
    from dm.prvrd_omni_features_usage u
    join ({aud}) c
        on c.contact_id = u.customer_id::bigint
    where 1=1
    and event_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and target_cnt > 0
    and feature != 'QR'
)   

select 
    contact_id
    , sum(1) :: int omni_features_days_count_{month}
    , avg(omni_features_lag) :: real omni_features_lag_avg_{month}
from omni_features
group by contact_id
"""

fav_omni_features_create_query = """
with f_sum as (
    select 
        customer_id :: bigint contact_id
        , feature
        , sum(1) :: int feature_days_count  -- equals to count(distinct event_date) - tested
    from dm.prvrd_omni_features_usage u
    join ({aud}) c
        on c.contact_id = u.customer_id::bigint
    where 1=1
    and event_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and target_cnt > 0
    and feature != 'QR'
    group by customer_id, feature
),

f_max as (
    select 
        contact_id
        , feature
        , feature_days_count
        , max(feature_days_count) over (partition by contact_id)::int feature_days_count_max
    from f_sum
)

select 
    contact_id
    , feature
from f_max
where feature_days_count = feature_days_count_max
"""

fav_omni_features_select_query = """
with f_sum as (
    select 
        customer_id :: bigint contact_id
        , feature
        , sum(1) :: int feature_days_count
    from dm.prvrd_omni_features_usage u
    join ({aud}) c
        on c.contact_id = u.customer_id::bigint
    where 1=1
    and event_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    and target_cnt > 0
    group by customer_id, feature
)

select 
    u.contact_id::bigint
    , sum(u.feature_days_count)::int omni_fav_features_days_count_{month}
from f_sum u
join {fav_omni_features_table} f
    on true
    and f.contact_id = u.contact_id
    and f.feature = u.feature
group by u.contact_id
"""

omni_unique_features_count_query = """
select 
    customer_id :: bigint contact_id
    , count(distinct feature) :: int omni_unique_features_count_{month}
from dm.prvrd_omni_features_usage u
join ({aud}) c
    on c.contact_id = u.customer_id::bigint
where 1=1
and event_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
and target_cnt > 0
and feature != 'QR'
group by customer_id
"""

omni_goals_query = """
select 
    contact_id :: bigint contact_id
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
with aud as (
    {aud}
),

accept as (
    select 
        contact_id
        , count(distinct offer_pk) accept_count
    from dm.offer_contact oc
    join dm.offer o using(offer_pk)
    join aud using(contact_id)
    where oc.created_on :: date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    group by contact_id
    ---------
    union all
    ---------
    select 
        contact_id
        , count(distinct rule_pk) accept_count
    from dm.offer_perspromo
    join aud using(contact_id)
    where true
    and is_accept = 1
    and datetime_accept :: date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
    group by contact_id
)

select 
    contact_id :: bigint
    , sum(accept_count)::int accept_count_{month}
from accept
group by contact_id
"""

bonus_query = """
with aud as (
    {aud}
),

card as (
    select 
        contact_id
        , card_number
    from dm.card 
)

select 
    contact_id::bigint
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
select 
    contact_id::bigint
    , avg(level_id) :: real avg_level_{month}
from dm.contact_loyality_lvl_month
join ({aud}) aud using(contact_id)
where true
and month_start_date between date'{date}' - interval '{month} month' and date'{date}' - interval '1 day'
group by contact_id
"""

static_features_query = """
with aud as (
    {aud}
),

cohort as (
    select 
        contact_id::bigint  -- uniques positive
        , extract(year from age(date'{date}', first_dac_month)) * 12 + extract(month from age(date'{date}', first_dac_month)) :: int dac_age_months
    from dm.prvdr_dac_cohorts
),

demo as (
    select 
        contact_id::bigint  -- uniques negative
        , max(extract(year from age(date'{date}', birth_date)))::int cust_age_years
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
with aud as (
    {aud}
), 

monthly_activity as (
    select
        d.contact_id :: bigint AS contact_id,
        d.date_month,
        case
            when d.has_transaction_activity = 1
            and (
                  d.has_mobapp_activity = 1
                  OR d.has_pwa_activity = 1
                  OR d.vcoff_trn_cnt > 0
             )
            then 1 ELSE 0
            end :: int AS is_dac_month
    from dm.prvdr_dac_monthly d
    join aud using(contact_id)
    where d.date_month < date_trunc('month', date'{date}')
), 

window_agg AS (
    select
        contact_id,
        sum(is_dac_month) :: int as dac_months_count,
        sum(
            case
                when is_dac_month = 1
                and date_month >= date_trunc('month', date'{date}') - interval '3 month'
                then 1 
                else 0
                end
        ) :: int as dac_months_last_3,
        sum(
            case
                when is_dac_month = 1
                and date_month >= date_trunc('month', date'{date}') - interval '6 month'
                then 1 
                else 0
                end
        ) :: int as dac_months_last_6,
        sum(
            case
                when is_dac_month = 1
                and date_month >= date_trunc('month', date'{date}') - interval '12 month'
                then 1 
                else 0
                end
        ) :: int AS dac_months_last_12
    from monthly_activity
    group by contact_id
)
select
    aud.contact_id :: bigint,
    coalesce(w.dac_months_count, 0) :: int AS dac_months_count,
    coalesce(w.dac_months_last_3, 0) :: int AS dac_months_last_3,
    coalesce(w.dac_months_last_6, 0) :: int AS dac_months_last_6,
    coalesce(w.dac_months_last_12, 0) :: int AS dac_months_last_12
from aud
left join window_agg w using(contact_id)
"""

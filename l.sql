with aud as (
    select distinct
        contact_id::int
    from dm.prvdr_dac_monthly
    where date_month = date'2026-02-01'
      and has_transaction_activity = 1
      and (
          has_mobapp_activity = 1
          or has_pwa_activity = 1
          or vcoff_trn_cnt > 0
      )
      and contact_id is not null
),

target_by_monthly as (
    select
        aud.contact_id,
        case
            when d.contact_id is not null then 0
            else 1
        end::int as target_by_monthly
    from aud
    left join (
        select distinct
            contact_id::int
        from dm.prvdr_dac_monthly
        where date_month = date'2026-03-01'
          and has_transaction_activity = 1
          and (
              has_mobapp_activity = 1
              or has_pwa_activity = 1
              or vcoff_trn_cnt > 0
          )
          and contact_id is not null
    ) d using(contact_id)
),

target_by_churn_table as (
    select
        aud.contact_id,
        case
            when c.status_churn is not null then 1
            else 0
        end::int as target_by_churn
    from aud
    left join pa_core_provider.prvdr_dac_churn c
        on aud.contact_id = c.contact_id
       and c.date_month = date'2026-03-01'
)

select
    m.target_by_monthly,
    c.target_by_churn,
    count(*) as clients,
    count(*) * 1.0 / sum(count(*)) over () as share
from target_by_monthly m
join target_by_churn_table c using(contact_id)
group by
    m.target_by_monthly,
    c.target_by_churn
order by
    m.target_by_monthly,
    c.target_by_churn;

with aud as (
    select distinct
        contact_id::int
    from dm.prvdr_dac_monthly
    where date_month = date'2026-02-01'
      and has_transaction_activity = 1
      and (
          has_mobapp_activity = 1
          or has_pwa_activity = 1
          or vcoff_trn_cnt > 0
      )
      and contact_id is not null
)

select
    count(*) as base_audience,
    count(c.contact_id) as found_in_churn_table,
    count(*) - count(c.contact_id) as missing_in_churn_table
from aud
left join pa_core_provider.prvdr_dac_churn c
    on aud.contact_id = c.contact_id
   and c.date_month = date'2026-03-01';
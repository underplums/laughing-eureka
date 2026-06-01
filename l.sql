with aud as (
    select distinct
        contact_id::int as contact_id
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

m_target as (
    select
        a.contact_id,
        case
            when d.contact_id is not null then 0
            else 1
        end::int as target_monthly
    from aud a
    left join (
        select distinct
            contact_id::int as contact_id
        from dm.prvdr_dac_monthly
        where date_month = date'2026-03-01'
          and has_transaction_activity = 1
          and (
              has_mobapp_activity = 1
              or has_pwa_activity = 1
              or vcoff_trn_cnt > 0
          )
          and contact_id is not null
    ) d
        on a.contact_id = d.contact_id
),

c_target as (
    select
        a.contact_id,
        case
            when c.status_churn is not null then 1
            else 0
        end::int as target_churn_table
    from aud a
    left join pa_core_provider.prvdr_dac_churn c
        on a.contact_id = c.contact_id
       and c.date_month = date'2026-03-01'
)

select
    m.target_monthly,
    c.target_churn_table,
    count(*) as clients,
    count(*) * 1.0 / sum(count(*)) over () as share
from m_target m
join c_target c
    on m.contact_id = c.contact_id
group by
    m.target_monthly,
    c.target_churn_table
order by
    m.target_monthly,
    c.target_churn_table;
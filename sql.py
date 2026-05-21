WITH aud AS (
    select distinct contact_id :: int as contact_id
    from ({aud}) a
    where contact_id is not null
), cheques AS (
    SELECT
        c.contact_id :: int,
        date'{date}' - max(c.datetime::date) AS cheque_recency
    FROM cvm_sbx.v_cheque_filtered c
    JOIN aud a
        ON a.contact_id = c.contact_id :: int
    WHERE true
        AND c.datetime :: date BETWEEN date'{date}' - interval '{month} month'
                                  AND date'{date}' - interval '1 day'
        AND c.contact_id IS NOT NULL
        AND c.sale_channel_id IN (0, 1, 2)
    GROUP BY c.contact_id :: int
), logins AS (
    SELECT
        d.contact_id :: int,
        date'{date}' - max(date(d.ptn_dt)) AS login_recency
    FROM dm.prvrd_dau_main_metrics d
    JOIN aud a
        ON a.contact_id = d.contact_id :: int
    WHERE true
        AND d.ptn_dt BETWEEN date'{date}' - interval '{month} month'
                         AND date'{date}' - interval '1 day'
        AND d.contact_id IS NOT NULL
    GROUP BY d.contact_id :: int
), omni_qr_and_features AS (
    SELECT
        u.customer_id :: int AS contact_id,
        date'{date}' - max(CASE WHEN u.feature = 'QR' THEN date(u.event_date) END) AS omni_qr_recency,
        date'{date}' - max(CASE WHEN u.feature != 'QR' THEN date(u.event_date) END) AS omni_features_recency
    FROM dm.prvrd_omni_features_usage u
    JOIN aud a
        ON a.contact_id = u.customer_id :: int
    WHERE true
        AND u.event_date BETWEEN date'{date}' - interval '{month} month'
                             AND date'{date}' - interval '1 day'
        AND u.target_cnt > 0
        AND u.customer_id IS NOT NULL
    GROUP BY u.customer_id :: int
)
SELECT
    aud.contact_id,
    cheques.cheque_recency,
    logins.login_recency,
    omni_qr_and_features.omni_qr_recency,
    omni_qr_and_features.omni_features_recency
FROM aud
LEFT JOIN cheques USING(contact_id)
LEFT JOIN logins USING(contact_id)
LEFT JOIN omni_qr_and_features USING(contact_id)
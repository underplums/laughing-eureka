RANDOM_STATE = 42

# Periods for feature gathering
preperiod_months = [24, 12, 6, 3, 2, 1]

# Classification threshold
threshold = 0.3

# Names
project_name = "cvm_churn-from-dac_binary-class_mvp"
model_type = "catboost"
jira = "CVMB-22227"

# Template to get full s3a path
template = "s3a://{bucket}/{prefix}"

# Audience table
#chudac_table = "cvm_sbx.v_cvm_churn_dac_dynamic_churn_markup"
chudac_table = "cvm_sbx.dac_dynamic_churn_markup"

# S3 permanent suffixes
model_predictions_suffix = "model_predictions"
train_data_stat_suffix = "data_stat/train"
inference_data_stat_suffix = "data_stat/inference"

# S3 temporary suffixes
aud_suffix = 'aud'
input_suffix = 'input'

# S3 permanent tables
uplift_rate_prefix = 's3://cvm-current-customer-data/personal/chernov_ke/uplift-churn-from-dac/perm/uplift_rate_chunked'

# Preprocess tables
aud_table = f"cvm_sbx.{project_name.replace('-', '_') + '_aud'}"
fav_omni_features_table = f"cvm_sbx.{project_name.replace('-', '_') + '_fav_omni'}"

# Artifacts for mlflow
artifacts_dir = "../artifacts"



# Мэппинг с названиями моделей
segments_mapping = {
    'no app': {
        'M1-2': 'no_app_m1-2',
        'M3-12': 'no_app_m3-12',
    },
    'no app / no transactions': {
        'M1-2': 'no_both_m1-2',
        'M3-12': 'no_both_m3-12',
    },
    'no transactions': {
        'M1-2': 'no_trns_m1-2',
        'M3-12': 'no_trns_m3-12',
    },
}

# Important columns
target = 'target_churn_from_dac'
score = 'score'

# Features
features = [
    'cheque_recency',
    'login_recency',
    'omni_qr_recency',
    'omni_features_recency',
    'rto_24',
    'aov_24',
    'trans_lag_avg_24',
    'whs_count_24',
    'format_count_24',
    'location_count_24',
    'whs_age_avg_24',
    'whs_is_closed_avg_24',
    'whs_in_capital_avg_24',
    'whs_in_city_avg_24',
    'population_avg_24',
    'competitor_echelon_avg_24',
    'aov_12',
    'trans_lag_avg_12',
    'aov_6',
    'trans_lag_avg_6',
    'aov_3',
    'trans_lag_avg_3',
    'trans_lag_avg_2',
    'trans_lag_avg_1',
    'login_count_24',
    'login_lag_avg_24',
    'login_count_12',
    'login_lag_avg_12',
    'login_count_6',
    'login_lag_avg_6',
    'login_count_3',
    'login_lag_avg_3',
    'login_lag_avg_2',
    'login_lag_avg_1',
    'omni_qr_days_count_24',
    'omni_qr_lag_avg_24',
    'omni_qr_days_count_12',
    'omni_qr_lag_avg_12',
    'omni_qr_days_count_6',
    'omni_qr_lag_avg_6',
    'omni_qr_days_count_2',
    'omni_features_days_count_24',
    'omni_features_lag_avg_24',
    'omni_features_lag_avg_12',
    'omni_features_lag_avg_6',
    'omni_features_lag_avg_3',
    'omni_features_lag_avg_2',
    'omni_unique_features_count_24',
    'omni_unique_features_count_2',
    'omni_unique_features_count_1',
    'accept_count_24',
    'accept_count_12',
    'accept_count_6',
    'bonus_accrued_sum_24',
    'bonus_accrued_offer_sum_24',
    'bonus_expired_sum_24',
    'bonus_accrued_sum_12',
    'bonus_expired_sum_12',
    'bonus_expired_sum_6',
    'bonus_accrued_sum_3',
    'bonus_expired_sum_3',
    'bonus_accrued_sum_2',
    'bonus_accrued_sum_1',
    'bonus_accrued_offer_sum_1',
    'bonus_expired_sum_1',
    'transaction_tendency',
    'login_tendency',
    'omni_qr_tendency',
    'omni_features_tendency',
    'dac_age_months',
    'cust_age_years',
    'is_male',
    'dac_months_count',
    'dac_months_per_dac_age_ratio',
    'uplift_rate',
    'trans_count_24',
    'trans_count_12',
    'trans_count_6',
    'trans_count_3',
    'trans_count_2',
    'trans_count_1',
    'square_trade_avg_24',
    'competitor_age_avg_24',
    'competitor_dist_avg_24',
    'omni_fav_features_days_count_24',
    'omni_fav_features_days_count_12',
    'omni_fav_features_days_count_6',
    'omni_fav_features_days_count_3',
    'omni_fav_features_days_count_2',
    'omni_fav_features_days_count_1',
    'omni_goals_activated_count_12',
    'omni_goals_updated_count_12',
    'omni_goals_finished_count_12',
    'omni_goals_activated_count_6',
    'omni_goals_updated_count_6',
    'omni_goals_finished_count_6',
    'omni_goals_activated_count_3',
    'omni_goals_updated_count_3',
    'omni_goals_finished_count_3',
    'omni_goals_activated_count_2',
    'omni_goals_updated_count_2',
    'omni_goals_finished_count_2',
    'omni_goals_activated_count_1',
    'omni_goals_updated_count_1',
    'omni_goals_finished_count_1',
    'bonus_redeemed_sum_24',
    'bonus_redeemed_sum_12',
    'bonus_redeemed_sum_6',
    'bonus_redeemed_sum_3',
    'bonus_redeemed_sum_2',
    'bonus_redeemed_sum_1',
    'avg_level_24',
    'avg_level_12',
    'avg_level_6',
    'avg_level_3',
    'avg_level_2',
    'avg_level_1',
    'max_consecutive_dac_months',
    'avg_consecutive_dac_months',
    'current_dac_streak',
    'dac_months_last_3',
    'dac_months_last_6',
    'dac_months_last_12',
    'dac_share_last_3',
    'dac_share_last_6',
    'dac_share_last_12',
    'is_stable_dac',
    'is_regular_dac',
    'is_unstable_dac',
    'is_new_dac',
    'transaction_active_months_last_12',
    'mobapp_active_months_last_12',
    'pwa_active_months_last_12',
    'vcoff_active_months_last_12',
    'base_has_transaction_activity',
    'base_has_mobapp_activity',
    'base_has_pwa_activity',
    'base_has_vcoff_activity',
    'base_vcoff_trn_cnt',
    'base_digital_mechanism_count',
]
features_for_outliers = [
    'rto_24',
    'trans_count_24',
    'rto_12',
    'trans_count_12',
    'rto_6',
    'trans_count_6',
    'rto_3',
    'trans_count_3',
    'rto_2',
    'trans_count_2',
    'rto_1',
    'trans_count_1'
]
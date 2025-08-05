FIRST_MODULE = [
    'survival_time_days',
    'event_indicator_all', 
    'dataset_split',
    'naics_cd',
    'age_at_vantage',
    'tenure_at_vantage_days',
    'baseline_salary'
]

SECOND_MODULE = [
    'baseline_salary',
    'salary_growth_rate_12m', 
    'peer_salary_ratio',
    'compensation_volatility',
    'compensation_percentile_company',
    'age_at_vantage',
    'tenure_at_vantage_days',
    'time_with_current_manager',
    'tenure_in_current_role',
    'pay_grade_stagnation_months',
    'team_size',
    'work_location_changes_count',
    'team_avg_turn_days'
] + [
    'pay_rate_type_cd',
    'career_stage',
    'generation_cohort',
    'gender_cd',
    'hire_date_seasonality',
    'full_tm_part_tm_cd',
    'reg_temp_cd',
    'flsa_stus_cd',
    'fscl_actv_ind',
    'company_size_tier',
    'naics_cd'
] + [
    'survival_time_days',
    'event_indicator_vol'
]

THIRD_MODULE = ['survival_time_days', 'event_indicator_all']

FOURTH_MODULE = [
    # Promotion intervention features
    'job_level',
    'mngr_lvl_cd', 
    'time_since_last_promotion',
    'promotion_velocity',
    
    # Salary intervention features
    'baseline_salary',
    'salary_growth_rate_12m',
    'compensation_percentile_company',
    'peer_salary_ratio',
    
    # Role change intervention features
    'role_complexity_score',
    'decision_making_authority_indicator',
    'tenure_in_current_role'
]


RAW_TO_NORMALIZED_MAPPING = {
    "pay_temp_cd": {"keep": ["T", "R"]},
    "naics_cd": {"drop": ["null"]},
    "full_tm_part_tm_cd": {"keep": ["F", "P"]},
    "quarter_effect": {"keep": ["Q1", "Q2", "Q3", "Q4"]},
    "fscl_actv_ind": {"keep": ["Y", "N"]},
    "gender_cd": {"keep": ["F", "M"]},
}

MISSING_VALUE_LIT = float("nan")
RAW_COLUMN_SUFFIX = "_raw"

# Event code mappings for feature engineering
EVENT_CODE_MAPPINGS = {
    "promotion": ["PRO"],
    "demotion": ["DEM"], 
    "transfer": ["XFR"],
    "job_change": ["JTC", "PJC", "POS", "JRC"],
    "performance": ["OPR", "OUTSTANDING PERFORMANCE", "PTP", "OP", "PER"],
    "title_change": ["JOB", "TC", "T", "PRO", "M28", "TCH", "TTL", "PTC", "PNP", "JTC", "OFC"],
    "market_adjustment": ["MKT", "MRK", "MKA"],
    "company_reorg": ["RES", "REO", "MOR", "TRP", "ORG", "MPC", "ROS"],
    "performance_issues": [
        "USP", "UNSATISFACTORY PERFORMANCE", "PER", "DEMOTE - PERFORMANCE",
        "PERFORMANCE", "UNS", "DEMOTE PERFORMANCE", "UNSATISFACTORY PERFORMANCE - USP",
        "301", "USJ", "UP", "PERFORMANCE-DRIVEN", "DUP", "PEF", "TPR", "S07", "PNU"
    ],
    "employee_request": ["EER", "EMPLOYEE REQUEST", "ER2", "ER1", "EE"],
    "skill_based": ["TRANSFER - SKILL-BASED"],
    "relocation": ["RELOCATION", "REL"],
    "assignment": ["TMP", "EXPATRIATE ASSIGNMENT", "ASC", "EXP", "1", "SAB", "SPA", "IPA"],
    "ft_to_pt": ["FTPT", "FT2PT"],
    "nonexempt_to_exempt": ["NEX", "NXX"]
}

# Feature validation rules
FEATURE_VALIDATION_RULES = {
    # Compensation features
    "baseline_salary": {"min": 0, "max": 1000000, "allow_null": False},
    "salary_growth_rate_12m": {"min": -1.0, "max": 5.0, "allow_null": True},
    "compensation_percentile_company": {"min": 0.0, "max": 1.0, "allow_null": True},
    "compensation_percentile_industry": {"min": 0.0, "max": 1.0, "allow_null": True},
    "compensation_volatility": {"min": 0.0, "max": None, "allow_null": True},
    "pay_grade_stagnation_months": {"min": 0, "max": 600, "allow_null": True},
    "total_compensation_growth": {"min": -1.0, "max": 5.0, "allow_null": True},
    "pay_frequency_preference": {"min": 0, "max": 2, "allow_null": False},
    "avg_salary_last_quarter": {"min": 0, "max": None, "allow_null": True},
    "pay_freq_consistency_score": {"min": 0.0, "max": 1.0, "allow_null": True},
    
    # Career progression features
    "time_since_last_promotion": {"min": 0, "max": 10000, "allow_null": True},
    "promotion_velocity": {"min": 0.0, "max": None, "allow_null": True},
    "promot_2yr_ind": {"min": 0, "max": 1, "allow_null": False},
    "demot_2yr_ind": {"min": 0, "max": 1, "allow_null": False},
    "num_promot_2yr": {"min": 0, "max": 20, "allow_null": False},
    "num_demot_2yr": {"min": 0, "max": 20, "allow_null": False},
    "promot_2yr_perf_ind": {"min": 0, "max": 1, "allow_null": False},
    "promot_2yr_titlechng_ind": {"min": 0, "max": 1, "allow_null": False},
    "promot_2yr_mktadjst_ind": {"min": 0, "max": 1, "allow_null": False},
    "demot_2yr_compreorg_ind": {"min": 0, "max": 1, "allow_null": False},
    "demot_2yr_perfissue_ind": {"min": 0, "max": 1, "allow_null": False},
    "days_since_promot": {"min": 0, "max": 10000, "allow_null": True},
    "promot_veloc": {"min": 0.0, "max": None, "allow_null": True},
    "promot_rt": {"min": 0.0, "max": 1.0, "allow_null": True},
    
    # Demographic features
    "age_at_vantage": {"min": 16, "max": 100, "allow_null": True},
    "career_stage": {"values": ["Early", "Mid", "Late", "Senior"], "allow_null": True},
    "retirement_eligibility_years": {"min": 0, "max": 50, "allow_null": True},
    "generation_cohort": {"values": ["Gen Z", "Millennial", "Gen X", "Baby Boomer", "Silent Generation"], "allow_null": True},
    "tenure_age_ratio": {"min": 0.0, "max": 1.0, "allow_null": True},
    "career_joiner_stage": {"values": ["early_career", "mid_career_joiner", "late_career_joiner", "experienced_loyal", "other"], "allow_null": True},
    "tenure_age_risk": {"values": ["burnout_risk", "retirement_on_disengagement_risk", "normal"], "allow_null": True},
    
    # Job characteristics
    "job_level": {"min": 1, "max": 10, "allow_null": True},
    "job_family_turnover_rate": {"min": 0.0, "max": 1.0, "allow_null": True},
    "role_complexity_score": {"min": 1, "max": 3, "allow_null": False},
    "job_stability_ind": {"min": 0, "max": 1, "allow_null": False},
    
    # Manager environment
    "time_with_current_manager": {"min": 0, "max": 10000, "allow_null": True},
    "manager_tenure_days": {"min": 0, "max": 20000, "allow_null": True},
    "manager_span_control": {"min": 0, "max": 1000, "allow_null": True},
    "manager_changes_count": {"min": 0, "max": 20, "allow_null": False},
    "is_manager_ind": {"min": 0, "max": 1, "allow_null": True},
    
    # Team environment
    "team_size": {"min": 1, "max": 1000, "allow_null": True},
    "team_turnover_rate_12m": {"min": 0.0, "max": 1.0, "allow_null": True},
    "team_avg_tenure": {"min": 0, "max": 20000, "allow_null": True},
    "peer_salary_ratio": {"min": 0.1, "max": 10.0, "allow_null": True},
    "team_avg_comp": {"min": 0, "max": None, "allow_null": True},
    "team_turnover_rate": {"min": 0.0, "max": 1.0, "allow_null": True},
    
    # Tenure dynamics
    "tenure_at_vantage_days": {"min": 0, "max": 20000, "allow_null": False},
    "tenure_in_current_role": {"min": 0, "max": 20000, "allow_null": True},
    "company_tenure_percentile": {"min": 1, "max": 99, "allow_null": True},
    
    # Work patterns
    "assignment_frequency_12m": {"min": 0, "max": 50, "allow_null": False},
    "work_location_changes_count": {"min": 0, "max": 20, "allow_null": False},
    "num_city_chng": {"min": 0, "max": 20, "allow_null": False},
    "num_state_chng": {"min": 0, "max": 20, "allow_null": False},
    "days_since_transfer": {"min": 0, "max": 10000, "allow_null": True},
    "transfer_2yr_ind": {"min": 0, "max": 1, "allow_null": False},
    "num_transfer_2yr": {"min": 0, "max": 20, "allow_null": False},
    
    # Company factors
    "company_size_tier": {"values": ["Small", "Medium", "Large", "Enterprise", "Unknown"], "allow_null": False},
    "company_layoff_indicator": {"min": 0, "max": 1, "allow_null": False},
    
    # Job change events
    "job_chng_2yr_ind": {"min": 0, "max": 1, "allow_null": False},
    "num_job_chng_2yr": {"min": 0, "max": 20, "allow_null": False},
    "job_chng_fulltopart_ind": {"min": 0, "max": 1, "allow_null": False},
    "job_chng_nexmptoexmp_ind": {"min": 0, "max": 1, "allow_null": False},
    
    # External features
    "salary_growth_rate12m_to_cpi_rate": {"min": -10.0, "max": 10.0, "allow_null": True},
    "sal_nghb_ratio": {"min": 0.1, "max": 20.0, "allow_null": True},
    
    # Target variables
    "survival_time_days": {"min": 0, "max": 365, "allow_null": False},
    "event_indicator_all": {"min": 0, "max": 1, "allow_null": False},
    "event_indicator_vol": {"min": 0, "max": 1, "allow_null": False},
}

# External data configuration
DATA_DIR_PATH = "/path/to/external/data"  # Update with actual path

EXTERNAL_DATA_CONFIG = {
    "state_to_region_file": "US_State_to_Region_Mapping.csv",
    "cpi_by_region_file": "US_CPI_Change_By_Regions.csv",
    "flsa_mapping_file": "flsa_client_mapping.csv",
    "census_income_file": "US_Census_Median_Income.csv",
}

# Fallback values when external data is unavailable
EXTERNAL_DATA_FALLBACKS = {
    "default_cpi_ratio": 1.0,
    "default_neighborhood_ratio": 1.0,
    "default_flsa_desc": "others",
    "default_region": "Unknown",
    "default_median_income": 50000,
}

# Temporal consistency validation rules
TEMPORAL_VALIDATION_RULES = {
    "max_future_days": 0,  # No features should use future data
    "required_temporal_columns": ["vantage_date"],
    "event_date_columns": ["event_eff_dt", "rec_eff_start_dt_mod", "termination_date"],
}

# Business logic validation thresholds
BUSINESS_LOGIC_THRESHOLDS = {
    "max_age": 100,
    "min_age": 16,
    "max_tenure_years": 50,
    "max_salary": 1000000,
    "min_salary": 10000,
    "max_manager_span": 1000,
    "max_promotion_velocity": 10.0,  # promotions per year
    "max_job_changes_2yr": 20,
    "outlier_percentile": 99.5,
}

BASE_COLS = [
    "person_composite_id", "dataset_split", "vantage_date", "event_indicator_all", "event_indicator_vol", "birth_dt", "sal_grd_cd", "headcount", "job_cd", "birth_dt",
    "event_indicator_vol", "work_loc_cd", "pers_clsfn_cd", "pers_stus_cd", "min_salary_to_date", "manager_changes_count", "pay_grp_cd", "cmpny_cd", "annl_cmpn_amt",
    "pay_rt_type_cd", "is_first_client", "dc_regn_bank_ky", "work_asgmnt_stus_desc", "mnly_cmpn_amt", "func_styl_desc", "func_cd", "trent_cd",
    "work_asgmnt_nm", "work_asgmnt_stus_cd", "prmry_job_titl_confndc_scor", "rec_eff_strt_dt", "clnt_live_ind", "has_jobd_cd", "full_tm_eqv_val", "max_salary_to_date",
    "empl_pers_obj_id", "team_avg_thur_days", "job_desc", "mnthly_cmpn_amt", "rec_eff_start_dt_mod", "sal_admln_plan_cd", "bus_cntl", "pay_actv_typ_cd", "ffte_cnt",
    "supvr_pers_obj_id", "salary_growth_ratio", "compa_rt", "termination_type", "job_changes_count", "term_type_cd", "mgr_lvl", "headcount_band", "ecol_job_catg_cd",
    "empl_obj_cd", "baseline_mobility", "mgr_chg_ctv", "distinct_assignments_count", "rec_eff_end_dt", "num_dt", "clnt_src_sys_nm", "glob_wscld_ind", "nm_generation",
    "ltst_hire_dt", "reg_tmp_cd", "ecol_job_cat_cd", "work_asgmnt_actv_ind", "ben_mrk_f", "termination_date", "new_range_band_y", "state_prov_cd", "state_prov_nm",
    "position", "cpi_2023",
]

SELECT_FEATURE_COLS = [
    # Compensation features
    "baseline_salary", "salary_growth_rate_12m", "compensation_percentile_company", "compensation_percentile_industry",
    "compensation_volatility", "pay_grade_stagnation_months", "total_compensation_growth", "pay_frequency_preference",
    "avg_salary_last_quarter", "pay_freq_consistency_score",
    
    # Career progression features
    "time_since_last_promotion", "promotion_velocity",
    
    # Event-based career features
    "promot_2yr_ind", "demot_2yr_ind", "transfer_2yr_ind", "num_promot_2yr", "num_demot_2yr", "num_transfer_2yr",
    "promot_2yr_perf_ind", "promot_2yr_titlechng_ind", "promot_2yr_mktadjst_ind", 
    "demot_2yr_compreorg_ind", "demot_2yr_perfissue_ind", "days_since_promot", "days_since_transfer",
    "promot_veloc", "promot_rt",
    
    # Demographic features
    "age_at_vantage", "gender_cd", "career_stage", "retirement_eligibility_years", "generation_cohort",
    "tenure_age_ratio", "career_joiner_stage", "tenure_age_risk",
    
    # Job characteristics
    "job_level", "job_family_turnover_rate", "flsa_stus_cd", "full_tm_part_tm_cd", "reg_temp_cd", "role_complexity_score",
    "job_stability_ind",
    
    # Manager environment
    "time_with_current_manager", "manager_tenure_days", "manager_span_control", "manager_changes_count", "is_manager_ind",
    
    # Team environment
    "team_size", "team_turnover_rate_12m", "team_avg_tenure", "peer_salary_ratio", "team_avg_comp", "team_turnover_rate",
    
    # Tenure dynamics
    "tenure_at_vantage_days", "tenure_in_current_role", "company_tenure_percentile",
    
    # Work patterns
    "assignment_frequency_12m", "work_location_changes_count", "num_city_chng", "num_state_chng",
    
    # Company factors
    "naics_cd", "company_size_tier", "company_layoff_indicator", "fscl_actv_ind",
    
    # Temporal features
    "hire_date_seasonality", "fiscal_year_effect", "quarter_effect",
    
    # Job change events
    "job_chng_2yr_ind", "num_job_chng_2yr", "job_chng_fulltopart_ind", "job_chng_nexmptoexmp_ind",
    
    # External features
    "salary_growth_rate12m_to_cpi_rate", "flsa_status_desc", "sal_nghb_ratio",
    
    # Target variables
    "survival_time_days", "event_indicator_all", "event_indicator_vol",
]

RAW_COLS = [f"{col}{RAW_COLUMN_SUFFIX}" for col in RAW_TO_NORMALIZED_MAPPING]

FINAL_COLS = list(set(BASE_COLS).union(set(SELECT_FEATURE_COLS)).union(set(RAW_COLS)))

# Sample IDs for testing (placeholder)
SAMPLE_IDS = []

# Table names
ONEDATA_CATALOG_NM = "onedata_us_east_1_shared_prod"
TOP_SCHEMA_NM = "datacloud_raw_oneai_turnoverprobability_prod"

# Feature groups for organized processing
FEATURE_GROUPS = {
    "compensation": [
        "baseline_salary", "salary_growth_rate_12m", "compensation_percentile_company", 
        "compensation_percentile_industry", "compensation_volatility", "pay_grade_stagnation_months",
        "total_compensation_growth", "pay_frequency_preference", "avg_salary_last_quarter", 
        "pay_freq_consistency_score"
    ],
    "career_progression": [
        "time_since_last_promotion", "promotion_velocity", "promot_2yr_ind", "demot_2yr_ind", 
        "transfer_2yr_ind", "num_promot_2yr", "num_demot_2yr", "num_transfer_2yr",
        "promot_2yr_perf_ind", "promot_2yr_titlechng_ind", "promot_2yr_mktadjst_ind",
        "demot_2yr_compreorg_ind", "demot_2yr_perfissue_ind", "days_since_promot", 
        "days_since_transfer", "promot_veloc", "promot_rt"
    ],
    "demographic": [
        "age_at_vantage", "gender_cd", "career_stage", "retirement_eligibility_years", 
        "generation_cohort", "tenure_age_ratio", "career_joiner_stage", "tenure_age_risk"
    ],
    "job_characteristics": [
        "job_level", "job_family_turnover_rate", "flsa_stus_cd", "full_tm_part_tm_cd", 
        "reg_temp_cd", "role_complexity_score", "job_stability_ind"
    ],
    "manager_environment": [
        "time_with_current_manager", "manager_tenure_days", "manager_span_control", 
        "manager_changes_count", "is_manager_ind"
    ],
    "team_environment": [
        "team_size", "team_turnover_rate_12m", "team_avg_tenure", "peer_salary_ratio", 
        "team_avg_comp", "team_turnover_rate"
    ],
    "tenure_dynamics": [
        "tenure_at_vantage_days", "tenure_in_current_role", "company_tenure_percentile"
    ],
    "work_patterns": [
        "assignment_frequency_12m", "work_location_changes_count", "num_city_chng", "num_state_chng"
    ],
    "company_factors": [
        "naics_cd", "company_size_tier", "company_layoff_indicator", "fscl_actv_ind"
    ],
    "temporal": [
        "hire_date_seasonality", "fiscal_year_effect", "quarter_effect"
    ],
    "job_change_events": [
        "job_chng_2yr_ind", "num_job_chng_2yr", "job_chng_fulltopart_ind", "job_chng_nexmptoexmp_ind"
    ],
    "external": [
        "salary_growth_rate12m_to_cpi_rate", "flsa_status_desc", "sal_nghb_ratio"
    ],
    "target": [
        "survival_time_days", "event_indicator_all", "event_indicator_vol"
    ]
}

# Critical features that must be present
CRITICAL_FEATURES = [
    "person_composite_id", "vantage_date", "baseline_salary", "tenure_at_vantage_days",
    "age_at_vantage", "job_level", "survival_time_days", "event_indicator_all"
]

# Feature engineering configuration
FEATURE_CONFIG = {
    "promotion_lookback_days": 730,  # 2 years
    "salary_history_days": 365,     # 1 year
    "team_analysis_days": 730,      # 2 years for team turnover
    "assignment_frequency_days": 365, # 1 year
    "gap_fill_threshold_days": 30,
    "max_manager_span": 1000,
    "retirement_age": 85,
    "working_age_start": 25,
}

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    "max_null_percentage": 0.95,  # Reject features with >95% nulls
    "min_variance": 0.001,        # Reject features with minimal variance
    "max_correlation": 0.99,      # Flag highly correlated features
    "outlier_std_threshold": 5,   # Flag outliers beyond 5 standard deviations
}
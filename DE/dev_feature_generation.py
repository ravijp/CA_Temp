# Import requirements
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
from top_etl.common.uc_volume_io import *

# Pre-defined Functions

def create_promotion_demotion_count_features(df):
    """
    Create employee promotion features from raw employee data
    Returns one record per person_composite_id with all engineered features
    
    Parameters:
    df: Input PySpark DataFrame with columns:
        - person_composite_id: employee identifier
        - event_cd: event code ('PRO' for promotion, 'DEM' for demotion)
        - vantage_date: reference date
        - event_eff_dt: date when promotion/demotion occurred
        - lst_promo_dt: date of last promotion
    
    Returns:
    DataFrame with one record per employee containing all features
    """
    
    print("Starting feature engineering process...")
    
    # Convert date columns to proper date format if they're strings
    print("Converting date columns to proper format...")
    df = df.withColumn("vantage_date", F.to_date(F.col("vantage_date"))) \
           .withColumn("event_eff_dt", F.to_date(F.col("event_eff_dt"))) \
           .withColumn("lst_promo_dt", F.to_date(F.col("lst_promo_dt")))
    
    # Filter for events that happened before or on vantage date
    print("Filtering events that occurred before or on vantage date...")
    df_filtered = df.filter(F.col("event_eff_dt") <= F.col("vantage_date"))
    
    # Define window specifications
    print("Setting up window specifications for aggregations...")
    window_spec_person = Window.partitionBy("person_composite_id")
    window_spec_person_ordered = Window.partitionBy("person_composite_id").orderBy("event_eff_dt")
    
    # Step 1: Generate all features for each record
    print("Step 1: Generating individual feature flags for each record...")
    features_expanded = (
        df_filtered
        .withColumn(
            "promot_2yr_ind_raw",
            F.when((F.col("event_cd") == "PRO") & 
                 (F.col("event_eff_dt") >= F.add_months(F.col("vantage_date"), -24)), 1).otherwise(0)
        ).withColumn(
            "demot_2yr_ind_raw",
            F.when((F.col("event_cd") == "DEM") & 
                 (F.col("event_eff_dt") >= F.add_months(F.col("vantage_date"), -24)), 1).otherwise(0)
        ).withColumn(
            "num_promot_2yr_raw",
            F.when((F.col("event_cd") == "PRO") & 
                 (F.col("event_eff_dt") >= F.add_months(F.col("vantage_date"), -24)), 1).otherwise(0)
        ).withColumn(
            "num_demot_2yr_raw",
            F.when((F.col("event_cd") == "DEM") & 
                 (F.col("event_eff_dt") >= F.add_months(F.col("vantage_date"), -24)), 1).otherwise(0)
        )
    )
    
    print("Step 1 completed: Individual feature flags generated")
    
    # **LINES 97-105: UPDATED LOGIC FOR DAYS SINCE LAST PROMOTION**
    # First, get the most recent promotion date for each person from event records
    print("Step 2: Calculating days since last promotion with fallback logic...")
    latest_promotion_from_events = features_expanded.filter(F.col("event_cd") == "PRO") \
        .withColumn("latest_promo_from_events",
                    min("event_eff_dt").over(window_spec_person)) \
        .select("person_composite_id", "latest_promo_from_events").distinct()
    
    # Join back to get the latest promotion date from events
    features_expanded = features_expanded.join(
        latest_promotion_from_events, on="person_composite_id", how="left"
    )
    
    # Step 2: Calculate promotion intervals for average days calculation
    print("Step 3: Calculating promotion intervals for average days between promotions...")
    promotion_records = features_expanded.filter(F.col("event_cd") == "PRO")
    
    # Add previous promotion date using window function
    print("Adding lag values to calculate intervals between consecutive promotions...")
    promotion_with_lag = promotion_records.withColumn(
        "prev_promo_date",
        F.lag("event_eff_dt").over(window_spec_person_ordered)
    ).withColumn(
        "days_between_current_and_prev",
        F.when(F.col("prev_promo_date").isNotNull(),
             F.datediff(F.col("event_eff_dt"), F.col("prev_promo_date"))).otherwise(None)
    )
    
    # Calculate running statistics for average days between promotions
    print("Calculating average days between promotions for each employee...")
    promotion_with_stats = promotion_with_lag.withColumn(
        "promotion_rank",
        F.row_number().over(window_spec_person_ordered)
    ).withColumn(
        "total_promotions_for_person",
        F.count("*").over(window_spec_person)
    ).withColumn(
        "sum_days_between_promos",
        sum("days_between_current_and_prev").over(window_spec_person)
    ).withColumn(
        "count_intervals",
        F.count("days_between_current_and_prev").over(window_spec_person)
    )
    
    # Step 4: Aggregate all features per person using window functions to get final values
    print("Step 5: Aggregating all features per person using window functions...")
    features_aggregated = (features_expanded
        .withColumn(
            "promot_2yr_ind",
            max("promot_2yr_ind_raw").over(window_spec_person)
        ).withColumn(
            "demot_2yr_ind",
            max("demot_2yr_ind_raw").over(window_spec_person)
        ).withColumn(
            "num_promot_2yr",
            sum("num_promot_2yr_raw").over(window_spec_person)
        ).withColumn(
            "num_demot_2yr",
            sum("num_demot_2yr_raw").over(window_spec_person)
        )
    )
    
    # Step 5: Get one record per person with final aggregated features
    print("Step 6: Creating final dataset with one record per employee...")
    final_features = features_aggregated.select(
        "person_composite_id",
        "vantage_date",
        "promot_2yr_ind",
        "demot_2yr_ind",
        "num_promot_2yr",
        "num_demot_2yr",
    ).distinct()
    
    print("Feature engineering completed successfully!")
    return final_features


def create_promotion_demotion_transfer_indicators_features(df):
    """
    Generate promotion, demotion, and transfer features for employee dataset
    
    Args:
        df: PySpark DataFrame containing employee data
        
    Returns:
        PySpark DataFrame with additional features
    """
    
    print("Starting feature generation for promotion, demotion, and transfer...")
    
    # Filter data to include only events before vantage_date
    print("Filtering events that occurred before vantage_date...")
    df_filtered = df.filter(F.col("event_eff_dt") < F.col("vantage_date"))
    
    # Create binary flags for event types
    print("Creating event type flags...")
    df_with_flags = df_filtered.withColumn(
        "is_promotion",
        F.when(F.col("event_cd") == "PRO", 1).otherwise(0)
    ).withColumn(
        "is_demotion",
        F.when(F.col("event_cd") == "DEM", 1).otherwise(0)
    ).withColumn(
        "is_transfer",
        F.when(F.col("event_cd") == "XFR", 1).otherwise(0)
    )
    
    # Create reason code flags for detailed categorization
    print("Creating detailed reason code flags...")
    df_with_reasons = df_with_flags.withColumn(
        "is_outstanding_performance",
        F.when(F.col("event_rsn_cd").isin(["OPR", "Outstanding Performance", "PTP", "OP", "PER"]), 1).otherwise(0)
    ).withColumn(
        "is_title_change_only",
        F.when(F.col("event_rsn_cd").isin(["JOB", "TC", "T", "PRO", "M28", "TCH", "TTL", "PTC", "PNP", "JTC", "OFC"]), 1).otherwise(0)
    ).withColumn(
        "is_market_adjustment",
        F.when(F.col("event_rsn_cd").isin(["MKT", "MRK", "MKA"]), 1).otherwise(0)
    ).withColumn(
        "is_company_reorg",
        F.when(F.col("event_rsn_cd").isin(["RES", "REO", "MOR", "TRP", "ORG", "MPC", "ROS"]), 1).otherwise(0)
    ).withColumn(
        "is_performance_issues",
        F.when(F.col("event_rsn_cd").isin(["USP", "Unsatisfactory Performance", "PER", "Demote - Performance",
                                          "Performance", "UNS", "Demote Performance", "Unsatisfactory Performance - USP",
                                          "301", "USJ", "UP", "Performance-Driven", "DUP", "PEF", "TPR", "S07", "PNU"]), 1).otherwise(0)
    ).withColumn(
        "is_employee_request",
        F.when(F.col("event_rsn_cd").isin(["EER", "Employee Request", "ER2", "ER1", "EE"]), 1).otherwise(0)
    ).withColumn(
        "is_skill_based",
        F.when(F.col("event_rsn_cd").contains("Transfer - Skill-based"), 1).otherwise(0)
    ).withColumn(
        "is_relocation",
        F.when(F.col("event_rsn_cd").isin(["Relocation", "REL"]), 1).otherwise(0)
    ).withColumn(
        "is_assignment",
        F.when(F.col("event_rsn_cd").isin(["TMP", "Expatriate Assignment", "ASC", "EXP", "1", "SAB", "SPA", "IPA"]), 1).otherwise(0)
    )
    
    # Calculate last 2 years window
    print("Creating 2-year lookback window...")
    two_years_ago = F.date_sub(F.col("vantage_date"), 730)  # 2 years = 730 days
    
    df_with_lookback = df_with_reasons.withColumn(
        "is_within_2_years",
        F.when(F.col("event_eff_dt") >= two_years_ago, 1).otherwise(0)
    )
    
    # Calculate days since last transfer for each person
    print("Calculating days since last transfer...")
    transfer_events = df_with_lookback.filter(F.col("is_transfer") == 1)
    
    # Get the most recent transfer date for each person
    latest_transfer_per_person = transfer_events.groupBy("person_composite_id", "vantage_date").agg(
        F.max("event_eff_dt").alias("last_transfer_date")
    )
    
    # Generate aggregate features per person
    print("Generating aggregate features per person...")
    person_aggregates = df_with_lookback.groupBy("person_composite_id", "vantage_date").agg(
        # Promotion features
        F.max(F.when((F.col("is_promotion") == 1) & (F.col("is_outstanding_performance") == 1), 1).otherwise(0)).alias("promot_2yr_perf_ind"),
        F.max(F.when((F.col("is_promotion") == 1) & (F.col("is_title_change_only") == 1), 1).otherwise(0)).alias("promot_2yr_titlechng_ind"),
        F.max(F.when((F.col("is_promotion") == 1) & (F.col("is_market_adjustment") == 1), 1).otherwise(0)).alias("promot_2yr_mktadjst_ind"),
        
        # Demotion features
        F.max(F.when((F.col("is_demotion") == 1) & (F.col("is_company_reorg") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("demot_2yr_compreorg_ind"),
        F.max(F.when((F.col("is_demotion") == 1) & (F.col("is_performance_issues") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("demot_2yr_perfissue_ind"),
        
        # Transfer features
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("transfer_2yr_ind"),
        F.sum(F.when((F.col("is_transfer") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("num_transfer_2yr"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_company_reorg") == 1), 1).otherwise(0)).alias("transfer_2yr_reorg_ind"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_employee_request") == 1), 1).otherwise(0)).alias("transfer_2yr_req_ind"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_skill_based") == 1), 1).otherwise(0)).alias("transfer_2yr_skill_ind"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_relocation") == 1), 1).otherwise(0)).alias("transfer_2yr_reloc_ind"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_assignment") == 1), 1).otherwise(0)).alias("transfer_2yr_asgnmt_ind")
    )
    
    # Get the original dataframe structure (one record per person)
    print("Creating base dataframe with one record per person...")
    base_df = df_filtered.select("person_composite_id", "vantage_date").distinct()
    
    # Join aggregated features back to base dataframe
    print("Joining features back to base dataframe...")
    result_df = base_df.join(person_aggregates, ["person_composite_id", "vantage_date"], "left")
    
    # Join days since last transfer
    print("Adding days since last transfer...")
    latest_transfer_per_person = latest_transfer_per_person.filter(F.col("last_transfer_date") < F.col("vantage_date"))
    result_df = result_df.join(latest_transfer_per_person, ["person_composite_id", "vantage_date"], "left").withColumn(
        "days_since_transfer",
        F.when(F.col("last_transfer_date").isNotNull(),
               F.datediff(F.col("vantage_date"), F.col("last_transfer_date")))
        .otherwise(F.lit(None))
    ).drop("last_transfer_date")
    
    # Fill null values with 0 for binary features and appropriate defaults for others
    print("Filling null values with appropriate defaults...")
    feature_columns = [
        "promot_2yr_perf_ind",
        "promot_2yr_titlechng_ind", 
        "promot_2yr_mktadjst_ind",
        "demot_2yr_compreorg_ind",
        "demot_2yr_perfissue_ind",
        "transfer_2yr_ind",
        "transfer_2yr_reorg_ind",
        "transfer_2yr_req_ind",
        "transfer_2yr_skill_ind",
        "transfer_2yr_reloc_ind",
        "transfer_2yr_asgnmt_ind"
    ]
    
    for col in feature_columns:
        result_df = result_df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    
    # Fill total_transfers_in_last2_years with 0
    result_df = result_df.withColumn("num_transfer_2yr", F.coalesce(F.col("num_transfer_2yr"), F.lit(0)))
    print(result_df.filter(F.col("days_since_transfer") < 0).count(), "check check")
    print("Feature generation completed successfully!")
    return result_df


def create_promotion_time_interaction_features(df):
    """
    Generate promotion-related features for employees
    Features:
    1. promot_veloc: Average days between promotions
    2. promotion_rate: Number of promotions over a period of time
    3. time_since_last_promotion: Number of days since last promotion from vantage date
    """
    
    print("Starting feature_set_3 generation...")
    
    # Filter data for relevant events before vantage date
    print("Filtering data for promotion/demotion/transfer events before vantage date...")
    filtered_df = df.filter(F.col("event_eff_dt") < F.col("vantage_date"))
    
    # Create window specification for each person ordered by event date
    person_window = Window.partitionBy("person_composite_id").orderBy("event_eff_dt")
    person_window_desc = Window.partitionBy("person_composite_id").orderBy(F.desc("event_eff_dt"))
    
    print("Calculating promotion-specific metrics...")
    
    # Filter for promotions only
    promotions_df = filtered_df.filter(F.col("event_cd") == "PRO")
    
    # Add row numbers and lag functions for promotion calculations
    promotions_with_metrics = promotions_df.withColumn(
        "row_num", F.row_number().over(person_window)
    ).withColumn(
        "prev_promo_date", F.F.lag("event_eff_dt", 1).over(person_window)
    ).withColumn(
        "days_between_promotions",
        F.when(F.col("prev_promo_date").isNotNull(),
               F.datediff(F.col("event_eff_dt"), F.col("prev_promo_date")))
    )
    
    # Calculate promotion velocity (average days between promotions)
    promotion_velocity = promotions_with_metrics.filter(
        F.col("days_between_promotions").isNotNull()
    ).groupBy("person_composite_id", "vantage_date").agg(
        F.avg("days_between_promotions").alias("promot_veloc")
    )
    
    print("Calculating promotion rate...")
    
    # Calculate promotion rate (total number of promotions)
    promotion_rate = promotions_df.groupBy("person_composite_id", "vantage_date").agg(
        F.count("*").alias("num_promotions"),
        F.min("event_eff_dt").alias("first_promo_date"),
        F.max("event_eff_dt").alias("last_promo_date")
    ).withColumn(
        "promot_rt",
        F.when(
            F.datediff(F.col("last_promo_date"), F.col("first_promo_date")) > 0,
            F.col("num_promotions") / F.datediff(F.col("last_promo_date"), F.col("first_promo_date"))
        ).otherwise(F.lit(0.0))
    ).select("person_composite_id", "vantage_date", "promot_rt")
    
    print("Calculating time since last promotion...")
    
    # Get the most recent promotion date for each person
    last_promotion = promotions_df.withColumn(
        "row_num", F.row_number().over(person_window_desc)
    ).filter(F.col("row_num") == 1).select(
        "person_composite_id",
        "event_eff_dt",
        "vantage_date"
    )
    
    # Calculate days since last promotion
    time_since_last_promotion = last_promotion.withColumn(
        "days_since_promot",
        F.datediff(F.col("vantage_date"), F.col("event_eff_dt"))
    ).select("person_composite_id", "vantage_date", "days_since_promot")
    
    print("Getting unique persons from original dataset...")
    
    # Get all unique persons with their vantage dates
    unique_persons = df.select("person_composite_id", "vantage_date").distinct()
    
    print("Joining all promotion features...")
    
    # Join all features together
    result_df = unique_persons.join(
        promotion_velocity, ["person_composite_id", "vantage_date"], "left"
    ).join(
        promotion_rate, ["person_composite_id", "vantage_date"], "left"
    ).join(
        time_since_last_promotion, ["person_composite_id", "vantage_date"], "left"
    )
    
    print("Filling null values and finalizing features...")
    
    # Fill null values with appropriate defaults
    final_df = result_df.fillna({
        "promot_veloc": 0.0,
        "promot_rt": 0,
        "days_since_promot": 0
    })
    
    # Handle case where promotion_velocity might be null for single promotions
    final_df = final_df.withColumn(
        "promot_veloc",
        F.when(
            (F.col("promot_rt") == 1) & (F.col("promot_veloc") == 0.0),
            F.lit(None).cast("double")
        ).otherwise(F.col("promot_veloc"))
    )
    
    print("Feature generation completed successfully!")
    print("Features created:")
    print("- promot_veloc: Average days between promotions")
    print("- promotion_rate: Number of promotions over time period")
    print("- days_since_promot: Days since last promotion from vantage date")
    
    return final_df


def add_work_patterns_features_s(df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
    # Contains incremental work pattern features
    
    # Read work location dimension table
    work_loc_mapping = spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
    
    # Getting counts at 3 joining keys
    work_loc_counts = (
        work_loc_mapping
        .groupBy("db_schema", "clnt_obj_id", "work_loc_cd")
        .count()
    )
    
    # Removing records from mapping where the count is not 1
    work_loc_counts = work_loc_counts.filter(F.col("count")==1)
    work_loc_mapping = work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"]).select(["db_schema", "clnt_obj_id", "work_loc_cd", "CITY_NM", "state_prov_nm"])
    
    # Add city and state names to start stop format
    start_stop_df = start_stop_df.join(work_loc_mapping, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how="left")
    
    # Work city changes count
    city_location_changes = (
        start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
        .filter(F.col("city_nm").isNotNull())
        .filter(~F.upper(F.col("city_nm")).isin(["UNKNOWN", "MISSING", "NULL"]))
        .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "city_nm")
        .distinct()
        .withColumn(
            "prev_city_location",
            F.F.lag("city_nm").over(
                Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
            ),
        )
        .withColumn(
            "location_changed",
            F.when(
                (F.col("prev_city_location").isNotNull()) & (F.col("city_nm") != F.col("prev_city_location")), 1
            ).otherwise(0),
        )
        .groupBy("person_composite_id", "vantage_date")
        .agg(F.sum("location_changed").alias("num_city_chng"))
        .fillna(0, subset=["num_city_chng"])
    )
    
    # Work state changes count
    state_location_changes = (
        start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
        .filter(F.col("state_prov_nm").isNotNull())
        .filter(~F.col("state_prov_nm").isin(["UNKNOWN", "MISSING", "NULL"]))
        .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "state_prov_nm")
        .distinct()
        .withColumn(
            "prev_state_location",
            F.F.lag("state_prov_nm").over(
                Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
            ),
        )
        .withColumn(
            "location_changed",
            F.when(
                (F.col("prev_state_location").isNotNull()) & (F.col("state_prov_nm") != F.col("prev_state_location")), 1
            ).otherwise(0),
        )
        .groupBy("person_composite_id", "vantage_date")
        .agg(F.sum("location_changed").alias("num_state_chng"))
        .fillna(0, subset=["num_state_chng"])
    )
    
    # Join work patterns features
    df = (
        df.join(city_location_changes, ["person_composite_id", "vantage_date"], "left")
        .join(state_location_changes, ["person_composite_id", "vantage_date"], "left")
        .fillna(0, subset=["num_city_chng", "num_state_chng"])
    )
    
    df = df.select("person_composite_id", "vantage_date", "num_city_chng", "num_state_chng")
    
    return df


def add_age_tenure_features_s(df: DataFrame) -> DataFrame:
    """Add age and tenure features to the base data."""
    
    # Add age feature
    df = df.withColumn("tenure_age_ratio", ((F.col("tenure_at_vantage_days") / 365.25) / F.col("age_at_vantage")))
    df = df.select(['person_composite_id', 'vantage_date', 'tenure_age_ratio'])
    
    return df


def add_manager_environment_features_s(df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
    
    # Read employee main monthly
    ebm_table = spark.table("us_east_1_prd_ds_blue_landing_base.employee_base_monthly")
    
    # Filter on 2023 Jan and keep selected columns
    ebm_2023 = ebm_table.filter(F.col("yyyymm")=="202301").select("ooid", "aoid", "source_hr", "is_manager_").distinct()
    
    # Dedupe EBM based on Dinesh's inputs - using source_hr
    window_spec = Window.partitionBy("aoid", "ooid").orderBy(F.col("source_hr").asc_nulls_last())
    ebm_2023_dedupe = ebm_2023.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
    
    # Filter on 2024 Jan and keep selected columns
    ebm_2024 = ebm_table.filter(F.col("yyyymm")=="202401").select("ooid", "aoid", "source_hr", "is_manager_").distinct()
    
    # Dedupe EBM based on Dinesh's inputs - using source_hr
    window_spec = Window.partitionBy("aoid", "ooid").orderBy(F.col("source_hr").asc_nulls_last())
    ebm_2024_dedupe = ebm_2024.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
    
    # Add vantage date column to 2023 and 2024 EBM data; Rename columns
    ebm_2023_dedupe = ebm_2023_dedupe.withColumn("vantage_date", F.lit("2023-01-01"))
    ebm_2023_dedupe = ebm_2023_dedupe.withColumnRenamed("ooid", "clnt_obj_id")
    ebm_2023_dedupe = ebm_2023_dedupe.withColumnRenamed("aoid", "prsn_obj_id")
    
    ebm_2024_dedupe = ebm_2024_dedupe.withColumn("vantage_date", F.lit("2024-01-01"))
    ebm_2024_dedupe = ebm_2024_dedupe.withColumnRenamed("ooid", "clnt_obj_id")
    ebm_2024_dedupe = ebm_2024_dedupe.withColumnRenamed("aoid", "prsn_obj_id")
    
    # Concat both EBM data
    ebm_comb = ebm_2023_dedupe.union(ebm_2024_dedupe)
    
    # Derive columns from modeling data
    df = df.withColumn("clnt_obj_id" , F.split(F.col("person_composite_id"), "_")[2])
    df = df.withColumn("prsn_obj_id" ,F.split(F.col("person_composite_id"), "_")[3])
    
    # Join with modeling data
    df = df.join(ebm_comb, on = ["clnt_obj_id", "prsn_obj_id", "vantage_date"], how = 'left')
    df = df.withColumnRenamed("is_manager_", "is_manager_ind")
    
    df = df.select("person_composite_id", "vantage_date", "is_manager_ind")
    return df


def add_external_features_1s(df: DataFrame) -> DataFrame:
    """
    Add external features
    """
    
    # Read work location table
    work_loc_mapping = spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
    work_loc_mapping = work_loc_mapping.select("clnt_obj_id", "work_loc_cd", "db_schema", "pstl_cd", "state_prov_cd", "state_prov_nm")
    
    # Remove records from work location table which have multiple state values
    work_loc_counts = (
        work_loc_mapping
        .groupBy("db_schema", "clnt_obj_id", "work_loc_cd")
        .count())
    work_loc_counts = work_loc_counts.filter(F.col("count")==1)
    work_loc_mapping = work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"])
    
    # Read State-to-region mapping
    state_to_region_mapping = spark.createDataFrame(pd.read_csv('US_State_to_Region_Mapping.csv', encoding='ISO-8859-1'))
    
    # Change State name to uppercase in work_loc_mapping and state-to-region mapping
    work_loc_mapping = work_loc_mapping.withColumn("state_upper", F.upper(F.col("state_prov_nm")))
    state_to_region_mapping = state_to_region_mapping.withColumn("state_upper", F.upper(F.col("State")))
    
    # Join work_loc_mapping and state-to-region mapping to get Region
    df_region = work_loc_mapping.join(state_to_region_mapping, on = 'state_upper', how='left')
    
    # Read external data having region-wise % CPI change for 2023 and 2024
    df_cpi = pd.read_csv('US_CPI_Change_By_Regions.csv', encoding='ISO-8859-1')
    df_cpi = df_cpi.rename(columns={'1_Jan_23': 'CPI_2023_', '1_Jan_24':'CPI_2024_' } )
    
    #  Join df_region to external CPI data on region
    df_wcpi = df_region.join(spark.createDataFrame(df_cpi), on=["Region"], how = "left")
    
    # Extract db_schema as the first element from person_composite_id separated by "_"
    df = df.withColumn(
        "db_schema" ,
        F.concat(F.split(F.col("person_composite_id"), "_")[0], F.lit("_util"))
    )
    
    df = df.withColumn(
        "clnt_obj_id" ,
        F.split(F.col("person_composite_id"), "_")[2]
    )
    
    # Join df data with df_wcpi on db_schema, client id and work location code
    df_all = df.join(df_wcpi, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how = "left")
    
    # Calculate feature -> salary growth rate to CPI growth rate
    # df_all = df_all.withColumn("salary_growth_rate12m_to_cpi_rate"
    #                           , F.when(F.col("CPI_2023").isNotNull()
    #                                  , F.col("salary_growth_rate_12m") / F.col("CPI_2023")).otherwise(0.0))
    
    # Updated logic
    df_all = df_all.withColumn(
        "CPI",
        F.when(F.col("dataset_split").isin(["train", "val"]), F.col("CPI_2023_"))
        .otherwise(F.col("CPI_2024_"))
    )
    df_all = df_all.withColumn("sal_cpi_rt_ratio_1yr", F.when(F.col("CPI").isNotNull(), F.col("salary_growth_rate_12m") / F.col("CPI")).otherwise(0.0))
    
    # Keep only required columns
    df_wselected_cols = df_all.select("person_composite_id", "vantage_date", "sal_cpi_rt_ratio_1yr")
    
    return df_wselected_cols

def add_external_features_2s(df: DataFrame) -> DataFrame:
    """
    Add external features
    """
    
    # Read ADP internal geo-code table (table contains census ID per employee guid)
    geo_code = spark.table("us_east_1_prd_ven_blue_landing_base.pb_geocode_result_monthly_gz")
    
    # Dedupe table and keep selected columns and filtered data (dedupe logic shared by Blair; Discussed)
    geo_code = geo_code.filter(F.col("yyyymm") == "202301").withColumn("rn", F.row_number().over(
        Window.partitionBy("employee_guid", "yyyymm").orderBy(F.col("home_congressional_district"))
    )).filter(F.col("rn") == 1
    ).select("employee_guid", "home_census_block_group", "yyyymm")
    
    # Read US Census data
    # Download the csv file from https://data.census.gov/table/ACSST5Y2023.S1903?q=ACS+data&t=Income+and+Poverty&g=010XX00US$1400000
    us_census = spark.createDataFrame(pd.read_csv('US_Census_Median_Income.csv', encoding='ISO-8859-1'))
    
    ## Join ADP internal geo-code table with US Census data
    # create mapping key in geo_code data
    geo_code = geo_code.withColumn("census_mapping_key", F.substring("home_census_block_group", 1, 11))
    
    # create mapping key in us_census data
    us_census = us_census.withColumn("census_mapping_key", F.expr("substring(GEO_ID, -11, 11)"))
    
    # Join on census_mapping_key
    geo_code_wext = geo_code.join(us_census, "census_mapping_key", "left")
    
    # Read Employee Base Monthly
    ebm_table = spark.table("us_east_1_prd_ds_blue_landing_base.employee_base_monthly")
    
    # # Drop all MAP type columns before performing distinct
    # map_columns = [field.name for field in ebm_table.schema.fields if str(field.dataType).startswith("MapType")]
    # ebm_table_no_map = ebm_table.drop(*map_columns)
    
    # Filter on 2023 Jan and keep selected columns. Only 2023 data available on US Census website
    ebm_table_no_map_2023 = ebm_table.filter(F.col("yyyymm")=="202301").select("ooid", "aoid", "employee_guid", "source_hr").distinct()
    
    # Dedupe EBM based on Dinesh's inputs -> Use source_hr column and take non-null entry
    window_spec = Window.partitionBy("aoid", "ooid").orderBy(F.col("source_hr").asc_nulls_last())
    ebm_table_no_map_2023_dedupe = ebm_table_no_map_2023.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
    
    # Join EBM table with geo-code table having external data
    ebm_w_ext = ebm_table_no_map_2023_dedupe.join(geo_code_wext,on = "employee_guid", how = "left")
    
    # Create mapping columns in df
    df = df.withColumn(
        "aoid" ,
        F.split(F.col("person_composite_id"), "_")[3])
    df = df.withColumn(
        "ooid" ,
        F.split(F.col("person_composite_id"), "_")[2])
    
    # Join ebm_w_ext with df
    df = df.join(ebm_w_ext, on = ["ooid", "aoid"], how = "left")
    df = df.withColumnRenamed("S1903_C03_001E", "median_household_income")
    
    # Create feature sal_nghb_ratio
    df = df.withColumn("sal_nghb_ratio", F.when(F.col("median_household_income").isNotNull(), F.col("baseline_salary") / F.col("median_household_income")).otherwise(0.0))
    
    df = df.select("person_composite_id", "vantage_date", "sal_nghb_ratio")
    # because we dont have 2024 US Census data, vantage_date does not make much sense; but still kept it
    
    return df

def add_work_patterns_features_s(df:DataFrame, start_stop_df:DataFrame) -> DataFrame:
    # Read work location table
    work_loc_mapping = spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
    work_loc_mapping = work_loc_mapping.select("clnt_obj_id", "work_loc_cd", "db_schema", "pstl_cd", "state_prov_cd", "state_prov_nm")

    # Remove records from work location table which have multiple state values
    work_loc_counts = (
        work_loc_mapping
        .groupBy("db_schema", "clnt_obj_id", "work_loc_cd")
        .count())
    work_loc_counts = work_loc_counts.filter(F.col("count")==1)
    work_loc_mapping = work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"])

    # Read State-to-region mapping
    state_to_region_mapping = spark.createDataFrame(pd.read_csv('US_State_to_Region_Mapping.csv', encoding='ISO-8859-1'))

    # Change State name to uppercase in work_loc_mapping and state-to-region mapping
    work_loc_mapping = work_loc_mapping.withColumn("state_upper", F.upper(F.col("state_prov_nm")))
    state_to_region_mapping = state_to_region_mapping.withColumn("state_upper", F.upper(F.col("State")))

    # Join work_loc_mapping and state-to-region mapping to get Region
    df_region = work_loc_mapping.join(state_to_region_mapping, on = 'state_upper', how='left')

    # Read external data having region-wise % CPI change for 2023 and 2024
    df_cpi = pd.read_csv('US_CPI_Change_By_Regions.csv', encoding='ISO-8859-1')
    df_cpi = df_cpi.rename(columns={'1_Jan_23': 'CPI_2023_', '1_Jan_24':'CPI_2024_' } )

    #  Join df_region to external CPI data on region
    df_wcpi = df_region.join(spark.createDataFrame(df_cpi), on=["Region"], how = "left")

    # Extract db_schema as the first element from person_composite_id separated by "_"
    df = df.withColumn(
        "db_schema" ,
        F.concat(F.split(F.col("person_composite_id"), "_")[0], F.lit("_util"))
    )

    df = df.withColumn(
        "clnt_obj_id" ,
        F.split(F.col("person_composite_id"), "_")[2]
    )

    # Join df data with df_wcpi on db_schema, client_id and work location code
    df_all = df.join(df_wcpi, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how = "left")

    # Calculate feature -> salary growth rate to CPI growth rate
    # df_all = df_all.withColumn("salary_growth_rate12m_to_cpi_rate"
    #                           , F.when(F.col("CPI_2023").isNotNull()
    #                                  , F.col("salary_growth_rate_12m") / F.col("CPI_2023")).otherwise(0.0))

    # Updated logic
    df_all = df_all.withColumn(
        "CPI",
        F.when(F.col("dataset_split").isin(["train", "val"]), F.col("CPI_2023_"))
        .otherwise(F.col("CPI_2024_"))
    )
    df_all = df_all.withColumn("sal_cpi_rt_ratio_1yr", F.when(F.col("CPI").isNotNull(), F.col("salary_growth_rate_12m") / F.col("CPI")).otherwise(0.0))

    # Keep only required columns
    df_wselected_cols = df_all.select("person_composite_id", "vantage_date", "sal_cpi_rt_ratio_1yr")

    return df_wselected_cols


def create_job_change_features_s(df: DataFrame) -> DataFrame:
    # Read event fact table
    event_dim = spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_event")
    event_fact = spark.table("us_east_1_prd_ds_blue_raw.dwh_t_fact_work_event")
    
    event_dim = event_dim.select(["clnt_obj_id", "db_schema", "event_cd", "event_dsc", "event_rsn_cd", "event_rsn_dsc"])
    
    # join the two tables
    event_fact_dim = event_fact.join(event_dim, on = ['clnt_obj_id', 'db_schema', 'event_cd', 'event_rsn_cd'], how = 'left')
    
    # create joining keys in modeling data
    df = df.withColumn(
        "db_schema" ,
        F.concat(F.split(F.col("person_composite_id"), "_")[0], F.lit("_util"))
    )
    
    df = df.withColumn(
        "clnt_obj_id" ,
        F.split(F.col("person_composite_id"), "_")[2]
    )
    
    df = df.withColumn(
        "pers_obj_id" ,
        F.split(F.col("person_composite_id"), "_")[3]
    )
    
    # Join event master with modeling df
    df_temp = df.join(event_fact_dim, on = ['clnt_obj_id', 'db_schema', 'pers_obj_id'], how = 'left')
    
    # Filter for events that happened 24 months before or on vantage date
    df_24m = df_temp.filter((F.col("event_eff_dt") >= F.F.add_months(F.col("vantage_date"), -24)) & (F.col("event_eff_dt") <= F.col("vantage_date")))
    
    # create 1/0 flag to indicate if job changed
    job_change_list = ['JTC', 'PJC', 'POS', 'JRC']
    df_24m = df_24m.withColumn(
        "job_change_flag",
        F.when(F.col("event_cd").isin(job_change_list), 1).otherwise(0)
    )
    
    # Take max of flag at person composite id X vantage date
    df_job_change_in_last_24m = df_24m.groupBy("person_composite_id", "vantage_date").agg(F.max("job_change_flag").alias("job_chng_2yr_ind"))
    
    # Count of distinct event_eff_dt where event_cd is in job_change_list per person_composite_id X vantage date
    df_number_of_job_changes_in_last_24m = (
        df_24m.filter(F.col("event_cd").isin(job_change_list))
        .groupBy("person_composite_id", "vantage_date")
        .agg(F.countDistinct("event_eff_dt").alias("num_job_chng_2yr"))
    )
    
    event_reason_cd_list = ['FTPT', 'FT2PT']
    df_24m = df_24m.withColumn(
        "job_change_flag_due_toftpt",
        F.when(F.col("event_cd").isin(job_change_list) & F.col("event_rsn_cd").isin(event_reason_cd_list), 1).otherwise(0)
    )
    
    # Take max of flag at person composite id X vantage date
    df_job_change_ft_to_pt = df_24m.groupBy("person_composite_id", "vantage_date").agg(F.max("job_change_flag_due_toftpt").alias("job_chng_fulltopart_ind"))
    
    non_exempt_to_exempt = ['NEX', 'NXX']
    df_24m = df_24m.withColumn(
        "job_change_flag_due_to_non_exempt_to_exempt",
        F.when(F.col("event_cd").isin(job_change_list) & F.col("event_rsn_cd").isin(non_exempt_to_exempt), 1).otherwise(0)
    )
    
    # Take max of flag at person composite id X vantage date
    df_job_change_nonexe_to_exe = df_24m.groupBy("person_composite_id", "vantage_date").agg(F.max("job_change_flag_due_to_non_exempt_to_exempt").alias("job_chng_nexmptoexmp_ind"))
    
    # Join back features to modeling data
    df = df.join(df_job_change_in_last_24m, on = ['person_composite_id', 'vantage_date'], how = 'left')
    df = df.join(df_number_of_job_changes_in_last_24m, on = ['person_composite_id', 'vantage_date'], how = 'left')
    df = df.join(df_job_change_ft_to_pt, on = ['person_composite_id', 'vantage_date'], how = 'left')
    df = df.join(df_job_change_nonexe_to_exe, on = ['person_composite_id', 'vantage_date'], how = 'left')
    
    df = df.select('person_composite_id', 'vantage_date', 'job_chng_2yr_ind', 'num_job_chng_2yr', 'job_chng_fulltopart_ind', 'job_chng_nexmptoexmp_ind')
    
    return df


def feature_generate_master():
    """
    Master function to generate and join all feature engineering outputs for employee turnover modeling.
    Reads the base modeling data, applies multiple feature engineering functions (promotion, work patterns, tenure, manager, external, job change, etc.),
    and returns a Spark DataFrame with all features joined.
    """
    
    # Import parquet file
    modeling_master_data_parquet = "employee_features_comprehensive.parquet"
    DEFAULT_CATALOG_NAME = "onedata_us_east_1_shared_prod"
    DEFAULT_SCHEMA_NAME = "datacloud_raw_oneai_turnoverprobability_prod"
    DEFAULT_VOLUME_NAME = "datacloud_raw_oneai_turnoverprobability_prod-volume"
    
    # modeling_df = read_from_unity_catalog(
    #     filename=modeling_master_data_parquet,
    #     spark=spark,
    #     catalog=DEFAULT_CATALOG_NAME,
    #     schema=DEFAULT_SCHEMA_NAME,
    #     volume=DEFAULT_VOLUME_NAME
    # )
    
    modeling_df = spark.read.parquet("dbfs:/Volumes/onedata_us_east_1_shared_prod/datacloud_raw_oneai_turnoverprobability_prod/datacloud_raw_oneai_turnoverprobability_prod-volume/employee_features_comprehensive")
    # drop the "promotion_velocity", "time_since_last_promotion", "salary_growth_rate12m_to_cpi_rate" features as the logic is not correct
    # these features will be created again in the below function call
    modeling_df = modeling_df.drop("promotion_velocity", "time_since_last_promotion", "salary_growth_rate12m_to_cpi_rate")
    
    modeling_df = modeling_df.withColumn(
        "age_at_vantage",
        F.when(F.col("age_at_vantage") < 0, F.lit(None)).otherwise(F.col("age_at_vantage"))
    )
    # Import start stop format
    start_stop_compressed = spark.table("onedata_us_east_1_shared_prod.datacloud_raw_oneai_turnoverprobability_prod.zenon_start_stop_compressed")
    
    # # Add key joining fields to start stop format
    start_stop_compressed = start_stop_compressed.withColumn(
        "db_schema" ,
        F.concat(F.split(F.col("person_composite_id"), "_")[0], F.lit("_util"))
    )
    
    start_stop_compressed = start_stop_compressed.withColumn(
        "clnt_obj_id" ,
        F.split(F.col("person_composite_id"), "_")[2]
    )
    
    base_data = (modeling_df
                 .filter(F.col("work_loc_cntry_cd").isin(["USA", "CAN"]))
                 # .filter(~F.col("mngr_pers_obj_id").isin(["UNKNOWN", "ONE", "", "0"]))
                 # .filter(F.col("dataset_split").isin(["train", "val"]))
                 )
    
    print("Total records", base_data.count(), "...")
    print("Unique records at person composite id", base_data.select("person_composite_id").distinct().count())
    
    # load event data
    df_event = spark.table("us_east_1_prd_ds_blue_raw.dwh_t_fact_work_event")
    df_event = df_event.withColumn(
        "person_composite_id",
        F.concat(F.col("db_schema"), F.lit("_"), F.col("clnt_obj_id"), F.lit("_"), F.col("pers_obj_id"))
    )
    df_event = df_event.filter(F.col("event_cd").isin(["PRO", "DEM", "XFR"]))
    df_event = df_event.select("person_composite_id", "event_cd", "event_rsn_cd", "event_eff_dt", "lst_promo_dt")
    
    # join event data
    df_joined = base_data.join(df_event, "person_composite_id", "left")
    print("Size of filtered data after joining event data",df_joined.count())
    
    df_feature_1 = create_promotion_demotion_count_features(df_joined)
    base_data = base_data.join(df_feature_1, ["person_composite_id", "vantage_date"], "left")
    
    df_feature_2 = create_promotion_demotion_transfer_indicators_features(df_joined)
    base_data = base_data.join(df_feature_2, ["person_composite_id", "vantage_date"], "left")
    
    df_feature_3 = create_promotion_time_interaction_features(df_joined)
    base_data = base_data.join(df_feature_3, ["person_composite_id", "vantage_date"], "left")
    
    work_df = add_work_patterns_features_s(base_data, start_stop_compressed)
    base_data = base_data.join(work_df, on = ["person_composite_id", "vantage_date"], how = 'left')
    # "num_city_chng", "num_state_chng"
    print("Total records post joining work location features", base_data.count())
    
    tenure_df = add_age_tenure_features_s(base_data)
    base_data = base_data.join(tenure_df, on = ["person_composite_id", "vantage_date"], how = 'left')
    # tenure_age_ratio
    print("Total records post joining age-tenure features", base_data.count())
    
    manager_df = add_manager_environment_features_s(base_data, start_stop_compressed)
    base_data = base_data.join(manager_df, on = ["person_composite_id", "vantage_date"], how = 'left')
    # is_manager_ind
    print("Total records post joining manager features", base_data.count())
    
    external_df1 = add_external_features_1s(base_data)
    base_data = base_data.join(external_df1, on = ["person_composite_id", "vantage_date"], how = 'left')
    # sal_cpi_rt_ratio_1yr
    print("Total records post joining external features", base_data.count())
    
    external_df2 = add_external_features_2s(base_data)
    base_data = base_data.join(external_df2, on = ["person_composite_id", "vantage_date"], how = 'left')
    # sal_nghb_ratio
    print("Total records post joining 2nd external features", base_data.count())
    
    job_change_df = create_job_change_features_s(base_data)
    base_data = base_data.join(job_change_df, on = ["person_composite_id", "vantage_date"], how = 'left')
    # 'job_chng_2yr_ind', 'num_job_chng_2yr', 'job_chng_fulltopart_ind', 'job_chng_nexmptoexmp_ind'
    print("Total records post joining job change features", base_data.count())
    
    return base_data

import pyspark.sql.functions as F
from pyspark.sql.functions import when, lit
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import *
from typing import Tuple
import logging

from top_etl.feature_engg.constants import (
    SAMPLE_IDS,
    FINAL_COLS,
    ONEDATA_CATALOG_NM,
    TOP_SCHEMA_NM,
    EVENT_CODE_MAPPINGS,
    FEATURE_VALIDATION_RULES,
    EXTERNAL_DATA_CONFIG,
)

from top_etl.feature_engg.external_features import ExternalFeatures

from top_etl.common.utils import save_to_table

from top_etl.common.uc_volume_io import write_to_unity_catalog
from top_etl.feature_engg.normalize_features import FeatureNormalizer

logger = logging.getLogger(__name__)


class FeatureGenerator:
    def __init__(
        self,
        spark: SparkSession,
        use_select_cols: bool = True,
        use_sample: bool = False,
        save_to_table: bool = True,
        normalize: bool = True,
        fail_on_feature_error: bool = False,
    ):
        self.spark = spark
        self.sample_list = SAMPLE_IDS if use_sample else None
        self.feature_cols = FINAL_COLS if use_select_cols else None
        self.save_to_table = save_to_table
        self.normalize = normalize
        self.fail_on_feature_error = fail_on_feature_error

    def get_cleaned_data_with_vantage_date(self):
        cleaned_data = self.spark.table(
            "onedata_us_east_1_shared_prod.datacloud_raw_oneai_turnoverprobability_prod.zenon_cleaned_data"
        )
        split_assignments = self.spark.table(
            "onedata_us_east_1_shared_prod.datacloud_raw_oneai_turnoverprobability_prod.zenon_split_assignments"
        )

        return cleaned_data.join(split_assignments, on="person_composite_id", how="inner").withColumn(
            "vantage_date",
            when(split_assignments["dataset_split"].isin(["train", "val"]), lit("2023-01-01"))
            .when(split_assignments["dataset_split"] == "oot", lit("2024-01-01"))
            .otherwise(lit(None)),
        )

    def load_source_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # Load base datasets
        employee_level = self.spark.table(f"{ONEDATA_CATALOG_NM}.{TOP_SCHEMA_NM}.zenon_employee_level")
        start_stop_compressed = self.spark.table(f"{ONEDATA_CATALOG_NM}.{TOP_SCHEMA_NM}.zenon_start_stop_compressed")

        # Apply base filters
        base_data = (
            employee_level.filter(F.col("work_loc_cntry_cd").isin(["USA", "CAN"]))
            .filter(~F.col("mngr_pers_obj_id").isin(["UNKNOWN", "ONE", "", "0"]))
            # .filter(F.col("dataset_split").isin(["train", "val"]))
        )

        if self.sample_list:
            base_data = base_data.filter(F.col("person_composite_id").isin(self.sample_list))

        # Extract clnt_obj_id as the third element from person_composite_id separated by "_"
        # TODO: Remove as clnt_obj_id should come in etl output tables
        base_data = base_data.withColumn("clnt_obj_id", F.split(F.col("person_composite_id"), "_")[2])

        start_stop_compressed = start_stop_compressed.join(
            base_data.select("person_composite_id", "vantage_date"), ["person_composite_id", "vantage_date"], "inner"
        )

        cleaned_data = self.get_cleaned_data_with_vantage_date()

        return base_data, start_stop_compressed, cleaned_data

    def load_event_data(self) -> DataFrame:
        """Load work event data for event-based feature calculations"""
        try:
            event_fact = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_fact_work_event")
            event_dim = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_event")
            
            # Create composite person ID and join with dimension table
            event_data = (
                event_fact.withColumn(
                    "person_composite_id",
                    F.concat(F.col("db_schema"), F.lit("_"), F.col("clnt_obj_id"), F.lit("_"), F.col("pers_obj_id"))
                )
                .join(
                    event_dim.select(["clnt_obj_id", "db_schema", "event_cd", "event_dsc", "event_rsn_cd", "event_rsn_dsc"]),
                    on=["clnt_obj_id", "db_schema", "event_cd", "event_rsn_cd"],
                    how="left"
                )
                .select(
                    "person_composite_id", "event_cd", "event_rsn_cd", "event_eff_dt", 
                    "event_dsc", "event_rsn_dsc", "lst_promo_dt"
                )
            )
            
            logger.info(f"Successfully loaded {event_data.count():,} event records")
            return event_data
            
        except Exception as e:
            logger.error(f"Failed to load event data: {e}")
            if self.fail_on_feature_error:
                raise
            # Return empty DataFrame with required schema
            return self.spark.createDataFrame([], StructType([
                StructField("person_composite_id", StringType(), True),
                StructField("event_cd", StringType(), True),
                StructField("event_rsn_cd", StringType(), True),
                StructField("event_eff_dt", DateType(), True),
                StructField("event_dsc", StringType(), True),
                StructField("event_rsn_dsc", StringType(), True),
                StructField("lst_promo_dt", DateType(), True)
            ]))

    def create_comprehensive_features(self) -> DataFrame:
        """
        Create comprehensive feature set for employee turnover prediction
        
        Returns:
            DataFrame with comprehensive features
        """
        logger.info("Starting comprehensive feature generation...")

        try:
            base_data, start_stop_compressed, cleaned_data = self.load_source_data()
            event_data = self.load_event_data()

            # Apply all feature engineering functions with error handling
            df = self._safe_add_features(base_data, start_stop_compressed, cleaned_data, event_data)

            if self.normalize:
                logger.info("Applying feature normalization...")
                feature_normalizer = FeatureNormalizer(spark=self.spark)
                df = feature_normalizer.normalize(df)

            if self.save_to_table:
                df, table_name = self.save_feature_engineered_data(df)

            logger.info(f"Feature generation completed. Final dataset: {df.count():,} records")
            return df

        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            raise

    def _safe_add_features(self, base_data: DataFrame, start_stop_compressed: DataFrame, 
                          cleaned_data: DataFrame, event_data: DataFrame) -> DataFrame:
        """Safely add all features with error handling"""
        
        df = base_data
        feature_methods = [
            ("compensation", self.add_compensation_features, [start_stop_compressed]),
            ("career_progression", self.add_career_progression_features, [start_stop_compressed, event_data]),
            ("demographic", self.add_demographic_features, []),
            ("job_characteristics", self.add_job_characteristics_features, [cleaned_data]),
            ("manager_environment", self.add_manager_environment_features, [start_stop_compressed]),
            ("team_environment", self.add_team_environment_features, [cleaned_data]),
            ("tenure_dynamics", self.add_tenure_dynamics_features, [start_stop_compressed]),
            ("work_patterns", self.add_work_patterns_features, [start_stop_compressed]),
            ("company_factors", self.add_company_factors_features, []),
            ("temporal", self.add_temporal_features, []),
            ("promotion_events", self.add_promotion_event_features, [event_data]),
            ("job_change_events", self.add_job_change_event_features, [event_data]),
            ("external", self.add_external_features, []),
        ]
        
        for feature_name, method, args in feature_methods:
            try:
                logger.info(f"Adding {feature_name} features...")
                df = method(df, *args)
                logger.info(f"✓ Successfully added {feature_name} features")
            except Exception as e:
                logger.error(f"✗ Failed to add {feature_name} features: {e}")
                if self.fail_on_feature_error:
                    raise
                logger.warning(f"Continuing without {feature_name} features...")
        
        return df

    def add_promotion_event_features(self, df: DataFrame, event_data: DataFrame) -> DataFrame:
        """Add event-based promotion, demotion, and transfer features"""
        try:
            if event_data.count() == 0:
                logger.warning("No event data available, skipping promotion event features")
                return df
                
            # Join event data with base data
            df_with_events = df.join(event_data, on="person_composite_id", how="left")
            
            # Filter events that occurred before vantage date (temporal safety)
            df_filtered = df_with_events.filter(
                F.col("event_eff_dt").isNull() | (F.col("event_eff_dt") <= F.col("vantage_date"))
            )
            
            # Create 2-year lookback window
            two_years_ago = F.date_sub(F.col("vantage_date"), 730)
            
            # Basic promotion/demotion indicators (2-year window)
            df_indicators = (
                df_filtered.withColumn(
                    "is_promotion_2yr",
                    F.when(
                        (F.upper(F.col("event_cd")) == F.lit("PRO")) & 
                        (F.col("event_eff_dt") >= two_years_ago) &
                        (F.col("event_eff_dt") <= F.col("vantage_date")), 1
                    ).otherwise(0)
                ).withColumn(
                    "is_demotion_2yr",
                    F.when(
                        (F.upper(F.col("event_cd")) == F.lit("DEM")) & 
                        (F.col("event_eff_dt") >= two_years_ago) &
                        (F.col("event_eff_dt") <= F.col("vantage_date")), 1
                    ).otherwise(0)
                ).withColumn(
                    "is_transfer_2yr",
                    F.when(
                        (F.upper(F.col("event_cd")) == F.lit("XFR")) & 
                        (F.col("event_eff_dt") >= two_years_ago) &
                        (F.col("event_eff_dt") <= F.col("vantage_date")), 1
                    ).otherwise(0)
                )
            )
            
            # Detailed promotion categorization
            df_detailed = (
                df_indicators.withColumn(
                    "is_promotion_performance",
                    F.when(
                        (F.col("is_promotion_2yr") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["performance"]])), 1
                    ).otherwise(0)
                ).withColumn(
                    "is_promotion_title_change",
                    F.when(
                        (F.col("is_promotion_2yr") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["title_change"]])), 1
                    ).otherwise(0)
                ).withColumn(
                    "is_promotion_market_adjust",
                    F.when(
                        (F.col("is_promotion_2yr") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["market_adjustment"]])), 1
                    ).otherwise(0)
                ).withColumn(
                    "is_demotion_reorg",
                    F.when(
                        (F.col("is_demotion_2yr") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["company_reorg"]])), 1
                    ).otherwise(0)
                ).withColumn(
                    "is_demotion_performance",
                    F.when(
                        (F.col("is_demotion_2yr") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["performance_issues"]])), 1
                    ).otherwise(0)
                )
            )
            
            # Calculate time since last events
            promotion_dates = (
                df_detailed.filter(F.col("is_promotion_2yr") == 1)
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.max("event_eff_dt").alias("last_promotion_date"))
            )
            
            transfer_dates = (
                df_detailed.filter(F.col("is_transfer_2yr") == 1)
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.max("event_eff_dt").alias("last_transfer_date"))
            )
            
            # Calculate promotion velocity (average days between promotions)
            promotion_velocity = (
                df_detailed.filter(F.col("is_promotion_2yr") == 1)
                .withColumn(
                    "prev_promotion_date",
                    F.lag("event_eff_dt").over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy("event_eff_dt")
                    )
                )
                .withColumn(
                    "days_between_promotions",
                    F.when(F.col("prev_promotion_date").isNotNull(),
                           F.datediff(F.col("event_eff_dt"), F.col("prev_promotion_date")))
                )
                .filter(F.col("days_between_promotions").isNotNull())
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.avg("days_between_promotions").alias("promot_veloc"))
            )
            
            # Aggregate features by person
            aggregated_features = (
                df_detailed.groupBy("person_composite_id", "vantage_date")
                .agg(
                    # Basic indicators
                    F.max("is_promotion_2yr").alias("promot_2yr_ind"),
                    F.max("is_demotion_2yr").alias("demot_2yr_ind"),
                    F.max("is_transfer_2yr").alias("transfer_2yr_ind"),
                    
                    # Counts
                    F.sum("is_promotion_2yr").alias("num_promot_2yr"),
                    F.sum("is_demotion_2yr").alias("num_demot_2yr"),
                    F.sum("is_transfer_2yr").alias("num_transfer_2yr"),
                    
                    # Detailed categorization
                    F.max("is_promotion_performance").alias("promot_2yr_perf_ind"),
                    F.max("is_promotion_title_change").alias("promot_2yr_titlechng_ind"),
                    F.max("is_promotion_market_adjust").alias("promot_2yr_mktadjst_ind"),
                    F.max("is_demotion_reorg").alias("demot_2yr_compreorg_ind"),
                    F.max("is_demotion_performance").alias("demot_2yr_perfissue_ind"),
                )
            )
            
            # Calculate days since events
            df_with_time_features = (
                df.join(aggregated_features, ["person_composite_id", "vantage_date"], "left")
                .join(promotion_dates, ["person_composite_id", "vantage_date"], "left")
                .join(transfer_dates, ["person_composite_id", "vantage_date"], "left")
                .join(promotion_velocity, ["person_composite_id", "vantage_date"], "left")
                .withColumn(
                    "days_since_promot",
                    F.when(F.col("last_promotion_date").isNotNull(),
                           F.datediff(F.col("vantage_date"), F.col("last_promotion_date")))
                )
                .withColumn(
                    "days_since_transfer",
                    F.when(F.col("last_transfer_date").isNotNull(),
                           F.datediff(F.col("vantage_date"), F.col("last_transfer_date")))
                )
                .drop("last_promotion_date", "last_transfer_date")
            )
            
            # Fill null values with appropriate defaults
            null_fills = {
                "promot_2yr_ind": 0, "demot_2yr_ind": 0, "transfer_2yr_ind": 0,
                "num_promot_2yr": 0, "num_demot_2yr": 0, "num_transfer_2yr": 0,
                "promot_2yr_perf_ind": 0, "promot_2yr_titlechng_ind": 0, "promot_2yr_mktadjst_ind": 0,
                "demot_2yr_compreorg_ind": 0, "demot_2yr_perfissue_ind": 0,
                "promot_veloc": 0.0, "days_since_promot": 9999, "days_since_transfer": 9999
            }
            
            for col_name, fill_value in null_fills.items():
                df_with_time_features = df_with_time_features.withColumn(
                    col_name, F.coalesce(F.col(col_name), F.lit(fill_value))
                )
            
            logger.info("Successfully added promotion event features")
            return df_with_time_features
            
        except Exception as e:
            logger.error(f"Failed to add promotion event features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without promotion event features...")
            return df

    def add_job_change_event_features(self, df: DataFrame, event_data: DataFrame) -> DataFrame:
        """Add job change event features based on event codes"""
        try:
            if event_data.count() == 0:
                logger.warning("No event data available, skipping job change event features")
                return df
                
            # Join event data
            df_with_events = df.join(event_data, on="person_composite_id", how="left")
            
            # Filter for job change events within 2-year window
            two_years_ago = F.date_sub(F.col("vantage_date"), 730)
            job_change_codes = EVENT_CODE_MAPPINGS["job_change"]
            
            df_job_events = (
                df_with_events.filter(
                    (F.col("event_eff_dt").isNull()) | 
                    ((F.col("event_eff_dt") >= two_years_ago) & (F.col("event_eff_dt") <= F.col("vantage_date")))
                )
                .withColumn(
                    "is_job_change",
                    F.when(
                        F.upper(F.col("event_cd")).isin([x.upper() for x in job_change_codes]), 1
                    ).otherwise(0)
                )
                .withColumn(
                    "is_ft_to_pt_change",
                    F.when(
                        (F.col("is_job_change") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["ft_to_pt"]])), 1
                    ).otherwise(0)
                )
                .withColumn(
                    "is_nonexempt_to_exempt_change",
                    F.when(
                        (F.col("is_job_change") == 1) &
                        (F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in EVENT_CODE_MAPPINGS["nonexempt_to_exempt"]])), 1
                    ).otherwise(0)
                )
            )
            
            # Aggregate job change features
            job_change_features = (
                df_job_events.groupBy("person_composite_id", "vantage_date")
                .agg(
                    F.max("is_job_change").alias("job_chng_2yr_ind"),
                    F.sum("is_job_change").alias("num_job_chng_2yr"),
                    F.max("is_ft_to_pt_change").alias("job_chng_fulltopart_ind"),
                    F.max("is_nonexempt_to_exempt_change").alias("job_chng_nexmptoexmp_ind"),
                )
            )
            
            # Join back to main dataset
            result_df = (
                df.join(job_change_features, ["person_composite_id", "vantage_date"], "left")
                .fillna({
                    "job_chng_2yr_ind": 0,
                    "num_job_chng_2yr": 0,
                    "job_chng_fulltopart_ind": 0,
                    "job_chng_nexmptoexmp_ind": 0
                })
            )
            
            logger.info("Successfully added job change event features")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to add job change event features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without job change event features...")
            return df

    def add_compensation_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add compensation-related features (Flag = 1 priority)
        """
        try:
            # Get historical salary data with temporal filtering
            salary_history = (
                start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .filter(F.col("annl_cmpn_amt").isNotNull())
                .select("person_composite_id", "rec_eff_start_dt_mod", "annl_cmpn_amt", "mnthly_cmpn_amt", "vantage_date")
                .distinct()
            )

            # Calculate salary growth rate over 12 months
            salary_12m_ago = F.date_sub(F.col("vantage_date"), 365)

            salary_growth = (
                salary_history.withColumn("days_diff", F.datediff(F.col("vantage_date"), F.col("rec_eff_start_dt_mod")))
                .filter(F.col("days_diff") >= 0)
                .withColumn(
                    "rn",
                    F.row_number().over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy(F.col("days_diff"))
                    ),
                )
                .filter(F.col("rn") == 1)
                .select("person_composite_id", "vantage_date", F.col("annl_cmpn_amt").alias("current_salary"))
            )

            # Get salary from 12 months ago
            salary_12m = (
                salary_history.filter(F.col("rec_eff_start_dt_mod") <= salary_12m_ago)
                .withColumn("days_diff", F.datediff(salary_12m_ago, F.col("rec_eff_start_dt_mod")))
                .filter(F.col("days_diff") >= 0)
                .withColumn(
                    "rn",
                    F.row_number().over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy(F.col("days_diff"))
                    ),
                )
                .filter(F.col("rn") == 1)
                .select("person_composite_id", "vantage_date", F.col("annl_cmpn_amt").alias("salary_12m_ago"))
            )

            # Calculate salary growth rate
            salary_metrics = (
                salary_growth.join(salary_12m, ["person_composite_id", "vantage_date"], "left")
                .withColumn(
                    "salary_growth_rate_12m",
                    F.when(
                        F.col("salary_12m_ago").isNotNull(),
                        (F.col("current_salary") - F.col("salary_12m_ago")) / F.col("salary_12m_ago"),
                    ).otherwise(0.0),
                )
                .select("person_composite_id", "vantage_date", "salary_growth_rate_12m")
            )

            # Company-level salary percentiles
            company_salary_percentiles = (
                df.filter(F.col("baseline_salary").isNotNull())
                .select("person_composite_id", "clnt_obj_id", "vantage_date", "baseline_salary")
                .withColumn(
                    "compensation_percentile_company",
                    F.percent_rank().over(Window.partitionBy("clnt_obj_id", "vantage_date").orderBy("baseline_salary")),
                )
            )

            # Industry-level salary percentiles
            industry_salary_percentiles = (
                df.filter(F.col("baseline_salary").isNotNull())
                .filter(F.col("naics_cd").isNotNull())
                .select("person_composite_id", "naics_cd", "baseline_salary", "vantage_date")
                .withColumn(
                    "compensation_percentile_industry",
                    F.percent_rank().over(Window.partitionBy("naics_cd", "vantage_date").orderBy("baseline_salary")),
                )
            )

            # Get distinct salary records only
            dist_sal_hist = (
                salary_history
                .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "annl_cmpn_amt")
                .withColumn(
                    "p_sal",
                    F.lag("annl_cmpn_amt").over(
                        Window.partitionBy("person_composite_id", "vantage_date")
                        .orderBy("rec_eff_start_dt_mod")
                    ),
                )
                # keep only records where salary actually changed (or first record)
                .filter(
                    (F.col("p_sal").isNull()) |
                    (F.col("annl_cmpn_amt") != F.col("p_sal"))
                )
                .drop("p_sal")
            )

            # Compensation change frequency - captures instability differently
            comp_chng_freq = (
                dist_sal_hist
                .groupBy("person_composite_id", "vantage_date")
                .agg(
                    (F.count("*") - 1).alias('comp_chng_cnt'),
                    F.max("rec_eff_start_dt_mod").alias("last_chng_dt"),
                    F.min("rec_eff_start_dt_mod").alias("first_sal_dt")
                )
                .withColumn(
                    "comp_chang_freq_per_year",
                    F.when(
                        F.datediff(F.col("last_chng_dt"), F.col("first_sal_dt")) > 365,
                        F.col("comp_chng_cnt") * 365.0 / F.datediff(F.col("last_chng_dt"), F.col("first_sal_dt"))
                    ).otherwise(F.col("comp_chng_cnt").cast("double"))  # less than 1 year tenure
                )
                .select("person_composite_id", "vantage_date", "comp_chang_freq_per_year")
            )

            # Calculate compensation volatility
            compensation_volatility = (
                salary_history.groupBy("person_composite_id", "vantage_date")
                .agg(F.stddev("annl_cmpn_amt").alias("compensation_volatility"))
                .fillna(0.0, subset=["compensation_volatility"])
            )

            # Define the start and end dates for the last quarter
            last_quarter_start = F.date_sub(F.col("vantage_date"), 90)
            # Filter for last quarter and calculate average salary per person
            salary_last_quarter = (
                salary_history.filter(
                    (F.col("rec_eff_start_dt_mod") >= last_quarter_start)
                    & (F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                )
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.avg("mnthly_cmpn_amt").alias("avg_salary_last_quarter"))
            )

            # Calculate Gini entropy for hr_cmpn_freq_cd per person_composite_id
            pay_freq_counts = (
                start_stop_df.filter(F.col("hr_cmpn_freq_cd").isNotNull())
                .filter(
                    (F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                    & (F.col("rec_eff_start_dt_mod") >= F.date_sub(F.col("vantage_date"), 365))
                )
                .groupBy("person_composite_id", "vantage_date", "hr_cmpn_freq_cd")
                .count()
            )

            total_counts = pay_freq_counts.groupBy("person_composite_id", "vantage_date").agg(F.sum("count").alias("total"))

            pay_freq_probs = pay_freq_counts.join(total_counts, on=["person_composite_id", "vantage_date"]).withColumn(
                "prob", F.col("count") / F.col("total")
            )

            pay_freq_consistency = (
                pay_freq_probs
                .groupBy("person_composite_id", "vantage_date")
                .agg((1 - F.sum(F.col("prob") ** 2)).alias("pay_freq_diversity_score"))
                .withColumn(
                    "pay_freq_consistency_score",
                    F.when(F.col("pay_freq_diversity_score") <= 0.2, 1.0)  # very consistent
                    .when(F.col("pay_freq_diversity_score") <= 0.5, 0.7)  # moderate consistent
                    .when(F.col("pay_freq_diversity_score") <= 0.8, 0.4)  # inconsistent
                    .otherwise(0.1)  # very inconsistent
                )
                .select("person_composite_id", "vantage_date", "pay_freq_consistency_score")
            )

            # Pay grade stagnation (months since last increase)
            pay_grade_stagnation = (
                dist_sal_hist
                .withColumn(
                    "prev_salary",
                    F.lag("annl_cmpn_amt").over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                    ),
                )
                .withColumn(
                    "salary_change",
                    F.when(
                        F.col("prev_salary").isNotNull(),
                        F.col("annl_cmpn_amt") - F.col("prev_salary")
                    ).otherwise(0.0)
                )
                .filter(F.col("salary_change") > 0)
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.max("rec_eff_start_dt_mod").alias("last_increase_date"))
                .withColumn("pay_grade_stagnation_months", F.months_between(F.col("vantage_date"), F.col("last_increase_date")))
                .select("person_composite_id", "vantage_date", "pay_grade_stagnation_months")
            )

            # Join all compensation features
            df = (
                df.join(salary_metrics, ["person_composite_id", "vantage_date"], "left")
                .join(
                    company_salary_percentiles.select(
                        "person_composite_id", "vantage_date", "compensation_percentile_company"
                    ),
                    ["person_composite_id", "vantage_date"],
                    "left",
                )
                .join(
                    industry_salary_percentiles.select(
                        "person_composite_id", "vantage_date", "compensation_percentile_industry"
                    ),
                    ["person_composite_id", "vantage_date"],
                    "left",
                )
                .join(compensation_volatility, ["person_composite_id", "vantage_date"], "left")
                .join(comp_chng_freq, ["person_composite_id", "vantage_date"], "left")
                .join(pay_grade_stagnation, ["person_composite_id", "vantage_date"], "left")
                .join(salary_last_quarter, on=["person_composite_id", "vantage_date"], how="left")
                .join(pay_freq_consistency, on=["person_composite_id", "vantage_date"], how="left")
                .fillna(0.0, subset=["salary_growth_rate_12m", "compensation_volatility", "pay_grade_stagnation_months"])
            )

            # Add total compensation growth (assuming bonus data is available)
            df = df.withColumn("total_compensation_growth", F.col("salary_growth_rate_12m"))  # Placeholder

            # Add pay frequency preference alignment
            df = df.withColumn(
                "pay_frequency_preference",
                F.when(F.col("pay_rt_type_cd") == "H", 1).when(F.col("pay_rt_type_cd") == "S", 2).otherwise(0),
            )

            logger.info("Successfully added compensation features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add compensation features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without compensation features...")
            return df

    def add_career_progression_features(self, df: DataFrame, start_stop_df: DataFrame, event_data: DataFrame) -> DataFrame:
        """
        Add career progression features (Flag = 1 priority)
        FIXED: Now uses event-based promotion detection instead of manager level comparison
        """
        try:
            # Get promotion events with temporal filtering
            promotion_events = (
                event_data.filter(F.upper(F.col("event_cd")) == F.lit("PRO"))
                .filter(F.col("event_eff_dt") <= F.col("vantage_date"))
                .select("person_composite_id", "event_eff_dt", "vantage_date")
            )
            
            if promotion_events.count() == 0:
                logger.warning("No promotion events found, using fallback logic")
                # Fallback to job level history if no events available
                job_history = (
                    start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                    .filter(F.col("mngr_lvl_cd").isNotNull())
                    .select("person_composite_id", "rec_eff_start_dt_mod", "mngr_lvl_cd", "job_cd", "vantage_date")
                    .distinct()
                )
                
                promotion_events = (
                    job_history.withColumn(
                        "prev_level",
                        F.lag("mngr_lvl_cd").over(
                            Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                        ),
                    )
                    .withColumn("is_promotion", F.when(F.col("mngr_lvl_cd") > F.col("prev_level"), 1).otherwise(0))
                    .filter(F.col("is_promotion") == 1)
                    .select("person_composite_id", F.col("rec_eff_start_dt_mod").alias("event_eff_dt"), "vantage_date")
                )

            # Time since last promotion
            time_since_last_promotion = (
                promotion_events.groupBy("person_composite_id", "vantage_date")
                .agg(F.max("event_eff_dt").alias("last_promotion_date"))
                .withColumn("time_since_last_promotion", F.datediff(F.col("vantage_date"), F.col("last_promotion_date")))
                .select("person_composite_id", "vantage_date", "time_since_last_promotion")
            )

            # Promotion velocity
            promotion_velocity = (
                promotion_events.withColumn(
                    "prev_promotion_date",
                    F.lag("event_eff_dt").over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy("event_eff_dt")
                    )
                )
                .withColumn(
                    "days_between_promotions",
                    F.when(F.col("prev_promotion_date").isNotNull(),
                           F.datediff(F.col("event_eff_dt"), F.col("prev_promotion_date")))
                )
                .filter(F.col("days_between_promotions").isNotNull())
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.avg("days_between_promotions").alias("promotion_velocity"))
            )

            # Join career progression features
            df = (
                df.join(time_since_last_promotion, ["person_composite_id", "vantage_date"], "left")
                .join(promotion_velocity, ["person_composite_id", "vantage_date"], "left")
                .fillna({
                    "time_since_last_promotion": 9999.0,  # Large number for never promoted
                    "promotion_velocity": 0.0
                })
            )

            logger.info("Successfully added career progression features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add career progression features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without career progression features...")
            return df

    def add_demographic_features(self, df: DataFrame) -> DataFrame:
        """
        Add demographic features (Flag = 1 priority)
        """
        try:
            vantage_dt = F.to_date(F.col("vantage_date"))

            # Age at vantage date
            df = df.withColumn(
                "age_at_vantage",
                F.when(F.col("birth_dt").isNotNull(),
                       F.months_between(vantage_dt, F.col("birth_dt")) / 12).otherwise(F.lit(None))
            )

            # Career stage based on age
            df = df.withColumn(
                "career_stage",
                F.when(F.col("age_at_vantage") < 30, "Early")
                .when(F.col("age_at_vantage") < 45, "Mid")
                .when(F.col("age_at_vantage") < 60, "Late")
                .otherwise("Senior"),
            )

            # Retirement eligibility (assuming 65 as retirement age)
            df = df.withColumn(
                "retirement_eligibility_years",
                F.when(F.col("age_at_vantage").isNotNull(), F.greatest(F.lit(0), 65 - F.col("age_at_vantage")))
                .otherwise(F.lit(None)),
            )

            df = df.withColumn("birth_year", F.year("birth_dt"))
            # Generation cohort
            df = df.withColumn(
                "generation_cohort",
                F.when(F.col("birth_year") >= 1997, "Gen Z")
                .when(F.col("birth_year") >= 1981, "Millennial")
                .when(F.col("birth_year") >= 1965, "Gen X")
                .when(F.col("birth_year") >= 1946, "Baby Boomer")
                .otherwise("Silent Generation"),
            )

            # tenure age interaction
            df = df.withColumn("tenure_at_vantage_years", F.col("tenure_at_vantage_days") / 365.25)
            df = df.withColumn("tenure_age_ratio", 
                              F.when(
                                  (F.col("age_at_vantage").isNotNull()) & (F.col("age_at_vantage") > 0),
                                  F.col("tenure_at_vantage_years") / F.col("age_at_vantage")).otherwise(0.0)
                              )

            # Calculate working age
            df = df.withColumn("working_age",
                              F.when(
                                  F.col("age_at_vantage").isNotNull(),
                                  F.greatest(F.lit(0), F.col("age_at_vantage") - 25)).otherwise(0.0)
                              )

            # Define career stage
            df = df.withColumn(
                "career_joiner_stage",
                F.when((F.col("working_age") <= 5) & (F.col("tenure_at_vantage_years") <= 2), "early_career")
                .when(
                    (F.col("working_age") > 5) & (F.col("working_age") <= 15) & (F.col("tenure_at_vantage_years") <= 5),
                    "mid_career_joiner",
                )
                .when((F.col("working_age") > 15) & (F.col("tenure_at_vantage_years") <= 5), "late_career_joiner")
                .when(F.col("tenure_at_vantage_years") > 5, "experienced_loyal")
                .otherwise("other"),
            )

            df = df.withColumn(
                "tenure_age_risk",
                F.when((F.col("age_at_vantage") <= 30) & (F.col("tenure_at_vantage_years") >= 5), "burnout_risk")
                .when((F.col("age_at_vantage") >= 50) & (F.col("tenure_at_vantage_years") <= 2), "retirement_on_disengagement_risk")
                .otherwise("normal"),
            )

            logger.info("Successfully added demographic features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add demographic features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without demographic features...")
            return df

    def add_job_characteristics_features(self, df: DataFrame, cleaned_data: DataFrame) -> DataFrame:
        """
        Add job characteristics features (Flag = 1 priority)
        """
        try:
            # Job level from manager level code
            df = df.withColumn("job_level", F.col("mngr_lvl_cd"))

            # Add role complexity score based on manager level
            df = df.withColumn(
                "role_complexity_score",
                F.when(F.col("mngr_lvl_cd") >= 5, 3)  # High complexity
                .when(F.col("mngr_lvl_cd") >= 3, 2)  # Medium complexity
                .otherwise(1),  # Low complexity
            )

            historical_start_date = F.date_sub(F.col("vantage_date"), 365 * 2)  # 2 YEARS before vantage date.

            historical_job_family_turnover = (
                cleaned_data.filter(F.col("termination_date").isNotNull())
                .filter(F.col("termination_date").between(historical_start_date, F.col("vantage_date")))
                .groupBy("job_cd", "vantage_date")
                .agg(F.countDistinct("person_composite_id").alias("turnover_cnt"))
            )

            # job family turnover rate
            # Calculate total employees and turnover count per job_cd
            job_family_counts = df.groupBy("job_cd", "vantage_date").agg(
                F.countDistinct("person_composite_id").alias("active_cnt"),
            )

            # Calculate turnover rate
            job_family_counts = (
                job_family_counts.join(historical_job_family_turnover, ["job_cd", "vantage_date"], "left")
                .fillna(0, subset=["turnover_cnt"])
                .withColumn("job_family_turnover_rate", F.col("turnover_cnt") / (F.col("turnover_cnt") + F.col("active_cnt")))
                .select("job_cd", "job_family_turnover_rate", "vantage_date")
            )

            # Join turnover rate back to df
            df = df.join(job_family_counts, on=["job_cd", "vantage_date"], how="left")

            df = df.withColumn(
                "job_stability_ind",
                F.when(
                    (F.col("full_tm_part_tm_cd")=="F") &
                    (F.col("reg_temp_cd")=="R") &
                    (F.col("flsa_stus_cd").isin(["E", "N"])), 1  #high stability
                )
                .when(F.col("reg_temp_cd")=='T', 0)  # temporary
                .otherwise(0.5)  #medium stability
            )
            
            logger.info("Successfully added job characteristics features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add job characteristics features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without job characteristics features...")
            return df

    def add_manager_environment_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add manager environment features (Flag = 1 priority)
        """
        try:
            # Count unique manager changes per person_composite_id and vantage_data
            manager_changes_df = (
                start_stop_df.filter(~F.col("mngr_pers_obj_id").isin(["UNKNOWN", "ONE", "", "0"]))
                .filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.countDistinct("mngr_pers_obj_id").alias("manager_changes_count"))
            )

            # Join manager_changes_count to df
            df = df.join(manager_changes_df, ["person_composite_id", "vantage_date"], "left").fillna(
                {"manager_changes_count": 0}
            )

            # Time with current manager
            current_manager_start = (
                start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .filter(F.col("mngr_pers_obj_id").isNotNull())
                .withColumn(
                    "rn", 
                    F.row_number().over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy(F.desc("rec_eff_start_dt_mod"))
                    ),
                )
                .filter(F.col("rn")==1)
                .select("person_composite_id", "vantage_date", "mngr_pers_obj_id", "rec_eff_start_dt_mod")
            )
                
            # find when current manager relationship started
            manager_relationship_start = (
                start_stop_df.join(
                    current_manager_start.select(
                        "person_composite_id", "vantage_date", F.col("mngr_pers_obj_id").alias("current_manager")
                    ),
                    ["person_composite_id", "vantage_date"],
                )
                .filter(F.col("mngr_pers_obj_id") == F.col("current_manager"))
                .filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.min("rec_eff_start_dt_mod").alias("manager_start_date"))
                .withColumn("time_with_current_manager", F.datediff(F.col("vantage_date"), F.col("manager_start_date")))
                .select("person_composite_id", "vantage_date", "time_with_current_manager")
            )

            # manager's tenure
            person_composite_id_split = F.split(F.col("person_composite_id"), "_")
            df = df.withColumn("employee_id", F.element_at(person_composite_id_split, -1))
            df = df.withColumn("clnt_obj_id", F.element_at(person_composite_id_split, -2))
            # Extract db_schema by concatenating all elements from index 1 up to and including -2 (excluding the last element)
            db_schema_expr = F.expr(
                "concat_ws('_', slice(split(person_composite_id, '_'), 1, size(split(person_composite_id, '_')) - 2))"
            )
            df = df.withColumn("db_schema", db_schema_expr)

            # Create a mapping of employee_id to tenure
            emp_tenure_df = df.select(
                F.col("employee_id").alias("mgr_sup_id"),
                F.col("tenure_at_vantage_days").alias("mgr_tenure"),
                F.col("vantage_date").alias("vantage_date"),
                F.col("db_schema").alias("db_schema"),
                F.col("clnt_obj_id").alias("clnt_obj_id"),
            )

            # Join to get manager's tenure
            df = df.join(
                emp_tenure_df, ["mgr_sup_id", "clnt_obj_id", "db_schema", "vantage_date"], "left"
            ).withColumnRenamed("mgr_tenure", "manager_tenure_days")

            # manager span of control
            reportee_counts = df.groupBy(["mgr_sup_id", "vantage_date", "clnt_obj_id", "db_schema"]).agg(
                F.countDistinct("person_composite_id").alias("manager_span_control")
            )

            # Join the reportee_counts back to the original dataframe on manager_id
            df = df.join(reportee_counts, on=["mgr_sup_id", "clnt_obj_id", "db_schema", "vantage_date"], how="left")

            # Add manager indicator feature from EBM table
            try:
                ebm_table = self.spark.table("us_east_1_prd_ds_blue_landing_base.employee_base_monthly")
                
                # Process 2023 and 2024 data separately
                ebm_2023 = (
                    ebm_table.filter(F.col("yyyymm") == "202301")
                    .select("ooid", "aoid", "source_hr", "is_manager_")
                    .distinct()
                    .withColumn("vantage_date", F.lit("2023-01-01"))
                )
                
                ebm_2024 = (
                    ebm_table.filter(F.col("yyyymm") == "202401")
                    .select("ooid", "aoid", "source_hr", "is_manager_")
                    .distinct()
                    .withColumn("vantage_date", F.lit("2024-01-01"))
                )
                
                # Dedupe based on source_hr
                window_spec = Window.partitionBy("aoid", "ooid", "vantage_date").orderBy(F.col("source_hr").asc_nulls_last())
                
                ebm_combined = (
                    ebm_2023.union(ebm_2024)
                    .withColumn("rank", F.row_number().over(window_spec))
                    .filter(F.col("rank") == 1)
                    .drop("rank", "source_hr")
                    .withColumnRenamed("ooid", "clnt_obj_id")
                    .withColumnRenamed("aoid", "prsn_obj_id")
                    .withColumnRenamed("is_manager_", "is_manager_ind")
                )
                
                # Join with main data
                df = df.join(ebm_combined, on=["clnt_obj_id", "prsn_obj_id", "vantage_date"], how="left")
                
            except Exception as e:
                logger.warning(f"Could not load manager indicator from EBM table: {e}")
                df = df.withColumn("is_manager_ind", F.lit(None))

            # Join manager environment features
            df = df.join(manager_relationship_start, ["person_composite_id", "vantage_date"], "left")

            logger.info("Successfully added manager environment features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add manager environment features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without manager environment features...")
            return df

    def add_team_environment_features(self, df: DataFrame, cleaned_data: DataFrame) -> DataFrame:
        """
        Add team environment features (Flag = 1 priority)
        """
        try:
            # Team size, team avg_comp, team avg tenure already exist in the data
            # Add peer salary ratio
            df = df.withColumn(
                "peer_salary_ratio",
                F.when(F.col("team_avg_comp") > 0, F.col("baseline_salary") / F.col("team_avg_comp")).otherwise(1.0),
            )

            historical_start_date = F.date_sub(F.col("vantage_date"), 365 * 2)

            historical_team_turnover = (
                cleaned_data.filter(F.col("latest_termination_date").isNotNull())
                .filter(F.col("latest_termination_date").between(historical_start_date, F.col("vantage_date")))
                .groupBy("mngr_pers_obj_id", "vantage_date")
                .agg(
                    F.countDistinct("person_composite_id").alias("team_turnover_count"),
                )
            )

            # Calculate total employees and turnover count per team
            team_counts = df.groupBy("mngr_pers_obj_id", "vantage_date").agg(
                F.countDistinct("person_composite_id").alias("team_count")
            )

            # Calculate turnover rate
            team_counts = (
                team_counts.join(historical_team_turnover, ["mngr_pers_obj_id", "vantage_date"], "left")
                .fillna(0, subset=["team_turnover_count"])
                .withColumn(
                    "team_turnover_rate",
                    F.col("team_turnover_count") / (F.col("team_turnover_count") + F.col("team_count")),
                )
                .select("mngr_pers_obj_id", "team_turnover_rate", "vantage_date", "team_turnover_count")
            )

            # Join turnover rate back to df
            df = df.join(team_counts, on=["mngr_pers_obj_id", "vantage_date"], how="left")

            logger.info("Successfully added team environment features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add team environment features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without team environment features...")
            return df

    def add_tenure_dynamics_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add tenure dynamics features (Flag = 1 priority)
        """
        try:
            # Tenure in current role (job code)
            role_tenure = (
                start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .filter(F.col("job_cd").isNotNull())
                .withColumn(
                    "rn",
                    F.row_number().over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy(F.desc("rec_eff_start_dt_mod"))
                    ),
                )
                .filter(F.col("rn") == 1)
                .select("person_composite_id", "job_cd", "rec_eff_start_dt_mod", "vantage_date")
            )

            # Find start of current role
            current_role_start = (
                start_stop_df.join(
                    role_tenure.select("person_composite_id", "vantage_date", F.col("job_cd").alias("current_job")),
                    ["person_composite_id", "vantage_date"],
                )
                .filter(F.col("job_cd") == F.col("current_job"))
                .filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.min("rec_eff_start_dt_mod").alias("role_start_date"))
                .withColumn("tenure_in_current_role", F.datediff(F.col("vantage_date"), F.col("role_start_date")))
                .select("person_composite_id", "vantage_date", "tenure_in_current_role")
            )

            # Company tenure percentile
            company_tenure_percentiles = (
                df.filter(F.col("tenure_at_vantage_days").isNotNull())
                .select("person_composite_id", "clnt_obj_id", "tenure_at_vantage_days", "vantage_date")
                .withColumn(
                    "_tmp_company_tenurepct_0_1",
                    F.percent_rank().over(
                        Window.partitionBy("clnt_obj_id", "vantage_date").orderBy("tenure_at_vantage_days")
                    ),
                )
                .withColumn("company_tenure_percentile", F.col("_tmp_company_tenurepct_0_1") * F.lit(99) + F.lit(1))
                .drop("_tmp_company_tenurepct_0_1")
            )

            # Join tenure dynamics features
            df = (
                df.join(current_role_start, ["person_composite_id", "vantage_date"], "left")
                .join(
                    company_tenure_percentiles.select("person_composite_id", "company_tenure_percentile", "vantage_date"),
                    ["person_composite_id", "vantage_date"],
                    "left",
                )
                .fillna(0, subset=["tenure_in_current_role"])
                .fillna(0.0, subset=["company_tenure_percentile"])
            )

            logger.info("Successfully added tenure dynamics features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add tenure dynamics features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without tenure dynamics features...")
            return df

    def add_work_patterns_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add work patterns features (Flag = 1 priority)
        Enhanced with location change tracking
        """
        try:
            # Assignment frequency in last 12 months
            assignment_frequency = (
                start_stop_df.filter(F.col("rec_eff_start_dt_mod") >= F.date_sub(F.col("vantage_date"), 365))
                .filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.countDistinct("work_asgmnt_nbr").alias("assignment_frequency_12m"))
                .fillna(0, subset=["assignment_frequency_12m"])
            )

            # Work location changes count
            location_changes = (
                start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .filter(F.col("work_loc_cd").isNotNull())
                .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "work_loc_cd")
                .distinct()
                .withColumn(
                    "prev_location",
                    F.lag("work_loc_cd").over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                    ),
                )
                .withColumn(
                    "location_changed",
                    F.when(
                        (F.col("prev_location").isNotNull()) & (F.col("work_loc_cd") != F.col("prev_location")), 1
                    ).otherwise(0),
                )
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.sum("location_changed").alias("work_location_changes_count"))
                .fillna(0, subset=["work_location_changes_count"])
            )

            # Add city and state location change tracking
            try:
                work_loc_mapping = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
                
                # Clean work location mapping
                work_loc_counts = (
                    work_loc_mapping.groupBy("db_schema", "clnt_obj_id", "work_loc_cd").count()
                    .filter(F.col("count") == 1)
                )
                
                work_loc_clean = (
                    work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"])
                    .select(["db_schema", "clnt_obj_id", "work_loc_cd", "city_nm", "state_prov_nm"])
                )
                
                # Add location info to start_stop data
                start_stop_with_loc = start_stop_df.join(
                    work_loc_clean, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how="left"
                )
                
                # City location changes
                city_changes = (
                    start_stop_with_loc.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                    .filter(F.col("city_nm").isNotNull())
                    .filter(~F.upper(F.col("city_nm")).isin(["UNKNOWN", "MISSING", "NULL"]))
                    .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "city_nm")
                    .distinct()
                    .withColumn(
                        "prev_city",
                        F.lag("city_nm").over(
                            Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                        ),
                    )
                    .withColumn(
                        "city_changed",
                        F.when(
                            (F.col("prev_city").isNotNull()) & (F.col("city_nm") != F.col("prev_city")), 1
                        ).otherwise(0),
                    )
                    .groupBy("person_composite_id", "vantage_date")
                    .agg(F.sum("city_changed").alias("num_city_chng"))
                    .fillna(0, subset=["num_city_chng"])
                )
                
                # State location changes
                state_changes = (
                    start_stop_with_loc.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                    .filter(F.col("state_prov_nm").isNotNull())
                    .filter(~F.upper(F.col("state_prov_nm")).isin(["UNKNOWN", "MISSING", "NULL"]))
                    .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "state_prov_nm")
                    .distinct()
                    .withColumn(
                        "prev_state",
                        F.lag("state_prov_nm").over(
                            Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                        ),
                    )
                    .withColumn(
                        "state_changed",
                        F.when(
                            (F.col("prev_state").isNotNull()) & (F.col("state_prov_nm") != F.col("prev_state")), 1
                        ).otherwise(0),
                    )
                    .groupBy("person_composite_id", "vantage_date")
                    .agg(F.sum("state_changed").alias("num_state_chng"))
                    .fillna(0, subset=["num_state_chng"])
                )
                
            except Exception as e:
                logger.warning(f"Could not load work location dimension table: {e}")
                # Create empty DataFrames with appropriate schema
                city_changes = df.select("person_composite_id", "vantage_date").withColumn("num_city_chng", F.lit(0))
                state_changes = df.select("person_composite_id", "vantage_date").withColumn("num_state_chng", F.lit(0))

            # Join work patterns features
            df = (
                df.join(assignment_frequency, ["person_composite_id", "vantage_date"], "left")
                .join(location_changes, ["person_composite_id", "vantage_date"], "left")
                .join(city_changes, ["person_composite_id", "vantage_date"], "left")
                .join(state_changes, ["person_composite_id", "vantage_date"], "left")
                .fillna(0, subset=["assignment_frequency_12m", "work_location_changes_count", "num_city_chng", "num_state_chng"])
            )

            logger.info("Successfully added work patterns features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add work patterns features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without work patterns features...")
            return df

    def add_company_factors_features(self, df: DataFrame) -> DataFrame:
        """
        Add company factors features (Flag = 1 priority)
        """
        try:
            # Company size tier based on number of employees
            company_size = (
                df.groupBy("clnt_obj_id", "vantage_date")
                .agg(F.count("*").alias("company_employee_count"))
                .withColumn(
                    "company_size_tier",
                    F.when(F.col("company_employee_count") < 100, "Small")
                    .when(F.col("company_employee_count") < 1000, "Medium")
                    .when(F.col("company_employee_count") < 10000, "Large")
                    .otherwise("Enterprise"),
                )
            )

            # Company layoff indicator (placeholder - would need termination event data)
            company_layoffs = (
                df.select("clnt_obj_id", "vantage_date").distinct().withColumn("company_layoff_indicator", F.lit(0))
            )  # Placeholder

            # Join company factors
            df = (
                df.join(
                    company_size.select("clnt_obj_id", "vantage_date", "company_size_tier"),
                    ["clnt_obj_id", "vantage_date"],
                    "left",
                )
                .join(company_layoffs, ["clnt_obj_id", "vantage_date"], "left")
                .fillna("Unknown", subset=["company_size_tier"])
                .fillna(0, subset=["company_layoff_indicator"])
            )

            logger.info("Successfully added company factors features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add company factors features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without company factors features...")
            return df

    def add_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Add temporal features (Flag = 0.5 priority)
        """
        try:
            # Hire date seasonality
            df = df.withColumn("hire_month", F.month("ltst_hire_dt"))
            df = df.withColumn("hire_quarter", F.quarter("ltst_hire_dt"))
            df = df.withColumn(
                "hire_date_seasonality",
                F.when(F.col("hire_month").isin([12, 1, 2]), "Winter")
                .when(F.col("hire_month").isin([3, 4, 5]), "Spring")
                .when(F.col("hire_month").isin([6, 7, 8]), "Summer")
                .otherwise("Fall"),
            )

            # Fiscal year effect calculations
            df = (
                df.withColumn(
                    "fiscal_year",
                    F.when(
                        (F.col("work_loc_cntry_cd") == "USA") & (F.month("ltst_hire_dt") >= 10), F.year("ltst_hire_dt") + 1
                    )
                    .when((F.col("work_loc_cntry_cd") == "USA"), F.year("ltst_hire_dt"))
                    .when(
                        (F.col("work_loc_cntry_cd") == "CAN") & (F.month("ltst_hire_dt") >= 4), F.year("ltst_hire_dt") + 1
                    )
                    .when((F.col("work_loc_cntry_cd") == "CAN"), F.year("ltst_hire_dt")),
                )
                .withColumn(
                    "fiscal_date",
                    F.when(
                        F.col("work_loc_cntry_cd") == "USA",
                        F.to_date(F.concat_ws("-", F.col("fiscal_year"), F.lit("09"), F.lit("30"))),
                    ).when(
                        F.col("work_loc_cntry_cd") == "CAN",
                        F.to_date(F.concat_ws("-", F.col("fiscal_year"), F.lit("03"), F.lit("31"))),
                    ),
                )
                .withColumn("fiscal_year_effect", F.datediff(F.col("fiscal_date"), F.col("ltst_hire_dt")))
            )

            # Quarter effect
            df = df.withColumn(
                "quarter_effect",
                F.when(
                    F.col("work_loc_cntry_cd") == "USA",
                    F.when(F.month("ltst_hire_dt").between(10, 12), F.lit("Q1"))
                    .when(F.month("ltst_hire_dt").between(1, 3), F.lit("Q2"))
                    .when(F.month("ltst_hire_dt").between(4, 6), F.lit("Q3"))
                    .when(F.month("ltst_hire_dt").between(7, 9), F.lit("Q4")),
                ).when(
                    F.col("work_loc_cntry_cd") == "CAN",
                    F.when(F.month("ltst_hire_dt").between(4, 6), F.lit("Q1"))
                    .when(F.month("ltst_hire_dt").between(7, 9), F.lit("Q2"))
                    .when(F.month("ltst_hire_dt").between(10, 12), F.lit("Q3"))
                    .when(F.month("ltst_hire_dt").between(1, 3), F.lit("Q4")),
                ),
            )

            logger.info("Successfully added temporal features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add temporal features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without temporal features...")
            return df

    def add_external_features(self, df: DataFrame) -> DataFrame:
        """
        Add features created using data present in CSV files
        and not UC tables
        """
        try:
            ext_feature_generator = ExternalFeatures(self.spark)

            df_with_cpi = ext_feature_generator.create_salary_growth_to_cpi_feature(df)

            df_with_normalized_flsa = ext_feature_generator.create_normalized_flsa_desc(df_with_cpi)
            
            # Add neighborhood salary ratio feature
            df_with_neighborhood = ext_feature_generator.create_neighborhood_salary_ratio_feature(df_with_normalized_flsa)

            logger.info("Successfully added external features")
            return df_with_neighborhood
            
        except Exception as e:
            logger.error(f"Failed to add external features: {e}")
            if self.fail_on_feature_error:
                raise
            logger.warning("Continuing without external features...")
            return df

    def save_feature_engineered_data(
        self,
        df: DataFrame,
        table_name: str = "employee_features_comprehensive",
        parquet_name: str = "employee_features_comprehensive",
        dest: list = ["table", "volume"],
    ) -> Tuple[DataFrame, str]:
        """
        Save the feature-engineered dataset
        """
        try:
            # Select final feature columns
            if not self.feature_cols:
                df_to_save = df
            else:
                feature_columns = list(set(self.feature_cols))
                # Select only columns that exist in the dataframe
                existing_columns = df.columns
                final_columns = [col for col in feature_columns if col in existing_columns]

                logger.info(f"Feature-engineered dataset prepared with {len(final_columns)} features")
                logger.info(f"Features: {final_columns}")

                df_to_save = df.select(*final_columns)

            dest = set(dest)

            if "table" in dest:
                save_to_table(df=df_to_save, table_name=table_name)
                logger.info(f"Saved to table: {table_name}")

            if "volume" in dest:
                write_to_unity_catalog(df_to_save, filename=parquet_name, format_type="parquet", spark=self.spark)
                logger.info(f"Saved to volume: {parquet_name}")

            return df_to_save, table_name
            
        except Exception as e:
            logger.error(f"Failed to save feature-engineered data: {e}")
            raise
        
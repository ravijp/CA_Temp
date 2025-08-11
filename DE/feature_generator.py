import pyspark.sql.functions as F
from pyspark.sql.functions import when, lit
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import *
from typing import Tuple

from top_etl.feature_engg.constants import (
    SAMPLE_IDS,
    FINAL_COLS,
    ONEDATA_CATALOG_NM,
    TOP_SCHEMA_NM,
)

from top_etl.feature_engg.external_features import ExternalFeatures

from top_etl.common.utils import save_to_table

from top_etl.common.uc_volume_io import write_to_unity_catalog
from top_etl.feature_engg.normalize_features import FeatureNormalizer


class FeatureGenerator:
    def __init__(
        self,
        spark: SparkSession,
        use_select_cols: bool = True,
        use_sample: bool = False,
        save_to_table: bool = True,
        normalize: bool = True,
    ):
        self.spark = spark
        self.sample_list = SAMPLE_IDS if use_sample else None
        self.feature_cols = FINAL_COLS if use_select_cols else None
        self.save_to_table = save_to_table
        self.normalize = normalize
        pass

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

    def create_comprehensive_features(self) -> DataFrame:
        """
        Create comprehensive feature set for employee turnover prediction
        
        Args:
            spark: SparkSession
            vantage_date: Reference date for feature calculation
        
        Returns:
            DataFrame with comprehensive features
        """

        base_data, start_stop_compressed, cleaned_data = self.load_source_data()

        # Apply all feature engineering functions
        df = self.add_compensation_features(base_data, start_stop_compressed)

        df = self.add_career_progression_features(df, start_stop_compressed)

        df = self.add_demographic_features(df)

        df = self.add_job_characteristics_features(df, cleaned_data)

        df = self.add_manager_environment_features(df, start_stop_compressed)

        df = self.add_team_environment_features(df, cleaned_data)

        df = self.add_tenure_dynamics_features(df, start_stop_compressed)

        df = self.add_work_patterns_features(df, start_stop_compressed)

        df = self.add_company_factors_features(df)

        df = self.add_temporal_features(df)

        df = self.add_external_features(df)

        if self.normalize:
            feature_normalizer = FeatureNormalizer(spark=self.spark)
            df = feature_normalizer.normalize(df)

        if self.save_to_table:
            df, table_name = self.save_feature_engineered_data(df)

        return df

    def add_compensation_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add compensation-related features (Flag = 1 priority)
        """

        # Get historical salary data
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
                    F.col("comp_chng_cnt") * 365.0,
                    F.datediff(F.col("last_chng_dt"), F.col("first_sal_dt"))
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

        # df = df.join(salary_last_quarter, on="person_composite_id", how="left")

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
            .agg((1 - F.sum(F.col("prob") ** 2)).alias("pay_freq_diversity_score")
        ).withColumn(
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
                F.col("annl_cmpn_amt")
                - F.lag("annl_cmpn_amt").over(
                    Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                ),
            )
            .withColumn(
                "salary_change",
                F.when(
                    "prev_salary".isNotNull(),
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

        return df

    def add_career_progression_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add career progression features (Flag = 1 priority)
        """

        # Get job level history
        job_history = (
            start_stop_df.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
            .filter(F.col("mngr_lvl_cd").isNotNull())
            .select("person_composite_id", "rec_eff_start_dt_mod", "mngr_lvl_cd", "job_cd", "vantage_date")
            .distinct()
        )

        # Time since last promotion (job level increase)
        promotion_history = (
            job_history.withColumn(
                "prev_level",
                F.lag("mngr_lvl_cd").over(
                    Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                ),
            )
            .withColumn("is_promotion", F.when(F.col("mngr_lvl_cd") > F.col("prev_level"), 1).otherwise(0))  # 
            # assuming higher code indicates promotion --> need to replace with event table PRomotion events
            .filter(F.col("is_promotion") == 1)
            .groupBy("person_composite_id", "vantage_date")
            .agg(F.max("rec_eff_start_dt_mod").alias("last_promotion_date"))
            .withColumn("time_since_last_promotion", F.datediff(F.col("vantage_date"), F.col("last_promotion_date")))
            .select("person_composite_id", "vantage_date", "time_since_last_promotion")
        )

        # Promotion velocity
        promotion_velocity = (
            job_history.withColumn(
                "prev_level",
                F.lag("mngr_lvl_cd").over(
                    Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                ),
            )
            .withColumn("is_promotion", F.when(F.col("mngr_lvl_cd") > F.col("prev_level"), 1).otherwise(0))
            .filter(F.col("is_promotion") == 1)
            .withColumn(
                "tenure_years",
                F.datediff(
                    "rec_eff_start_dt_mod",
                    F.first("rec_eff_start_dt_mod").over(
                        Window.partitionBy("person_composite_id", "vantage_date").orderBy("rec_eff_start_dt_mod")
                    ),
                ) / 365.25
            )
            .groupBy("person_composite_id", "vantage_date")
            .agg(
                F.count("*").alias("promotion_count"),
                F.max("tenure_years").alias("total_tenure_years")
            )
            .withColumn(
                "promotion_velocity",  # promotion_velocity_per_year
                F.when(
                    (F.col("promotion_count") > 0) & (F.col('total_tenure_years') > 0),
                    F.col("promotion_count") / F.col("total_tenure_years")
                ).otherwise(0.0)
            )
            .select("person_composite_id", "vantage_date", "promotion_velocity")
        )

        # Join career progression features
        df = (
            df.join(promotion_history, ["person_composite_id", "vantage_date"], "left")
            .join(promotion_velocity, ["person_composite_id", "vantage_date"], "left")
            .fillna({
                "time_since_last_promotion": 9999.0,  # Large number for never promoted
                "promotion_velocity" : 0.0
            })
        )

        return df

    def add_demographic_features(self, df: DataFrame) -> DataFrame:
        """
        Add demographic features (Flag = 1 priority)
        """

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
            F.when(F.col("age_at_vantage").isNotNull(), F.greatest(F.lit(0), 65 - F.col("age_at_vantage"))).
            otherwise(F.lit(None)
            ),
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
                (F.col("working_age") > 5) & (F.col("working_age") <= 15) & (F.col("tenure at vantage years") <= 5),
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

        return df

    def add_job_characteristics_features(self, df: DataFrame, cleaned_data: DataFrame) -> DataFrame:
        """
        Add job characteristics features (Flag = 1 priority)
        """

        # Job level from manager level code
        df = df.withColumn("job_level", F.col("mngr_lvl_cd"))

        # Job family turnover rate (placeholder - would need historical data)
        # df = df.withColumn("job_family_turnover_rate", F.lit(0.15))  # Placeholder

        # FLSA status, full-time/part-time, regular/temporary already exist in the data
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
            .agg(F.countDistinct("person_composite_id")
                .alias("turnover_cnt"))
        )

        # job family turnover rate
        # Calculate total employees and turnover count per job_cd
        job_family_counts = df.groupBy("job_cd", "vantage_date").agg(
            F.countDistinct("person_composite_id").alias("active_cnt"),
        )

        # Calculate turnover rate
        job_family_counts = (
            job_family_counts.join(historical_job_family_turnover, ["job_cd", "vantage_date"], "left")
            .fillna(0, subset=["turnover_count"])
            .withColumn("job_family_turnover_rate", F.col("turnover_cnt") / (F.col("turnover_cnt") + F.col("active_cnt")))
            .select("job_cd", "job_family_turnover_rate", "vantage_date")
        )

        # Join turnover rate back to df
        df = df.join(job_family_counts, on=["job_cd", "vantage_date"], how="left")

        # # decision_making_authority_indicator
        # df = df.withColumn("decision_making_authority_indicator", F.col("mngr_lvl_cd"))

        df = df.withColumn(
            "job_stability_ind",
            F.when(
                (F.col("full_tm_part_tm_cd")=="F") &
                (F.col("reg_temp_cd")=="R") &
                (F.col("flsa_stus_cd").isin(["F", "N"])), 1  #high stability
            )
            .when(F.col("reg_temp_cd")=='T', 0)  # temporary
            .otherwise(0.5)  #medium stability
        )
        return df

    def add_manager_environment_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add manager environment features (Flag = 1 priority)
        """

        # Count unique manager changes per person_composite_id and vantage_data
        manager_changes_df = (
            start_stop_df.filter(~F.col("mngr_pers_obj_id").isin(["UNKNOWN", "ONE", "", "0"]))
            .groupBy("person_composite_id", "vantage_date")
            .agg(F.countDistinct("mngr_pers_obj_id").alias("manager_changes_count_vantage"))
        )

        # Join manager_changes_count to df
        df = df.join(manager_changes_df, ["person_composite_id", "vantage_date"], "left").fillna(
            {"manager_changes_count_vantage": 0}
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
        # TODO: Remove as these fts should come in etl output tables
        person_composite_id_split = F.split(F.col("person_composite_id"), "_")
        df = df.withColumn("employee_id", F.element_at(person_composite_id_split, -1))
        df = df.withColumn("clnt_obj_id", F.element_at(person_composite_id_split, -2))
        # Extract db_schema by concatenating all elements from index 1 up to and including -2 (excluding the last element)
        db_schema_expr = F.expr(
            "concat_ws('_', slice(split(person_composite_id, '_'), 1, size(split(person_composite_id, '_')) - 2))"
        )
        df = df.withColumn("db_schema", db_schema_expr)

        # Extract employee_id from person_composite_id
        # df = df.withColumn("employee_id", F.split(F.col("person_composite_id"), "_")[3])

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
        # Step 1: Identify the manager of each employee (assuming 'manager_id' column exists)
        # Step 2: For each manager, count the distinct employees reporting to them
        reportee_counts = df.groupBy(["mgr_sup_id", "vantage_date", "clnt_obj_id", "db_schema"]).agg(
            F.countDistinct("person_composite_id").alias("manager_span_control")
        )

        # Step 3: Join the reportee_counts back to the original dataframe on manager_id
        df = df.join(reportee_counts, on=["mgr_sup_id", "clnt_obj_id", "db_schema", "vantage_date"], how="left")

        # Join manager environment features
        df = (
            df.alias("df1")
            # .join(manager_changes.alias("df2"), ["person_composite_id", "vantage_date"], "left")
            .join(manager_relationship_start.alias("df3"), ["person_composite_id", "vantage_date"], "left")
        )
        #     .fillna(0, subset=["df2.manager_changes_count", "time_with_current_manager"]))

        return df

    def add_team_environment_features(self, df: DataFrame, cleaned_data: DataFrame) -> DataFrame:
        """
        Add team environment features (Flag = 1 priority)
        """

        # Team size, team avg_comp, team avg tenure already exist in the data
        # Add peer salary ratio
        df = df.withColumn(
            "peer_salary_ratio",
            F.when(F.col("team_avg_comp") > 0, F.col("baseline_salary") / F.col("team_avg_comp")).otherwise(1.0),
        )

        # Team turnover rate calculation (placeholder - would need historical termination data)
        # df = df.withColumn("team_turnover_rate_12m", F.lit(0.12))  # Placeholder

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

        return df

    def add_tenure_dynamics_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add tenure dynamics features (Flag = 1 priority)
        """

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
        # Define window partitioned by industry code and ordered by tenure
        w = (
            Window.partitionBy("naics_cd")
            .orderBy("tenure_at_vantage_days")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        # Calculate percentile rank for tenure within each industry
        df = df.withColumn("tenure_percentile_industry", F.percent_rank().over(w))

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

        return df

    def add_work_patterns_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """
        Add work patterns features (Flag = 1 priority)
        """

        # Assignment frequency in last 12 months
        assignment_frequency = (
            start_stop_df.filter(F.col("rec_eff_start_dt_mod") >= F.date_sub(F.col("vantage_date"), 365))
            .filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
            .groupBy("person_composite_id", "vantage_date")
            .agg(F.countDistinct("work_asgnmt_nbr").alias("assignment_frequency_12m"))
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

        # Join work patterns features
        df = (
            df.join(assignment_frequency, ["person_composite_id", "vantage_date"], "left")
            .join(location_changes, ["person_composite_id", "vantage_date"], "left")
            .fillna(0, subset=["assignment_frequency_12m", "work_location_changes_count"])
        )

        return df

    def add_company_factors_features(self, df: DataFrame) -> DataFrame:
        """
        Add company factors features (Flag = 1 priority)
        """

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
            df.alias("df1")
            .join(
                company_size.select("clnt_obj_id", "vantage_date", "company_size_tier").alias("df2"),
                ["clnt_obj_id", "vantage_date"],
                "left",
            )
            .join(company_layoffs.alias("df3"), ["clnt_obj_id", "vantage_date"], "left")
            .fillna("Unknown", subset=["company_size_tier"])
            .fillna(0, subset=["company_layoff_indicator"])
        )

        return df

    def add_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Add temporal features (Flag = 0.5 priority)
        """

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

        # Fiscal year effect
        # df = df.withColumn("fiscal_year_effect",
        #     F.when(F.col("fscl_actv_ind") == "Y", 1).otherwise(0))

        # Assume fiscal year ends on June 30 (fiscal date for a year is 'yyyy-06-30')

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
        # df = df.withColumn("quarter_effect", F.quarter(F.col("vantage_date")))

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

        return df

    def add_external_features(self, df: DataFrame) -> DataFrame:
        """
        Add features creating using data present in CSV files
        and not UC tables
        """

        ext_feature_generator = ExternalFeatures(self.spark)

        df_with_cpi = ext_feature_generator.create_salary_growth_to_cpi_feature(df)

        df_with_normalized_flsa = ext_feature_generator.create_normalized_flsa_desc(df_with_cpi)

        return df_with_normalized_flsa

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

        # Select final feature columns
        if not self.feature_cols:
            df_to_save = df
        else:
            feature_columns = list(set(self.feature_cols))
            # Select only columns that exist in the dataframe
            existing_columns = df.columns
            final_columns = [col for col in feature_columns if col in existing_columns]

            # # Save the feature-engineered dataset
            print(f"Feature-engineered dataset saved with {len(final_columns)} features")
            print(f"Features: {final_columns}")

            df_to_save = df.select(*final_columns)

        dest = set(dest)

        if "table" in dest:
            save_to_table(df=df_to_save, table_name=table_name)

        if "volume" in dest:
            write_to_unity_catalog(df_to_save, filename=parquet_name, format_type="parquet", spark=self.spark)

        return df_to_save, table_name
    
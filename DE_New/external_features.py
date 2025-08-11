# external_features.py
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
import pandas as pd
from pathlib import Path
import logging

from top_etl.feature_engg.constants import (
    DATA_DIR_PATH, 
    EXTERNAL_DATA_CONFIG,
    FT_TO_PT_REASONS,
    NON_EXEMPT_TO_EXEMPT_REASONS
)
from top_etl.common.utils import read_csv_from_volume

logger = logging.getLogger(__name__)

class ExternalFeatures:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = EXTERNAL_DATA_CONFIG

    def load_work_loc_dim_tbl(self) -> DataFrame:
        """Load work location dimension table with error handling"""
        try:
            # Read work location table
            work_loc_mapping = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
            work_loc_mapping = work_loc_mapping.select(
                "clnt_obj_id", "work_loc_cd", "db_schema", "pstl_cd", "state_prov_cd", "state_prov_nm", "city_nm"
            )
            
            # Remove records from work location table which have multiple state values
            work_loc_counts = work_loc_mapping.groupBy("db_schema", "clnt_obj_id", "work_loc_cd").count()
            work_loc_counts = work_loc_counts.filter(F.col("count") == 1)
            work_loc_mapping = work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"])
            
            logger.info("Successfully loaded work location dimension table")
            return work_loc_mapping
            
        except Exception as e:
            logger.error(f"Failed to load work location dimension table: {e}")
            # Return empty DataFrame with expected schema
            return self.spark.createDataFrame([], schema=StructType([
                StructField("clnt_obj_id", StringType(), True),
                StructField("work_loc_cd", StringType(), True),
                StructField("db_schema", StringType(), True),
                StructField("pstl_cd", StringType(), True),
                StructField("state_prov_cd", StringType(), True),
                StructField("state_prov_nm", StringType(), True),
                StructField("city_nm", StringType(), True)
            ]))

    def load_external_csv_data(self, filename: str, encoding: str = "ISO-8859-1") -> DataFrame:
        """Load external CSV data with comprehensive error handling"""
        try:
            file_path = Path(DATA_DIR_PATH) / filename
            
            # Try to read the CSV file
            csv_data = read_csv_from_volume(
                self.spark, file_path, encoding=encoding
            )
            
            if csv_data is None:
                raise Exception(f"read_csv_from_volume returned None for {filename}")
                
            # Convert pandas DataFrame to Spark DataFrame if needed
            if isinstance(csv_data, pd.DataFrame):
                if csv_data.empty:
                    logger.warning(f"CSV file {filename} is empty")
                    return None
                csv_data = self.spark.createDataFrame(csv_data)
                
            logger.info(f"Successfully loaded external CSV: {filename} with {csv_data.count()} rows")
            return csv_data
            
        except FileNotFoundError:
            logger.error(f"External CSV file not found: {filename}")
            return None
        except Exception as e:
            logger.error(f"Failed to load external CSV {filename}: {e}")
            return None

    def create_salary_growth_to_cpi_feature(self, df: DataFrame) -> DataFrame:
        """Create salary growth to CPI ratio feature with robust error handling"""
        try:
            logger.info("Starting salary growth to CPI feature calculation")
            
            # Load work location dimension table
            work_loc_mapping = self.load_work_loc_dim_tbl()
            if work_loc_mapping.count() == 0:
                logger.warning("Work location mapping table is empty, skipping CPI feature")
                return df.withColumn("salary_growth_rate12m_to_cpi_rate", F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))
            
            # Load state to region mapping
            state_to_region_mapping = self.load_external_csv_data(self.config["state_to_region_file"])
            if state_to_region_mapping is None:
                logger.warning("State to region mapping unavailable, using fallback values")
                return df.withColumn("salary_growth_rate12m_to_cpi_rate", F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))
            
            # Load CPI data
            cpi_data = self.load_external_csv_data(self.config["cpi_data_file"])
            if cpi_data is None:
                logger.warning("CPI data unavailable, using fallback values")
                return df.withColumn("salary_growth_rate12m_to_cpi_rate", F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))
            
            # Data validation
            required_state_cols = ["State", "Region"]
            missing_state_cols = [col for col in required_state_cols if col not in state_to_region_mapping.columns]
            if missing_state_cols:
                logger.error(f"Missing required columns in state mapping: {missing_state_cols}")
                return df.withColumn("salary_growth_rate12m_to_cpi_rate", F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))
            
            required_cpi_cols = ["Region", "1_Jan_23", "1_Jan_24"]
            missing_cpi_cols = [col for col in required_cpi_cols if col not in cpi_data.columns]
            if missing_cpi_cols:
                logger.error(f"Missing required columns in CPI data: {missing_cpi_cols}")
                return df.withColumn("salary_growth_rate12m_to_cpi_rate", F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))
            
            # Process data transformations
            # Change State name to uppercase in work_loc_mapping and state-to-region mapping
            work_loc_mapping = work_loc_mapping.withColumn("state_upper", F.upper(F.col("state_prov_nm")))
            state_to_region_mapping = state_to_region_mapping.withColumn("state_upper", F.upper(F.col("State")))
            
            # Join work_loc_mapping and state-to-region mapping to get Region
            df_region = work_loc_mapping.join(state_to_region_mapping, on="state_upper", how="left")
            
            # Rename CPI columns
            cpi_data = cpi_data.withColumnRenamed("1_Jan_23", "CPI_2023").withColumnRenamed("1_Jan_24", "CPI_2024")
            
            # Join df_region to external CPI data on region
            df_wcpi = df_region.join(cpi_data, on=["Region"], how="left")
            
            # Join df data with df_wcpi on db_schema, client_id and work location code
            df_all = df.join(df_wcpi, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how="left")
            
            # Calculate feature based on dataset split
            df_all = df_all.withColumn(
                "CPI",
                F.when(F.col("dataset_split").isin(["train", "val"]), F.col("CPI_2023"))
                .otherwise(F.col("CPI_2024"))
            )
            
            # Calculate salary growth rate to CPI rate ratio
            df_all = df_all.withColumn(
                "salary_growth_rate12m_to_cpi_rate",
                F.when(
                    (F.col("CPI").isNotNull()) & (F.col("CPI") != 0) & (F.col("salary_growth_rate_12m").isNotNull()),
                    F.col("salary_growth_rate_12m") / F.col("CPI")
                ).otherwise(F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))
            )
            
            logger.info("Successfully calculated salary growth to CPI feature")
            return df_all
            
        except Exception as e:
            logger.error(f"Failed to calculate salary growth to CPI feature: {e}")
            return df.withColumn("salary_growth_rate12m_to_cpi_rate", F.lit(self.config["fallback_values"]["sal_cpi_rt_ratio_1yr"]))

    def create_normalized_flsa_desc(self, df: DataFrame) -> DataFrame:
        """Create normalized FLSA description with error handling"""
        try:
            logger.info("Starting normalized FLSA description feature")
            
            normalized_flsa_mapping = self.load_external_csv_data(self.config["flsa_mapping_file"])
            if normalized_flsa_mapping is None:
                logger.warning("FLSA mapping data unavailable, using fallback values")
                return df.withColumn("flsa_status_desc", F.lit(self.config["fallback_values"]["flsa_status_desc"]))
            
            # Data validation
            required_cols = ["clnt_obj_id", "flsa_stus_cd", "flsa_stus_cd_normalized"]
            missing_cols = [col for col in required_cols if col not in normalized_flsa_mapping.columns]
            if missing_cols:
                logger.error(f"Missing required columns in FLSA mapping: {missing_cols}")
                return df.withColumn("flsa_status_desc", F.lit(self.config["fallback_values"]["flsa_status_desc"]))
            
            df_with_normalized_flsa = (
                df.join(normalized_flsa_mapping, ["clnt_obj_id", "flsa_stus_cd"], "left")
                .fillna({"flsa_stus_cd_normalized": self.config["fallback_values"]["flsa_status_desc"]})
                .drop("flsa_stus_cd")
                .withColumnRenamed("flsa_stus_cd_normalized", "flsa_status_desc")
            )
            
            logger.info("Successfully created normalized FLSA description feature")
            return df_with_normalized_flsa
            
        except Exception as e:
            logger.error(f"Failed to create normalized FLSA description: {e}")
            return df.withColumn("flsa_status_desc", F.lit(self.config["fallback_values"]["flsa_status_desc"]))

    def create_neighborhood_salary_ratio_feature(self, df: DataFrame) -> DataFrame:
        """Create salary to neighborhood median income ratio with error handling"""
        try:
            logger.info("Starting neighborhood salary ratio feature calculation")
            
            # Load ADP internal geo-code table
            try:
                geo_code = self.spark.table("us_east_1_prd_ven_blue_landing_base.pb_geocode_result_monthly_gz")
                
                # Dedupe table and keep selected columns and filtered data
                geo_code = geo_code.filter(F.col("yyyymm") == "202301").withColumn("rn", F.row_number().over(
                    Window.partitionBy("employee_guid", "yyyymm").orderBy(F.col("home_congressional_district"))
                )).filter(F.col("rn") == 1
                ).select("employee_guid", "home_census_block_group", "yyyymm")
                
                logger.info("Successfully loaded geo-code table")
                
            except Exception as e:
                logger.error(f"Failed to load geo-code table: {e}")
                return df.withColumn("sal_nghb_ratio", F.lit(self.config["fallback_values"]["sal_nghb_ratio"]))
            
            # Load US Census data
            us_census = self.load_external_csv_data(self.config["census_income_file"])
            if us_census is None:
                logger.warning("US Census data unavailable, using fallback values")
                return df.withColumn("sal_nghb_ratio", F.lit(self.config["fallback_values"]["sal_nghb_ratio"]))
            
            # Data validation for census data
            required_census_cols = ["GEO_ID", "S1903_C03_001E"]
            missing_census_cols = [col for col in required_census_cols if col not in us_census.columns]
            if missing_census_cols:
                logger.error(f"Missing required columns in Census data: {missing_census_cols}")
                return df.withColumn("sal_nghb_ratio", F.lit(self.config["fallback_values"]["sal_nghb_ratio"]))
            
            # Join ADP internal geo-code table with US Census data
            # create mapping key in geo_code data
            geo_code = geo_code.withColumn("census_mapping_key", F.substring("home_census_block_group", 1, 11))
            
            # create mapping key in us_census data
            us_census = us_census.withColumn("census_mapping_key", F.expr("substring(GEO_ID, -11, 11)"))
            
            # Join on census_mapping_key
            geo_code_wext = geo_code.join(us_census, "census_mapping_key", "left")
            
            # Load Employee Base Monthly
            try:
                ebm_table = self.spark.table("us_east_1_prd_ds_blue_landing_base.employee_base_monthly")
                
                # Filter on 2023 Jan and keep selected columns
                ebm_table_2023 = ebm_table.filter(F.col("yyyymm")=="202301").select("ooid", "aoid", "employee_guid", "source_hr").distinct()
                
                # Dedupe EBM based on source_hr column
                window_spec = Window.partitionBy("aoid", "ooid").orderBy(F.col("source_hr").asc_nulls_last())
                ebm_table_2023_dedupe = ebm_table_2023.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
                
                logger.info("Successfully loaded and processed EBM table")
                
            except Exception as e:
                logger.error(f"Failed to load EBM table: {e}")
                return df.withColumn("sal_nghb_ratio", F.lit(self.config["fallback_values"]["sal_nghb_ratio"]))
            
            # Join EBM table with geo-code table having external data
            ebm_w_ext = ebm_table_2023_dedupe.join(geo_code_wext, on="employee_guid", how="left")
            
            # Create mapping columns in df
            df = df.withColumn("aoid", F.split(F.col("person_composite_id"), "_")[3])
            df = df.withColumn("ooid", F.split(F.col("person_composite_id"), "_")[2])
            
            # Join ebm_w_ext with df
            df = df.join(ebm_w_ext, on=["ooid", "aoid"], how="left")
            df = df.withColumnRenamed("S1903_C03_001E", "median_household_income")
            
            # Create feature sal_nghb_ratio
            df = df.withColumn(
                "sal_nghb_ratio", 
                F.when(
                    (F.col("median_household_income").isNotNull()) & 
                    (F.col("median_household_income") > 0) & 
                    (F.col("baseline_salary").isNotNull()),
                    F.col("baseline_salary") / F.col("median_household_income")
                ).otherwise(F.lit(self.config["fallback_values"]["sal_nghb_ratio"]))
            )
            
            logger.info("Successfully calculated neighborhood salary ratio feature")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate neighborhood salary ratio: {e}")
            return df.withColumn("sal_nghb_ratio", F.lit(self.config["fallback_values"]["sal_nghb_ratio"]))

    def create_work_location_change_features(self, df: DataFrame, start_stop_df: DataFrame) -> DataFrame:
        """Create work location change features with error handling"""
        try:
            logger.info("Starting work location change features calculation")
            
            # Load work location dimension table
            work_loc_mapping = self.load_work_loc_dim_tbl()
            if work_loc_mapping.count() == 0:
                logger.warning("Work location mapping table is empty, using zero values")
                return df.withColumn("num_city_chng", F.lit(0)).withColumn("num_state_chng", F.lit(0))
            
            # Add city and state names to start stop format
            start_stop_enhanced = start_stop_df.join(work_loc_mapping, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how="left")
            
            # Work city changes count
            city_location_changes = (
                start_stop_enhanced.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .filter(F.col("city_nm").isNotNull())
                .filter(~F.upper(F.col("city_nm")).isin([x.upper() for x in ["UNKNOWN", "MISSING", "NULL"]]))
                .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "city_nm")
                .distinct()
                .withColumn(
                    "prev_city_location",
                    F.lag("city_nm").over(
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
                start_stop_enhanced.filter(F.col("rec_eff_start_dt_mod") <= F.col("vantage_date"))
                .filter(F.col("state_prov_nm").isNotNull())
                .filter(~F.upper(F.col("state_prov_nm")).isin([x.upper() for x in ["UNKNOWN", "MISSING", "NULL"]]))
                .select("person_composite_id", "vantage_date", "rec_eff_start_dt_mod", "state_prov_nm")
                .distinct()
                .withColumn(
                    "prev_state_location",
                    F.lag("state_prov_nm").over(
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
            df_with_location_changes = (
                df.join(city_location_changes, ["person_composite_id", "vantage_date"], "left")
                .join(state_location_changes, ["person_composite_id", "vantage_date"], "left")
                .fillna(0, subset=["num_city_chng", "num_state_chng"])
            )
            
            logger.info("Successfully calculated work location change features")
            return df_with_location_changes
            
        except Exception as e:
            logger.error(f"Failed to calculate work location change features: {e}")
            return df.withColumn("num_city_chng", F.lit(0)).withColumn("num_state_chng", F.lit(0))

    def create_manager_indicator_feature(self, df: DataFrame) -> DataFrame:
        """Create manager indicator feature with error handling"""
        try:
            logger.info("Starting manager indicator feature calculation")
            
            # Load Employee Base Monthly tables for both years
            ebm_table = self.spark.table("us_east_1_prd_ds_blue_landing_base.employee_base_monthly")
            
            # Process 2023 data
            ebm_2023 = ebm_table.filter(F.col("yyyymm")=="202301").select("ooid", "aoid", "source_hr", "is_manager_").distinct()
            window_spec = Window.partitionBy("aoid", "ooid").orderBy(F.col("source_hr").asc_nulls_last())
            ebm_2023_dedupe = ebm_2023.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
            ebm_2023_dedupe = ebm_2023_dedupe.withColumn("vantage_date", F.lit("2023-01-01"))
            ebm_2023_dedupe = ebm_2023_dedupe.withColumnRenamed("ooid", "clnt_obj_id").withColumnRenamed("aoid", "prsn_obj_id")
            
            # Process 2024 data
            ebm_2024 = ebm_table.filter(F.col("yyyymm")=="202401").select("ooid", "aoid", "source_hr", "is_manager_").distinct()
            ebm_2024_dedupe = ebm_2024.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
            ebm_2024_dedupe = ebm_2024_dedupe.withColumn("vantage_date", F.lit("2024-01-01"))
            ebm_2024_dedupe = ebm_2024_dedupe.withColumnRenamed("ooid", "clnt_obj_id").withColumnRenamed("aoid", "prsn_obj_id")
            
            # Combine both datasets
            ebm_combined = ebm_2023_dedupe.union(ebm_2024_dedupe)
            
            # Derive columns from modeling data
            df_with_keys = df.withColumn("clnt_obj_id", F.split(F.col("person_composite_id"), "_")[2])
            df_with_keys = df_with_keys.withColumn("prsn_obj_id", F.split(F.col("person_composite_id"), "_")[3])
            
            # Join with modeling data
            df_with_manager_flag = df_with_keys.join(ebm_combined, on=["clnt_obj_id", "prsn_obj_id", "vantage_date"], how="left")
            df_with_manager_flag = df_with_manager_flag.withColumnRenamed("is_manager_", "is_manager_ind")
            
            # Fill null values with 0 (assume non-manager if not found)
            df_with_manager_flag = df_with_manager_flag.fillna({"is_manager_ind": 0})
            
            logger.info("Successfully calculated manager indicator feature")
            return df_with_manager_flag
            
        except Exception as e:
            logger.error(f"Failed to calculate manager indicator feature: {e}")
            return df.withColumn("is_manager_ind", F.lit(0))

    def create_job_change_features(self, df: DataFrame) -> DataFrame:
        """Create job change features with error handling"""
        try:
            logger.info("Starting job change features calculation")
            
            # Load event dimension and fact tables
            try:
                event_dim = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_event")
                event_fact = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_fact_work_event")
                
                event_dim = event_dim.select(["clnt_obj_id", "db_schema", "event_cd", "event_dsc", "event_rsn_cd", "event_rsn_dsc"])
                
                # Join the two tables
                event_fact_dim = event_fact.join(event_dim, on=['clnt_obj_id', 'db_schema', 'event_cd', 'event_rsn_cd'], how='left')
                
                logger.info("Successfully loaded event tables")
                
            except Exception as e:
                logger.error(f"Failed to load event tables: {e}")
                return df.withColumn("job_chng_2yr_ind", F.lit(0)).withColumn("num_job_chng_2yr", F.lit(0)).withColumn("job_chng_fulltopart_ind", F.lit(0)).withColumn("job_chng_nexmptoexmp_ind", F.lit(0))
            
            # Create joining keys in modeling data
            df_with_keys = df.withColumn("db_schema", F.concat(F.split(F.col("person_composite_id"), "_")[0], F.lit("_util")))
            df_with_keys = df_with_keys.withColumn("clnt_obj_id", F.split(F.col("person_composite_id"), "_")[2])
            df_with_keys = df_with_keys.withColumn("pers_obj_id", F.split(F.col("person_composite_id"), "_")[3])
            
            # Join event master with modeling df
            df_temp = df_with_keys.join(event_fact_dim, on=['clnt_obj_id', 'db_schema', 'pers_obj_id'], how='left')
            
            # Filter for events that happened 24 months before or on vantage date
            df_24m = df_temp.filter(
                (F.col("event_eff_dt") >= F.add_months(F.col("vantage_date"), -24)) & 
                (F.col("event_eff_dt") <= F.col("vantage_date"))
            )
            
            # Create job change flag
            job_change_list = ['JTC', 'PJC', 'POS', 'JRC']
            df_24m = df_24m.withColumn(
                "job_change_flag",
                F.when(F.upper(F.col("event_cd")).isin([x.upper() for x in job_change_list]), 1).otherwise(0)
            )
            
            # Take max of flag at person composite id X vantage date
            df_job_change_in_last_24m = df_24m.groupBy("person_composite_id", "vantage_date").agg(
                F.max("job_change_flag").alias("job_chng_2yr_ind")
            )
            
            # Count of distinct event_eff_dt where event_cd is in job_change_list
            df_number_of_job_changes_in_last_24m = (
                df_24m.filter(F.upper(F.col("event_cd")).isin([x.upper() for x in job_change_list]))
                .groupBy("person_composite_id", "vantage_date")
                .agg(F.countDistinct("event_eff_dt").alias("num_job_chng_2yr"))
            )
            
            # Full-time to part-time changes
            df_24m = df_24m.withColumn(
                "job_change_flag_due_toftpt",
                F.when(
                    F.upper(F.col("event_cd")).isin([x.upper() for x in job_change_list]) & 
                    F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in FT_TO_PT_REASONS]), 1
                ).otherwise(0)
            )
            
            df_job_change_ft_to_pt = df_24m.groupBy("person_composite_id", "vantage_date").agg(
                F.max("job_change_flag_due_toftpt").alias("job_chng_fulltopart_ind")
            )
            
            # Non-exempt to exempt changes
            df_24m = df_24m.withColumn(
                "job_change_flag_due_to_non_exempt_to_exempt",
                F.when(
                    F.upper(F.col("event_cd")).isin([x.upper() for x in job_change_list]) & 
                    F.upper(F.col("event_rsn_cd")).isin([x.upper() for x in NON_EXEMPT_TO_EXEMPT_REASONS]), 1
                ).otherwise(0)
            )
            
            df_job_change_nonexe_to_exe = df_24m.groupBy("person_composite_id", "vantage_date").agg(
                F.max("job_change_flag_due_to_non_exempt_to_exempt").alias("job_chng_nexmptoexmp_ind")
            )
            
            # Join back features to modeling data
            df_result = (
                df.join(df_job_change_in_last_24m, on=['person_composite_id', 'vantage_date'], how='left')
                .join(df_number_of_job_changes_in_last_24m, on=['person_composite_id', 'vantage_date'], how='left')
                .join(df_job_change_ft_to_pt, on=['person_composite_id', 'vantage_date'], how='left')
                .join(df_job_change_nonexe_to_exe, on=['person_composite_id', 'vantage_date'], how='left')
                .fillna(0, subset=["job_chng_2yr_ind", "num_job_chng_2yr", "job_chng_fulltopart_ind", "job_chng_nexmptoexmp_ind"])
            )
            
            logger.info("Successfully calculated job change features")
            return df_result
            
        except Exception as e:
            logger.error(f"Failed to calculate job change features: {e}")
            return df.withColumn("job_chng_2yr_ind", F.lit(0)).withColumn("num_job_chng_2yr", F.lit(0)).withColumn("job_chng_fulltopart_ind", F.lit(0)).withColumn("job_chng_nexmptoexmp_ind", F.lit(0))
        
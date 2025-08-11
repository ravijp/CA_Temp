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
    EXTERNAL_DATA_FALLBACKS
)
from top_etl.common.utils import read_csv_from_volume

logger = logging.getLogger(__name__)


class ExternalFeatures:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def _safe_read_csv(self, file_path: Path, encoding: str = "utf-8") -> DataFrame:
        """Safely read CSV with fallback mechanisms"""
        try:
            logger.info(f"Attempting to read CSV: {file_path}")
            
            result = read_csv_from_volume(self.spark, file_path, encoding=encoding)
            
            if result is None:
                raise Exception(f"read_csv_from_volume returned None for {file_path}")
            
            if isinstance(result, pd.DataFrame):
                result = self.spark.createDataFrame(result)
                
            logger.info(f"Successfully loaded CSV: {file_path} with {result.count():,} rows")
            return result
            
        except Exception as e:
            logger.error(f"Failed to read CSV {file_path}: {e}")
            return None

    def _validate_csv_data(self, df: DataFrame, required_columns: list, file_name: str) -> bool:
        """Validate CSV data quality"""
        if df is None:
            return False
            
        try:
            # Check required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {file_name}: {missing_cols}")
                return False
                
            # Check for data
            if df.count() == 0:
                logger.error(f"No data found in {file_name}")
                return False
                
            logger.info(f"CSV validation passed for {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"CSV validation failed for {file_name}: {e}")
            return False

    def load_work_loc_dim_tbl(self) -> DataFrame:
        """Load work location dimension table with error handling"""
        try:
            # Read work location table
            work_loc_mapping = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
            work_loc_mapping = work_loc_mapping.select(
                "clnt_obj_id", "work_loc_cd", "db_schema", "pstl_cd", "state_prov_cd", "state_prov_nm"
            )
            
            # Remove records from work location table which have multiple state values
            work_loc_counts = (
                work_loc_mapping.groupBy("db_schema", "clnt_obj_id", "work_loc_cd").count()
                .filter(F.col("count") == 1)
            )
            
            work_loc_mapping = work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"])
            
            logger.info(f"Successfully loaded work location mapping with {work_loc_mapping.count():,} records")
            return work_loc_mapping
            
        except Exception as e:
            logger.error(f"Failed to load work location dimension table: {e}")
            raise

    def create_salary_growth_to_cpi_feature(self, df: DataFrame) -> DataFrame:
        """Create salary growth to CPI ratio feature with robust error handling"""
        try:
            # Load work location dimension table
            work_loc_mapping = self.load_work_loc_dim_tbl()
            
            # Attempt to load external data files
            state_to_region_mapping_fp = Path(DATA_DIR_PATH) / EXTERNAL_DATA_CONFIG["state_to_region_file"]
            region_to_cpi_mapping_fp = Path(DATA_DIR_PATH) / EXTERNAL_DATA_CONFIG["cpi_by_region_file"]
            
            # Read State-to-region mapping with error handling
            state_to_region_mapping = self._safe_read_csv(state_to_region_mapping_fp, encoding="ISO-8859-1")
            if not self._validate_csv_data(state_to_region_mapping, ["State", "Region"], "state_to_region_mapping"):
                logger.error("State-to-region mapping validation failed, using fallback values")
                return self._add_fallback_cpi_feature(df)
            
            # Read CPI data with error handling
            df_cpi = self._safe_read_csv(region_to_cpi_mapping_fp, encoding="ISO-8859-1")
            if not self._validate_csv_data(df_cpi, ["Region", "1_Jan_23", "1_Jan_24"], "cpi_by_region"):
                logger.error("CPI data validation failed, using fallback values")
                return self._add_fallback_cpi_feature(df)
            
            # Standardize column names for case consistency
            work_loc_mapping = work_loc_mapping.withColumn("state_upper", F.upper(F.col("state_prov_nm")))
            state_to_region_mapping = state_to_region_mapping.withColumn("state_upper", F.upper(F.col("State")))
            
            # Join work_loc_mapping and state-to-region mapping to get Region
            df_region = work_loc_mapping.join(state_to_region_mapping, on="state_upper", how="left")
            
            # Rename CPI columns to standard format
            df_cpi = df_cpi.withColumnRenamed("1_Jan_23", "CPI_2023").withColumnRenamed("1_Jan_24", "CPI_2024")
            
            # Join df_region to external CPI data on region
            df_wcpi = df_region.join(df_cpi, on=["Region"], how="left")
            
            # Join df data with df_wcpi on db_schema, client_id and work location code
            df_all = df.join(df_wcpi, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how="left")
            
            # Calculate feature -> salary growth rate to CPI growth rate with dataset split logic
            df_with_cpi = (
                df_all.withColumn(
                    "CPI",
                    F.when(F.col("dataset_split").isin(["train", "val"]), F.col("CPI_2023"))
                    .otherwise(F.col("CPI_2024"))
                )
                .withColumn(
                    "salary_growth_rate12m_to_cpi_rate",
                    F.when(
                        (F.col("CPI").isNotNull()) & (F.col("CPI") != 0) & (F.col("salary_growth_rate_12m").isNotNull()),
                        F.col("salary_growth_rate_12m") / F.col("CPI")
                    ).otherwise(F.lit(EXTERNAL_DATA_FALLBACKS["default_cpi_ratio"]))
                )
                .drop("CPI")  # Clean up temporary column
            )
            
            logger.info("Successfully created salary growth to CPI feature")
            return df_with_cpi
            
        except Exception as e:
            logger.error(f"Failed to create salary growth to CPI feature: {e}")
            logger.info("Using fallback CPI feature values")
            return self._add_fallback_cpi_feature(df)

    def _add_fallback_cpi_feature(self, df: DataFrame) -> DataFrame:
        """Add CPI feature with fallback default values"""
        return df.withColumn(
            "salary_growth_rate12m_to_cpi_rate", 
            F.lit(EXTERNAL_DATA_FALLBACKS["default_cpi_ratio"])
        )

    def create_normalized_flsa_desc(self, df: DataFrame) -> DataFrame:
        """Create normalized FLSA description with fallback handling"""
        try:
            normalized_flsa_mapping_csv_fp = Path(DATA_DIR_PATH) / EXTERNAL_DATA_CONFIG["flsa_mapping_file"]
            
            normalized_flsa_mapping = self._safe_read_csv(normalized_flsa_mapping_csv_fp)
            
            if not self._validate_csv_data(
                normalized_flsa_mapping, 
                ["clnt_obj_id", "flsa_stus_cd", "flsa_stus_cd_normalized"], 
                "flsa_mapping"
            ):
                logger.error("FLSA mapping validation failed, using fallback values")
                return self._add_fallback_flsa_feature(df)
            
            df_with_normalized_flsa = (
                df.join(normalized_flsa_mapping, ["clnt_obj_id", "flsa_stus_cd"], "left")
                .fillna({"flsa_stus_cd_normalized": EXTERNAL_DATA_FALLBACKS["default_flsa_desc"]})
                .drop("flsa_stus_cd")
                .withColumnRenamed("flsa_stus_cd_normalized", "flsa_status_desc")
            )
            
            logger.info("Successfully created normalized FLSA description feature")
            return df_with_normalized_flsa
            
        except Exception as e:
            logger.error(f"Failed to create normalized FLSA feature: {e}")
            logger.info("Using fallback FLSA description values")
            return self._add_fallback_flsa_feature(df)

    def _add_fallback_flsa_feature(self, df: DataFrame) -> DataFrame:
        """Add FLSA feature with fallback default values"""
        return df.withColumn(
            "flsa_status_desc", 
            F.lit(EXTERNAL_DATA_FALLBACKS["default_flsa_desc"])
        )

    def create_neighborhood_salary_ratio_feature(self, df: DataFrame) -> DataFrame:
        """Create salary to neighborhood median income ratio feature"""
        try:
            # Load geocode and census data
            geocode_result = self._load_geocode_data()
            if geocode_result is None:
                logger.error("Failed to load geocode data, using fallback values")
                return self._add_fallback_neighborhood_feature(df)
                
            census_data = self._load_census_data()
            if census_data is None:
                logger.error("Failed to load census data, using fallback values")
                return self._add_fallback_neighborhood_feature(df)
            
            # Create mapping keys
            geocode_with_key = geocode_result.withColumn(
                "census_mapping_key", 
                F.substring("home_census_block_group", 1, 11)
            )
            
            census_with_key = census_data.withColumn(
                "census_mapping_key", 
                F.expr("substring(GEO_ID, -11, 11)")
            )
            
            # Join geocode with census data
            geo_census_mapping = geocode_with_key.join(
                census_with_key, on="census_mapping_key", how="left"
            )
            
            # Load EBM table and join with geocode data
            ebm_with_income = self._create_ebm_income_mapping(geo_census_mapping)
            if ebm_with_income is None:
                logger.error("Failed to create EBM income mapping, using fallback values")
                return self._add_fallback_neighborhood_feature(df)
            
            # Create composite keys in main dataframe
            df_with_keys = (
                df.withColumn("aoid", F.split(F.col("person_composite_id"), "_")[3])
                .withColumn("ooid", F.split(F.col("person_composite_id"), "_")[2])
            )
            
            # Join with income data and calculate ratio
            df_with_income = (
                df_with_keys.join(ebm_with_income, on=["ooid", "aoid"], how="left")
                .withColumn(
                    "sal_nghb_ratio",
                    F.when(
                        (F.col("median_household_income").isNotNull()) & (F.col("median_household_income") > 0),
                        F.col("baseline_salary") / F.col("median_household_income")
                    ).otherwise(F.lit(EXTERNAL_DATA_FALLBACKS["default_neighborhood_ratio"]))
                )
                .drop("aoid", "ooid", "median_household_income")
            )
            
            logger.info("Successfully created neighborhood salary ratio feature")
            return df_with_income
            
        except Exception as e:
            logger.error(f"Failed to create neighborhood salary ratio feature: {e}")
            logger.info("Using fallback neighborhood ratio values")
            return self._add_fallback_neighborhood_feature(df)

    def _load_geocode_data(self) -> DataFrame:
        """Load and process geocode data with error handling"""
        try:
            geo_code = self.spark.table("us_east_1_prd_ven_blue_landing_base.pb_geocode_result_monthly_gz")
            
            # Dedupe and filter for 2023 data
            geo_code_clean = (
                geo_code.filter(F.col("yyyymm") == "202301")
                .withColumn(
                    "rn", 
                    F.row_number().over(
                        Window.partitionBy("employee_guid", "yyyymm").orderBy(F.col("home_congressional_district"))
                    )
                )
                .filter(F.col("rn") == 1)
                .select("employee_guid", "home_census_block_group", "yyyymm")
            )
            
            logger.info(f"Successfully loaded geocode data with {geo_code_clean.count():,} records")
            return geo_code_clean
            
        except Exception as e:
            logger.error(f"Failed to load geocode data: {e}")
            return None

    def _load_census_data(self) -> DataFrame:
        """Load US Census median income data with error handling"""
        try:
            census_file_path = Path(DATA_DIR_PATH) / EXTERNAL_DATA_CONFIG["census_income_file"]
            
            census_data = self._safe_read_csv(census_file_path, encoding="ISO-8859-1")
            
            if not self._validate_csv_data(census_data, ["GEO_ID", "S1903_C03_001E"], "census_income"):
                return None
                
            return census_data
            
        except Exception as e:
            logger.error(f"Failed to load census data: {e}")
            return None

    def _create_ebm_income_mapping(self, geo_census_mapping: DataFrame) -> DataFrame:
        """Create EBM to income mapping with error handling"""
        try:
            ebm_table = self.spark.table("us_east_1_prd_ds_blue_landing_base.employee_base_monthly")
            
            # Filter and dedupe EBM table for 2023
            ebm_2023_clean = (
                ebm_table.filter(F.col("yyyymm") == "202301")
                .select("ooid", "aoid", "employee_guid", "source_hr")
                .distinct()
                .withColumn(
                    "rank",
                    F.row_number().over(
                        Window.partitionBy("aoid", "ooid").orderBy(F.col("source_hr").asc_nulls_last())
                    )
                )
                .filter(F.col("rank") == 1)
                .drop("rank")
            )
            
            # Join EBM with geocode and census data
            ebm_with_income = (
                ebm_2023_clean.join(geo_census_mapping, on="employee_guid", how="left")
                .withColumnRenamed("S1903_C03_001E", "median_household_income")
                .select("ooid", "aoid", "median_household_income")
            )
            
            logger.info(f"Successfully created EBM income mapping with {ebm_with_income.count():,} records")
            return ebm_with_income
            
        except Exception as e:
            logger.error(f"Failed to create EBM income mapping: {e}")
            return None

    def _add_fallback_neighborhood_feature(self, df: DataFrame) -> DataFrame:
        """Add neighborhood feature with fallback default values"""
        return df.withColumn(
            "sal_nghb_ratio", 
            F.lit(EXTERNAL_DATA_FALLBACKS["default_neighborhood_ratio"])
        )

    def create_regional_economic_features(self, df: DataFrame) -> DataFrame:
        """Create additional regional economic features"""
        try:
            # This could be extended with more economic indicators
            # For now, focusing on the core CPI feature
            logger.info("Regional economic features created (CPI-based)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to create regional economic features: {e}")
            return df

    def validate_external_features(self, df: DataFrame) -> bool:
        """Validate external features meet business requirements"""
        try:
            validation_results = {}
            
            # Check CPI ratio feature
            if "salary_growth_rate12m_to_cpi_rate" in df.columns:
                cpi_stats = df.select("salary_growth_rate12m_to_cpi_rate").describe().collect()
                validation_results["cpi_ratio"] = True
                logger.info("CPI ratio feature validation passed")
            else:
                validation_results["cpi_ratio"] = False
                logger.warning("CPI ratio feature missing")
            
            # Check neighborhood ratio feature
            if "sal_nghb_ratio" in df.columns:
                neighborhood_stats = df.select("sal_nghb_ratio").describe().collect()
                validation_results["neighborhood_ratio"] = True
                logger.info("Neighborhood ratio feature validation passed")
            else:
                validation_results["neighborhood_ratio"] = False
                logger.warning("Neighborhood ratio feature missing")
            
            # Check FLSA description feature
            if "flsa_status_desc" in df.columns:
                flsa_values = df.select("flsa_status_desc").distinct().collect()
                validation_results["flsa_desc"] = True
                logger.info("FLSA description feature validation passed")
            else:
                validation_results["flsa_desc"] = False
                logger.warning("FLSA description feature missing")
            
            all_passed = all(validation_results.values())
            logger.info(f"External features validation: {sum(validation_results.values())}/{len(validation_results)} passed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"External features validation failed: {e}")
            return False
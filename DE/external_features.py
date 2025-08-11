# external_features.py
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
import pandas as pd
from pathlib import Path

from top_etl.feature_engg.constants import DATA_DIR_PATH
from top_etl.common.utils import read_csv_from_volume

class ExternalFeatures:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_work_loc_dim_tbl(self) -> DataFrame:
        # Read work location table
        work_loc_mapping = self.spark.table("us_east_1_prd_ds_blue_raw.dwh_t_dim_work_loc")
        work_loc_mapping = work_loc_mapping.select(
            "clnt_obj_id", "work_loc_cd", "db_schema", "pst1_cd", "state_prov_cd", "state_prov_nm"
        )
        
        # Remove records from work location table which have multiple state values
        work_loc_counts = work_loc_mapping.groupBy("db_schema", "clnt_obj_id", "work_loc_cd").count()
        work_loc_counts = work_loc_counts.filter(F.col("count") == 1)
        work_loc_mapping = work_loc_mapping.join(work_loc_counts, on=["db_schema", "clnt_obj_id", "work_loc_cd"])
        
        return work_loc_mapping

    def create_salary_growth_to_cpi_feature(self, df: DataFrame) -> DataFrame:
        # Load work location dimension table
        work_loc_mapping = self.load_work_loc_dim_tbl()
        
        state_to_region_mapping_fp = Path(DATA_DIR_PATH) / "US_State_to_Region_Mapping.csv"
        region_to_cpi_mapping_fp = Path(DATA_DIR_PATH) / "US_CPI_Change_By_Regions.csv"
        
        # Read State-to-region mapping
        try:
            state_to_region_mapping = read_csv_from_volume(
                self.spark, state_to_region_mapping_fp, encoding="ISO-8859-1"
            )
            if state_to_region_mapping is None:
                raise Exception("read_state_to_region_mapping: read_csv_from_volume() returned None")
            elif isinstance(state_to_region_mapping, pd.DataFrame):
                state_to_region_mapping = self.spark.createDataFrame(state_to_region_mapping)
        except Exception as e:
            print(f"error reading state_to_region_mapping csv file into dataframe: {e}")
            print("skipping adding external features....")
            return df
        
        # Change State name to uppercase in work_loc_mapping and state-to-region mapping
        work_loc_mapping = work_loc_mapping.withColumn("state_upper", F.upper(F.col("state_prov_nm")))
        state_to_region_mapping = state_to_region_mapping.withColumn("state_upper", F.upper(F.col("State")))
        
        # Join work_loc_mapping and state-to-region mapping to get Region
        df_region = work_loc_mapping.join(state_to_region_mapping, on="state_upper", how="left")
        
        # Read external data having region-wise % CPI change for 2023 and 2024
        try:
            df_cpi = read_csv_from_volume(self.spark, region_to_cpi_mapping_fp, encoding="ISO-8859-1")
            if df_cpi is None:
                raise Exception("read df_cpi: read_csv_from_volume() returned None")
            elif isinstance(df_cpi, pd.DataFrame):
                df_cpi = self.spark.createDataFrame(df_cpi)
        except Exception as e:
            print(f"error reading region_to_cpi_mapping csv file into dataframe: {e}")
            print("skipping adding external features....")
            return df
        
        df_cpi = df_cpi.withColumnRenamed("1_Jan_23", "CPI_2023").withColumnRenamed("1_Jan_24", "CPI_2024")
        
        # Join df_region to external CPI data on region
        df_wcpi = df_region.join(df_cpi, on=["Region"], how="left")
        
        # Join df data with df_wcpi on db_schema, client_id and work location code
        df_all = df.join(df_wcpi, on=["db_schema", "clnt_obj_id", "work_loc_cd"], how="left")
        
        # Calculate feature -> salary growth rate to CPI growth rate
        df_all = df_all.withColumn(
            "salary_growth_rate12m_to_cpi_rate",
            F.when(F.col("CPI_2023").isNotNull(), F.col("salary_growth_rate_12m") / F.col("CPI_2023")).otherwise(0.0)
        )
        
        return df_all

    def create_normalized_flsa_desc(self, df: DataFrame) -> DataFrame:
        normalized_flsa_mapping_csv_fp = Path(DATA_DIR_PATH) / "flsa_client_mapping.csv"
        try:
            normalized_flsa_mapping = read_csv_from_volume(self.spark, normalized_flsa_mapping_csv_fp)
            if normalized_flsa_mapping is None:
                raise Exception("read normalized_flsa_mapping: read_csv_from_volume() returned None")
            elif isinstance(normalized_flsa_mapping, pd.DataFrame):
                normalized_flsa_mapping = self.spark.createDataFrame(normalized_flsa_mapping)
        except Exception as e:
            print(f"error reading normalized_flsa_mapping csv file into dataframe: {e}")
            print("skipping adding external features....")
            return df
        
        df_with_normalized_flsa = (
            df.join(normalized_flsa_mapping, ["clnt_obj_id", "flsa_stus_cd"], "left")
            .fillna({"flsa_stus_cd_normalized": "others"})
            .drop("flsa_stus_cd")
            .withColumnRenamed("flsa_stus_cd_normalized", "flsa_status_desc")
        )
        
        return df_with_normalized_flsa

import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from typing import Dict

from top_etl.feature_engg.constants import RAW_TO_NORMALIZED_MAPPING, MISSING_VALUE_LIT, RAW_COLUMN_SUFFIX

import logging
logger = logging.getLogger(__name__)


class FeatureNormalizer:
    def __init__(self, spark: SparkSession, mapping: Dict[str, Dict] = RAW_TO_NORMALIZED_MAPPING):
        self.spark = spark
        self.mapping = mapping
        self.missing_value_lit = MISSING_VALUE_LIT
        self.orig_column_suffix = RAW_COLUMN_SUFFIX

    def normalize(self, df: DataFrame) -> DataFrame:
        logger.info(f"Starting normalization for {len(self.mapping)} columns")
        for col_name, rules in self.mapping.items():
            if col_name not in set(df.columns):
                logger.info(f"{col_name} not found in dataframe column list... skipping")
                continue

            logger.info(f"normalizing {col_name}")

            keep_vals = rules.get("keep", [])
            drop_vals = rules.get("drop", [])

            raw_col = f"{col_name}{self.orig_column_suffix}"
            df = df.withColumnRenamed(col_name, raw_col)

            if keep_vals:
                df = df.withColumn(
                    col_name,
                    F.when(F.col(raw_col).isin(keep_vals), F.col(raw_col)).otherwise(F.lit(self.missing_value_lit)),
                )

            if drop_vals:
                df = df.withColumn(
                    col_name,
                    F.when(F.col(raw_col).isin(drop_vals), F.lit(self.missing_value_lit)).otherwise(F.col(raw_col)),
                )

            logger.info(f"normalized {col_name}")

        logger.info("Completed normalization process")
        return df
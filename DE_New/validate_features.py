import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import *
import logging

from top_etl.feature_engg.constants import FEATURE_VALIDATION_RULES, FEATURE_GROUPS

logger = logging.getLogger(__name__)

class FeatureValidator:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validation_rules = FEATURE_VALIDATION_RULES
        self.feature_groups = FEATURE_GROUPS
        self.validators = [
            {"name": "duplicate_rows", "fn": self.has_duplicate_rows},
            {"name": "duplicate_persons", "fn": self.has_duplicate_persons},
            {"name": "temporal_consistency", "fn": self.validate_temporal_consistency},
            {"name": "feature_value_ranges", "fn": self.validate_feature_ranges},
            {"name": "business_logic", "fn": self.validate_business_logic},
            {"name": "data_leakage", "fn": self.detect_data_leakage},
            {"name": "feature_distributions", "fn": self.validate_feature_distributions},
        ]

    def run_validations(self, df: DataFrame):
        """Run comprehensive validation suite"""
        logger.info("Starting comprehensive feature validation")
        all_pass = True
        validation_results = {}
        
        for v in self.validators:
            vname, vfunc = v["name"], v["fn"]
            logger.info(f"Running validation: {vname}")
            
            if vfunc and callable(vfunc):
                try:
                    vres = vfunc(df)
                    validation_results[vname] = vres
                except Exception as e:
                    logger.error(f"Error invoking validator: {vname}: {e}")
                    vres = False
                    validation_results[vname] = False
                    continue
            else:
                logger.error(f"Function for validator {vname} is either NoneType or not callable: {vfunc}")
                vres = False
                validation_results[vname] = False
                continue
                
            if vres:
                logger.info(f"Validation {vname} passed")
            else:
                logger.error(f"Validation {vname} failed")
            
            all_pass = all_pass and vres
        
        if all_pass:
            logger.info("All validations passed!")
        else:
            logger.error("One or more validations have failed. Check above messages for details")
            
        return validation_results

    def has_duplicate_rows(self, df: DataFrame) -> bool:
        """Check for duplicate rows in the dataset"""
        try:
            orig_count = df.count()
            dedup_count = df.dropDuplicates().count()
            if orig_count != dedup_count:
                logger.error(f"Found {orig_count - dedup_count} duplicate rows. Original: {orig_count:,}, After dedup: {dedup_count:,}")
                return False
            else:
                logger.info("No duplicate rows found")
                return True
        except Exception as e:
            logger.error(f"Error checking duplicate rows: {e}")
            return False

    def has_duplicate_persons(self, df: DataFrame) -> bool:
        """Check for duplicate person records per vantage date"""
        try:
            orig_count = df.count()
            dedup_count = df.dropDuplicates(["person_composite_id", "vantage_date"]).count()
            if orig_count != dedup_count:
                logger.error(f"Found {orig_count - dedup_count} duplicate person records. Original: {orig_count:,}, After dedup: {dedup_count:,}")
                
                # Show examples of duplicates
                duplicates = (
                    df.groupBy("person_composite_id", "vantage_date")
                    .count()
                    .filter(F.col("count") > 1)
                    .orderBy(F.desc("count"))
                    .limit(5)
                )
                logger.error("Sample duplicate person records:")
                duplicates.show(truncate=False)
                return False
            else:
                logger.info("No duplicate person records found")
                return True
        except Exception as e:
            logger.error(f"Error checking duplicate persons: {e}")
            return False

    def validate_temporal_consistency(self, df: DataFrame) -> bool:
        """Validate temporal consistency - no future data usage"""
        try:
            logger.info("Validating temporal consistency...")
            
            # Check if vantage_date column exists
            if "vantage_date" not in df.columns:
                logger.warning("vantage_date column not found, skipping temporal validation")
                return True
            
            # Check for any dates that are after vantage_date (potential data leakage)
            date_columns = [
                "event_eff_dt", "rec_eff_start_dt_mod", "rec_eff_end_dt", 
                "termination_date", "hire_date", "last_promotion_date"
            ]
            
            issues_found = False
            
            for date_col in date_columns:
                if date_col in df.columns:
                    future_dates_count = df.filter(
                        (F.col(date_col).isNotNull()) &
                        (F.col(date_col) > F.col("vantage_date"))
                    ).count()
                    
                    if future_dates_count > 0:
                        logger.error(f"Found {future_dates_count} records with {date_col} after vantage_date")
                        issues_found = True
                    else:
                        logger.info(f"{date_col} temporal consistency validated")
            
            if not issues_found:
                logger.info("All temporal consistency checks passed")
                return True
            else:
                logger.error("Temporal consistency issues detected")
                return False
                
        except Exception as e:
            logger.error(f"Error validating temporal consistency: {e}")
            return False

    def validate_feature_ranges(self, df: DataFrame) -> bool:
        """Validate feature values are within expected ranges"""
        try:
            logger.info("Validating feature value ranges...")
            
            issues_found = False
            
            for feature_name, rules in self.validation_rules.items():
                if feature_name not in df.columns:
                    continue
                    
                # Check minimum values
                if "min" in rules:
                    min_val = rules["min"]
                    below_min_count = df.filter(
                        (F.col(feature_name).isNotNull()) &
                        (F.col(feature_name) < min_val)
                    ).count()
                    
                    if below_min_count > 0:
                        logger.error(f"Feature {feature_name}: {below_min_count} values below minimum {min_val}")
                        issues_found = True
                
                # Check maximum values
                if "max" in rules:
                    max_val = rules["max"]
                    if max_val is not None:
                        above_max_count = df.filter(
                            (F.col(feature_name).isNotNull()) &
                            (F.col(feature_name) > max_val)
                        ).count()
                        
                        if above_max_count > 0:
                            logger.error(f"Feature {feature_name}: {above_max_count} values above maximum {max_val}")
                            issues_found = True
                
                # Check null allowance
                if "allow_null" in rules and not rules["allow_null"]:
                    null_count = df.filter(F.col(feature_name).isNull()).count()
                    
                    if null_count > 0:
                        logger.error(f"Feature {feature_name}: {null_count} null values found (nulls not allowed)")
                        issues_found = True
            
            if not issues_found:
                logger.info("All feature range validations passed")
                return True
            else:
                logger.error("Feature range validation issues detected")
                return False
                
        except Exception as e:
            logger.error(f"Error validating feature ranges: {e}")
            return False

    def validate_business_logic(self, df: DataFrame) -> bool:
        """Validate business logic relationships between features"""
        try:
            logger.info("Validating business logic...")
            
            issues_found = False
            
            # Validate age is reasonable
            if "age_at_vantage" in df.columns:
                unreasonable_age_count = df.filter(
                    (F.col("age_at_vantage") < 16) | (F.col("age_at_vantage") > 85)
                ).count()
                
                if unreasonable_age_count > 0:
                    logger.error(f"Found {unreasonable_age_count} records with unreasonable age values")
                    issues_found = True
            
            # Validate tenure cannot exceed age
            if "tenure_at_vantage_days" in df.columns and "age_at_vantage" in df.columns:
                invalid_tenure_count = df.filter(
                    (F.col("tenure_at_vantage_days") / 365.25) > (F.col("age_at_vantage") - 14)
                ).count()
                
                if invalid_tenure_count > 0:
                    logger.error(f"Found {invalid_tenure_count} records where tenure exceeds reasonable work age")
                    issues_found = True
            
            # Validate promotion counts are reasonable
            if "num_promot_2yr" in df.columns:
                excessive_promotions_count = df.filter(F.col("num_promot_2yr") > 5).count()
                
                if excessive_promotions_count > 0:
                    logger.warning(f"Found {excessive_promotions_count} records with >5 promotions in 2 years")
            
            # Validate salary percentiles are between 0 and 1
            percentile_cols = ["compensation_percentile_company", "compensation_percentile_industry"]
            for col_name in percentile_cols:
                if col_name in df.columns:
                    invalid_percentile_count = df.filter(
                        (F.col(col_name) < 0) | (F.col(col_name) > 1)
                    ).count()
                    
                    if invalid_percentile_count > 0:
                        logger.error(f"Found {invalid_percentile_count} invalid percentile values in {col_name}")
                        issues_found = True
            
            # Validate event indicators are binary
            binary_cols = ["event_indicator_all", "event_indicator_vol", "promot_2yr_ind", "demot_2yr_ind"]
            for col_name in binary_cols:
                if col_name in df.columns:
                    invalid_binary_count = df.filter(
                        (F.col(col_name).isNotNull()) &
                        (~F.col(col_name).isin([0, 1]))
                    ).count()
                    
                    if invalid_binary_count > 0:
                        logger.error(f"Found {invalid_binary_count} non-binary values in {col_name}")
                        issues_found = True
            
            if not issues_found:
                logger.info("All business logic validations passed")
                return True
            else:
                logger.error("Business logic validation issues detected")
                return False
                
        except Exception as e:
            logger.error(f"Error validating business logic: {e}")
            return False

    def detect_data_leakage(self, df: DataFrame) -> bool:
        """Detect potential data leakage patterns"""
        try:
            logger.info("Detecting potential data leakage...")
            
            leakage_found = False
            
            # Check for perfect correlations with target variables
            if "event_indicator_all" in df.columns:
                # Look for features that might be derived from future information
                suspicious_features = [
                    "termination_date", "survival_time_days", 
                    "future_salary", "post_termination_indicator"
                ]
                
                for feature in suspicious_features:
                    if feature in df.columns and feature != "survival_time_days":
                        logger.error(f"Suspicious feature found that may cause leakage: {feature}")
                        leakage_found = True
            
            # Check for features calculated using future data
            if "vantage_date" in df.columns:
                # Features that should not use data beyond vantage date
                time_sensitive_features = [
                    "days_since_promot", "days_since_transfer", 
                    "time_since_last_promotion", "manager_changes_count"
                ]
                
                # This is a heuristic check - in practice you'd need more sophisticated detection
                for feature in time_sensitive_features:
                    if feature in df.columns:
                        # Check if all values make sense given vantage date constraints
                        negative_values = df.filter(F.col(feature) < 0).count()
                        if negative_values > 0:
                            logger.warning(f"Found {negative_values} negative values in {feature} - possible temporal inconsistency")
            
            # Check survival time consistency
            if "survival_time_days" in df.columns and "event_indicator_all" in df.columns:
                # For non-events, survival time should equal follow-up period
                inconsistent_survival = df.filter(
                    (F.col("event_indicator_all") == 0) & 
                    (F.col("survival_time_days") != 365)
                ).count()
                
                if inconsistent_survival > 0:
                    logger.warning(f"Found {inconsistent_survival} records with inconsistent survival times for non-events")
            
            if not leakage_found:
                logger.info("No obvious data leakage patterns detected")
                return True
            else:
                logger.error("Potential data leakage issues detected")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting data leakage: {e}")
            return False

    def validate_feature_distributions(self, df: DataFrame) -> bool:
        """Validate feature distributions for anomalies"""
        try:
            logger.info("Validating feature distributions...")
            
            issues_found = False
            
            # Check for features with excessive null rates
            null_threshold = 0.95
            for col_name in df.columns:
                if col_name in ["person_composite_id", "vantage_date", "dataset_split"]:
                    continue
                    
                total_count = df.count()
                null_count = df.filter(F.col(col_name).isNull()).count()
                null_rate = null_count / total_count if total_count > 0 else 0
                
                if null_rate > null_threshold:
                    logger.error(f"Feature {col_name} has excessive null rate: {null_rate:.3f}")
                    issues_found = True
            
            # Check for features with zero or near-zero variance
            numeric_features = [
                f.name for f in df.schema.fields 
                if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]
            ]
            
            for feature in numeric_features:
                if feature in df.columns:
                    try:
                        distinct_count = df.select(feature).distinct().count()
                        if distinct_count <= 1:
                            logger.error(f"Feature {feature} has zero variance (only {distinct_count} distinct values)")
                            issues_found = True
                        elif distinct_count <= 3:
                            logger.warning(f"Feature {feature} has very low variance ({distinct_count} distinct values)")
                    except:
                        continue
            
            # Check for features with extreme outliers
            for feature in numeric_features[:10]:  # Limit to first 10 for performance
                if feature in df.columns:
                    try:
                        percentiles = df.select(
                            F.expr(f"percentile_approx({feature}, 0.01)").alias("p1"),
                            F.expr(f"percentile_approx({feature}, 0.99)").alias("p99"),
                            F.mean(feature).alias("mean"),
                            F.stddev(feature).alias("stddev")
                        ).collect()[0]
                        
                        if percentiles.stddev and percentiles.mean:
                            # Check for extreme outliers (more than 5 standard deviations from mean)
                            extreme_outliers = df.filter(
                                F.abs(F.col(feature) - percentiles.mean) > 5 * percentiles.stddev
                            ).count()
                            
                            if extreme_outliers > 0:
                                outlier_rate = extreme_outliers / df.count()
                                if outlier_rate > 0.01:  # More than 1% outliers
                                    logger.warning(f"Feature {feature} has {extreme_outliers} extreme outliers ({outlier_rate:.3f})")
                    except:
                        continue
            
            # Check categorical features for excessive cardinality
            categorical_features = [
                f.name for f in df.schema.fields 
                if f.dataType == StringType()
            ]
            
            for feature in categorical_features:
                if feature in df.columns and feature not in ["person_composite_id"]:
                    try:
                        distinct_count = df.select(feature).distinct().count()
                        total_count = df.count()
                        cardinality_ratio = distinct_count / total_count if total_count > 0 else 0
                        
                        if cardinality_ratio > 0.8:  # More than 80% unique values
                            logger.warning(f"Categorical feature {feature} has high cardinality: {distinct_count} distinct values ({cardinality_ratio:.3f} ratio)")
                    except:
                        continue
            
            if not issues_found:
                logger.info("Feature distribution validation passed")
                return True
            else:
                logger.error("Feature distribution issues detected")
                return False
                
        except Exception as e:
            logger.error(f"Error validating feature distributions: {e}")
            return False

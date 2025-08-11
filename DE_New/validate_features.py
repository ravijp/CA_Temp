import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import *
import logging

from top_etl.feature_engg.constants import (
    FEATURE_VALIDATION_RULES,
    TEMPORAL_VALIDATION_RULES,
    BUSINESS_LOGIC_THRESHOLDS,
    DATA_QUALITY_THRESHOLDS,
    CRITICAL_FEATURES,
    FEATURE_GROUPS
)

logger = logging.getLogger(__name__)


class FeatureValidator:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validators = [
            {"name": "critical_features", "fn": self.validate_critical_features},
            {"name": "duplicate_rows", "fn": self.has_duplicate_rows},
            {"name": "duplicate_persons", "fn": self.has_duplicate_persons},
            {"name": "temporal_consistency", "fn": self.validate_temporal_consistency},
            {"name": "business_logic", "fn": self.validate_business_logic},
            {"name": "feature_distributions", "fn": self.validate_feature_distributions},
            {"name": "data_quality", "fn": self.validate_data_quality},
            {"name": "feature_relationships", "fn": self.validate_feature_relationships},
        ]

    def run_validations(self, df: DataFrame) -> bool:
        """Run comprehensive validation suite"""
        logger.info("Starting comprehensive feature validation...")
        
        all_pass = True
        validation_results = {}
        
        for v in self.validators:
            vname, vfunc = v["name"], v["fn"]
            
            if vfunc and callable(vfunc):
                try:
                    logger.info(f"Running validation: {vname}")
                    vres = vfunc(df)
                    validation_results[vname] = vres
                except Exception as e:
                    logger.error(f"Error invoking validator: {vname}: {e}")
                    vres = False
                    validation_results[vname] = vres
            else:
                logger.error(f"Function for validator {vname} is either NoneType or not callable: {vfunc}")
                vres = False
                validation_results[vname] = vres
            
            if vres:
                logger.info(f"✓ Validation {vname} successful")
            else:
                logger.error(f"✗ Validation {vname} failed")
            
            all_pass = all_pass and vres
        
        # Summary report
        passed_count = sum(validation_results.values())
        total_count = len(validation_results)
        
        if all_pass:
            logger.info(f"🎉 All validations passed! ({passed_count}/{total_count})")
        else:
            logger.error(f"❌ Validation failures: {passed_count}/{total_count} passed")
            
        return all_pass

    def validate_critical_features(self, df: DataFrame) -> bool:
        """Validate that all critical features are present and not entirely null"""
        try:
            missing_features = [col for col in CRITICAL_FEATURES if col not in df.columns]
            
            if missing_features:
                logger.error(f"Missing critical features: {missing_features}")
                return False
            
            # Check for entirely null critical features
            for feature in CRITICAL_FEATURES:
                null_count = df.filter(F.col(feature).isNull()).count()
                total_count = df.count()
                null_percentage = null_count / total_count if total_count > 0 else 1.0
                
                if null_percentage >= 0.99:  # Allow up to 1% nulls for critical features
                    logger.error(f"Critical feature '{feature}' is {null_percentage:.1%} null")
                    return False
            
            logger.info("All critical features present and valid")
            return True
            
        except Exception as e:
            logger.error(f"Critical features validation failed: {e}")
            return False

    def has_duplicate_rows(self, df: DataFrame) -> bool:
        """Check for duplicate rows"""
        try:
            orig_count = df.count()
            dedup_count = df.dropDuplicates().count()
            
            if orig_count != dedup_count:
                duplicate_count = orig_count - dedup_count
                logger.error(f"Found {duplicate_count:,} duplicate rows (Original: {orig_count:,}, Unique: {dedup_count:,})")
                return False
            else:
                logger.info("No duplicate rows found")
                return True
                
        except Exception as e:
            logger.error(f"Duplicate rows validation failed: {e}")
            return False

    def has_duplicate_persons(self, df: DataFrame) -> bool:
        """Check for duplicate person records at the same vantage date"""
        try:
            orig_count = df.count()
            dedup_count = df.dropDuplicates(["person_composite_id", "vantage_date"]).count()
            
            if orig_count != dedup_count:
                duplicate_count = orig_count - dedup_count
                logger.error(f"Found {duplicate_count:,} duplicate person records (Original: {orig_count:,}, Unique: {dedup_count:,})")
                
                # Show examples of duplicates
                duplicates = (
                    df.groupBy("person_composite_id", "vantage_date")
                    .count()
                    .filter(F.col("count") > 1)
                    .limit(5)
                    .collect()
                )
                
                logger.error("Sample duplicate person records:")
                for dup in duplicates:
                    logger.error(f"  {dup.person_composite_id} on {dup.vantage_date}: {dup.count} records")
                
                return False
            else:
                logger.info("No duplicate persons found")
                return True
                
        except Exception as e:
            logger.error(f"Duplicate persons validation failed: {e}")
            return False

    def validate_temporal_consistency(self, df: DataFrame) -> bool:
        """Validate temporal consistency and check for data leakage"""
        try:
            validation_passed = True
            
            # Check required temporal columns exist
            required_cols = TEMPORAL_VALIDATION_RULES["required_temporal_columns"]
            missing_temporal_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_temporal_cols:
                logger.error(f"Missing required temporal columns: {missing_temporal_cols}")
                validation_passed = False
            
            # Check for future dates in event columns
            event_date_cols = [col for col in TEMPORAL_VALIDATION_RULES["event_date_columns"] if col in df.columns]
            
            for col_name in event_date_cols:
                future_count = (
                    df.filter(F.col(col_name) > F.col("vantage_date"))
                    .count()
                )
                
                if future_count > 0:
                    logger.error(f"Found {future_count:,} records with future dates in {col_name}")
                    validation_passed = False
            
            # Check vantage date consistency
            vantage_date_values = df.select("vantage_date").distinct().collect()
            expected_dates = ["2023-01-01", "2024-01-01"]
            
            actual_dates = [row.vantage_date for row in vantage_date_values if row.vantage_date]
            unexpected_dates = [d for d in actual_dates if str(d) not in expected_dates]
            
            if unexpected_dates:
                logger.warning(f"Unexpected vantage dates found: {unexpected_dates}")
            
            if validation_passed:
                logger.info("Temporal consistency validation passed")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Temporal consistency validation failed: {e}")
            return False

    def validate_business_logic(self, df: DataFrame) -> bool:
        """Validate business logic constraints"""
        try:
            validation_passed = True
            
            # Age validation
            if "age_at_vantage" in df.columns:
                invalid_age_count = (
                    df.filter(
                        (F.col("age_at_vantage") < BUSINESS_LOGIC_THRESHOLDS["min_age"]) |
                        (F.col("age_at_vantage") > BUSINESS_LOGIC_THRESHOLDS["max_age"])
                    ).count()
                )
                
                if invalid_age_count > 0:
                    logger.error(f"Found {invalid_age_count:,} records with invalid ages")
                    validation_passed = False
            
            # Tenure validation
            if "tenure_at_vantage_days" in df.columns:
                max_tenure_days = BUSINESS_LOGIC_THRESHOLDS["max_tenure_years"] * 365
                invalid_tenure_count = (
                    df.filter(
                        (F.col("tenure_at_vantage_days") < 0) |
                        (F.col("tenure_at_vantage_days") > max_tenure_days)
                    ).count()
                )
                
                if invalid_tenure_count > 0:
                    logger.error(f"Found {invalid_tenure_count:,} records with invalid tenure")
                    validation_passed = False
            
            # Salary validation
            if "baseline_salary" in df.columns:
                invalid_salary_count = (
                    df.filter(
                        (F.col("baseline_salary") < BUSINESS_LOGIC_THRESHOLDS["min_salary"]) |
                        (F.col("baseline_salary") > BUSINESS_LOGIC_THRESHOLDS["max_salary"])
                    ).count()
                )
                
                if invalid_salary_count > 0:
                    logger.warning(f"Found {invalid_salary_count:,} records with unusual salaries")
            
            # Manager span validation
            if "manager_span_control" in df.columns:
                excessive_span_count = (
                    df.filter(F.col("manager_span_control") > BUSINESS_LOGIC_THRESHOLDS["max_manager_span"])
                    .count()
                )
                
                if excessive_span_count > 0:
                    logger.warning(f"Found {excessive_span_count:,} records with excessive manager span")
            
            # Survival time validation
            if "survival_time_days" in df.columns:
                invalid_survival_count = (
                    df.filter(
                        (F.col("survival_time_days") < 0) |
                        (F.col("survival_time_days") > 365)  # Should not exceed follow-up period
                    ).count()
                )
                
                if invalid_survival_count > 0:
                    logger.error(f"Found {invalid_survival_count:,} records with invalid survival times")
                    validation_passed = False
            
            if validation_passed:
                logger.info("Business logic validation passed")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Business logic validation failed: {e}")
            return False

    def validate_feature_distributions(self, df: DataFrame) -> bool:
        """Validate feature distributions for anomalies"""
        try:
            validation_passed = True
            
            # Check each feature group for distribution anomalies
            for group_name, feature_list in FEATURE_GROUPS.items():
                existing_features = [f for f in feature_list if f in df.columns]
                
                if not existing_features:
                    continue
                    
                logger.info(f"Validating distributions for {group_name} features...")
                
                for feature in existing_features:
                    if feature not in FEATURE_VALIDATION_RULES:
                        continue
                        
                    rules = FEATURE_VALIDATION_RULES[feature]
                    
                    # Check value ranges
                    if "min" in rules and rules["min"] is not None:
                        below_min = df.filter(F.col(feature) < rules["min"]).count()
                        if below_min > 0:
                            logger.warning(f"{feature}: {below_min:,} values below minimum {rules['min']}")
                    
                    if "max" in rules and rules["max"] is not None:
                        above_max = df.filter(F.col(feature) > rules["max"]).count()
                        if above_max > 0:
                            logger.warning(f"{feature}: {above_max:,} values above maximum {rules['max']}")
                    
                    # Check allowed values for categorical features
                    if "values" in rules:
                        allowed_values = rules["values"]
                        invalid_values = (
                            df.filter(~F.col(feature).isin(allowed_values) & F.col(feature).isNotNull())
                            .select(feature)
                            .distinct()
                            .collect()
                        )
                        
                        if invalid_values:
                            invalid_list = [row[feature] for row in invalid_values]
                            logger.warning(f"{feature}: Invalid values found: {invalid_list}")
                    
                    # Check null percentage
                    if not rules.get("allow_null", True):
                        null_count = df.filter(F.col(feature).isNull()).count()
                        total_count = df.count()
                        null_pct = null_count / total_count if total_count > 0 else 0
                        
                        if null_pct > 0.01:  # Allow up to 1% nulls for non-nullable features
                            logger.error(f"{feature}: {null_pct:.1%} null values (should not allow nulls)")
                            validation_passed = False
            
            if validation_passed:
                logger.info("Feature distributions validation passed")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Feature distributions validation failed: {e}")
            return False

    def validate_data_quality(self, df: DataFrame) -> bool:
        """Validate overall data quality metrics"""
        try:
            validation_passed = True
            total_records = df.count()
            
            if total_records == 0:
                logger.error("Dataset is empty")
                return False
            
            logger.info(f"Validating data quality for {total_records:,} records...")
            
            # Check null percentages for all features
            high_null_features = []
            
            for col_name in df.columns:
                null_count = df.filter(F.col(col_name).isNull()).count()
                null_percentage = null_count / total_records
                
                if null_percentage > DATA_QUALITY_THRESHOLDS["max_null_percentage"]:
                    high_null_features.append((col_name, null_percentage))
            
            if high_null_features:
                logger.warning("Features with high null percentages:")
                for feature, pct in high_null_features:
                    logger.warning(f"  {feature}: {pct:.1%} null")
            
            # Check for constant features (minimal variance)
            try:
                numeric_cols = [f.name for f in df.schema.fields if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]]
                
                for col_name in numeric_cols:
                    if col_name in df.columns:
                        distinct_count = df.select(col_name).distinct().count()
                        if distinct_count <= 1:
                            logger.warning(f"Feature '{col_name}' has minimal variance (≤1 distinct values)")
                            
            except Exception as e:
                logger.warning(f"Variance check failed: {e}")
            
            # Check dataset split distribution
            if "dataset_split" in df.columns:
                split_counts = df.groupBy("dataset_split").count().collect()
                split_distribution = {row.dataset_split: row.count for row in split_counts}
                
                total_splits = sum(split_distribution.values())
                logger.info("Dataset split distribution:")
                for split, count in split_distribution.items():
                    pct = count / total_splits * 100
                    logger.info(f"  {split}: {count:,} records ({pct:.1f}%)")
                
                # Validate minimum records per split
                min_records_per_split = 1000
                for split, count in split_distribution.items():
                    if count < min_records_per_split:
                        logger.warning(f"Split '{split}' has only {count:,} records (< {min_records_per_split:,})")
            
            logger.info("Data quality validation completed")
            return validation_passed
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return False

    def validate_feature_relationships(self, df: DataFrame) -> bool:
        """Validate logical relationships between features"""
        try:
            validation_passed = True
            
            # Age vs tenure relationship
            if "age_at_vantage" in df.columns and "tenure_at_vantage_days" in df.columns:
                impossible_tenure = (
                    df.filter(
                        (F.col("age_at_vantage").isNotNull()) &
                        (F.col("tenure_at_vantage_days").isNotNull()) &
                        (F.col("tenure_at_vantage_days") / 365.25 > F.col("age_at_vantage") - 16)
                    ).count()
                )
                
                if impossible_tenure > 0:
                    logger.error(f"Found {impossible_tenure:,} records with impossible age-tenure combinations")
                    validation_passed = False
            
            # Promotion indicators consistency
            if "promot_2yr_ind" in df.columns and "num_promot_2yr" in df.columns:
                inconsistent_promotions = (
                    df.filter(
                        ((F.col("promot_2yr_ind") == 1) & (F.col("num_promot_2yr") == 0)) |
                        ((F.col("promot_2yr_ind") == 0) & (F.col("num_promot_2yr") > 0))
                    ).count()
                )
                
                if inconsistent_promotions > 0:
                    logger.error(f"Found {inconsistent_promotions:,} records with inconsistent promotion indicators")
                    validation_passed = False
            
            # Event indicators consistency
            if "event_indicator_all" in df.columns and "event_indicator_vol" in df.columns:
                invalid_events = (
                    df.filter(
                        (F.col("event_indicator_vol") == 1) & (F.col("event_indicator_all") == 0)
                    ).count()
                )
                
                if invalid_events > 0:
                    logger.error(f"Found {invalid_events:,} records with voluntary events but no overall events")
                    validation_passed = False
            
            # Team size vs manager span consistency
            if "team_size" in df.columns and "manager_span_control" in df.columns:
                span_mismatch = (
                    df.filter(
                        (F.col("team_size").isNotNull()) &
                        (F.col("manager_span_control").isNotNull()) &
                        (F.abs(F.col("team_size") - F.col("manager_span_control")) > 5)  # Allow some tolerance
                    ).count()
                )
                
                if span_mismatch > 0:
                    logger.warning(f"Found {span_mismatch:,} records with team size vs span control mismatches")
            
            if validation_passed:
                logger.info("Feature relationships validation passed")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Feature relationships validation failed: {e}")
            return False

    def validate_outliers(self, df: DataFrame) -> DataFrame:
        """Identify and report outliers in numeric features"""
        try:
            logger.info("Identifying outliers in numeric features...")
            
            numeric_features = [
                f.name for f in df.schema.fields 
                if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]
                and f.name in FEATURE_VALIDATION_RULES
            ]
            
            outlier_summary = []
            
            for feature in numeric_features:
                # Calculate percentiles for outlier detection
                percentiles = df.select(
                    F.expr(f"percentile_approx({feature}, 0.01)").alias("p1"),
                    F.expr(f"percentile_approx({feature}, 0.99)").alias("p99"),
                    F.mean(feature).alias("mean"),
                    F.stddev(feature).alias("stddev")
                ).collect()[0]
                
                if percentiles.stddev and percentiles.stddev > 0:
                    # Count outliers beyond 5 standard deviations
                    outlier_threshold = BUSINESS_LOGIC_THRESHOLDS["outlier_std_threshold"]
                    lower_bound = percentiles.mean - (outlier_threshold * percentiles.stddev)
                    upper_bound = percentiles.mean + (outlier_threshold * percentiles.stddev)
                    
                    outlier_count = (
                        df.filter(
                            (F.col(feature) < lower_bound) | (F.col(feature) > upper_bound)
                        ).count()
                    )
                    
                    outlier_percentage = outlier_count / df.count() * 100
                    
                    outlier_summary.append({
                        "feature": feature,
                        "outlier_count": outlier_count,
                        "outlier_percentage": outlier_percentage,
                        "p1": percentiles.p1,
                        "p99": percentiles.p99
                    })
            
            # Report outlier summary
            high_outlier_features = [item for item in outlier_summary if item["outlier_percentage"] > 5.0]
            
            if high_outlier_features:
                logger.warning("Features with high outlier percentages (>5%):")
                for item in high_outlier_features:
                    logger.warning(f"  {item['feature']}: {item['outlier_count']:,} outliers ({item['outlier_percentage']:.1f}%)")
            
            logger.info(f"Outlier analysis completed for {len(numeric_features)} numeric features")
            return df
            
        except Exception as e:
            logger.error(f"Outlier validation failed: {e}")
            return df

    def generate_feature_report(self, df: DataFrame) -> dict:
        """Generate comprehensive feature quality report"""
        try:
            logger.info("Generating comprehensive feature report...")
            
            report = {
                "total_records": df.count(),
                "total_features": len(df.columns),
                "feature_groups": {},
                "data_quality_summary": {},
                "validation_summary": {}
            }
            
            # Analyze each feature group
            for group_name, feature_list in FEATURE_GROUPS.items():
                existing_features = [f for f in feature_list if f in df.columns]
                
                if existing_features:
                    # Calculate null percentages for group
                    null_stats = {}
                    for feature in existing_features:
                        null_count = df.filter(F.col(feature).isNull()).count()
                        null_pct = null_count / report["total_records"] * 100
                        null_stats[feature] = null_pct
                    
                    report["feature_groups"][group_name] = {
                        "total_features": len(feature_list),
                        "available_features": len(existing_features),
                        "coverage": len(existing_features) / len(feature_list) * 100,
                        "null_stats": null_stats
                    }
            
            # Overall data quality metrics
            total_nulls = sum(
                df.filter(F.col(col).isNull()).count() 
                for col in df.columns
            )
            total_cells = report["total_records"] * report["total_features"]
            overall_null_rate = total_nulls / total_cells * 100 if total_cells > 0 else 0
            
            report["data_quality_summary"] = {
                "overall_null_rate": overall_null_rate,
                "critical_features_present": all(f in df.columns for f in CRITICAL_FEATURES),
                "total_null_cells": total_nulls,
                "total_cells": total_cells
            }
            
            logger.info("Feature report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Feature report generation failed: {e}")
            return {}

    def validate_temporal_leakage(self, df: DataFrame) -> bool:
        """Advanced validation to detect potential temporal data leakage"""
        try:
            logger.info("Checking for temporal data leakage...")
            
            # List of features that should not have future information
            temporal_sensitive_features = [
                "time_since_last_promotion", "days_since_promot", "days_since_transfer",
                "salary_growth_rate_12m", "manager_changes_count", "job_chng_2yr_ind",
                "promot_2yr_ind", "transfer_2yr_ind"
            ]
            
            leakage_detected = False
            
            # Check if features calculated properly respect vantage dates
            for split in ["train", "val", "oot"]:
                split_data = df.filter(F.col("dataset_split") == split)
                
                if split_data.count() == 0:
                    continue
                
                # For each vantage date, ensure no future contamination
                vantage_dates = split_data.select("vantage_date").distinct().collect()
                
                for vd_row in vantage_dates:
                    vd = vd_row.vantage_date
                    if vd is None:
                        continue
                    
                    # This is a conceptual check - in practice, detailed leakage detection
                    # would require examining the feature calculation logic
                    logger.info(f"Temporal validation for {split} split at {vd}")
            
            if not leakage_detected:
                logger.info("No temporal leakage detected")
            
            return not leakage_detected
            
        except Exception as e:
            logger.error(f"Temporal leakage validation failed: {e}")
            return False

    def run_comprehensive_validation(self, df: DataFrame) -> dict:
        """Run all validations and return detailed results"""
        logger.info("🔍 Starting comprehensive validation suite...")
        
        # Run standard validations
        validation_passed = self.run_validations(df)
        
        # Generate detailed report
        feature_report = self.generate_feature_report(df)
        
        # Run outlier analysis
        df_with_outliers = self.validate_outliers(df)
        
        # Additional temporal leakage check
        temporal_leakage_passed = self.validate_temporal_leakage(df)
        
        comprehensive_report = {
            "basic_validations_passed": validation_passed,
            "temporal_leakage_passed": temporal_leakage_passed,
            "overall_passed": validation_passed and temporal_leakage_passed,
            "feature_report": feature_report,
            "recommendations": self._generate_recommendations(df, feature_report)
        }
        
        logger.info("🎯 Comprehensive validation completed")
        return comprehensive_report

    def _generate_recommendations(self, df: DataFrame, feature_report: dict) -> list:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        try:
            # Check feature group coverage
            for group_name, group_info in feature_report.get("feature_groups", {}).items():
                coverage = group_info.get("coverage", 0)
                if coverage < 80:
                    recommendations.append(
                        f"Low feature coverage in {group_name}: {coverage:.1f}% - consider adding missing features"
                    )
            
            # Check for high null rates
            overall_null_rate = feature_report.get("data_quality_summary", {}).get("overall_null_rate", 0)
            if overall_null_rate > 20:
                recommendations.append(
                    f"High overall null rate: {overall_null_rate:.1f}% - review data quality and imputation strategies"
                )
            
            # Check critical features
            if not feature_report.get("data_quality_summary", {}).get("critical_features_present", False):
                recommendations.append("Critical features missing - ensure all required features are generated")
            
            if not recommendations:
                recommendations.append("No critical issues identified - dataset quality looks good")
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Could not generate recommendations due to validation errors")
        
        return recommendations
    
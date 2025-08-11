import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

from top_etl.feature_engg.constants import (
    FEATURE_GROUPS,
    FEATURE_VALIDATION_RULES,
    BUSINESS_LOGIC_THRESHOLDS,
    DATA_QUALITY_THRESHOLDS
)

logger = logging.getLogger(__name__)


class FeatureEDD:
    """Enhanced Exploratory Data Description for feature validation and profiling"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def generate_feature_edd(self, df: DataFrame) -> Dict:
        """
        Generate comprehensive EDD report for feature-engineered dataset
        
        Args:
            df: Feature-engineered DataFrame
            
        Returns:
            Dictionary containing comprehensive EDD results
        """
        logger.info("🔍 Starting comprehensive feature EDD generation...")
        
        try:
            edd_results = {
                "dataset_overview": self._get_dataset_overview(df),
                "feature_group_analysis": self._analyze_feature_groups(df),
                "data_quality_metrics": self._calculate_data_quality_metrics(df),
                "distribution_analysis": self._analyze_feature_distributions(df),
                "temporal_analysis": self._analyze_temporal_patterns(df),
                "business_logic_checks": self._validate_business_logic_edd(df),
                "feature_completeness": self._analyze_feature_completeness(df),
                "outlier_analysis": self._analyze_outliers(df),
                "correlation_analysis": self._analyze_correlations(df),
            }
            
            # Generate summary report
            self._print_edd_summary(edd_results)
            
            logger.info("✅ Feature EDD generation completed successfully")
            return edd_results
            
        except Exception as e:
            logger.error(f"❌ Feature EDD generation failed: {e}")
            return {}

    def _get_dataset_overview(self, df: DataFrame) -> Dict:
        """Get basic dataset overview statistics"""
        try:
            total_records = df.count()
            total_features = len(df.columns)
            
            # Get dataset split distribution
            split_distribution = {}
            if "dataset_split" in df.columns:
                splits = df.groupBy("dataset_split").count().collect()
                split_distribution = {row.dataset_split: row.count for row in splits}
            
            # Get vantage date distribution
            vantage_distribution = {}
            if "vantage_date" in df.columns:
                vantages = df.groupBy("vantage_date").count().collect()
                vantage_distribution = {str(row.vantage_date): row.count for row in vantages}
            
            overview = {
                "total_records": total_records,
                "total_features": total_features,
                "unique_persons": df.select("person_composite_id").distinct().count() if "person_composite_id" in df.columns else 0,
                "split_distribution": split_distribution,
                "vantage_distribution": vantage_distribution,
                "schema_summary": {f.name: str(f.dataType) for f in df.schema.fields}
            }
            
            logger.info(f"Dataset overview: {total_records:,} records, {total_features} features")
            return overview
            
        except Exception as e:
            logger.error(f"Failed to get dataset overview: {e}")
            return {}

    def _analyze_feature_groups(self, df: DataFrame) -> Dict:
        """Analyze feature availability and quality by group"""
        try:
            group_analysis = {}
            
            for group_name, feature_list in FEATURE_GROUPS.items():
                available_features = [f for f in feature_list if f in df.columns]
                missing_features = [f for f in feature_list if f not in df.columns]
                
                # Calculate null rates for available features
                null_rates = {}
                for feature in available_features:
                    null_count = df.filter(F.col(feature).isNull()).count()
                    null_rate = null_count / df.count() * 100
                    null_rates[feature] = null_rate
                
                group_analysis[group_name] = {
                    "total_expected": len(feature_list),
                    "available": len(available_features),
                    "missing": len(missing_features),
                    "coverage_percentage": len(available_features) / len(feature_list) * 100,
                    "available_features": available_features,
                    "missing_features": missing_features,
                    "null_rates": null_rates,
                    "avg_null_rate": sum(null_rates.values()) / len(null_rates) if null_rates else 0
                }
            
            logger.info("Feature group analysis completed")
            return group_analysis
            
        except Exception as e:
            logger.error(f"Feature group analysis failed: {e}")
            return {}

    def _calculate_data_quality_metrics(self, df: DataFrame) -> Dict:
        """Calculate comprehensive data quality metrics"""
        try:
            total_records = df.count()
            total_features = len(df.columns)
            total_cells = total_records * total_features
            
            # Calculate overall null rate
            total_nulls = 0
            feature_null_rates = {}
            
            for col_name in df.columns:
                null_count = df.filter(F.col(col_name).isNull()).count()
                null_rate = null_count / total_records * 100
                feature_null_rates[col_name] = null_rate
                total_nulls += null_count
            
            overall_null_rate = total_nulls / total_cells * 100
            
            # Identify problem features
            high_null_features = {k: v for k, v in feature_null_rates.items() if v > 50}
            zero_variance_features = []
            
            # Check for zero variance in numeric features
            numeric_features = [
                f.name for f in df.schema.fields 
                if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]
            ]
            
            for feature in numeric_features:
                if feature in df.columns:
                    distinct_count = df.select(feature).distinct().count()
                    if distinct_count <= 1:
                        zero_variance_features.append(feature)
            
            quality_metrics = {
                "overall_null_rate": overall_null_rate,
                "feature_null_rates": feature_null_rates,
                "high_null_features": high_null_features,
                "zero_variance_features": zero_variance_features,
                "total_cells": total_cells,
                "total_nulls": total_nulls,
                "quality_score": 100 - overall_null_rate  # Simple quality score
            }
            
            logger.info(f"Data quality metrics: {overall_null_rate:.2f}% null rate")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Data quality metrics calculation failed: {e}")
            return {}

    def _analyze_feature_distributions(self, df: DataFrame) -> Dict:
        """Analyze feature distributions and identify anomalies"""
        try:
            logger.info("Analyzing feature distributions...")
            
            distribution_analysis = {}
            
            # Analyze numeric features
            numeric_features = [
                f.name for f in df.schema.fields 
                if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]
            ]
            
            for feature in numeric_features[:20]:  # Limit to first 20 for performance
                if feature in df.columns:
                    try:
                        stats = df.select(feature).describe().collect()
                        percentiles = df.select(
                            F.expr(f"percentile_approx({feature}, array(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))").alias("percentiles")
                        ).collect()[0].percentiles
                        
                        distribution_analysis[feature] = {
                            "count": int(stats[0][feature]) if stats[0][feature] else 0,
                            "mean": float(stats[1][feature]) if stats[1][feature] else 0,
                            "stddev": float(stats[2][feature]) if stats[2][feature] else 0,
                            "min": float(stats[3][feature]) if stats[3][feature] else 0,
                            "max": float(stats[4][feature]) if stats[4][feature] else 0,
                            "percentiles": {
                                "p1": percentiles[0], "p5": percentiles[1], "p25": percentiles[2],
                                "p50": percentiles[3], "p75": percentiles[4], "p95": percentiles[5], "p99": percentiles[6]
                            } if percentiles else {}
                        }
                    except Exception as e:
                        logger.warning(f"Could not analyze distribution for {feature}: {e}")
                        distribution_analysis[feature] = {"error": str(e)}
            
            # Analyze categorical features
            categorical_features = [
                f.name for f in df.schema.fields 
                if f.dataType == StringType()
            ]
            
            for feature in categorical_features[:10]:  # Limit to first 10 for performance
                if feature in df.columns:
                    try:
                        value_counts = (
                            df.groupBy(feature)
                            .count()
                            .orderBy(F.desc("count"))
                            .limit(10)
                            .collect()
                        )
                        
                        distribution_analysis[feature] = {
                            "type": "categorical",
                            "unique_values": df.select(feature).distinct().count(),
                            "top_values": [(row[feature], row.count) for row in value_counts]
                        }
                    except Exception as e:
                        logger.warning(f"Could not analyze categorical distribution for {feature}: {e}")
                        distribution_analysis[feature] = {"error": str(e)}
            
            logger.info(f"Distribution analysis completed for {len(distribution_analysis)} features")
            return distribution_analysis
            
        except Exception as e:
            logger.error(f"Feature distribution analysis failed: {e}")
            return {}

    def _analyze_temporal_patterns(self, df: DataFrame) -> Dict:
        """Analyze temporal patterns in the dataset"""
        try:
            temporal_analysis = {}
            
            if "vantage_date" in df.columns and "dataset_split" in df.columns:
                # Analyze records by vantage date and split
                temporal_dist = (
                    df.groupBy("vantage_date", "dataset_split")
                    .count()
                    .orderBy("vantage_date", "dataset_split")
                    .collect()
                )
                
                temporal_analysis["vantage_split_distribution"] = [
                    {"vantage_date": str(row.vantage_date), "split": row.dataset_split, "count": row.count}
                    for row in temporal_dist
                ]
            
            # Analyze survival time distributions
            if "survival_time_days" in df.columns:
                survival_stats = df.select("survival_time_days").describe().collect()
                temporal_analysis["survival_time_stats"] = {
                    "count": int(survival_stats[0]["survival_time_days"]) if survival_stats[0]["survival_time_days"] else 0,
                    "mean": float(survival_stats[1]["survival_time_days"]) if survival_stats[1]["survival_time_days"] else 0,
                    "stddev": float(survival_stats[2]["survival_time_days"]) if survival_stats[2]["survival_time_days"] else 0,
                    "min": float(survival_stats[3]["survival_time_days"]) if survival_stats[3]["survival_time_days"] else 0,
                    "max": float(survival_stats[4]["survival_time_days"]) if survival_stats[4]["survival_time_days"] else 0,
                }
            
            # Analyze event indicators
            if "event_indicator_all" in df.columns and "event_indicator_vol" in df.columns:
                event_summary = (
                    df.groupBy("event_indicator_all", "event_indicator_vol")
                    .count()
                    .collect()
                )
                
                temporal_analysis["event_distribution"] = [
                    {"all_events": row.event_indicator_all, "vol_events": row.event_indicator_vol, "count": row.count}
                    for row in event_summary
                ]
            
            logger.info("Temporal pattern analysis completed")
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {}

    def _validate_business_logic_edd(self, df: DataFrame) -> Dict:
        """Validate business logic constraints for EDD"""
        try:
            business_validation = {}
            
            # Age validation
            if "age_at_vantage" in df.columns:
                age_issues = (
                    df.filter(
                        (F.col("age_at_vantage") < BUSINESS_LOGIC_THRESHOLDS["min_age"]) |
                        (F.col("age_at_vantage") > BUSINESS_LOGIC_THRESHOLDS["max_age"])
                    ).count()
                )
                business_validation["invalid_ages"] = age_issues
            
            # Salary validation
            if "baseline_salary" in df.columns:
                salary_issues = (
                    df.filter(
                        (F.col("baseline_salary") < BUSINESS_LOGIC_THRESHOLDS["min_salary"]) |
                        (F.col("baseline_salary") > BUSINESS_LOGIC_THRESHOLDS["max_salary"])
                    ).count()
                )
                business_validation["invalid_salaries"] = salary_issues
            
            # Tenure validation
            if "tenure_at_vantage_days" in df.columns:
                max_tenure_days = BUSINESS_LOGIC_THRESHOLDS["max_tenure_years"] * 365
                tenure_issues = (
                    df.filter(
                        (F.col("tenure_at_vantage_days") < 0) |
                        (F.col("tenure_at_vantage_days") > max_tenure_days)
                    ).count()
                )
                business_validation["invalid_tenure"] = tenure_issues
            
            # Manager span validation
            if "manager_span_control" in df.columns:
                excessive_span = (
                    df.filter(F.col("manager_span_control") > BUSINESS_LOGIC_THRESHOLDS["max_manager_span"])
                    .count()
                )
                business_validation["excessive_manager_span"] = excessive_span
            
            logger.info("Business logic validation for EDD completed")
            return business_validation
            
        except Exception as e:
            logger.error(f"Business logic EDD validation failed: {e}")
            return {}

    def _analyze_feature_completeness(self, df: DataFrame) -> Dict:
        """Analyze feature completeness across different dimensions"""
        try:
            completeness_analysis = {}
            
            # Overall completeness
            total_records = df.count()
            
            # Completeness by dataset split
            if "dataset_split" in df.columns:
                split_completeness = {}
                for split in ["train", "val", "oot"]:
                    split_df = df.filter(F.col("dataset_split") == split)
                    split_count = split_df.count()
                    
                    if split_count > 0:
                        split_features = {}
                        for col_name in df.columns:
                            non_null_count = split_df.filter(F.col(col_name).isNotNull()).count()
                            completeness_pct = non_null_count / split_count * 100
                            split_features[col_name] = completeness_pct
                        
                        split_completeness[split] = {
                            "record_count": split_count,
                            "feature_completeness": split_features,
                            "avg_completeness": sum(split_features.values()) / len(split_features)
                        }
                
                completeness_analysis["by_split"] = split_completeness
            
            # Completeness by vantage date
            if "vantage_date" in df.columns:
                vantage_completeness = {}
                vantage_dates = [row.vantage_date for row in df.select("vantage_date").distinct().collect()]
                
                for vd in vantage_dates:
                    if vd:
                        vd_df = df.filter(F.col("vantage_date") == F.lit(str(vd)))
                        vd_count = vd_df.count()
                        
                        if vd_count > 0:
                            vd_features = {}
                            for col_name in df.columns:
                                non_null_count = vd_df.filter(F.col(col_name).isNotNull()).count()
                                completeness_pct = non_null_count / vd_count * 100
                                vd_features[col_name] = completeness_pct
                            
                            vantage_completeness[str(vd)] = {
                                "record_count": vd_count,
                                "feature_completeness": vd_features,
                                "avg_completeness": sum(vd_features.values()) / len(vd_features)
                            }
                
                completeness_analysis["by_vantage_date"] = vantage_completeness
            
            logger.info("Feature completeness analysis completed")
            return completeness_analysis
            
        except Exception as e:
            logger.error(f"Feature completeness analysis failed: {e}")
            return {}

    def _analyze_outliers(self, df: DataFrame) -> Dict:
        """Analyze outliers in numeric features"""
        try:
            outlier_analysis = {}
            
            numeric_features = [
                f.name for f in df.schema.fields 
                if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]
                and f.name in FEATURE_VALIDATION_RULES
            ]
            
            for feature in numeric_features:
                try:
                    # Calculate IQR-based outliers
                    percentiles = df.select(
                        F.expr(f"percentile_approx({feature}, 0.25)").alias("q1"),
                        F.expr(f"percentile_approx({feature}, 0.75)").alias("q3"),
                        F.mean(feature).alias("mean"),
                        F.stddev(feature).alias("stddev")
                    ).collect()[0]
                    
                    if percentiles.q1 is not None and percentiles.q3 is not None:
                        iqr = percentiles.q3 - percentiles.q1
                        lower_bound = percentiles.q1 - 1.5 * iqr
                        upper_bound = percentiles.q3 + 1.5 * iqr
                        
                        outlier_count = (
                            df.filter(
                                (F.col(feature) < lower_bound) | (F.col(feature) > upper_bound)
                            ).count()
                        )
                        
                        outlier_percentage = outlier_count / df.count() * 100
                        
                        outlier_analysis[feature] = {
                            "outlier_count": outlier_count,
                            "outlier_percentage": outlier_percentage,
                            "q1": percentiles.q1,
                            "q3": percentiles.q3,
                            "iqr": iqr,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "mean": percentiles.mean,
                            "stddev": percentiles.stddev
                        }
                
                except Exception as e:
                    logger.warning(f"Could not analyze outliers for {feature}: {e}")
                    outlier_analysis[feature] = {"error": str(e)}
            
            logger.info(f"Outlier analysis completed for {len(outlier_analysis)} features")
            return outlier_analysis
            
        except Exception as e:
            logger.error(f"Outlier analysis failed: {e}")
            return {}

    def _analyze_correlations(self, df: DataFrame) -> Dict:
        """Analyze correlations between numeric features"""
        try:
            logger.info("Analyzing feature correlations...")
            
            # Select numeric features for correlation analysis
            numeric_features = [
                f.name for f in df.schema.fields 
                if f.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]
                and f.name not in ["person_composite_id"]  # Exclude ID columns
            ]
            
            # Limit to first 20 features for performance
            correlation_features = numeric_features[:20]
            
            if len(correlation_features) < 2:
                return {"message": "Insufficient numeric features for correlation analysis"}
            
            # Calculate correlation matrix (simplified version)
            high_correlations = []
            
            for i, feat1 in enumerate(correlation_features):
                for feat2 in correlation_features[i+1:]:
                    try:
                        corr_result = df.select(F.corr(feat1, feat2).alias("correlation")).collect()
                        if corr_result and corr_result[0].correlation is not None:
                            correlation = abs(corr_result[0].correlation)
                            if correlation > 0.8:  # High correlation threshold
                                high_correlations.append({
                                    "feature1": feat1,
                                    "feature2": feat2,
                                    "correlation": correlation
                                })
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation between {feat1} and {feat2}: {e}")
            
            correlation_analysis = {
                "features_analyzed": len(correlation_features),
                "high_correlations": high_correlations,
                "high_correlation_count": len(high_correlations)
            }
            
            logger.info(f"Correlation analysis completed: {len(high_correlations)} high correlations found")
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {}

    def _print_edd_summary(self, edd_results: Dict) -> None:
        """Print comprehensive EDD summary"""
        try:
            print("\n" + "="*80)
            print("📊 COMPREHENSIVE FEATURE EDD SUMMARY")
            print("="*80)
            
            # Dataset Overview
            overview = edd_results.get("dataset_overview", {})
            print(f"\n📈 DATASET OVERVIEW:")
            print(f"  Total Records: {overview.get('total_records', 0):,}")
            print(f"  Total Features: {overview.get('total_features', 0)}")
            print(f"  Unique Persons: {overview.get('unique_persons', 0):,}")
            
            if overview.get('split_distribution'):
                print(f"  Split Distribution:")
                for split, count in overview['split_distribution'].items():
                    pct = count / overview.get('total_records', 1) * 100
                    print(f"    {split}: {count:,} ({pct:.1f}%)")
            
            # Feature Group Analysis
            group_analysis = edd_results.get("feature_group_analysis", {})
            print(f"\n🎯 FEATURE GROUP COVERAGE:")
            for group_name, group_info in group_analysis.items():
                coverage = group_info.get("coverage_percentage", 0)
                available = group_info.get("available", 0)
                total = group_info.get("total_expected", 0)
                avg_null = group_info.get("avg_null_rate", 0)
                
                status = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
                print(f"  {status} {group_name}: {available}/{total} features ({coverage:.1f}%) - Avg Null: {avg_null:.1f}%")
                
                if group_info.get("missing_features"):
                    missing = group_info["missing_features"][:3]  # Show first 3
                    more = len(group_info["missing_features"]) - 3
                    missing_str = ", ".join(missing)
                    if more > 0:
                        missing_str += f", +{more} more"
                    print(f"      Missing: {missing_str}")
            
            # Data Quality Summary
            quality = edd_results.get("data_quality_metrics", {})
            print(f"\n🔍 DATA QUALITY SUMMARY:")
            print(f"  Overall Null Rate: {quality.get('overall_null_rate', 0):.2f}%")
            print(f"  Quality Score: {quality.get('quality_score', 0):.1f}/100")
            
            if quality.get('high_null_features'):
                print(f"  High Null Features ({len(quality['high_null_features'])}): ", end="")
                high_null_names = list(quality['high_null_features'].keys())[:3]
                print(", ".join(high_null_names))
            
            if quality.get('zero_variance_features'):
                print(f"  Zero Variance Features: {len(quality['zero_variance_features'])}")
            
            # Business Logic Issues
            business_logic = edd_results.get("business_logic_checks", {})
            print(f"\n⚖️ BUSINESS LOGIC VALIDATION:")
            total_issues = sum(v for v in business_logic.values() if isinstance(v, int))
            if total_issues == 0:
                print("  ✅ No business logic violations detected")
            else:
                print(f"  ⚠️ Found {total_issues:,} business logic issues:")
                for check, count in business_logic.items():
                    if isinstance(count, int) and count > 0:
                        print(f"    {check}: {count:,} records")
            
            # Correlation Analysis
            correlations = edd_results.get("correlation_analysis", {})
            high_corr_count = correlations.get("high_correlation_count", 0)
            if high_corr_count > 0:
                print(f"\n🔗 HIGH CORRELATIONS DETECTED: {high_corr_count}")
                for corr in correlations.get("high_correlations", [])[:5]:
                    print(f"    {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Failed to print EDD summary: {e}")

    def export_edd_report(self, edd_results: Dict, output_path: str = None) -> bool:
        """Export EDD results to file for further analysis"""
        try:
            if output_path:
                import json
                with open(output_path, 'w') as f:
                    # Convert to JSON-serializable format
                    serializable_results = self._make_json_serializable(edd_results)
                    json.dump(serializable_results, f, indent=2, default=str)
                
                logger.info(f"EDD report exported to: {output_path}")
                return True
            else:
                logger.info("No output path specified, skipping export")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export EDD report: {e}")
            return False

    def _make_json_serializable(self, data) -> any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif hasattr(data, 'isoformat'):  # datetime objects
            return data.isoformat()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def create_feature_profile_report(self, df: DataFrame) -> DataFrame:
        """Create detailed feature profile report as Spark DataFrame"""
        try:
            logger.info("Creating detailed feature profile report...")
            
            feature_profiles = []
            
            for col_name in df.columns:
                col_type = str(df.schema[col_name].dataType)
                
                # Basic statistics
                total_count = df.count()
                null_count = df.filter(F.col(col_name).isNull()).count()
                non_null_count = total_count - null_count
                null_percentage = null_count / total_count * 100 if total_count > 0 else 0
                
                profile = {
                    "feature_name": col_name,
                    "data_type": col_type,
                    "total_count": total_count,
                    "null_count": null_count,
                    "non_null_count": non_null_count,
                    "null_percentage": null_percentage,
                    "feature_group": self._get_feature_group(col_name),
                    "is_critical": col_name in CRITICAL_FEATURES,
                    "has_validation_rules": col_name in FEATURE_VALIDATION_RULES
                }
                
                # Add type-specific statistics
                if col_type in ["IntegerType", "DoubleType", "FloatType", "LongType"]:
                    try:
                        stats = df.select(col_name).describe().collect()
                        if len(stats) >= 5:
                            profile.update({
                                "min_value": float(stats[3][col_name]) if stats[3][col_name] else None,
                                "max_value": float(stats[4][col_name]) if stats[4][col_name] else None,
                                "mean_value": float(stats[1][col_name]) if stats[1][col_name] else None,
                                "stddev_value": float(stats[2][col_name]) if stats[2][col_name] else None
                            })
                    except:
                        pass
                
                elif col_type == "StringType":
                    try:
                        distinct_count = df.select(col_name).distinct().count()
                        profile["distinct_values"] = distinct_count
                        profile["cardinality"] = distinct_count / non_null_count if non_null_count > 0 else 0
                    except:
                        pass
                
                feature_profiles.append(profile)
            
            # Convert to Spark DataFrame
            profile_df = self.spark.createDataFrame(feature_profiles)
            
            logger.info(f"Feature profile report created with {len(feature_profiles)} feature profiles")
            return profile_df
            
        except Exception as e:
            logger.error(f"Feature profile report creation failed: {e}")
            return self.spark.createDataFrame([], StructType([
                StructField("feature_name", StringType(), True),
                StructField("error", StringType(), True)
            ]))

    def _get_feature_group(self, feature_name: str) -> str:
        """Determine which feature group a feature belongs to"""
        for group_name, feature_list in FEATURE_GROUPS.items():
            if feature_name in feature_list:
                return group_name
        return "other"

    def create_validation_dashboard_data(self, df: DataFrame) -> Dict[str, DataFrame]:
        """Create data for validation dashboard visualization"""
        try:
            logger.info("Creating validation dashboard data...")
            
            dashboard_data = {}
            
            # Feature completeness by group
            group_completeness = []
            for group_name, feature_list in FEATURE_GROUPS.items():
                available_features = [f for f in feature_list if f in df.columns]
                coverage = len(available_features) / len(feature_list) * 100
                
                group_completeness.append({
                    "feature_group": group_name,
                    "total_features": len(feature_list),
                    "available_features": len(available_features),
                    "coverage_percentage": coverage
                })
            
            dashboard_data["group_completeness"] = self.spark.createDataFrame(group_completeness)
            
            # Null rate distribution
            null_rates = []
            for col_name in df.columns:
                null_count = df.filter(F.col(col_name).isNull()).count()
                null_rate = null_count / df.count() * 100
                
                null_rates.append({
                    "feature_name": col_name,
                    "null_percentage": null_rate,
                    "feature_group": self._get_feature_group(col_name)
                })
            
            dashboard_data["null_rates"] = self.spark.createDataFrame(null_rates)
            
            # Dataset split metrics
            if "dataset_split" in df.columns:
                split_metrics = (
                    df.groupBy("dataset_split")
                    .agg(
                        F.count("*").alias("record_count"),
                        F.countDistinct("person_composite_id").alias("unique_persons")
                    )
                )
                dashboard_data["split_metrics"] = split_metrics
            
            logger.info("Validation dashboard data created successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Validation dashboard data creation failed: {e}")
            return {}
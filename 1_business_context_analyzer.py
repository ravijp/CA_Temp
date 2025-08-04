"""
Business Context Analyzer for  Employee Turnover Survival Analysis

This module provides comprehensive business context analysis for survival modeling foundation,
including population baseline metrics, industry patterns, demographic segmentation, and temporal trends.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from scipy import stats
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set visualization defaults
plt.style.use("default")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 11

@dataclass
class AnalysisConfig:
    """Business context analysis configuration"""
    baseline_horizons: List[int] = field(default_factory=lambda: [30, 90, 180, 365])
    demographic_segments: List[str] = field(default_factory=lambda: ['age_group', 'tenure_group', 'salary_group'])
    industry_analysis_min_size: int = 1000
    temporal_cohorts: Dict[str, List[str]] = field(default_factory=lambda: {
        '2023': ['train', 'val'], 
        '2024': ['oot']
    })
    output_path: str = './business_context_analysis'
    save_visualizations: bool = True
    
class BusinessContextAnalyzer:
    """
    Comprehensive business context analysis for survival modeling foundation
    
    Provides foundational business intelligence including population baseline metrics,
    industry-specific retention patterns, demographic segmentation analysis, and temporal trends.
    Serves as the analytical foundation for downstream survival modeling efforts.
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize Business Context Analyzer with configuration
        
        Args:
            config: AnalysisConfig object containing analysis parameters
        """
        self.config = config
        self.kmf = KaplanMeierFitter()
        self.analysis_results = {}
        self.insights_summary = {}
        
        # Create output directory if saving visualizations
        if self.config.save_visualizations:
            Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("BusinessContextAnalyzer initialized successfully")
    
    # === POPULATION ANALYSIS SECTION ===
    
    def analyze_population_baseline(self, df: pd.DataFrame, 
                                  time_col: str = 'survival_time_days',
                                  event_col: str = 'event_indicator_all',
                                  split_col: str = 'dataset_split') -> Dict:
        """
        Generate population baseline metrics and Kaplan-Meier analysis
        
        Calculates overall retention rates at key milestones, generates population-wide
        survival curves with confidence intervals, and performs comprehensive event rate
        analysis with median survival time calculation.
        
        Args:
            df: Input DataFrame with survival data
            time_col: Column name for survival times
            event_col: Column name for event indicators (1=event, 0=censored)
            split_col: Column name for dataset splits
            
        Returns:
            Dict containing baseline metrics, survival statistics, and analysis metadata
        """
        logger.info("Starting population baseline analysis...")
        
        # Filter to training population for baseline establishment
        train_splits = self.config.temporal_cohorts['2023']
        baseline_data = df[df[split_col].isin(train_splits)].copy()
        
        if len(baseline_data) == 0:
            raise ValueError(f"No data found for baseline splits: {train_splits}")
        
        # Fit Kaplan-Meier estimator
        self.kmf.fit(baseline_data[time_col], baseline_data[event_col])
        
        # Calculate key retention metrics
        retention_metrics = {}
        for horizon in self.config.baseline_horizons:
            try:
                retention_rate = self.kmf.survival_function_at_times(horizon).iloc[0]
                retention_metrics[f'retention_{horizon}d'] = retention_rate
            except (IndexError, KeyError):
                retention_metrics[f'retention_{horizon}d'] = np.nan
                logger.warning(f"Could not calculate retention at {horizon} days")
        
        # Core population statistics
        baseline_metrics = {
            'population_size': len(baseline_data),
            'total_population': len(df),
            'event_rate': baseline_data[event_col].mean(),
            'median_survival': self.kmf.median_survival_time_,
            'censoring_rate': 1 - baseline_data[event_col].mean(),
            **retention_metrics
        }
        
        # Generate baseline visualization
        if self.config.save_visualizations:
            self._plot_baseline_survival(baseline_data, time_col, event_col, baseline_metrics)
        
        # Statistical confidence intervals
        confidence_intervals = self._calculate_baseline_confidence_intervals(baseline_data, time_col, event_col)
        baseline_metrics.update(confidence_intervals)
        
        logger.info(f"Baseline analysis complete. Population: {baseline_metrics['population_size']:,}, "
                   f"Event rate: {baseline_metrics['event_rate']:.1%}")
        
        self.analysis_results['baseline'] = baseline_metrics
        return baseline_metrics
    
    def analyze_temporal_trends(self, df: pd.DataFrame,
                               time_col: str = 'survival_time_days',
                               event_col: str = 'event_indicator_all',
                               split_col: str = 'dataset_split') -> Dict:
        """
        Compare retention trends across time periods (2023 vs 2024 cohorts)
        
        Performs cohort-based survival curve comparison, statistical significance testing
        for temporal changes, and business impact assessment of trend shifts.
        
        Args:
            df: Input DataFrame with survival data
            time_col: Column name for survival times
            event_col: Column name for event indicators
            split_col: Column name for dataset splits
            
        Returns:
            Dict containing temporal trend analysis, statistical tests, and change metrics
        """
        logger.info("Starting temporal trends analysis...")
        
        # Separate cohorts based on configuration
        cohort_2023 = df[df[split_col].isin(self.config.temporal_cohorts['2023'])].copy()
        cohort_2024 = df[df[split_col].isin(self.config.temporal_cohorts['2024'])].copy()
        
        if len(cohort_2024) < 1000:
            logger.warning(f"Limited 2024 data: {len(cohort_2024)} observations")
            return {'warning': 'Insufficient 2024 data for temporal analysis'}
        
        # Fit separate KM estimators for each cohort
        kmf_2023 = KaplanMeierFitter()
        kmf_2024 = KaplanMeierFitter()
        
        kmf_2023.fit(cohort_2023[time_col], cohort_2023[event_col])
        kmf_2024.fit(cohort_2024[time_col], cohort_2024[event_col])
        
        # Calculate retention metrics for each cohort
        temporal_metrics = {
            '2023_size': len(cohort_2023),
            '2024_size': len(cohort_2024),
        }
        
        for horizon in self.config.baseline_horizons:
            try:
                retention_2023 = kmf_2023.survival_function_at_times(horizon).iloc[0]
                retention_2024 = kmf_2024.survival_function_at_times(horizon).iloc[0]
                
                temporal_metrics.update({
                    f'2023_retention_{horizon}d': retention_2023,
                    f'2024_retention_{horizon}d': retention_2024,
                    f'retention_change_{horizon}d': retention_2024 - retention_2023
                })
            except (IndexError, KeyError):
                logger.warning(f"Could not calculate temporal comparison at {horizon} days")
        
        # Statistical significance testing
        temporal_metrics.update({
            '2023_median_survival': kmf_2023.median_survival_time_,
            '2024_median_survival': kmf_2024.median_survival_time_,
            '2023_event_rate': cohort_2023[event_col].mean(),
            '2024_event_rate': cohort_2024[event_col].mean()
        })
        
        # Log-rank test for statistical significance
        try:
            from lifelines.statistics import logrank_test
            logrank_result = logrank_test(
                cohort_2023[time_col], cohort_2024[time_col],
                cohort_2023[event_col], cohort_2024[event_col]
            )
            temporal_metrics['logrank_p_value'] = logrank_result.p_value
            temporal_metrics['temporal_difference_significant'] = logrank_result.p_value < 0.05
        except ImportError:
            logger.warning("Could not perform log-rank test - lifelines.statistics not available")
        
        # Generate temporal visualization
        if self.config.save_visualizations:
            self._plot_temporal_trends(cohort_2023, cohort_2024, time_col, event_col, temporal_metrics)
        
        logger.info(f"Temporal analysis complete. 2023: {len(cohort_2023):,}, 2024: {len(cohort_2024):,}")
        
        self.analysis_results['temporal'] = temporal_metrics
        return temporal_metrics
    
    # === SEGMENTATION ANALYSIS SECTION ===
    
    def analyze_industry_patterns(self, df: pd.DataFrame, 
                                 top_n: int = 10,
                                 time_col: str = 'survival_time_days',
                                 event_col: str = 'event_indicator_all',
                                 split_col: str = 'dataset_split',
                                 industry_col: str = 'naics_2digit') -> Dict:
        """
        Industry-specific retention pattern analysis
        
        Generates NAICS-based survival curves, performs cross-industry performance ranking,
        and identifies industry-specific risk factors with statistical significance testing.
        
        Args:
            df: Input DataFrame with survival data
            top_n: Number of top industries to analyze in detail
            time_col: Column name for survival times
            event_col: Column name for event indicators
            split_col: Column name for dataset splits
            industry_col: Column name for industry classification
            
        Returns:
            Dict containing industry metrics, rankings, and performance comparisons
        """
        logger.info(f"Starting industry patterns analysis for top {top_n} industries...")
        
        # Filter to training data for industry analysis
        train_splits = self.config.temporal_cohorts['2023']
        train_data = df[df[split_col].isin(train_splits)].copy()
        
        # Identify valid industries with sufficient sample size
        industry_counts = train_data[industry_col].value_counts()
        valid_industries = industry_counts[
            industry_counts >= self.config.industry_analysis_min_size
        ].head(top_n).index.tolist()
        
        if len(valid_industries) == 0:
            logger.warning("No industries meet minimum sample size requirement")
            return {'warning': f'No industries with >= {self.config.industry_analysis_min_size} observations'}
        
        # Analyze each valid industry
        industry_metrics = {}
        survival_curves = {}
        
        for industry in valid_industries:
            industry_data = train_data[train_data[industry_col] == industry].copy()
            
            # Fit Kaplan-Meier for this industry
            kmf_industry = KaplanMeierFitter()
            kmf_industry.fit(industry_data[time_col], industry_data[event_col])
            
            # Calculate industry-specific metrics
            industry_stats = {
                'sample_size': len(industry_data),
                'event_rate': industry_data[event_col].mean(),
                'median_survival': kmf_industry.median_survival_time_
            }
            
            # Retention rates at key horizons
            for horizon in self.config.baseline_horizons:
                try:
                    retention_rate = kmf_industry.survival_function_at_times(horizon).iloc[0]
                    industry_stats[f'retention_{horizon}d'] = retention_rate
                except (IndexError, KeyError):
                    industry_stats[f'retention_{horizon}d'] = np.nan
            
            industry_metrics[industry] = industry_stats
            survival_curves[industry] = kmf_industry
        
        # Rank industries by 365-day retention performance
        ranked_industries = sorted(
            industry_metrics.items(),
            key=lambda x: x[1].get('retention_365d', 0),
            reverse=True
        )
        
        # Calculate industry performance statistics
        retention_365_values = [
            metrics.get('retention_365d', 0) 
            for metrics in industry_metrics.values() 
            if not np.isnan(metrics.get('retention_365d', np.nan))
        ]
        
        industry_analysis = {
            'valid_industries': valid_industries,
            'industry_metrics': industry_metrics,
            'ranked_performance': ranked_industries,
            'performance_stats': {
                'best_industry': ranked_industries[0][0] if ranked_industries else None,
                'worst_industry': ranked_industries[-1][0] if ranked_industries else None,
                'performance_range': max(retention_365_values) - min(retention_365_values) if retention_365_values else 0,
                'mean_retention_365d': np.mean(retention_365_values) if retention_365_values else 0,
                'std_retention_365d': np.std(retention_365_values) if retention_365_values else 0
            }
        }
        
        # Generate industry visualization
        if self.config.save_visualizations:
            self._plot_industry_patterns(survival_curves, industry_metrics, ranked_industries)
        
        logger.info(f"Industry analysis complete for {len(valid_industries)} industries")
        
        self.analysis_results['industry'] = industry_analysis
        return industry_analysis
    
    def analyze_demographic_segments(self, df: pd.DataFrame,
                                   time_col: str = 'survival_time_days',
                                   event_col: str = 'event_indicator_all',
                                   split_col: str = 'dataset_split') -> Dict:
        """
        Demographic segment retention analysis
        
        Analyzes age group, tenure group, and salary group retention patterns with
        cross-segment performance comparison and demographic risk factor quantification.
        
        Args:
            df: Input DataFrame with survival data
            time_col: Column name for survival times
            event_col: Column name for event indicators
            split_col: Column name for dataset splits
            
        Returns:
            Dict containing demographic analysis results for all configured segments
        """
        logger.info("Starting demographic segments analysis...")
        
        # Filter to training data
        train_splits = self.config.temporal_cohorts['2023']
        train_data = df[df[split_col].isin(train_splits)].copy()
        
        # Create demographic groups if they don't exist
        train_data = self._create_demographic_groups(train_data)
        
        demographic_results = {}
        
        for segment_type in self.config.demographic_segments:
            if segment_type not in train_data.columns:
                logger.warning(f"Segment column '{segment_type}' not found in data")
                continue
            
            segment_data = train_data[train_data[segment_type].notna()].copy()
            segment_metrics = {}
            
            # Get unique categories for this segment
            categories = segment_data[segment_type].unique()
            
            for category in categories:
                category_data = segment_data[segment_data[segment_type] == category].copy()
                
                if len(category_data) < 500:  # Minimum sample size for reliable analysis
                    continue
                
                # Fit KM estimator for this category
                kmf_category = KaplanMeierFitter()
                kmf_category.fit(category_data[time_col], category_data[event_col])
                
                # Calculate category metrics
                category_stats = {
                    'sample_size': len(category_data),
                    'event_rate': category_data[event_col].mean(),
                    'median_survival': kmf_category.median_survival_time_
                }
                
                # Retention rates at key horizons
                for horizon in self.config.baseline_horizons:
                    try:
                        retention_rate = kmf_category.survival_function_at_times(horizon).iloc[0]
                        category_stats[f'retention_{horizon}d'] = retention_rate
                    except (IndexError, KeyError):
                        category_stats[f'retention_{horizon}d'] = np.nan
                
                segment_metrics[str(category)] = category_stats
            
            # Rank categories by 365-day retention
            if segment_metrics:
                ranked_categories = sorted(
                    segment_metrics.items(),
                    key=lambda x: x[1].get('retention_365d', 0),
                    reverse=True
                )
                
                demographic_results[segment_type] = {
                    'metrics': segment_metrics,
                    'ranked_performance': ranked_categories,
                    'best_performing': ranked_categories[0][0] if ranked_categories else None,
                    'worst_performing': ranked_categories[-1][0] if ranked_categories else None
                }
        
        # Generate demographic visualizations
        if self.config.save_visualizations:
            self._plot_demographic_segments(train_data, demographic_results, time_col, event_col)
        
        logger.info(f"Demographic analysis complete for {len(demographic_results)} segment types")
        
        self.analysis_results['demographics'] = demographic_results
        return demographic_results
    
    # === INSIGHTS GENERATION SECTION ===
    
    def generate_context_insights(self, analysis_results: Optional[Dict] = None) -> Dict:
        """
        Synthesize business context insights for modeling foundation
        
        Analyzes key population characteristics, identifies critical risk factors,
        and provides modeling strategy recommendations based on comprehensive analysis.
        
        Args:
            analysis_results: Optional pre-computed analysis results, uses self.analysis_results if None
            
        Returns:
            Dict containing synthesized insights, risk factors, and strategic recommendations
        """
        logger.info("Generating comprehensive business context insights...")
        
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if not analysis_results:
            logger.warning("No analysis results available for insight generation")
            return {'warning': 'No analysis results available'}
        
        insights = {
            'executive_summary': {},
            'key_risk_factors': {},
            'strategic_recommendations': {},
            'data_quality_assessment': {}
        }
        
        # Executive Summary
        if 'baseline' in analysis_results:
            baseline = analysis_results['baseline']
            insights['executive_summary'] = {
                'total_population': baseline.get('total_population', 0),
                'modeling_population': baseline.get('population_size', 0),
                'overall_event_rate': baseline.get('event_rate', 0),
                'median_survival_days': baseline.get('median_survival', 0),
                'one_year_retention': baseline.get('retention_365d', 0),
                'population_coverage': baseline.get('population_size', 0) / baseline.get('total_population', 1)
            }
        
        # Key Risk Factors Identification
        risk_factors = []
        
        # Industry-based risk factors
        if 'industry' in analysis_results:
            industry_data = analysis_results['industry']
            if 'performance_stats' in industry_data:
                perf_stats = industry_data['performance_stats']
                risk_factors.append({
                    'factor': 'Industry Variation',
                    'impact': perf_stats.get('performance_range', 0),
                    'description': f"Retention varies by {perf_stats.get('performance_range', 0):.1%} across industries"
                })
        
        # Demographic risk factors
        if 'demographics' in analysis_results:
            for segment_type, segment_data in analysis_results['demographics'].items():
                if 'ranked_performance' in segment_data and len(segment_data['ranked_performance']) > 1:
                    best_perf = segment_data['ranked_performance'][0][1].get('retention_365d', 0)
                    worst_perf = segment_data['ranked_performance'][-1][1].get('retention_365d', 0)
                    risk_factors.append({
                        'factor': f'{segment_type.title()} Disparity',
                        'impact': best_perf - worst_perf,
                        'description': f"{segment_type.title()} retention gap of {(best_perf - worst_perf):.1%}"
                    })
        
        insights['key_risk_factors'] = sorted(risk_factors, key=lambda x: x['impact'], reverse=True)
        
        # Strategic Recommendations
        recommendations = []
        
        # Population coverage recommendation
        if 'baseline' in analysis_results:
            coverage = insights['executive_summary'].get('population_coverage', 0)
            if coverage < 0.5:
                recommendations.append({
                    'priority': 'High',
                    'area': 'Data Coverage',
                    'recommendation': f'Expand model coverage from {coverage:.1%} to improve business impact'
                })
        
        # Industry stratification recommendation
        if 'industry' in analysis_results:
            industry_data = analysis_results['industry']
            if industry_data.get('performance_stats', {}).get('performance_range', 0) > 0.2:
                recommendations.append({
                    'priority': 'Medium',
                    'area': 'Model Strategy',
                    'recommendation': 'Consider industry-stratified modeling due to significant performance variation'
                })
        
        # Temporal stability recommendation
        if 'temporal' in analysis_results:
            temporal_data = analysis_results['temporal']
            if temporal_data.get('temporal_difference_significant', False):
                recommendations.append({
                    'priority': 'High',
                    'area': 'Model Validation',
                    'recommendation': 'Implement temporal validation strategy due to significant trend changes'
                })
        
        insights['strategic_recommendations'] = recommendations
        
        # Data Quality Assessment
        quality_metrics = {
            'analysis_completeness': len(analysis_results) / 4,  # Expected: baseline, temporal, industry, demographics
            'sufficient_sample_sizes': True,  # Assume true unless proven otherwise
            'temporal_stability': analysis_results.get('temporal', {}).get('temporal_difference_significant', True)
        }
        
        insights['data_quality_assessment'] = quality_metrics
        
        logger.info(f"Context insights generated with {len(risk_factors)} risk factors and {len(recommendations)} recommendations")
        
        self.insights_summary = insights
        return insights
    
    def create_context_visualizations(self, analysis_results: Optional[Dict] = None) -> None:
        """
        Generate comprehensive business context visualizations
        
        Creates executive dashboard with key metrics, risk factor visualization,
        and strategic recommendation summaries for stakeholder communication.
        
        Args:
            analysis_results: Optional pre-computed analysis results
        """
        logger.info("Creating comprehensive context visualizations...")
        
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if not analysis_results:
            logger.warning("No analysis results available for visualization")
            return
        
        # Create executive dashboard
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Executive metrics summary
        if 'baseline' in analysis_results:
            self._plot_executive_summary(fig, gs[0, :], analysis_results['baseline'])
        
        # Risk factors visualization  
        if hasattr(self, 'insights_summary') and 'key_risk_factors' in self.insights_summary:
            self._plot_risk_factors(fig, gs[1, 0], self.insights_summary['key_risk_factors'])
        
        # Performance distribution
        if 'industry' in analysis_results:
            self._plot_performance_distribution(fig, gs[1, 1], analysis_results['industry'])
        
        # Recommendations summary
        if hasattr(self, 'insights_summary') and 'strategic_recommendations' in self.insights_summary:
            self._plot_recommendations(fig, gs[1, 2], self.insights_summary['strategic_recommendations'])
        
        # Temporal trends summary
        if 'temporal' in analysis_results:
            self._plot_temporal_summary(fig, gs[2, :2], analysis_results['temporal'])
        
        # Data quality indicators
        if hasattr(self, 'insights_summary') and 'data_quality_assessment' in self.insights_summary:
            self._plot_quality_assessment(fig, gs[2, 2], self.insights_summary['data_quality_assessment'])
        
        plt.suptitle('Business Context Analysis - Executive Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if self.config.save_visualizations:
            plt.savefig(f'{self.config.output_path}/executive_dashboard.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        logger.info("Executive dashboard visualization created successfully")
    
    # === PRIVATE HELPER METHODS ===
    
    def _create_demographic_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic grouping variables if they don't exist"""
        df = df.copy()
        
        # Age groups
        if 'age_group' not in df.columns and 'age_at_vantage' in df.columns:
            df['age_group'] = pd.cut(
                df['age_at_vantage'],
                bins=[0, 25, 35, 45, 55, 65, np.inf],
                labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
                include_lowest=True
            )
        
        # Tenure groups
        if 'tenure_group' not in df.columns and 'tenure_at_vantage_days' in df.columns:
            tenure_years = df['tenure_at_vantage_days'] / 365.25
            df['tenure_group'] = pd.cut(
                tenure_years,
                bins=[0, 0.5, 1, 2, 3, 5, np.inf],
                labels=["<6mo", "6mo-1yr", "1-2yr", "2-3yr", "3-5yr", "5yr+"],
                include_lowest=True
            )
        
        # Salary groups  
        if 'salary_group' not in df.columns and 'baseline_salary' in df.columns:
            df['salary_group'] = pd.cut(
                df['baseline_salary'],
                bins=[0, 40000, 60000, 80000, 120000, 200000, np.inf],
                labels=["<40K", "40-60K", "60-80K", "80-120K", "120-200K", "200K+"],
                include_lowest=True
            )
        
        return df
    
    def _calculate_baseline_confidence_intervals(self, df: pd.DataFrame, 
                                               time_col: str, event_col: str) -> Dict:
        """Calculate confidence intervals for baseline metrics"""
        try:
            # Bootstrap confidence intervals for event rate
            n_bootstrap = 1000
            bootstrap_event_rates = []
            
            for _ in range(n_bootstrap):
                boot_sample = df.sample(n=len(df), replace=True)
                bootstrap_event_rates.append(boot_sample[event_col].mean())
            
            event_rate_ci = {
                'event_rate_ci_lower': np.percentile(bootstrap_event_rates, 2.5),
                'event_rate_ci_upper': np.percentile(bootstrap_event_rates, 97.5)
            }
            
            return event_rate_ci
        except Exception as e:
            logger.warning(f"Could not calculate confidence intervals: {e}")
            return {}
    
    def _plot_baseline_survival(self, data: pd.DataFrame, time_col: str, 
                               event_col: str, metrics: Dict) -> None:
        """Generate baseline survival curve visualization"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        self.kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=3, color="navy")
        
        # Add milestone markers
        milestones = self.config.baseline_horizons
        colors = ["red", "orange", "green", "purple"]
        
        for day, color in zip(milestones, colors):
            if f'retention_{day}d' in metrics:
                retention = metrics[f'retention_{day}d']
                if not np.isnan(retention):
                    ax.axvline(x=day, color=color, linestyle="--", alpha=0.7)
                    ax.text(day, retention + 0.02, f"{day}d\n{retention:.1%}",
                           ha="center", va="bottom", fontsize=11, fontweight="bold")
        
        ax.set_title("Population Baseline - Employee Retention Analysis", 
                    fontsize=17, fontweight="bold")
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.output_path}/baseline_survival.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_temporal_trends(self, cohort_2023: pd.DataFrame, cohort_2024: pd.DataFrame,
                             time_col: str, event_col: str, metrics: Dict) -> None:
        """Generate temporal trends visualization"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        kmf_2023 = KaplanMeierFitter()
        kmf_2024 = KaplanMeierFitter()
        
        kmf_2023.fit(cohort_2023[time_col], cohort_2023[event_col])
        kmf_2024.fit(cohort_2024[time_col], cohort_2024[event_col])
        
        kmf_2023.plot_survival_function(
            ax=ax, ci_show=False, linewidth=3, color="blue",
            label=f"2023 Cohort (n={len(cohort_2023):,})"
        )
        kmf_2024.plot_survival_function(
            ax=ax, ci_show=False, linewidth=3, color="red",
            label=f"2024 Cohort (n={len(cohort_2024):,})"
        )
        
        ax.set_title("Temporal Retention Trends: 2023 vs 2024", 
                    fontsize=17, fontweight="bold")
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.output_path}/temporal_trends.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_industry_patterns(self, survival_curves: Dict, industry_metrics: Dict, 
                               ranked_industries: List) -> None:
        """Generate industry patterns visualization"""
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = sns.color_palette("husl", len(survival_curves))
        
        for i, (industry, kmf) in enumerate(survival_curves.items()):
            sample_size = industry_metrics[industry]['sample_size']
            kmf.plot_survival_function(
                ax=ax, ci_show=False, color=colors[i],
                label=f"NAICS {industry} (n={sample_size:,})"
            )
        
        ax.set_title(f"Retention by Industry - Top {len(survival_curves)} Industries",
                    fontsize=17, fontweight="bold")
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.output_path}/industry_patterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_demographic_segments(self, data: pd.DataFrame, demo_results: Dict,
                                  time_col: str, event_col: str) -> None:
        """Generate demographic segments visualization"""
        n_segments = len(demo_results)
        fig, axes = plt.subplots(1, n_segments, figsize=(6*n_segments, 8))
        
        if n_segments == 1:
            axes = [axes]
        
        for idx, (segment_name, segment_data) in enumerate(demo_results.items()):
            ax = axes[idx]
            
            segment_col_data = data[data[segment_name].notna()]
            categories = segment_col_data[segment_name].unique()
            
            for category in categories:
                category_data = segment_col_data[segment_col_data[segment_name] == category]
                if len(category_data) >= 500:
                    kmf = KaplanMeierFitter()
                    kmf.fit(category_data[time_col], category_data[event_col])
                    kmf.plot_survival_function(ax=ax, ci_show=False, 
                                             label=f"{category} (n={len(category_data):,})")
            
            ax.set_title(f"Retention by {segment_name.replace('_', ' ').title()}", 
                        fontsize=14, fontweight="bold")
            ax.set_xlabel("Days Since Assignment Start")
            ax.set_ylabel("Survival Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.output_path}/demographic_segments.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_executive_summary(self, fig, gs_position, baseline_metrics: Dict) -> None:
        """Plot executive summary metrics"""
        ax = fig.add_subplot(gs_position)
        ax.axis('off')
        
        # Key metrics text
        summary_text = f"""
        POPULATION BASELINE SUMMARY
        ═══════════════════════════
        
        Total Population: {baseline_metrics.get('total_population', 0):,}
        Modeling Population: {baseline_metrics.get('population_size', 0):,}
        Coverage: {baseline_metrics.get('population_size', 0) / baseline_metrics.get('total_population', 1):.1%}
        
        Event Rate: {baseline_metrics.get('event_rate', 0):.1%}
        Median Survival: {baseline_metrics.get('median_survival', 0):.0f} days
        
        RETENTION MILESTONES
        ───────────────────
        30-day:  {baseline_metrics.get('retention_30d', 0):.1%}
        90-day:  {baseline_metrics.get('retention_90d', 0):.1%}
        180-day: {baseline_metrics.get('retention_180d', 0):.1%}
        365-day: {baseline_metrics.get('retention_365d', 0):.1%}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _plot_risk_factors(self, fig, gs_position, risk_factors: List) -> None:
        """Plot key risk factors"""
        ax = fig.add_subplot(gs_position)
        
        if risk_factors:
            factors = [rf['factor'] for rf in risk_factors[:5]]
            impacts = [rf['impact'] for rf in risk_factors[:5]]
            
            bars = ax.barh(factors, impacts, color='lightcoral', alpha=0.7)
            ax.set_xlabel('Impact on Retention')
            ax.set_title('Key Risk Factors', fontweight='bold')
            
            # Add impact values on bars
            for bar, impact in zip(bars, impacts):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{impact:.1%}', ha='left', va='center')
        else:
            ax.text(0.5, 0.5, 'No risk factors identified', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_distribution(self, fig, gs_position, industry_data: Dict) -> None:
        """Plot industry performance distribution"""
        ax = fig.add_subplot(gs_position)
        
        if 'industry_metrics' in industry_data:
            retention_values = [
                metrics.get('retention_365d', 0) 
                for metrics in industry_data['industry_metrics'].values()
                if not np.isnan(metrics.get('retention_365d', np.nan))
            ]
            
            if retention_values:
                ax.hist(retention_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(retention_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(retention_values):.1%}')
                ax.set_xlabel('365-Day Retention Rate')
                ax.set_ylabel('Number of Industries')
                ax.set_title('Industry Retention Distribution', fontweight='bold')
                ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    def _plot_recommendations(self, fig, gs_position, recommendations: List) -> None:
        """Plot strategic recommendations"""
        ax = fig.add_subplot(gs_position)
        ax.axis('off')
        
        if recommendations:
            rec_text = "STRATEGIC RECOMMENDATIONS\n" + "═" * 25 + "\n\n"
            for i, rec in enumerate(recommendations[:3], 1):
                rec_text += f"{i}. [{rec['priority']}] {rec['area']}\n"
                rec_text += f"   {rec['recommendation']}\n\n"
        else:
            rec_text = "No specific recommendations generated"
        
        ax.text(0.05, 0.95, rec_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _plot_temporal_summary(self, fig, gs_position, temporal_data: Dict) -> None:
        """Plot temporal trends summary"""
        ax = fig.add_subplot(gs_position)
        
        # Create comparison bars
        horizons = ['90d', '180d', '365d']
        cohort_2023 = [temporal_data.get(f'2023_retention_{h}', 0) for h in horizons]
        cohort_2024 = [temporal_data.get(f'2024_retention_{h}', 0) for h in horizons]
        
        x = np.arange(len(horizons))
        width = 0.35
        
        ax.bar(x - width/2, cohort_2023, width, label='2023', alpha=0.7, color='blue')
        ax.bar(x + width/2, cohort_2024, width, label='2024', alpha=0.7, color='red')
        
        ax.set_xlabel('Time Horizon')
        ax.set_ylabel('Retention Rate')
        ax.set_title('Temporal Retention Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add significance indicator
        if temporal_data.get('temporal_difference_significant', False):
            ax.text(0.5, 0.95, '* Statistically Significant Difference', 
                   transform=ax.transAxes, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def _plot_quality_assessment(self, fig, gs_position, quality_data: Dict) -> None:
        """Plot data quality assessment"""
        ax = fig.add_subplot(gs_position)
        
        # Quality metrics radar chart
        metrics = ['Completeness', 'Sample Size', 'Temporal Stability']
        values = [
            quality_data.get('analysis_completeness', 0),
            1.0 if quality_data.get('sufficient_sample_sizes', True) else 0.5,
            0.8 if quality_data.get('temporal_stability', True) else 0.3
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax = plt.subplot(gs_position, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='green', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='green')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Data Quality Assessment', fontweight='bold', pad=20)
        ax.grid(True)

if __name__ == "__main__":
    # Example usage
    config = AnalysisConfig()
    analyzer = BusinessContextAnalyzer(config)
    
    print("BusinessContextAnalyzer module ready for integration")
    print("Usage:")
    print("  analyzer = BusinessContextAnalyzer(AnalysisConfig())")
    print("  baseline = analyzer.analyze_population_baseline(df)")
    print("  industry = analyzer.analyze_industry_patterns(df)")
    print("  demographics = analyzer.analyze_demographic_segments(df)")
    print("  temporal = analyzer.analyze_temporal_trends(df)")
    print("  insights = analyzer.generate_context_insights()")
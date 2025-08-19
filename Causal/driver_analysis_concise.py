"""
driver_analysis_concise.py

Individual-level driver analysis for employee turnover risk.
Provides actionable explanations for UI display.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskDrivers:
    """Individual employee risk drivers for UI display"""
    employee_id: Any
    risk_score: float
    risk_category: str
    top_risk_factors: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    intervention_recommendations: List[Dict[str, Any]]


class IndividualDriverAnalyzer:
    """
    Analyzes individual-level drivers of turnover risk.
    Focused on actionable insights for HR practitioners.
    """
    
    # Define modifiable features (from actual feature list)
    MODIFIABLE_FEATURES = [
        'baseline_salary', 'salary_growth_rate_12m', 'compensation_percentile_company',
        'peer_salary_ratio', 'time_since_last_promotion', 'promotion_velocity',
        'role_complexity_score', 'team_size', 'time_with_current_manager',
        'manager_span_control', 'comp_chang_freq_per_year', 'pay_grade_stagnation_months'
    ]
    
    # Feature display names for UI
    FEATURE_DISPLAY_NAMES = {
        'baseline_salary': 'Current Salary',
        'tenure_at_vantage_days': 'Company Tenure',
        'age_at_vantage': 'Age',
        'time_since_last_promotion': 'Time Since Promotion',
        'compensation_percentile_company': 'Salary Percentile (Company)',
        'compensation_percentile_industry': 'Salary Percentile (Industry)',
        'peer_salary_ratio': 'Salary vs Peers',
        'job_level': 'Job Level',
        'team_size': 'Team Size',
        'salary_growth_rate_12m': '12-Month Salary Growth',
        'career_stage': 'Career Stage',
        'tenure_in_current_role': 'Time in Current Role',
        'time_with_current_manager': 'Time with Manager',
        'promotion_velocity': 'Promotion Speed',
        'team_avg_comp': 'Team Average Salary',
        'team_turnover_rate': 'Team Turnover Rate',
        'role_complexity_score': 'Role Complexity',
        'pay_grade_stagnation_months': 'Pay Grade Stagnation'
    }
    
    def __init__(self, model_engine, causal_analyzer=None):
        """
        Initialize driver analyzer.
        
        Args:
            model_engine: Trained SurvivalModelEngine
            causal_analyzer: Optional CausalInterventionAnalyzer
        """
        self.model_engine = model_engine
        self.causal_analyzer = causal_analyzer
        
    def analyze_employee(self, employee_data: pd.Series, 
                        n_drivers: int = 5) -> RiskDrivers:
        """
        Analyze turnover risk drivers for an individual employee.
        
        Args:
            employee_data: Single employee's RAW features (before preprocessing)
            n_drivers: Number of top drivers to return
            
        Returns:
            RiskDrivers object with analysis results
        """
        # Get current risk - model handles preprocessing internally
        employee_df = pd.DataFrame([employee_data])
        risk_score = self.model_engine.predict_risk_scores(employee_df)[0]
        risk_category = self._categorize_risk(risk_score)
        
        # Calculate feature contributions using permutation
        # Note: We need to get the processed features to know which ones matter
        processed_df = self.model_engine._get_processed_features(employee_df)
        contributions = self._calculate_feature_contributions(employee_df, processed_df)
        
        # Map contributions back to original features
        original_contributions = self._map_to_original_features(contributions)
        
        # Separate risk and protective factors
        risk_factors = []
        protective_factors = []
        
        for feature, contrib_value in original_contributions.items():
            # Get original feature value from raw data
            value = employee_data.get(feature) if feature in employee_data.index else None
            
            factor = {
                'feature': feature,
                'display_name': self.FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title()),
                'value': value,
                'contribution': contrib_value,
                'modifiable': feature in self.MODIFIABLE_FEATURES
            }
            
            if contrib_value > 0:  # Increases risk
                risk_factors.append(factor)
            else:  # Decreases risk
                protective_factors.append(factor)
        
        # Sort by absolute contribution
        risk_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        protective_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Generate intervention recommendations
        recommendations = self._generate_recommendations(
            employee_df, risk_factors[:n_drivers], risk_score
        )
        
        return RiskDrivers(
            employee_id=employee_data.name if hasattr(employee_data, 'name') else 'Unknown',
            risk_score=risk_score,
            risk_category=risk_category,
            top_risk_factors=risk_factors[:n_drivers],
            top_protective_factors=protective_factors[:n_drivers],
            intervention_recommendations=recommendations
        )
    
    def _calculate_feature_contributions(self, employee_df: pd.DataFrame, 
                                        processed_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature contributions using permutation approach on processed features.
        
        Args:
            employee_df: Raw employee data
            processed_df: Processed features after pipeline
        """
        baseline_risk = self.model_engine.predict_risk_scores(employee_df)[0]
        contributions = {}
        
        # Get the actual processed feature columns
        processed_features = processed_df.columns.tolist()
        
        # Calculate contributions for processed features
        for feature in processed_features:
            if feature in ['survival_time_days', 'event_indicator_vol', 'dataset_split']:
                continue
            
            # Create permuted version in raw data
            permuted_df = employee_df.copy()
            
            # Find corresponding raw feature
            raw_feature = self._get_raw_feature_name(feature)
            if raw_feature and raw_feature in permuted_df.columns:
                # Permute the raw feature based on its type
                if permuted_df[raw_feature].dtype == 'object':
                    # Categorical - use mode or 'Unknown'
                    permuted_df[raw_feature] = 'Unknown'
                else:
                    # Numeric - use median or 0
                    permuted_df[raw_feature] = 0
            
                permuted_risk = self.model_engine.predict_risk_scores(permuted_df)[0]
                contribution = baseline_risk - permuted_risk
                contributions[feature] = contribution
        
        return contributions
    
    def _get_raw_feature_name(self, processed_feature: str) -> str:
        """Map processed feature name back to raw feature name"""
        # Handle different transformations
        if processed_feature.endswith('_cap'):
            return processed_feature[:-4]  # Remove _cap
        elif processed_feature.endswith('_log'):
            return processed_feature[:-4]  # Remove _log
        elif processed_feature.endswith('_win_cap'):
            return processed_feature[:-8]  # Remove _win_cap
        elif processed_feature.endswith('_encoded'):
            return processed_feature[:-8]  # Remove _encoded
        else:
            return processed_feature
    
    def _map_to_original_features(self, contributions: Dict[str, float]) -> Dict[str, float]:
        """Map contributions from processed features to original feature names"""
        original_contributions = {}
        
        for processed_feature, contrib in contributions.items():
            original_feature = self._get_raw_feature_name(processed_feature)
            
            # Aggregate contributions for same original feature
            if original_feature in original_contributions:
                # Take the max absolute contribution
                if abs(contrib) > abs(original_contributions[original_feature]):
                    original_contributions[original_feature] = contrib
            else:
                original_contributions[original_feature] = contrib
        
        return original_contributions
    
    def _generate_recommendations(self, employee_df: pd.DataFrame,
                                 risk_factors: List[Dict],
                                 current_risk: float) -> List[Dict]:
        """
        Generate intervention recommendations based on risk factors.
        """
        recommendations = []
        
        # Check if salary-related factors are prominent
        salary_factors = [f for f in risk_factors 
                         if f['feature'] in ['baseline_salary', 'compensation_percentile_company', 
                                            'peer_salary_ratio', 'salary_growth_rate_12m']]
        
        if salary_factors and self.causal_analyzer:
            # Use causal analyzer if available
            salary_effect = self.causal_analyzer.estimate_salary_intervention(
                employee_df, increase_pct=0.15, horizon=365
            )
            if salary_effect.ite_array[0] > 0:  # This employee would benefit
                recommendations.append({
                    'intervention': '15% Salary Increase',
                    'expected_risk_reduction': f"{salary_effect.ite_array[0]:.1%}",
                    'confidence': 'High' if salary_effect.significant else 'Medium',
                    'priority': 1 if salary_effect.ite_array[0] > 0.05 else 2
                })
        elif salary_factors:
            # Fallback without causal analyzer
            recommendations.append({
                'intervention': 'Review Compensation',
                'expected_risk_reduction': 'To be determined',
                'confidence': 'Medium',
                'priority': 2
            })
        
        # Check promotion-related factors
        promotion_factors = [f for f in risk_factors
                            if f['feature'] in ['time_since_last_promotion', 'promotion_velocity',
                                               'pay_grade_stagnation_months', 'days_since_promot']]
        
        if promotion_factors and self.causal_analyzer:
            promotion_effect = self.causal_analyzer.estimate_promotion_intervention(
                employee_df, horizon=365
            )
            if promotion_effect.ite_array[0] > 0:
                recommendations.append({
                    'intervention': 'Promotion',
                    'expected_risk_reduction': f"{promotion_effect.ite_array[0]:.1%}",
                    'confidence': 'High' if promotion_effect.significant else 'Medium',
                    'priority': 1 if promotion_effect.ite_array[0] > 0.05 else 2
                })
        elif promotion_factors:
            recommendations.append({
                'intervention': 'Career Development Review',
                'expected_risk_reduction': 'To be determined',
                'confidence': 'Medium',
                'priority': 2
            })
        
        # Check team/manager factors
        team_factors = [f for f in risk_factors
                       if f['feature'] in ['time_with_current_manager', 'team_turnover_rate',
                                          'team_size', 'manager_span_control']]
        
        if team_factors:
            recommendations.append({
                'intervention': 'Team/Manager Discussion',
                'expected_risk_reduction': 'Varies',
                'confidence': 'Low',
                'priority': 3
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into High/Medium/Low"""
        if risk_score >= 0.67:  # Top tercile
            return 'High'
        elif risk_score >= 0.33:  # Middle tercile
            return 'Medium'
        else:
            return 'Low'
    
    def format_for_ui(self, analysis: RiskDrivers) -> Dict:
        """
        Format analysis results for UI display.
        
        Returns:
            Dict ready for JSON serialization and UI rendering
        """
        def format_value(feature: str, value: Any) -> str:
            """Format feature value for display"""
            if value is None:
                return 'N/A'
            
            if 'salary' in feature.lower() or 'comp' in feature.lower():
                if isinstance(value, (int, float)) and not 'percentile' in feature:
                    return f"${value:,.0f}"
            elif 'days' in feature.lower() or 'tenure' in feature.lower():
                if isinstance(value, (int, float)):
                    return f"{value/365.25:.1f} years"
            elif 'percentile' in feature.lower() or 'ratio' in feature.lower():
                if isinstance(value, (int, float)):
                    return f"{value:.1f}"
            
            return str(value)
        
        return {
            'employee_id': analysis.employee_id,
            'risk_summary': {
                'score': f"{analysis.risk_score:.1%}",
                'category': analysis.risk_category,
                'color': {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}[analysis.risk_category]
            },
            'top_risk_factors': [
                {
                    'name': factor['display_name'],
                    'value': format_value(factor['feature'], factor['value']),
                    'impact': 'High' if abs(factor['contribution']) > 0.05 else 'Medium',
                    'modifiable': factor['modifiable']
                }
                for factor in analysis.top_risk_factors
            ],
            'top_protective_factors': [
                {
                    'name': factor['display_name'],
                    'value': format_value(factor['feature'], factor['value']),
                    'impact': 'High' if abs(factor['contribution']) > 0.05 else 'Medium'
                }
                for factor in analysis.top_protective_factors
            ],
            'recommendations': [
                {
                    'action': rec['intervention'],
                    'expected_impact': rec['expected_risk_reduction'],
                    'confidence': rec['confidence'],
                    'priority': ['High', 'Medium', 'Low'][rec['priority'] - 1]
                }
                for rec in analysis.intervention_recommendations
            ]
        }


if __name__ == "__main__":
    pass

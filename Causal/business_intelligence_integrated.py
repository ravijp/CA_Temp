"""
business_intelligence_integrated.py

Integrated business intelligence module combining causal inference and driver analysis.
Replaces the placeholder 4_business_intelligence.py with working implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Import the concise modules
from causal_inference_concise import CausalInterventionAnalyzer, InterventionEffect
from driver_analysis_concise import IndividualDriverAnalyzer, RiskDrivers

logger = logging.getLogger(__name__)


class BusinessIntelligence:
    """
    Integrated business intelligence framework for survival analysis.
    Combines causal inference and driver analysis for actionable insights.
    """
    
    def __init__(self, model_engine, evaluation=None):
        """
        Initialize business intelligence framework.
        
        Args:
            model_engine: Trained SurvivalModelEngine instance
            evaluation: Optional SurvivalEvaluation instance
        """
        self.model_engine = model_engine
        self.evaluation = evaluation
        
        # Initialize components
        self.causal_analyzer = CausalInterventionAnalyzer(
            model_engine, n_bootstrap=100, confidence_level=0.95
        )
        self.driver_analyzer = IndividualDriverAnalyzer(
            model_engine, self.causal_analyzer
        )
        
        logger.info("BusinessIntelligence framework initialized")
    
    def analyze_population(self, X: pd.DataFrame, sample_size: int = None) -> Dict:
        """
        Analyze intervention effects at population level.
        
        Args:
            X: Population features
            sample_size: Optional sampling for large datasets
            
        Returns:
            Dict with population-level insights
        """
        # Sample if needed for performance
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} from {len(X)} employees")
        else:
            X_sample = X
        
        # Calculate intervention effects
        salary_effect = self.causal_analyzer.estimate_salary_intervention(X_sample)
        promotion_effect = self.causal_analyzer.estimate_promotion_intervention(X_sample)
        
        # Risk distribution
        risk_scores = self.model_engine.predict_risk_scores(X_sample)
        
        return {
            'population_size': len(X_sample),
            'risk_distribution': {
                'mean': float(np.mean(risk_scores)),
                'std': float(np.std(risk_scores)),
                'high_risk_pct': float(np.mean(risk_scores >= 0.67) * 100),
                'medium_risk_pct': float(np.mean((risk_scores >= 0.33) & (risk_scores < 0.67)) * 100),
                'low_risk_pct': float(np.mean(risk_scores < 0.33) * 100)
            },
            'interventions': {
                'salary_increase_15pct': self._format_intervention_effect(salary_effect, 'Salary Increase'),
                'promotion': self._format_intervention_effect(promotion_effect, 'Promotion')
            },
            'recommendations': self._generate_population_recommendations(
                risk_scores, salary_effect, promotion_effect
            )
        }
    
    def analyze_individual(self, employee_data: pd.Series) -> Dict:
        """
        Analyze individual employee with drivers and interventions.
        
        Args:
            employee_data: Single employee features
            
        Returns:
            Dict with individual analysis for UI display
        """
        # Get driver analysis
        analysis = self.driver_analyzer.analyze_employee(employee_data)
        
        # Format for UI
        ui_data = self.driver_analyzer.format_for_ui(analysis)
        
        # Add intervention details if available
        employee_df = pd.DataFrame([employee_data])
        
        try:
            salary_ite = self.causal_analyzer.estimate_salary_intervention(
                employee_df, horizon=365
            ).ite_array[0]
            
            promotion_ite = self.causal_analyzer.estimate_promotion_intervention(
                employee_df, horizon=365
            ).ite_array[0]
            
            ui_data['intervention_details'] = {
                'salary_increase': {
                    'risk_reduction': f"{salary_ite:.1%}",
                    'recommended': salary_ite > 0.05
                },
                'promotion': {
                    'risk_reduction': f"{promotion_ite:.1%}",
                    'recommended': promotion_ite > 0.05
                }
            }
        except Exception as e:
            logger.warning(f"Could not calculate individual intervention effects: {e}")
        
        return ui_data
    
    def analyze_high_risk_cohort(self, X: pd.DataFrame, 
                                 risk_threshold: float = 0.67) -> Dict:
        """
        Focused analysis on high-risk employees.
        
        Args:
            X: Employee features
            risk_threshold: Threshold for high risk
            
        Returns:
            Dict with high-risk cohort analysis
        """
        # Identify high-risk employees
        risk_scores = self.model_engine.predict_risk_scores(X)
        high_risk_mask = risk_scores >= risk_threshold
        high_risk_data = X[high_risk_mask]
        
        if len(high_risk_data) == 0:
            return {'message': 'No high-risk employees found'}
        
        # Analyze interventions for high-risk group
        salary_effect = self.causal_analyzer.estimate_salary_intervention(high_risk_data)
        promotion_effect = self.causal_analyzer.estimate_promotion_intervention(high_risk_data)
        
        # Identify common risk factors
        sample_size = min(100, len(high_risk_data))
        sample_indices = np.random.choice(len(high_risk_data), sample_size, replace=False)
        
        common_factors = {}
        for idx in sample_indices:
            employee = high_risk_data.iloc[idx]
            analysis = self.driver_analyzer.analyze_employee(employee, n_drivers=3)
            
            for factor in analysis.top_risk_factors:
                feature = factor['feature']
                if feature not in common_factors:
                    common_factors[feature] = 0
                common_factors[feature] += 1
        
        # Sort by frequency
        common_factors = sorted(common_factors.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'cohort_size': len(high_risk_data),
            'percentage_of_population': float(np.mean(high_risk_mask) * 100),
            'intervention_effectiveness': {
                'salary_increase': self._format_intervention_effect(salary_effect, 'Salary'),
                'promotion': self._format_intervention_effect(promotion_effect, 'Promotion')
            },
            'common_risk_factors': [
                {
                    'factor': self.driver_analyzer.FEATURE_DISPLAY_NAMES.get(f[0], f[0]),
                    'frequency_pct': float(f[1] / sample_size * 100)
                }
                for f in common_factors
            ],
            'priority_actions': self._prioritize_interventions(salary_effect, promotion_effect, len(high_risk_data))
        }
    
    def _format_intervention_effect(self, effect: InterventionEffect, name: str) -> Dict:
        """Format intervention effect for reporting"""
        return {
            'name': name,
            'ate': f"{effect.ate:.2%}",
            'confidence_interval': f"({effect.ate_ci_lower:.2%}, {effect.ate_ci_upper:.2%})",
            'responders_pct': f"{effect.responders_pct:.1f}%",
            'significant': effect.significant,
            'p_value': float(effect.p_value),
            'sample_size': effect.sample_size
        }
    
    def _generate_population_recommendations(self, risk_scores: np.ndarray,
                                           salary_effect: InterventionEffect,
                                           promotion_effect: InterventionEffect) -> List[str]:
        """Generate population-level recommendations"""
        recommendations = []
        
        high_risk_pct = np.mean(risk_scores >= 0.67) * 100
        
        if high_risk_pct > 20:
            recommendations.append(f"URGENT: {high_risk_pct:.1f}% of employees are high risk")
        
        if salary_effect.significant and salary_effect.ate > 0.05:
            recommendations.append(f"Salary increases could reduce turnover by {salary_effect.ate:.1%}")
        
        if promotion_effect.significant and promotion_effect.ate > 0.05:
            recommendations.append(f"Promotions could reduce turnover by {promotion_effect.ate:.1%}")
        
        if salary_effect.ate > promotion_effect.ate:
            recommendations.append("Prioritize compensation reviews over promotions")
        elif promotion_effect.ate > salary_effect.ate:
            recommendations.append("Prioritize career development over compensation")
        
        return recommendations
    
    def _prioritize_interventions(self, salary_effect: InterventionEffect,
                                 promotion_effect: InterventionEffect,
                                 cohort_size: int) -> List[Dict]:
        """Prioritize interventions based on effectiveness and cost"""
        actions = []
        
        # Estimate costs (placeholder values - should be configured)
        salary_cost_per_person = 10000  # 15% of ~$67k average
        promotion_cost_per_person = 5000  # Training, transition costs
        
        if salary_effect.significant and salary_effect.ate > 0.03:
            prevented_turnover = int(cohort_size * salary_effect.ate)
            actions.append({
                'action': 'Implement 15% salary increases',
                'expected_retention': prevented_turnover,
                'estimated_cost': salary_cost_per_person * cohort_size,
                'roi_estimate': 'Positive' if salary_effect.ate > 0.10 else 'Neutral',
                'priority': 1 if salary_effect.ate > 0.10 else 2
            })
        
        if promotion_effect.significant and promotion_effect.ate > 0.03:
            prevented_turnover = int(cohort_size * promotion_effect.ate)
            actions.append({
                'action': 'Fast-track promotions',
                'expected_retention': prevented_turnover,
                'estimated_cost': promotion_cost_per_person * cohort_size,
                'roi_estimate': 'Positive' if promotion_effect.ate > 0.08 else 'Neutral',
                'priority': 1 if promotion_effect.ate > 0.08 else 2
            })
        
        # Sort by priority
        actions.sort(key=lambda x: x['priority'])
        
        return actions


if __name__ == "__main__":
    pass
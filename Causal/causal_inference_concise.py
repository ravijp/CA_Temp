"""
causal_inference_concise.py

Causal inference for employee turnover interventions using G-computation.
Focused implementation for salary increase and promotion interventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.utils import resample
import logging

logger = logging.getLogger(__name__)

@dataclass
class InterventionEffect:
    """Container for intervention effect estimates"""
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    ite_array: np.ndarray
    responders_pct: float
    p_value: float
    significant: bool
    sample_size: int


class CausalInterventionAnalyzer:
    """
    G-computation based causal inference for survival interventions.
    Uses actual features from the trained model.
    """
    
    # Confounders based on actual available features
    SALARY_CONFOUNDERS = [
        'age_at_vantage', 'tenure_at_vantage_days', 'job_level',
        'compensation_percentile_company', 'compensation_percentile_industry',
        'naics_cd', 'gender_cd', 'career_stage', 'team_avg_comp',
        'company_tenure_percentile', 'baseline_salary'
    ]
    
    PROMOTION_CONFOUNDERS = [
        'age_at_vantage', 'tenure_at_vantage_days', 'job_level',
        'time_since_last_promotion', 'days_since_promot', 'career_stage',
        'naics_cd', 'gender_cd', 'promotion_velocity', 'num_promot_2yr',
        'promot_2yr_ind', 'career_joiner_stage'
    ]
    
    def __init__(self, model_engine, n_bootstrap: int = 100, confidence_level: float = 0.95):
        """
        Initialize causal analyzer.
        
        Args:
            model_engine: Trained SurvivalModelEngine
            n_bootstrap: Number of bootstrap samples for CI
            confidence_level: Confidence level for intervals
        """
        self.model_engine = model_engine
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
    def estimate_salary_intervention(self, X: pd.DataFrame, 
                                    increase_pct: float = 0.15,
                                    horizon: int = 365) -> InterventionEffect:
        """
        Estimate causal effect of salary increase using G-computation.
        
        Args:
            X: Employee features (raw data before preprocessing)
            increase_pct: Salary increase (0.15 = 15%)
            horizon: Time horizon in days
            
        Returns:
            InterventionEffect with ATE, ITE, and confidence intervals
        """
        X = X.copy()
        
        # Check for confounders
        available_confounders = [c for c in self.SALARY_CONFOUNDERS if c in X.columns]
        if len(available_confounders) < 5:
            logger.warning(f"Only {len(available_confounders)} confounders available for salary intervention")
        
        # G-computation steps
        baseline_risk = self._get_risk_at_horizon(X, horizon)
        X_intervened = self._apply_salary_increase(X.copy(), increase_pct)
        intervened_risk = self._get_risk_at_horizon(X_intervened, horizon)
        
        # Individual Treatment Effects (positive = risk reduction)
        ite = baseline_risk - intervened_risk
        ate = np.mean(ite)
        
        # Bootstrap for confidence intervals
        ate_bootstrap = []
        for _ in range(self.n_bootstrap):
            idx = resample(range(len(X)), n_samples=len(X))
            ite_boot = ite[idx]
            ate_bootstrap.append(np.mean(ite_boot))
        
        # Calculate CI and p-value
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(ate_bootstrap, 100 * alpha / 2)
        ci_upper = np.percentile(ate_bootstrap, 100 * (1 - alpha / 2))
        
        # Two-sided test for ATE != 0
        se = np.std(ate_bootstrap)
        z_stat = ate / (se / np.sqrt(len(X))) if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return InterventionEffect(
            ate=ate,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            ite_array=ite,
            responders_pct=np.mean(ite > 0) * 100,
            p_value=p_value,
            significant=p_value < 0.05,
            sample_size=len(X)
        )
    
    def estimate_promotion_intervention(self, X: pd.DataFrame,
                                       horizon: int = 365) -> InterventionEffect:
        """
        Estimate causal effect of promotion using G-computation.
        
        Args:
            X: Employee features (raw data before preprocessing)
            horizon: Time horizon in days
            
        Returns:
            InterventionEffect with causal estimates
        """
        X = X.copy()
        
        # Check confounders
        available_confounders = [c for c in self.PROMOTION_CONFOUNDERS if c in X.columns]
        if len(available_confounders) < 5:
            logger.warning(f"Only {len(available_confounders)} confounders available for promotion intervention")
        
        # G-computation
        baseline_risk = self._get_risk_at_horizon(X, horizon)
        X_intervened = self._apply_promotion(X.copy())
        intervened_risk = self._get_risk_at_horizon(X_intervened, horizon)
        
        ite = baseline_risk - intervened_risk
        ate = np.mean(ite)
        
        # Bootstrap CI
        ate_bootstrap = []
        for _ in range(self.n_bootstrap):
            idx = resample(range(len(X)), n_samples=len(X))
            ate_bootstrap.append(np.mean(ite[idx]))
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(ate_bootstrap, 100 * alpha / 2)
        ci_upper = np.percentile(ate_bootstrap, 100 * (1 - alpha / 2))
        
        se = np.std(ate_bootstrap)
        z_stat = ate / (se / np.sqrt(len(X))) if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return InterventionEffect(
            ate=ate,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            ite_array=ite,
            responders_pct=np.mean(ite > 0) * 100,
            p_value=p_value,
            significant=p_value < 0.05,
            sample_size=len(X)
        )
    
    def _get_risk_at_horizon(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """Get turnover risk at specific horizon
        
        Note: X should be raw features - preprocessing happens inside predict_survival_curves
        """
        try:
            survival_curves = self.model_engine.predict_survival_curves(
                X, time_points=np.array([horizon])
            )
            return 1 - survival_curves[:, 0]  # Risk = 1 - Survival
        except Exception as e:
            logger.error(f"Failed to predict survival curves: {e}")
            raise
    
    def _apply_salary_increase(self, X: pd.DataFrame, increase_pct: float) -> pd.DataFrame:
        """Apply salary increase intervention to RAW features
        
        Modifies both direct salary features and related compensation metrics
        """
        X_modified = X.copy()
        
        # Direct salary features (numeric)
        if 'baseline_salary' in X_modified.columns:
            X_modified['baseline_salary'] *= (1 + increase_pct)
        
        if 'avg_salary_last_quarter' in X_modified.columns:
            X_modified['avg_salary_last_quarter'] *= (1 + increase_pct)
        
        # Salary growth features (numeric)
        if 'salary_growth_rate_12m' in X_modified.columns:
            current_growth = X_modified['salary_growth_rate_12m']
            X_modified['salary_growth_rate_12m'] = np.where(
                pd.isna(current_growth), 
                increase_pct,
                np.maximum(current_growth, increase_pct)
            )
        
        if 'salary_growth_rate12m_to_cpl_rate' in X_modified.columns:
            X_modified['salary_growth_rate12m_to_cpl_rate'] *= (1 + increase_pct)
        
        # Compensation percentiles (numeric) - approximate impact
        if 'compensation_percentile_company' in X_modified.columns:
            current = X_modified['compensation_percentile_company']
            # 15% raise might move someone up ~10 percentile points
            X_modified['compensation_percentile_company'] = np.minimum(current + increase_pct * 67, 95)
        
        if 'compensation_percentile_industry' in X_modified.columns:
            current = X_modified['compensation_percentile_industry']
            X_modified['compensation_percentile_industry'] = np.minimum(current + increase_pct * 67, 95)
        
        # Peer comparison ratios (numeric)
        if 'peer_salary_ratio' in X_modified.columns:
            X_modified['peer_salary_ratio'] *= (1 + increase_pct)
        
        if 'sal_nghb_ratio' in X_modified.columns:
            X_modified['sal_nghb_ratio'] *= (1 + increase_pct)
        
        # Team average compensation (indirect effect)
        if 'team_avg_comp' in X_modified.columns and 'team_size' in X_modified.columns:
            team_size = X_modified['team_size'].fillna(10)
            if 'baseline_salary' in X_modified.columns:
                salary_increase = X_modified['baseline_salary'] * increase_pct
                X_modified['team_avg_comp'] += salary_increase / team_size
        
        # Compensation volatility reduces after raise (numeric)
        if 'compensation_volatility' in X_modified.columns:
            X_modified['compensation_volatility'] *= 0.7
        
        return X_modified
    
    def _apply_promotion(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply promotion intervention to RAW features
        
        Handles both categorical (job_level) and numeric features
        """
        X_modified = X.copy()
        
        # Job level - categorical but with numeric-like values ('1', '2', '3')
        if 'job_level' in X_modified.columns:
            def increment_job_level(level):
                if pd.isna(level):
                    return level
                try:
                    # Try to convert to int and increment
                    current = int(level)
                    return str(min(current + 1, 10))
                except:
                    # If not numeric, leave as is
                    return level
            
            X_modified['job_level'] = X_modified['job_level'].apply(increment_job_level)
        
        # Career stage progression - categorical
        if 'career_stage' in X_modified.columns:
            stage_mapping = {'Early': 'Mid', 'Mid': 'Senior', 'Senior': 'Senior'}
            X_modified['career_stage'] = X_modified['career_stage'].map(
                lambda x: stage_mapping.get(x, x) if pd.notna(x) else x
            )
        
        # Promotion timing features (numeric)
        if 'time_since_last_promotion' in X_modified.columns:
            X_modified['time_since_last_promotion'] = 0
        
        if 'days_since_promot' in X_modified.columns:
            X_modified['days_since_promot'] = 0
        
        # Promotion indicators (numeric 0/1)
        if 'promot_2yr_ind' in X_modified.columns:
            X_modified['promot_2yr_ind'] = 1
        
        if 'promot_2yr_titlechng_ind' in X_modified.columns:
            X_modified['promot_2yr_titlechng_ind'] = 1
        
        if 'promot_2yr_perf_ind' in X_modified.columns:
            X_modified['promot_2yr_perf_ind'] = 1
        
        if 'promot_2yr_mktadjst_ind' in X_modified.columns:
            X_modified['promot_2yr_mktadjst_ind'] = 1
        
        # Promotion count (numeric)
        if 'num_promot_2yr' in X_modified.columns:
            X_modified['num_promot_2yr'] = np.minimum(X_modified['num_promot_2yr'] + 1, 3)
        
        # Promotion velocity (numeric)
        if 'promotion_velocity' in X_modified.columns:
            X_modified['promotion_velocity'] *= 1.5
        
        # Reset stagnation (numeric)
        if 'pay_grade_stagnation_months' in X_modified.columns:
            X_modified['pay_grade_stagnation_months'] = 0
        
        # Role complexity increases with promotion (numeric)
        if 'role_complexity_score' in X_modified.columns:
            X_modified['role_complexity_score'] = np.minimum(
                X_modified['role_complexity_score'] * 1.2, 5
            )
        
        # Reset tenure in current role (numeric)
        if 'tenure_in_current_role' in X_modified.columns:
            X_modified['tenure_in_current_role'] = 0
        
        # Job change indicators (numeric)
        if 'job_chng_2yr_ind' in X_modified.columns:
            X_modified['job_chng_2yr_ind'] = 1
        
        # Demotion indicator reset (numeric)
        if 'demot_2yr_ind' in X_modified.columns:
            X_modified['demot_2yr_ind'] = 0
        
        return X_modified


if __name__ == "__main__":
    pass
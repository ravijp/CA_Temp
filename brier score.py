import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from scipy.special import expit, logit
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.metrics import brier_score_loss
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class BrierResults:
    """Results container for Brier score analysis"""
    time_points: np.ndarray
    brier_scores: np.ndarray
    integrated_brier_score: float
    ipcw_weights: Optional[np.ndarray]
    scale_parameter: float
    confidence_intervals: Optional[Dict[str, np.ndarray]]
    metadata: Dict

class DistributionHandler(ABC):
    """Abstract base for survival distribution implementations"""
    
    @abstractmethod
    def estimate_scale_mle(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def estimate_scale_robust(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def survival_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        pass
    
    @abstractmethod
    def hazard_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        pass

class LogNormalHandler(DistributionHandler):
    """Handler for normal AFT distribution (log-normal survival times)"""
    
    def estimate_scale_mle(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """Maximum likelihood estimation of scale parameter"""
        uncensored_mask = events.astype(bool)
        if np.sum(uncensored_mask) < 10:
            return self.estimate_scale_robust(predictions, times, events)
        
        log_times = np.log(times[uncensored_mask])
        residuals = log_times - predictions[uncensored_mask]
        
        # MLE for normal distribution
        n = len(residuals)
        sigma_mle = np.sqrt(np.sum(residuals**2) / n)
        
        # Apply bias correction for small samples
        if n < 30:
            correction = np.sqrt(n / (n - 1))
            sigma_mle *= correction
            
        return max(sigma_mle, 0.1)  # Numerical stability
    
    def estimate_scale_robust(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """Robust scale estimation using MAD"""
        uncensored_mask = events.astype(bool)
        if np.sum(uncensored_mask) < 5:
            return 1.0  # Default fallback
            
        log_times = np.log(times[uncensored_mask])
        residuals = log_times - predictions[uncensored_mask]
        
        # Median Absolute Deviation approach
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma_mad = mad / stats.norm.ppf(0.75)
        
        return max(sigma_mad, 0.1)
    
    def survival_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        """S(t) = P(T > t) for log-normal distribution"""
        z = (np.log(t) - location) / scale
        return 1 - stats.norm.cdf(z)
    
    def hazard_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        """Hazard function h(t) = f(t) / S(t)"""
        z = (np.log(t) - location) / scale
        pdf = stats.norm.pdf(z) / (scale * t)
        survival = self.survival_function(t, location, scale)
        return pdf / (survival + 1e-8)

class LogLogisticHandler(DistributionHandler):
    """Handler for logistic AFT distribution (log-logistic survival times)"""
    
    def estimate_scale_mle(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """MLE estimation using scipy optimization"""
        uncensored_mask = events.astype(bool)
        if np.sum(uncensored_mask) < 10:
            return self.estimate_scale_robust(predictions, times, events)
        
        log_times = np.log(times[uncensored_mask])
        locations = predictions[uncensored_mask]
        
        def neg_log_likelihood(sigma):
            if sigma <= 0:
                return np.inf
            z = (log_times - locations) / sigma
            # Log-logistic log-likelihood
            ll = np.sum(-np.log(sigma) - z - 2 * np.log(1 + np.exp(-z)))
            return -ll
        
        result = optimize.minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), method='bounded')
        return max(result.x, 0.1)
    
    def estimate_scale_robust(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """Robust estimation using moment matching"""
        uncensored_mask = events.astype(bool)
        if np.sum(uncensored_mask) < 5:
            return 1.0
            
        log_times = np.log(times[uncensored_mask])
        residuals = log_times - predictions[uncensored_mask]
        
        # Theoretical relationship: Var(logistic) = (π²/3) * σ²
        sigma_moment = np.std(residuals) * np.sqrt(3) / np.pi
        return max(sigma_moment, 0.1)
    
    def survival_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        """S(t) for log-logistic distribution"""
        z = (np.log(t) - location) / scale
        return 1 / (1 + np.exp(z))
    
    def hazard_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        """Non-monotonic hazard function for log-logistic"""
        z = (np.log(t) - location) / scale
        exp_z = np.exp(z)
        return exp_z / (scale * t * (1 + exp_z))

class WeibullHandler(DistributionHandler):
    """Handler for extreme AFT distribution (Weibull survival times)"""
    
    def estimate_scale_mle(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """MLE using Weibull regression theory"""
        uncensored_mask = events.astype(bool)
        if np.sum(uncensored_mask) < 10:
            return self.estimate_scale_robust(predictions, times, events)
        
        # Transform to Gumbel distribution on log scale
        log_times = np.log(times[uncensored_mask])
        locations = predictions[uncensored_mask]
        residuals = log_times - locations
        
        # MLE for Gumbel distribution
        def gumbel_mle(sigma):
            if sigma <= 0:
                return np.inf
            z = residuals / sigma
            ll = np.sum(-np.log(sigma) - z - np.exp(-z))
            return -ll
        
        result = optimize.minimize_scalar(gumbel_mle, bounds=(0.1, 5.0), method='bounded')
        return max(result.x, 0.1)
    
    def estimate_scale_robust(self, predictions: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """Robust estimation using Gumbel moments"""
        uncensored_mask = events.astype(bool)
        if np.sum(uncensored_mask) < 5:
            return 1.0
            
        log_times = np.log(times[uncensored_mask])
        residuals = log_times - predictions[uncensored_mask]
        
        # Gumbel distribution: Var = (π²/6) * σ²
        sigma_moment = np.std(residuals) * np.sqrt(6) / np.pi
        return max(sigma_moment, 0.1)
    
    def survival_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        """S(t) for Weibull distribution"""
        z = (np.log(t) - location) / scale
        return np.exp(-np.exp(z))
    
    def hazard_function(self, t: np.ndarray, location: float, scale: float) -> np.ndarray:
        """Monotonically increasing hazard for Weibull"""
        z = (np.log(t) - location) / scale
        return np.exp(z) / (scale * t)

class IPCWEstimator:
    """Inverse Probability of Censoring Weighting implementation"""
    
    @staticmethod
    def kaplan_meier_weights(times: np.ndarray, events: np.ndarray, 
                           eval_times: np.ndarray) -> np.ndarray:
        """Standard KM-based IPCW weights"""
        try:
            kmf = KaplanMeierFitter()
            # Fit on censoring distribution (flip events)
            kmf.fit(times, 1 - events)
            
            weights = np.zeros(len(times))
            for i, (t, e) in enumerate(zip(times, events)):
                if e == 1:  # Event occurred
                    # Weight = 1 / P(C >= T_i)
                    surv_prob = kmf.survival_function_at_times(t).iloc[0]
                    weights[i] = 1.0 / max(surv_prob, 0.01)  # Avoid division by zero
                else:  # Censored
                    weights[i] = 0.0  # Censored observations get zero weight
                    
            return weights
            
        except Exception:
            # Fallback to uniform weights
            return np.ones(len(times))
    
    @staticmethod
    def cox_weights(times: np.ndarray, events: np.ndarray, 
                   covariates: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
        """Cox regression based IPCW weights"""
        try:
            # Prepare data for Cox model
            censor_data = pd.DataFrame(covariates)
            censor_data['duration'] = times
            censor_data['censored'] = 1 - events  # Flip for censoring model
            
            if censor_data['censored'].sum() < 5:
                return IPCWEstimator.kaplan_meier_weights(times, events, eval_times)
            
            cph = CoxPHFitter(penalizer=0.1)  # Small penalization for stability
            cph.fit(censor_data, 'duration', 'censored')
            
            # Predict censoring survival probabilities
            surv_func = cph.predict_survival_function(censor_data.drop(['duration', 'censored'], axis=1))
            
            weights = np.zeros(len(times))
            for i, (t, e) in enumerate(zip(times, events)):
                if e == 1:
                    # Find closest time point in survival function
                    closest_idx = np.argmin(np.abs(surv_func.index - t))
                    surv_prob = surv_func.iloc[closest_idx, i]
                    weights[i] = 1.0 / max(surv_prob, 0.01)
                else:
                    weights[i] = 0.0
                    
            return weights
            
        except Exception:
            return IPCWEstimator.kaplan_meier_weights(times, events, eval_times)

class AdvancedBrierScoreCalculator:
    """Brier score calculator for XGBoost AFT models"""
    
    DISTRIBUTION_HANDLERS = {
        'normal': LogNormalHandler,
        'logistic': LogLogisticHandler, 
        'extreme': WeibullHandler
    }
    
    def __init__(self, distribution_type: str = 'normal', 
                 scale_estimation_method: str = 'mle',
                 ipcw_method: str = 'kaplan_meier',
                 confidence_level: float = 0.95):
        """
        Initialize calculator with distribution and estimation methods
        
        Parameters:
        -----------
        distribution_type : {'normal', 'logistic', 'extreme'}
        scale_estimation_method : {'mle', 'robust'}
        ipcw_method : {'kaplan_meier', 'cox', 'none'}
        confidence_level : float, confidence level for intervals
        """
        if distribution_type not in self.DISTRIBUTION_HANDLERS:
            raise ValueError(f"Unknown distribution: {distribution_type}")
        
        self.distribution_type = distribution_type
        self.scale_method = scale_estimation_method
        self.ipcw_method = ipcw_method
        self.confidence_level = confidence_level
        
        self.handler = self.DISTRIBUTION_HANDLERS[distribution_type]()
        self._fitted_scale = None
        self._last_predictions = None
    
    def estimate_scale_parameter(self, predictions: np.ndarray, 
                               observed_times: np.ndarray, 
                               events: np.ndarray) -> float:
        """Estimate scale parameter using specified method"""
        if self.scale_method == 'mle':
            scale = self.handler.estimate_scale_mle(predictions, observed_times, events)
        elif self.scale_method == 'robust':
            scale = self.handler.estimate_scale_robust(predictions, observed_times, events)
        else:
            raise ValueError(f"Unknown scale estimation method: {self.scale_method}")
        
        self._fitted_scale = scale
        return scale
    
    def survival_probability_matrix(self, time_points: np.ndarray, 
                                  predictions: np.ndarray, 
                                  scale: float) -> np.ndarray:
        """Calculate survival probabilities S(t|x) for all time points and predictions"""
        n_obs = len(predictions)
        n_times = len(time_points)
        surv_matrix = np.zeros((n_obs, n_times))
        
        for i, pred in enumerate(predictions):
            surv_matrix[i, :] = self.handler.survival_function(time_points, pred, scale)
        
        return surv_matrix
    
    def _compute_time_dependent_brier(self, time_point: float,
                                    predictions: np.ndarray,
                                    observed_times: np.ndarray,
                                    events: np.ndarray,
                                    scale: float,
                                    weights: Optional[np.ndarray] = None) -> Dict:
        """Compute Brier score at specific time point"""
        n = len(predictions)
        
        # Survival probabilities at time_point
        surv_probs = self.handler.survival_function(time_point, predictions, scale)
        
        # Binary outcomes at time_point
        outcomes = ((observed_times <= time_point) & (events == 1)).astype(float)
        
        # Determine who is at risk at time_point
        at_risk = observed_times >= time_point
        
        if weights is None:
            weights = np.ones(n)
        
        # IPCW-adjusted Brier score calculation
        brier_components = np.zeros(n)
        valid_mask = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if events[i] == 1 and observed_times[i] <= time_point:
                # Event before time_point
                brier_components[i] = weights[i] * (surv_probs[i]**2)
                valid_mask[i] = True
            elif observed_times[i] > time_point:
                # At risk at time_point
                brier_components[i] = weights[i] * ((1 - surv_probs[i])**2)
                valid_mask[i] = True
        
        if valid_mask.sum() == 0:
            return {'brier_score': np.nan, 'n_at_risk': 0, 'n_events': 0}
        
        brier_score = np.mean(brier_components[valid_mask])
        n_at_risk = at_risk.sum()
        n_events = ((observed_times <= time_point) & (events == 1)).sum()
        
        return {
            'brier_score': brier_score,
            'n_at_risk': n_at_risk,
            'n_events': n_events,
            'surv_probs_mean': np.mean(surv_probs),
            'surv_probs_std': np.std(surv_probs)
        }
    
    def bootstrap_confidence_intervals(self, predictions: np.ndarray,
                                     observed_times: np.ndarray,
                                     events: np.ndarray,
                                     time_points: np.ndarray,
                                     scale: float,
                                     n_bootstrap: int = 200) -> Dict[str, np.ndarray]:
        """Bootstrap confidence intervals for Brier scores"""
        n = len(predictions)
        bootstrap_scores = np.zeros((n_bootstrap, len(time_points)))
        
        np.random.seed(42)  # Reproducibility
        
        for b in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            boot_pred = predictions[indices]
            boot_times = observed_times[indices] 
            boot_events = events[indices]
            
            # Compute IPCW weights for bootstrap sample
            if self.ipcw_method == 'kaplan_meier':
                boot_weights = IPCWEstimator.kaplan_meier_weights(
                    boot_times, boot_events, time_points
                )
            else:
                boot_weights = None
            
            # Calculate Brier scores
            for t_idx, t in enumerate(time_points):
                result = self._compute_time_dependent_brier(
                    t, boot_pred, boot_times, boot_events, scale, boot_weights
                )
                bootstrap_scores[b, t_idx] = result['brier_score']
        
        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.nanpercentile(bootstrap_scores, lower_percentile, axis=0)
        ci_upper = np.nanpercentile(bootstrap_scores, upper_percentile, axis=0)
        
        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_scores': bootstrap_scores
        }
    
    def calculate_brier_scores(self, model, X_val: np.ndarray, 
                             y_val: np.ndarray, events_val: np.ndarray,
                             time_points: Optional[np.ndarray] = None,
                             X_train: Optional[np.ndarray] = None,
                             y_train: Optional[np.ndarray] = None,
                             events_train: Optional[np.ndarray] = None,
                             use_ipcw: bool = True,
                             compute_ci: bool = False,
                             n_bootstrap: int = 200) -> BrierResults:
        """
        Calculate comprehensive Brier score analysis
        
        Parameters:
        -----------
        model : xgboost model
        X_val : validation features  
        y_val : validation survival times
        events_val : validation event indicators
        time_points : evaluation time points (optional)
        X_train : training features for scale estimation (optional)
        y_train : training survival times for scale estimation (optional) 
        events_train : training events for scale estimation (optional)
        use_ipcw : whether to use IPCW correction
        compute_ci : whether to compute confidence intervals
        n_bootstrap : number of bootstrap samples for CI
        """
        # Generate predictions
        import xgboost as xgb
        if isinstance(X_val, np.ndarray):
            dval = xgb.DMatrix(X_val)
        else:
            dval = X_val
        predictions = model.predict(dval)
        
        # Estimate scale parameter
        if X_train is not None and y_train is not None and events_train is not None:
            if isinstance(X_train, np.ndarray):
                dtrain = xgb.DMatrix(X_train)
            else:
                dtrain = X_train
            train_predictions = model.predict(dtrain)
            scale = self.estimate_scale_parameter(train_predictions, y_train, events_train)
        else:
            # Fall back to validation data for scale estimation
            scale = self.estimate_scale_parameter(predictions, y_val, events_val)
        
        # Default time points if not provided
        if time_points is None:
            max_time = min(np.max(y_val), 730)  # Cap at 2 years
            time_points = np.concatenate([
                np.arange(30, 91, 30),      # Monthly for first quarter
                np.arange(90, 366, 90),     # Quarterly for first year  
                np.arange(365, max_time, 182)  # Semi-annual beyond
            ])
        
        # Compute IPCW weights
        ipcw_weights = None
        if use_ipcw:
            if self.ipcw_method == 'kaplan_meier':
                ipcw_weights = IPCWEstimator.kaplan_meier_weights(
                    y_val, events_val, time_points
                )
            elif self.ipcw_method == 'cox' and X_val is not None:
                ipcw_weights = IPCWEstimator.cox_weights(
                    y_val, events_val, X_val, time_points
                )
        
        # Calculate time-dependent Brier scores
        brier_scores = []
        detailed_results = []
        
        for t in time_points:
            result = self._compute_time_dependent_brier(
                t, predictions, y_val, events_val, scale, ipcw_weights
            )
            brier_scores.append(result['brier_score'])
            detailed_results.append(result)
        
        brier_scores = np.array(brier_scores)
        
        # Calculate Integrated Brier Score
        valid_mask = ~np.isnan(brier_scores)
        if valid_mask.sum() < 2:
            ibs = np.nan
        else:
            valid_times = time_points[valid_mask]
            valid_scores = brier_scores[valid_mask]
            # Trapezoidal integration weighted by time interval
            ibs = integrate.trapz(valid_scores, valid_times) / (valid_times[-1] - valid_times[0])
        
        # Bootstrap confidence intervals
        confidence_intervals = None
        if compute_ci and not np.all(np.isnan(brier_scores)):
            confidence_intervals = self.bootstrap_confidence_intervals(
                predictions, y_val, events_val, time_points[valid_mask], 
                scale, n_bootstrap
            )
        
        # Compile metadata
        metadata = {
            'distribution_type': self.distribution_type,
            'scale_method': self.scale_method,
            'ipcw_method': self.ipcw_method if use_ipcw else 'none',
            'n_observations': len(y_val),
            'n_events': np.sum(events_val),
            'event_rate': np.mean(events_val),
            'scale_parameter': scale,
            'max_follow_up': np.max(y_val),
            'detailed_results': detailed_results
        }
        
        return BrierResults(
            time_points=time_points,
            brier_scores=brier_scores,
            integrated_brier_score=ibs,
            ipcw_weights=ipcw_weights,
            scale_parameter=scale,
            confidence_intervals=confidence_intervals,
            metadata=metadata
        )
    
    def compare_models(self, models: Dict[str, any], 
                      X_val: np.ndarray, y_val: np.ndarray, events_val: np.ndarray,
                      time_points: Optional[np.ndarray] = None,
                      **kwargs) -> pd.DataFrame:
        """Compare multiple models using Brier scores"""
        results = []
        
        for model_name, model in models.items():
            try:
                brier_result = self.calculate_brier_scores(
                    model, X_val, y_val, events_val, time_points, **kwargs
                )
                
                results.append({
                    'model': model_name,
                    'integrated_brier_score': brier_result.integrated_brier_score,
                    'mean_brier_score': np.nanmean(brier_result.brier_scores),
                    'scale_parameter': brier_result.scale_parameter,
                    'n_valid_timepoints': np.sum(~np.isnan(brier_result.brier_scores))
                })
                
            except Exception as e:
                results.append({
                    'model': model_name,
                    'integrated_brier_score': np.nan,
                    'mean_brier_score': np.nan,
                    'scale_parameter': np.nan,
                    'n_valid_timepoints': 0,
                    'error': str(e)
                })
        
        return pd.DataFrame(results).sort_values('integrated_brier_score')

# Convenience function for quick analysis
def quick_brier_analysis(model, X_val, y_val, events_val, 
                        distribution='normal', scale_method='mle',
                        use_ipcw=True) -> BrierResults:
    """Quick Brier score analysis with sensible defaults"""
    calculator = AdvancedBrierScoreCalculator(
        distribution_type=distribution,
        scale_estimation_method=scale_method,
        ipcw_method='kaplan_meier' if use_ipcw else 'none'
    )
    
    return calculator.calculate_brier_scores(
        model, X_val, y_val, events_val, use_ipcw=use_ipcw
    )

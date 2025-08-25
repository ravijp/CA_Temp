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
        """
        Correct IPCW weights using Kaplan-Meier estimator for censoring distribution
        
        Weight formula: w_i = δ_i / G(T_i) + (1-δ_i) / G(C_i)
        where G is the survival function of censoring distribution
        """
        try:
            n_samples = len(times)
            weights = np.ones(n_samples)
            
            # For very large datasets, use simplified calculation
            if n_samples > 1000000:
                print(f"Using simplified IPCW for {n_samples:,} samples")
                # Simplified approach: uniform weights adjusted by censoring rate
                censoring_rate = 1 - np.mean(events)
                base_weight = 1.0 / (1 - censoring_rate/2)  # Adjusted formula
                return np.full(n_samples, base_weight)
            
            # Fit KM on CENSORING distribution (not event distribution)
            kmf = KaplanMeierFitter()
            kmf.fit(times, 1 - events, label='Censoring')
            
            # For each observation, calculate appropriate weight
            for i in range(n_samples):
                t_i = times[i]
                
                try:
                    # Get censoring survival probability at time t_i
                    G_ti = kmf.survival_function_at_times(t_i).iloc[0]
                    
                    # Ensure G(t_i) doesn't get too small (numerical stability)
                    G_ti = max(G_ti, 0.01)
                    
                    # IPCW weight (same formula for both censored and uncensored)
                    # This is correct: ALL observations get weighted by censoring probability
                    weights[i] = 1.0 / G_ti
                    
                except Exception:
                    # If KM fails at this time point, use default weight
                    weights[i] = 1.0
            
            # Normalize weights to sum to n (optional but helps interpretation)
            weights = weights * (n_samples / weights.sum())
            
            return weights
            
        except Exception as e:
            print(f"Warning: IPCW calculation failed: {e}, using uniform weights")
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
    
    def _compute_time_dependent_brier(self,
                                    time_point: float,
                                    predictions: np.ndarray,
                                    observed_times: np.ndarray,
                                    events: np.ndarray,
                                    scale: float,
                                    weights: Optional[np.ndarray] = None) -> Dict:
        """
        IPCW Brier score at `time_point` (Graf et al., 1999).

        Keeps the same signature. `weights` (if provided) are *external*
        per-subject multipliers and are applied on top of IPCW.
        """
        n = len(predictions)

        # survival and event probabilities at t
        surv_probs = self.handler.survival_function(time_point, predictions, scale)
        event_probs = 1.0 - surv_probs

        # External weights (class weights etc.)
        if weights is None:
            w_ext = np.ones(n, dtype=float)
        else:
            w_ext = np.asarray(weights, dtype=float)

        T = np.asarray(observed_times, dtype=float)
        D = np.asarray(events, dtype=int)

        # ---- IPCW pieces: KM of censoring distribution G(t) ----
        # Cache across calls to avoid O(n log n) per time point on large data
        if not hasattr(self, "_ipcw_cache") or self._ipcw_cache.get("n", None) != n \
        or self._ipcw_cache.get("hash", None) != (T.__array_interface__['data'][0], D.__array_interface__['data'][0]):
            # Build KM for censoring using numpy (fast + no per-row loops)
            order = np.argsort(T)
            t_sorted = T[order]
            cens_sorted = (1 - D)[order]

            uniq, idx = np.unique(t_sorted, return_index=True)
            at_risk = n - idx
            d_c = np.add.reduceat(cens_sorted, idx)  # # censored at each uniq time

            with np.errstate(divide="ignore", invalid="ignore"):
                factors = 1.0 - d_c / at_risk
                factors = np.clip(factors, 0.0, 1.0)

            G_right = np.cumprod(factors)                 # right-continuous G(t)
            G_left = np.empty_like(G_right)               # left limits G(t-)
            G_left[0] = 1.0
            if len(G_right) > 1:
                G_left[1:] = G_right[:-1]

            self._ipcw_cache = {
                "uniq": uniq,
                "G_right": G_right,
                "G_left": G_left,
                "n": n,
                "hash": (T.__array_interface__['data'][0], D.__array_interface__['data'][0])
            }

        uniq = self._ipcw_cache["uniq"]
        G_right = self._ipcw_cache["G_right"]
        G_left = self._ipcw_cache["G_left"]

        # Step-function evaluation (right- or left-continuous)
        def _step_eval(x_times, y_vals, x, left=False):
            x = np.asarray(x, dtype=float)
            idx = np.searchsorted(x_times, x, side=("left" if left else "right")) - 1
            idx = np.clip(idx, -1, len(x_times) - 1)
            out = np.ones_like(x, dtype=float)
            m = idx >= 0
            out[m] = y_vals[idx[m]]
            return out

        # Denominators: G(t) and G(T_i^-)
        G_t = _step_eval(uniq, G_right, np.array([time_point]))[0]
        G_T_left = _step_eval(uniq, G_left, T, left=True)
        eps = 1e-12
        G_t = max(G_t, eps)
        G_T_left = np.maximum(G_T_left, eps)

        # Indicators
        died_by_t = (T <= time_point) & (D == 1)
        alive_at_t = T > time_point

        # Graf et al. IPCW terms
        term_event = np.zeros(n, dtype=float)
        term_event[died_by_t] = (event_probs[died_by_t] - 1.0) ** 2 / G_T_left[died_by_t]

        term_alive = np.zeros(n, dtype=float)
        term_alive[alive_at_t] = (event_probs[alive_at_t] - 0.0) ** 2 / G_t

        contrib = (term_event + term_alive) * w_ext
        brier_score = float(contrib.sum() / w_ext.sum())

        n_at_risk = int((T > time_point).sum())
        n_events = int(((T <= time_point) & (D == 1)).sum())
        event_rate_actual = n_events / (n_events + n_at_risk) if (n_events + n_at_risk) > 0 else 0.0

        return {
            "brier_score": brier_score,
            "n_at_risk": n_at_risk,
            "n_events": n_events,
            "event_probs_mean": float(np.mean(event_probs)),
            "event_probs_std": float(np.std(event_probs)),
            "actual_event_rate": float(event_rate_actual),
            "calibration_gap": float(abs(event_rate_actual - np.mean(event_probs)))
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
    
    def calculate_brier_scores(self, model, X_val: np.ndarray, y_val: np.ndarray, 
                            events_val: np.ndarray, time_points: Optional[np.ndarray] = None,
                            use_ipcw: bool = True, compute_ci: bool = False, 
                            n_bootstrap: int = 0) -> BrierResults:
        """
        Calculate time-dependent Brier scores with optional IPCW correction
        OPTIMIZED: Bootstrap removed by default for production speed
        
        Parameters:
        -----------
        model : XGBoost AFT model
        X_val : Validation features
        y_val : Observed survival times
        events_val : Event indicators (1=event, 0=censored)
        time_points : Evaluation time points (default: automatic selection)
        use_ipcw : Use inverse probability of censoring weighting
        compute_ci : Compute confidence intervals (set to False for speed)
        n_bootstrap : Number of bootstrap iterations (0 for no bootstrap)
        
        Returns:
        --------
        BrierResults with time-dependent scores and integrated Brier score
        """
        # Get predictions from model
        dmatrix = xgb.DMatrix(X_val)
        predictions = model.predict(dmatrix)
        
        # Estimate or use fitted scale parameter
        scale = self.estimate_scale_parameter(predictions, y_val, events_val)
        
        # Define evaluation time points if not provided
        if time_points is None:
            max_time = np.percentile(y_val[events_val == 1], 90) if events_val.sum() > 0 else np.max(y_val)
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
            # Ensure integration from ~0 (BS(0)=0 for proper survival models)
            if valid_times[0] > 0:
                valid_times = np.insert(valid_times, 0, 0.0)
                valid_scores = np.insert(valid_scores, 0, 0.0)
            ibs = integrate.trapz(valid_scores, valid_times) / valid_times[-1]

        
        # Skip bootstrap by default for production speed
        confidence_intervals = None
        if compute_ci and n_bootstrap > 0:
            print(f"Computing bootstrap confidence intervals with {n_bootstrap} iterations...")
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
            'bootstrap_iterations': n_bootstrap if compute_ci else 0,
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

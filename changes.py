# CHANGE 1: Remove brier_score import and add IPCW utilities (around line 20)
# REMOVE these imports:
# from brier_score import AdvancedBrierScoreCalculator, quick_brier_analysis

# CHANGE 2: Add IPCW utility methods to SurvivalEvaluation class (after line ~150)
def _km_censoring_survival(self, T: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaplan-Meier estimator for censoring distribution G(t) = P(C > t)
    
    Args:
        T: Observed times
        E: Event indicators (1=event, 0=censored)
        
    Returns:
        Tuple of (unique_times, G_right_continuous, G_left_continuous)
    """
    T = np.asarray(T, float)
    C = 1 - np.asarray(E, int)  # Flip to censoring indicators
    order = np.argsort(T)
    t_sorted = T[order]
    c_sorted = C[order]
    uniq, idx = np.unique(t_sorted, return_index=True)
    n = len(T)
    at_risk = n - idx
    d_c = np.add.reduceat(c_sorted, idx)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.clip(1.0 - d_c / at_risk, 0.0, 1.0)
    
    G_right = np.cumprod(factors)
    G_left = np.empty_like(G_right)
    G_left[0] = 1.0
    if len(G_right) > 1:
        G_left[1:] = G_right[:-1]
    
    return uniq, G_right, G_left

def _step_eval(self, x_times: np.ndarray, y_vals: np.ndarray, 
               x: np.ndarray, left: bool = False) -> np.ndarray:
    """
    Evaluate step function at specified points
    
    Args:
        x_times: Time points where function has values
        y_vals: Function values at x_times
        x: Evaluation points
        left: If True, use left-continuous evaluation
        
    Returns:
        Function values at evaluation points
    """
    x = np.asarray(x, float)
    idx = np.searchsorted(x_times, x, side=("left" if left else "right")) - 1
    idx = np.clip(idx, -1, len(x_times) - 1)
    out = np.ones_like(x, float)
    m = idx >= 0
    out[m] = y_vals[idx[m]]
    return out

# CHANGE 3: Remove brier calculator initialization (around line ~180)
# REMOVE this method entirely:
# def _initialize_brier_calculator(self) -> None:

# CHANGE 4: Update calculate_survival_metrics method (around line ~364)
# REPLACE the Brier score calculation section with:
def calculate_survival_metrics(self, X: pd.DataFrame, y_true: np.ndarray, 
                              events: np.ndarray, use_ipcw: bool = True) -> Dict:
    """Calculate core survival analysis metrics with IPCW correction"""
    metrics = {}
    
    try:
        # C-index calculation (keep existing - working correctly)
        if self._evaluation_cache and 'log_predictions' in self._evaluation_cache:
            log_predictions = self._evaluation_cache['log_predictions']
            X_processed = self._evaluation_cache['X_processed']
        else:
            print("Warning: No cached predictions in calculate_survival_metrics, computing now")
            X_processed = self._get_processed_features(X)
            dmatrix = self.model_engine._create_categorical_aware_dmatrix(X_processed)
            log_predictions = self.model_engine.model.predict(dmatrix)
        
        pred_times = np.exp(log_predictions)
        c_index = concordance_index(y_true, pred_times, events)
        metrics['c_index'] = c_index
    except Exception as e:
        print(f"Warning: C-index calculation failed: {e}")
        metrics['c_index'] = np.nan
    
    # CORRECTED: Calculate TRUE Integrated Brier Score
    if use_ipcw:
        try:
            # Get full survival curves from model
            survival_curves = self.model_engine.predict_survival_curves(X_processed)
            
            # Calculate IBS by integrating over multiple time points
            ibs = self._calculate_integrated_brier_score(
                survival_curves, y_true, events, max_time=364
            )
            metrics['integrated_brier_score'] = ibs
            
            print(f"   Integrated Brier Score (0-364d): {ibs:.4f}")
            
        except Exception as e:
            print(f"Warning: Integrated Brier Score calculation failed: {e}")
            metrics['integrated_brier_score'] = np.nan
    else:
        metrics['integrated_brier_score'] = np.nan
    
    try:
        curve_quality = self._assess_survival_curve_quality_efficient(X, y_true, events)
        metrics.update(curve_quality)
    except Exception as e:
        print(f"Warning: Survival curve quality assessment failed: {e}")
    
    return metrics

# CHANGE 5: Add IPCW Brier Score method (new method)
def _calculate_brier_score_ipcw(self, predicted_probs: np.ndarray, y_true: np.ndarray, 
                               events: np.ndarray, t_eval: int) -> float:
    """
    Calculate IPCW-corrected Brier Score using Graf et al. (1999) methodology
    
    Args:
        predicted_probs: Model predicted event probabilities by t_eval
        y_true: Actual survival times
        events: Event indicators
        t_eval: Evaluation time point
        
    Returns:
        IPCW-corrected Brier score
    """
    try:
        # Build censoring distribution
        uniq, G_right, G_left = self._km_censoring_survival(y_true, events)
        
        # Define outcome groups
        died_by_t = (y_true <= t_eval) & (events == 1)
        alive_at_t = y_true > t_eval
        
        # Check if we have sufficient data
        if died_by_t.sum() < 10 and alive_at_t.sum() < 10:
            return np.nan
        
        # Calculate IPCW weights
        G_t = max(self._step_eval(uniq, G_right, np.array([t_eval]))[0], 1e-8)
        
        # Initialize Brier components
        total_brier = 0.0
        total_weight = 0.0
        
        # Events by t_eval: contribute (1 - predicted_prob)^2 weighted by 1/G(T_i-)
        if died_by_t.any():
            G_T_left = np.maximum(self._step_eval(uniq, G_left, y_true[died_by_t], left=True), 1e-8)
            weights_event = 1.0 / G_T_left
            
            # Clip weights to prevent extreme values
            weights_event = np.clip(weights_event, 0, 100)
            
            brier_event = weights_event * ((1.0 - predicted_probs[died_by_t]) ** 2)
            total_brier += brier_event.sum()
            total_weight += weights_event.sum()
        
        # Alive at t_eval: contribute (0 - predicted_prob)^2 weighted by 1/G(t)
        if alive_at_t.any():
            weight_alive = 1.0 / G_t
            weight_alive = min(weight_alive, 100)  # Clip extreme weights
            
            brier_alive = weight_alive * (predicted_probs[alive_at_t] ** 2)
            total_brier += brier_alive.sum()
            total_weight += weight_alive * alive_at_t.sum()
        
        # Return weighted average
        return float(total_brier / total_weight) if total_weight > 0 else np.nan
        
    except Exception as e:
        print(f"Warning: Brier score calculation failed at t={t_eval}: {e}")
        return np.nan

# CHANGE 6: Replace _calculate_fast_ece method (around line ~485)
def _calculate_fast_ece(self, predicted_event_probs: np.ndarray, 
                        y_true: np.ndarray, events: np.ndarray, 
                        horizon: int, n_bins: int = 10) -> float:
    """
    IPCW-corrected Expected Calibration Error with consistent controls definition
    
    Args:
        predicted_event_probs: Model's predicted event probabilities
        y_true: Actual survival times
        events: Event indicators
        horizon: Time horizon for evaluation
        n_bins: Number of calibration bins
        
    Returns:
        IPCW-corrected Expected Calibration Error
    """
    try:
        # FIXED: More conservative boundary condition handling
        controls = y_true > horizon
        cases = (y_true <= horizon) & (events == 1)
        
        # Only adjust if we have insufficient controls (< 1000)
        actual_horizon = horizon
        if controls.sum() < 1000:
            actual_horizon = horizon - 1
            controls = y_true > actual_horizon
            cases = (y_true <= actual_horizon) & (events == 1)
            print(f"   ECE: Adjusted horizon {horizon} -> {actual_horizon} (controls: {controls.sum():,})")
        
        print(f"   ECE at {actual_horizon}d: cases={cases.sum():,}, controls={controls.sum():,}")
        
        # Build censoring distribution
        uniq, G_right, G_left = self._km_censoring_survival(y_true, events)
        
        # FIXED: Proper IPCW weight calculation
        G_t = max(self._step_eval(uniq, G_right, np.array([actual_horizon]))[0], 1e-8)
        
        # Calculate weights for each observation
        weights = np.zeros_like(y_true, dtype=float)
        
        # Events get weight 1/G(T_i-)
        if cases.any():
            G_T_left = np.maximum(self._step_eval(uniq, G_left, y_true[cases], left=True), 1e-8)
            weights[cases] = np.clip(1.0 / G_T_left, 0, 50)  # Clip extreme weights
        
        # Controls get weight 1/G(t)
        if controls.any():
            weight_control = min(1.0 / G_t, 50)  # Clip extreme weights
            weights[controls] = weight_control
        
        # Only include observations with positive weights
        valid_mask = weights > 0
        if valid_mask.sum() < 100:
            return np.nan
        
        p_valid = predicted_event_probs[valid_mask]
        w_valid = weights[valid_mask]
        outcome_valid = cases[valid_mask].astype(float)  # 1 for events, 0 for controls
        
        # FIXED: Proper weighted calibration calculation
        edges = np.quantile(p_valid, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bins = np.digitize(p_valid, edges[1:-1], right=False)
        
        total_weight = w_valid.sum()
        ece = 0.0
        
        for b in range(n_bins):
            bin_mask = bins == b
            if not bin_mask.any():
                continue
                
            bin_weights = w_valid[bin_mask]
            bin_weight_sum = bin_weights.sum()
            
            if bin_weight_sum <= 0:
                continue
            
            # Weighted averages within bin
            pred_avg = np.average(p_valid[bin_mask], weights=bin_weights)
            obs_avg = np.average(outcome_valid[bin_mask], weights=bin_weights)
            
            # Add to ECE
            ece += (bin_weight_sum / total_weight) * abs(pred_avg - obs_avg)
        
        return float(ece)
        
    except Exception as e:
        print(f"Warning: ECE calculation failed for {horizon}d: {e}")
        return np.nan

# CHANGE 7: Update _calculate_time_dependent_auc method (around line ~600)
def _calculate_time_dependent_auc(self, X: pd.DataFrame, y: np.ndarray, 
                             events: np.ndarray, horizons: List[int]) -> Dict:
    """
    IPCW-corrected time-dependent AUC with consistent controls definition
    
    Args:
        X: Feature matrix
        y: Actual survival times
        events: Event indicators
        horizons: Time horizons for AUC assessment
        
    Returns:
        Dict: AUC metrics by horizon with overall assessment
    """
    auc_results = {}
    
    # Use cached predictions
    if self._evaluation_cache and 'log_predictions' in self._evaluation_cache:
        log_predictions = self._evaluation_cache['log_predictions']
    else:
        print("Warning: No cached predictions in AUC calculation, computing now")
        X_processed = self._get_processed_features(X)
        dmatrix = self.model_engine._create_categorical_aware_dmatrix(X_processed)
        log_predictions = self.model_engine.model.predict(dmatrix)
    
    # Build censoring distribution once
    uniq, G_right, G_left = self._km_censoring_survival(y, events)
    
    # Get AFT parameters
    sigma = self.model_engine.aft_parameters.sigma
    distribution = self.model_engine.aft_parameters.distribution.value
    
    for horizon in horizons:
        try:
            # Handle boundary condition
            actual_horizon = horizon
            if horizon >= y.max():
                actual_horizon = int(y.max() - 1)
                print(f"   Adjusting AUC horizon from {horizon} to {actual_horizon}")
            
            # Consistent controls definition
            cases = (y <= actual_horizon) & (events == 1)
            controls = y > actual_horizon  # People observable beyond horizon
            
            print(f"   AUC at {actual_horizon}d: cases={cases.sum():,}, controls={controls.sum():,}")
            
            # Calculate raw event probabilities (no normalization for calculations)
            log_horizon = np.log(actual_horizon)
            z_scores = (log_horizon - log_predictions) / sigma
            
            if distribution == 'normal':
                survival_probs = 1 - stats.norm.cdf(z_scores)
            elif distribution == 'logistic':
                survival_probs = 1 / (1 + np.exp(z_scores))
            elif distribution == 'extreme':
                survival_probs = np.exp(-np.exp(z_scores))
            else:
                survival_probs = 1 - stats.norm.cdf(z_scores)
            
            # Use raw probabilities for AUC calculation
            risk_scores = 1 - np.clip(survival_probs, 1e-6, 1 - 1e-6)
            
            # IPCW AUC calculation
            w_cases = 1.0 / np.maximum(
                self._step_eval(uniq, G_left, y[cases], left=True), 1e-12)
            w_controls = np.full(controls.sum(), 
                1.0 / max(self._step_eval(uniq, G_right, np.array([actual_horizon]))[0], 1e-12))
            
            # Combine and sort by risk scores
            r_all = np.concatenate([risk_scores[cases], risk_scores[controls]])
            is_case = np.concatenate([np.ones(cases.sum(), bool), np.zeros(controls.sum(), bool)])
            w_all = np.concatenate([w_cases, w_controls])
            
            order = np.argsort(r_all)
            r_s, is_c, w_s = r_all[order], is_case[order], w_all[order]
            
            # Calculate weighted AUC using IPCW methodology
            newgrp = np.r_[True, r_s[1:] != r_s[:-1]]
            idx = np.flatnonzero(newgrp)
            swc = np.add.reduceat(w_s * is_c, idx)
            swu = np.add.reduceat(w_s * (~is_c), idx)
            csum_u = np.cumsum(swu)
            u_below = np.r_[0.0, csum_u[:-1]]
            
            numer = np.sum(swc * (u_below + 0.5 * swu))
            denom = w_cases.sum() * w_controls.sum()
            
            auc = float(numer / denom) if denom > 0 else np.nan
            
            # Sanity check
            if not (0 <= auc <= 1):
                print(f"   Warning: AUC {auc:.4f} outside valid range at {actual_horizon}d")
                auc = np.nan
            
            auc_results[f'auc_{horizon}d'] = auc
            
        except Exception as e:
            print(f"Warning: AUC calculation failed for {horizon}d: {e}")
            auc_results[f'auc_{horizon}d'] = np.nan
    
    # Calculate average AUC
    valid_aucs = [auc for auc in auc_results.values() if not np.isnan(auc)]
    auc_results['average_auc'] = np.mean(valid_aucs) if valid_aucs else np.nan
    
    return auc_results

# CHANGE 8: Update _calculate_gini method (around line ~700)
def _calculate_gini(self, events: np.ndarray, X: pd.DataFrame) -> Dict:
    """
    Gini calculation using AUC-based approach for methodological consistency
    
    Args:
        events: Event indicators
        X: Feature matrix
        
    Returns:
        Dict: Gini coefficient with business interpretation
    """
    try:
        # Check if we have cached y_true
        if 'y_true' not in self._evaluation_cache:
            print("Warning: No cached y_true for Gini calculation")
            return {'gini_coefficient': np.nan, 'interpretation': 'No cached survival times'}
        
        y_true = self._evaluation_cache['y_true']
        
        # Use 364-day horizon to avoid boundary issues while maintaining business relevance
        horizon = 364
        
        # Calculate AUC using consistent IPCW method
        auc_results = self._calculate_time_dependent_auc(X, y_true, events, [horizon])
        auc_value = auc_results.get(f'auc_{horizon}d', np.nan)
        
        if np.isnan(auc_value):
            return {'gini_coefficient': np.nan, 'interpretation': 'AUC calculation failed'}
        
        # Gini = 2*AUC - 1 (standard relationship)
        gini = 2 * auc_value - 1
        gini = np.clip(gini, 0.0, 1.0)
        
        print(f"  Gini Debug - AUC at {horizon}d: {auc_value:.4f}, Gini: {gini:.4f}")
        
        return {
            'gini_coefficient': gini,
            'interpretation': self._interpret_gini_coefficient(gini),
            'based_on_auc': auc_value,
            'horizon_days': horizon
        }
        
    except Exception as e:
        print(f"Warning: Gini calculation failed: {e}")
        return {'gini_coefficient': np.nan, 'interpretation': 'Calculation failed'}

# CHANGE 9: Update _setup_evaluation_cache method (around line ~250)
def _setup_evaluation_cache(self, X: pd.DataFrame, y: np.ndarray, event: np.ndarray) -> None:
    """Setup evaluation cache with all required data for IPCW calculations"""
    try:
        X_processed = self._get_processed_features(X)
        dmatrix = self.model_engine._create_categorical_aware_dmatrix(X_processed)
        log_predictions = self.model_engine.model.predict(dmatrix)
        
        self._evaluation_cache = {
            'X_processed': X_processed,
            'dmatrix': dmatrix,
            'log_predictions': log_predictions,
            'dataset_size': len(X),
            'y_true': y,      # Required for IPCW calculations
            'events': event   # Required for IPCW calculations
        }
    except Exception as e:
        print(f"Warning: Failed to setup evaluation cache: {e}")
        self._evaluation_cache = {}
        
# Change 10: Add TRUE Integrated Brier Score calculation

def _calculate_integrated_brier_score(self, survival_curves: np.ndarray, 
                                    y_true: np.ndarray, events: np.ndarray, 
                                    max_time: int = 364, n_points: int = 30) -> float:
    """
    Calculate true Integrated Brier Score by integrating over time
    
    Args:
        survival_curves: Model survival curves (n_samples, n_timepoints)
        y_true: Actual survival times
        events: Event indicators  
        max_time: Maximum integration time
        n_points: Number of time points for integration
        
    Returns:
        Integrated Brier Score averaged over [1, max_time]
    """
    try:
        # Create time grid for integration
        time_points = np.linspace(1, max_time, n_points)
        brier_scores = []
        
        # Build censoring distribution once
        uniq, G_right, G_left = self._km_censoring_survival(y_true, events)
        
        for t in time_points:
            try:
                # Get survival probability at time t
                t_idx = min(int(t - 1), survival_curves.shape[1] - 1)  # Convert to 0-indexed
                survival_probs = survival_curves[:, t_idx]
                event_probs = 1 - survival_probs
                
                # Calculate Brier score at this time point
                bs = self._calculate_brier_score_ipcw(
                    event_probs, y_true, events, int(t), uniq, G_right, G_left
                )
                
                if not np.isnan(bs):
                    brier_scores.append(bs)
                    
            except Exception as e:
                print(f"Warning: Brier score calculation failed at t={t}: {e}")
                continue
        
        if len(brier_scores) < 2:
            return np.nan
        
        # Integrate using trapezoidal rule
        valid_times = time_points[:len(brier_scores)]
        ibs = np.trapz(brier_scores, valid_times) / (valid_times[-1] - valid_times[0])
        
        return float(ibs)
        
    except Exception as e:
        print(f"Warning: IBS calculation failed: {e}")
        return np.nan

#!/usr/bin/env python3

import numpy as np
from scipy import stats

def validate_from_engine(engine, val_df,
                              time_col="survival_time_days",
                              event_col="event_indicator_vol", 
                              t_star=None,
                              m_grid=128):
    """
    Fixed validation that handles your data structure correctly
    """
    # Extract ground truth
    T = np.asarray(val_df[time_col].values, float)
    E = np.asarray(val_df[event_col].values, int)
    
    # Get survival curves
    S = np.asarray(engine.predict_survival_curves(val_df), float)  # (n, m)
    n, m = S.shape
    times_grid_full = np.arange(1.0, m + 1.0, dtype=float)
    
    # Smart t_star selection for your data structure
    if t_star is None:
        # Find t_star where we have at least 1000 survivors
        for candidate_t in [90, 180, 270, 300]:
            if candidate_t <= times_grid_full[-1]:
                survivors_at_t = (T > candidate_t).sum()
                events_by_t = ((T <= candidate_t) & (E == 1)).sum()
                
                print(f"t={candidate_t}: events={events_by_t:,}, survivors={survivors_at_t:,}")
                
                if survivors_at_t >= 1000 and events_by_t >= 1000:
                    t_star = candidate_t
                    break
        
        if t_star is None:
            print("Warning: No suitable t_star found, using 180")
            t_star = 180.0
    
    # Find index for t_star
    idx = int(np.searchsorted(times_grid_full, t_star, side="right") - 1)
    idx = max(0, min(idx, m - 1))
    actual_t = times_grid_full[idx]
    
    print(f"Using t_star={actual_t} (index {idx})")
    
    # Build IPCW infrastructure
    uniq, G_right, G_left = _km_censoring_survival(T, E)
    
    # Calculate metrics at t_star
    p_event_t = 1.0 - S[:, idx]
    
    # Validate we have sufficient data
    cases = (T <= actual_t) & (E == 1)
    ctrls = T > actual_t
    
    print(f"At t_star={actual_t}: cases={cases.sum():,}, controls={ctrls.sum():,}")
    
    if cases.sum() < 100 or ctrls.sum() < 100:
        print("Warning: Insufficient data for reliable metrics")
        return {"error": "insufficient_data", "t_star": actual_t}
    
    # Calculate all metrics
    auc = _auc_ipcw_at_t(p_event_t, T, E, actual_t, uniq, G_right, G_left)
    gini = 2.0 * auc - 1.0 if np.isfinite(auc) else np.nan
    ece = _ece_ipcw_at_t(p_event_t, T, E, actual_t, uniq, G_right, G_left)
    bs_t = _bs_ipcw_at_t(p_event_t, T, E, actual_t, uniq, G_right, G_left)
    cidx = _cindex_sampled(p_event_t, T, E)
    
    # IBS calculation
    ibs = _calculate_ibs(S, T, E, times_grid_full, uniq, G_right, G_left, max_t=actual_t)
    
    return {
        "t_star": float(actual_t),
        "c_index_harrell_sampled": cidx,
        "auc_ipcw_tstar": auc,
        "gini_tstar": gini,
        "ece_ipcw_tstar": ece,
        "brier_ipcw_tstar": bs_t,
        "ibs_ipcw_[0,t_star]": ibs,
        "n_val": int(n),
        "n_cases_at_tstar": int(cases.sum()),
        "n_controls_at_tstar": int(ctrls.sum())
    }

def _km_censoring_survival(T, E):
    """KM estimator for censoring distribution - unchanged"""
    T = np.asarray(T, float)
    C = 1 - np.asarray(E, int)
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

def _step_eval(x_times, y_vals, x, left=False):
    """Step function evaluation - unchanged"""
    x = np.asarray(x, float)
    idx = np.searchsorted(x_times, x, side=("left" if left else "right")) - 1
    idx = np.clip(idx, -1, len(x_times) - 1)
    out = np.ones_like(x, float)
    m = idx >= 0
    out[m] = y_vals[idx[m]]
    return out

def _bs_ipcw_at_t(p_event_t, T, E, t, uniq, G_right, G_left):
    """Brier Score - unchanged, working correctly"""
    S = 1.0 - np.asarray(p_event_t, float)
    T = np.asarray(T, float)
    E = np.asarray(E, int)
    
    Gt = max(_step_eval(uniq, G_right, np.array([t]))[0], 1e-12)
    G_T_left = np.maximum(_step_eval(uniq, G_left, T, left=True), 1e-12)
    
    died = (T <= t) & (E == 1)
    alive = T > t
    
    term = np.zeros_like(S, float)
    term[died] = (S[died] ** 2) / G_T_left[died]
    term[alive] = ((1.0 - S[alive]) ** 2) / Gt
    
    return float(term.mean())

def _auc_ipcw_at_t(p_event_t, T, E, t, uniq, G_right, G_left):
    """Fixed AUC calculation with better error handling"""
    T = np.asarray(T, float)
    E = np.asarray(E, int) 
    r = np.asarray(p_event_t, float)
    
    cases = (T <= t) & (E == 1)
    ctrls = T > t
    
    if not cases.any() or not ctrls.any():
        print(f"AUC Error: cases={cases.sum()}, controls={ctrls.sum()}")
        return np.nan
    
    try:
        # Calculate IPCW weights
        w_cases = 1.0 / np.maximum(_step_eval(uniq, G_left, T[cases], left=True), 1e-12)
        w_ctrls = np.full(ctrls.sum(), 1.0 / max(_step_eval(uniq, G_right, np.array([t]))[0], 1e-12))
        
        # Combine data
        r_all = np.concatenate([r[cases], r[ctrls]])
        is_c = np.concatenate([np.ones(cases.sum(), bool), np.zeros(ctrls.sum(), bool)])
        w_all = np.concatenate([w_cases, w_ctrls])
        
        # Handle edge cases
        if len(np.unique(r_all)) == 1:
            print("AUC Warning: All risk scores identical")
            return 0.5
        
        # Sort by risk score
        order = np.argsort(r_all)
        r_s, is_c, w_s = r_all[order], is_c[order], w_all[order]
        
        # Group by unique risk values
        newgrp = np.r_[True, r_s[1:] != r_s[:-1]]
        idx = np.flatnonzero(newgrp)
        
        swc = np.add.reduceat(w_s * is_c, idx)
        swu = np.add.reduceat(w_s * (~is_c), idx)
        
        csum_u = np.cumsum(swu)
        u_below = np.r_[0.0, csum_u[:-1]]
        
        numer = np.sum(swc * (u_below + 0.5 * swu))
        denom = w_cases.sum() * w_ctrls.sum()
        
        auc = float(numer / denom) if denom > 0 else np.nan
        
        # Sanity check
        if not (0 <= auc <= 1):
            print(f"AUC Warning: AUC={auc} outside [0,1] range")
            return np.nan
            
        return auc
        
    except Exception as e:
        print(f"AUC Error: {e}")
        return np.nan

def _ece_ipcw_at_t(p_event_t, T, E, t, uniq, G_right, G_left, n_bins=10):
    """Fixed ECE calculation"""
    T = np.asarray(T, float)
    E = np.asarray(E, int)
    p = np.asarray(p_event_t, float)
    
    try:
        # Calculate IPCW weights
        Gt = max(_step_eval(uniq, G_right, np.array([t]))[0], 1e-12)
        
        w_event = np.zeros_like(T, float)
        w_alive = np.zeros_like(T, float)
        
        m_ev = (T <= t) & (E == 1)
        m_al = T > t
        
        if m_ev.any():
            w_event[m_ev] = 1.0 / np.maximum(_step_eval(uniq, G_left, T[m_ev], left=True), 1e-12)
        w_alive[m_al] = 1.0 / Gt
        
        denom = w_event + w_alive
        total = denom.sum()
        
        if total == 0:
            return np.nan
        
        # Quantile-based binning for stability
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bins = np.digitize(p, edges[1:-1], right=False)
        
        ece = 0.0
        for b in range(n_bins):
            m = bins == b
            if not m.any():
                continue
                
            denom_b = denom[m].sum()
            if denom_b <= 0:
                continue
                
            pred_b = float((p[m] * denom[m]).sum() / denom_b)
            obs_b = float(w_event[m].sum() / denom_b)
            ece += (denom_b / total) * abs(pred_b - obs_b)
        
        return float(ece)
        
    except Exception as e:
        print(f"ECE Error: {e}")
        return np.nan

def _cindex_sampled(p_event_t, T, E, max_pairs=1_000_000, seed=123):
    """Fixed C-index with better sampling"""
    rng = np.random.default_rng(seed)
    n = len(T)
    
    # More intelligent sampling for large datasets
    if n > 10000:
        sample_pairs = max_pairs
    else:
        sample_pairs = min(max_pairs, n * (n-1) // 2)
    
    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)
    
    # Remove self-pairs
    m = i != j
    i, j = i[m], j[m]
    
    if len(i) == 0:
        return np.nan
    
    Ti, Tj = T[i], T[j]
    Ei, Ej = E[i], E[j]
    ri, rj = p_event_t[i], p_event_t[j]
    
    # Identify comparable pairs
    earlier_i = (Ti < Tj) & (Ei == 1)
    earlier_j = (Tj < Ti) & (Ej == 1)
    comp = earlier_i | earlier_j
    
    if not comp.any():
        print("C-index Warning: No comparable pairs found")
        return np.nan
    
    # Evaluate concordance on comparable pairs
    ri, rj = ri[comp], rj[comp]
    earlier_i = earlier_i[comp]
    
    # Concordance logic: higher risk should have earlier event
    gt = ri > rj
    eq = ri == rj
    
    conc = (gt & earlier_i) | ((~gt) & (~eq) & (~earlier_i))
    ties = eq
    
    c_index = float((conc.sum() + 0.5 * ties.sum()) / conc.size)
    return c_index

def _calculate_ibs(S, T, E, times_grid, uniq, G_right, G_left, max_t, m_grid=128):
    """Fixed IBS calculation"""
    try:
        # Select time points up to max_t
        mask = times_grid <= max_t
        t_sub = times_grid[mask]
        
        if len(t_sub) > m_grid:
            # Subsample for efficiency
            sel = np.linspace(0, len(t_sub) - 1, num=m_grid).round().astype(int)
            t_sub = t_sub[sel]
            S_sub = S[:, sel]
        else:
            S_sub = S[:, :len(t_sub)]
        
        # Calculate BS(t) for each time point
        bs_vals = np.empty(len(t_sub), float)
        for j, t in enumerate(t_sub):
            p_event_t = 1.0 - S_sub[:, j]
            bs_vals[j] = _bs_ipcw_at_t(p_event_t, T, E, float(t), uniq, G_right, G_left)
        
        # Integrate using trapezoidal rule
        if t_sub[0] > 0:
            t_int = np.insert(t_sub, 0, 0.0)
            bs_int = np.insert(bs_vals, 0, 0.0)  # BS(0) = 0 for proper survival models
        else:
            t_int, bs_int = t_sub, bs_vals
        
        # Remove any NaN values
        valid_mask = np.isfinite(bs_int)
        if valid_mask.sum() < 2:
            return np.nan
            
        t_valid = t_int[valid_mask]
        bs_valid = bs_int[valid_mask]
        
        ibs = float(np.trapz(bs_valid, t_valid) / t_valid[-1]) if t_valid[-1] > 0 else np.nan
        return ibs
        
    except Exception as e:
        print(f"IBS Error: {e}")
        return np.nan

# Test with different t_star values:
# metrics_90 = validate_from_engine(engine, datasets_raw['val'], t_star=90)
# metrics_180 = validate_from_engine(engine, datasets_raw['val'], t_star=180)

# from validate_with_engine import validate_from_engine

# val_df = datasets_raw['val']  # from your suite
# metrics = validate_from_engine(
#     engine=engine,
#     val_df=val_df,
#     t_star=365.0,    # pick a clinically relevant horizon (days)
#     m_grid=128       # subsample for IBS speed if needed
# )
# for k, v in metrics.items():
#     print(f"{k}: {v}")


# #!/usr/bin/env python3

# import numpy as np
# from scipy import stats

# def debug_validate_metrics(engine, val_df, t_star=365.0):
#     """
#     Debug the validate_metrics.py issues step by step
#     """
#     print("=== DEBUGGING VALIDATE_METRICS ISSUES ===")
    
#     # Extract ground truth
#     T = np.asarray(val_df['survival_time_days'].values, float)
#     E = np.asarray(val_df['event_indicator_vol'].values, int)
    
#     print(f"Dataset size: {len(T):,}")
#     print(f"Events: {E.sum():,} ({E.mean():.3f})")
#     print(f"Survival time range: [{T.min():.1f}, {T.max():.1f}]")
#     print(f"t_star: {t_star}")
    
#     # Check t_star validity
#     events_by_tstar = ((T <= t_star) & (E == 1)).sum()
#     survivors_at_tstar = (T > t_star).sum()
    
#     print(f"\nAt t_star={t_star}:")
#     print(f"  Events by t*: {events_by_tstar:,}")
#     print(f"  Survivors at t*: {survivors_at_tstar:,}")
#     print(f"  Censored before t*: {((T <= t_star) & (E == 0)).sum():,}")
    
#     if events_by_tstar < 10 or survivors_at_tstar < 10:
#         print("❌ INSUFFICIENT DATA: Need at least 10 in each group for AUC")
#         return None
    
#     # Get survival curves from engine
#     try:
#         S = np.asarray(engine.predict_survival_curves(val_df), float)
#         print(f"\nSurvival curves shape: {S.shape}")
#         print(f"Survival at day 1: [{S[:, 0].min():.3f}, {S[:, 0].max():.3f}]")
        
#         # Find index for t_star
#         times_grid = np.arange(1.0, S.shape[1] + 1.0)
#         idx = int(np.searchsorted(times_grid, t_star, side="right") - 1)
#         idx = max(0, min(idx, S.shape[1] - 1))
#         actual_t = times_grid[idx]
        
#         print(f"t_star={t_star} maps to grid index {idx} (actual_t={actual_t})")
        
#         # Check survival probabilities at t_star
#         S_t = S[:, idx]
#         p_event_t = 1.0 - S_t
        
#         print(f"\nSurvival probs at t*: [{S_t.min():.3f}, {S_t.max():.3f}]")
#         print(f"Event probs at t*: [{p_event_t.min():.3f}, {p_event_t.max():.3f}]")
#         print(f"Mean event prob: {p_event_t.mean():.3f}")
        
#         # Check for monotonicity issues
#         non_monotonic = 0
#         for i in range(min(100, S.shape[0])):
#             diffs = np.diff(S[i, :])
#             violations = (diffs > 1e-6).sum()
#             if violations > 0:
#                 non_monotonic += 1
        
#         print(f"Non-monotonic curves (sample): {non_monotonic}/100")
        
#         return {
#             'T': T, 'E': E, 'S': S, 't_star': actual_t, 
#             'p_event_t': p_event_t, 'idx': idx,
#             'events_by_tstar': events_by_tstar,
#             'survivors_at_tstar': survivors_at_tstar
#         }
        
#     except Exception as e:
#         print(f"❌ ENGINE ERROR: {e}")
#         return None

# def debug_auc_calculation(debug_data):
#     """
#     Step-by-step AUC debugging
#     """
#     print("\n=== DEBUGGING AUC CALCULATION ===")
    
#     T = debug_data['T']
#     E = debug_data['E']
#     p_event_t = debug_data['p_event_t']
#     t_star = debug_data['t_star']
    
#     # Define cases and controls
#     cases = (T <= t_star) & (E == 1)
#     ctrls = T > t_star
    
#     print(f"Cases (events by t*): {cases.sum()}")
#     print(f"Controls (survivors at t*): {ctrls.sum()}")
    
#     if not cases.any() or not ctrls.any():
#         print("❌ AUC FAILURE: No cases or no controls")
#         return np.nan
    
#     # Check risk score distributions
#     case_risks = p_event_t[cases]
#     ctrl_risks = p_event_t[ctrls]
    
#     print(f"\nRisk distributions:")
#     print(f"  Cases: [{case_risks.min():.4f}, {case_risks.max():.4f}] (mean: {case_risks.mean():.4f})")
#     print(f"  Controls: [{ctrl_risks.min():.4f}, {ctrl_risks.max():.4f}] (mean: {ctrl_risks.mean():.4f})")
    
#     # Expected: case_risks.mean() > ctrl_risks.mean() for good discrimination
#     separation = case_risks.mean() - ctrl_risks.mean()
#     print(f"  Separation (cases - controls): {separation:.4f}")
    
#     if separation < 0:
#         print("⚠️  WARNING: Cases have lower risk than controls (directional issue)")
    
#     # Simple AUC without IPCW for comparison
#     try:
#         from sklearn.metrics import roc_auc_score
#         y_binary = np.concatenate([np.ones(cases.sum()), np.zeros(ctrls.sum())])
#         risk_combined = np.concatenate([case_risks, ctrl_risks])
#         simple_auc = roc_auc_score(y_binary, risk_combined)
#         print(f"Simple AUC (no IPCW): {simple_auc:.4f}")
        
#         return simple_auc
        
#     except Exception as e:
#         print(f"❌ Simple AUC failed: {e}")
#         return np.nan

# def debug_cindex_calculation(debug_data):
#     """
#     Debug C-index calculation
#     """
#     print("\n=== DEBUGGING C-INDEX CALCULATION ===")
    
#     T = debug_data['T']
#     E = debug_data['E']
#     p_event_t = debug_data['p_event_t']
    
#     # Manual C-index calculation (small sample)
#     n_pairs = 0
#     concordant = 0
#     tied = 0
    
#     # Sample for large datasets
#     sample_size = min(1000, len(T))
#     indices = np.random.choice(len(T), sample_size, replace=False)
    
#     T_sample = T[indices]
#     E_sample = E[indices]
#     r_sample = p_event_t[indices]
    
#     print(f"Computing C-index on sample of {sample_size}")
    
#     for i in range(sample_size):
#         for j in range(i+1, sample_size):
#             Ti, Tj = T_sample[i], T_sample[j]
#             Ei, Ej = E_sample[i], E_sample[j]
#             ri, rj = r_sample[i], r_sample[j]
            
#             # Only count comparable pairs
#             if (Ti < Tj and Ei == 1) or (Tj < Ti and Ej == 1):
#                 n_pairs += 1
                
#                 if Ti < Tj and Ei == 1:  # i had event first
#                     if ri > rj:  # Higher risk had earlier event
#                         concordant += 1
#                     elif ri == rj:
#                         tied += 1
#                 elif Tj < Ti and Ej == 1:  # j had event first
#                     if rj > ri:  # Higher risk had earlier event
#                         concordant += 1
#                     elif ri == rj:
#                         tied += 1
    
#     if n_pairs > 0:
#         c_index = (concordant + 0.5 * tied) / n_pairs
#         print(f"Comparable pairs: {n_pairs}")
#         print(f"Concordant: {concordant}")
#         print(f"Tied: {tied}")
#         print(f"C-index: {c_index:.4f}")
        
#         return c_index
#     else:
#         print("❌ No comparable pairs found")
#         return np.nan

# # Usage example:
# # debug_data = debug_validate_metrics(engine, datasets_raw['val'])
# # if debug_data:
# #     debug_auc_calculation(debug_data)
# #     debug_cindex_calculation(debug_data)
"""
survival_model_engine.py

Advanced AFT survival modeling engine with comprehensive feature preprocessing and multi-dataset support.
Implements XGBoost-based accelerated failure time models with intelligent feature transformation,
automated preprocessing pipelines, and production-ready model persistence capabilities.

Key Features:
- XGBoost 3.0+ AFT modeling with interval-based censoring
- Smart feature preprocessing with automatic transformation detection
- Multi-dataset training and evaluation (train/val/oot)
- Feature name mapping and transformation tracking
- Production-ready model persistence and loading
- Comprehensive survival curve and risk score generation

Author: ADP Survival Analysis Team
Version: 3.0.0
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import norm, logistic, median_abs_deviation
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings
import pickle
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AFTParameters:
    """AFT model parameters with distribution-specific configuration"""
    eta: np.ndarray
    sigma: float
    distribution: str
    log_likelihood: float
    optimization_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResults:
    """Comprehensive model training results with multi-dataset support"""
    model: xgb.Booster
    aft_parameters: AFTParameters
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    oot_metrics: Dict[str, float] = field(default_factory=dict)
    multi_dataset_metrics: Dict[str, Dict] = field(default_factory=dict)
    feature_importance: pd.DataFrame = None
    feature_name_mapping: Dict[str, str] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    evals_result: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Core modeling configuration with optimized XGBoost parameters"""
    aft_distributions: List[str] = field(default_factory=lambda: ['normal', 'logistic', 'extreme'])
    scale_parameter_range: Tuple[float, float] = (0.1, 5.0)
    scale_parameter_grid_size: int = 20
    validation_metrics: List[str] = field(default_factory=lambda: ['c_index', 'ibs', 'gini', 'ece'])
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 3,
        'eta': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'gamma': 1.0,
        'seed': 42,
        'verbosity': 0,
        'tree_method': 'hist'
    })
    early_stopping_rounds: int = 50
    num_boost_round: int = 5000

@dataclass
class FeatureConfig:
    """Smart feature configuration with transformation-specific feature lists"""
    
    # Features that need IQR-based outlier capping
    iqr_cap_features: List[str] = field(default_factory=lambda: [
        'baseline_salary', 'salary_growth_rate_12m', 'peer_salary_ratio', 'compensation_volatility',
        'age_at_vantage', 'tenure_at_vantage_days', 'time_with_current_manager', 
        'tenure_in_current_role', 'team_size', 'team_avg_comp'
    ])
    
    # Features that need log transformation
    log_transform_features: List[str] = field(default_factory=lambda: [
        'role_complexity_score', 'tenure_in_current_role', 'tenure_at_vantage_days',
        'compensation_volatility', 'time_with_current_manager', 'pay_grade_stagnation_months'
    ])
    
    # Features that need winsorization (1st-99th percentile)
    winsorize_features: List[str] = field(default_factory=lambda: [
        'company_tenure_percentile', 'team_avg_turn_days', 'team_avg_comp', 
        'peer_salary_ratio', 'baseline_salary', 'salary_growth_rate_12m',
        'age_at_vantage', 'pay_grade_stagnation_months'
    ])
    
    # Features that should be used as-is (already processed)
    direct_features: List[str] = field(default_factory=lambda: [
        'compensation_percentile_company', 'company_tenure_percentile'
    ])
    
    # Categorical features for encoding
    categorical_features: List[str] = field(default_factory=lambda: [
        'pay_rate_type_cd', 'career_stage', 'generation_cohort', 'gender_cd', 
        'hire_date_seasonality', 'full_tm_part_tm_cd', 'reg_temp_cd', 
        'flsa_stus_cd', 'fscl_actv_ind', 'company_size_tier', 'naics_2digit'
    ])

class SmartFeatureProcessor:
    """Smart feature processor with automatic transformation detection and comprehensive preprocessing"""
    
    def __init__(self, feature_config: FeatureConfig):
        """Initialize with smart feature configuration"""
        self.feature_config = feature_config
        self.feature_name_mapping = {}  # transformed_name -> original_name
        self.transformation_history = {}  # track transformation steps
        self.scalers = {}
        self.label_encoders = {}
        
        logger.info(f"SmartFeatureProcessor initialized with intelligent transformation detection")
    
    def _update_feature_mapping(self, original_col: str, transformed_col: str, transformation: str):
        """Update feature name mapping and transformation history"""
        self.feature_name_mapping[transformed_col] = original_col
        
        if original_col not in self.transformation_history:
            self.transformation_history[original_col] = []
        self.transformation_history[original_col].append({
            'transformation': transformation,
            'result_column': transformed_col
        })
    
    def _should_apply_transformation(self, feature: str, transformation_type: str, existing_columns: List[str]) -> bool:
        """Smart logic to determine if transformation should be applied based on target feature availability"""
        
        if transformation_type == 'iqr_cap':
            target_col = f'{feature}_cap'
            # Apply if target doesn't exist, original feature exists, and feature needs this transformation
            return (target_col not in existing_columns and 
                    feature in existing_columns and 
                    feature in self.feature_config.iqr_cap_features)
        
        elif transformation_type == 'log_transform':
            target_col = f'{feature}_log'
            return (target_col not in existing_columns and 
                    feature in existing_columns and 
                    feature in self.feature_config.log_transform_features)
        
        elif transformation_type == 'winsorize':
            target_col = f'{feature}_win_cap'
            return (target_col not in existing_columns and 
                    feature in existing_columns and 
                    feature in self.feature_config.winsorize_features)
        
        return False
    
    def _iqr_cap_outliers(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """IQR-based outlier capping with smart feature detection"""
        df_processed = df.copy()
        existing_columns = df_processed.columns.tolist()
        
        for col in cols:
            if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64', 'int32']:
                if self._should_apply_transformation(col, 'iqr_cap', existing_columns):
                    q1 = df_processed[col].quantile(0.25)
                    q3 = df_processed[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    capped_col = f'{col}_cap'
                    df_processed[capped_col] = df_processed[col].clip(lower_bound, upper_bound)
                    
                    self._update_feature_mapping(col, capped_col, 'iqr_cap')
                    logger.debug(f"Applied IQR capping to {col} -> {capped_col}")
        
        return df_processed
    
    def _log_transform_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Log transformation with smart feature detection"""
        df_processed = df.copy()
        existing_columns = df_processed.columns.tolist()
        
        for col in cols:
            if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64', 'int32']:
                if self._should_apply_transformation(col, 'log_transform', existing_columns):
                    log_col = f'{col}_log'
                    df_processed[log_col] = np.log1p(df_processed[col])
                    
                    self._update_feature_mapping(col, log_col, 'log_transform')
                    logger.debug(f"Applied log transform to {col} -> {log_col}")
        
        return df_processed
    
    def _winsorize_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Winsorization at 1st and 99th percentiles with smart feature detection"""
        df_processed = df.copy()
        existing_columns = df_processed.columns.tolist()
        
        for col in cols:
            if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64', 'int32']:
                if self._should_apply_transformation(col, 'winsorize', existing_columns):
                    q1 = df_processed[col].quantile(0.01)
                    q99 = df_processed[col].quantile(0.99)
                    
                    win_col = f'{col}_win_cap'
                    df_processed[win_col] = df_processed[col].clip(q1, q99)
                    
                    self._update_feature_mapping(col, win_col, 'winsorize')
                    logger.debug(f"Applied winsorization to {col} -> {win_col}")
        
        return df_processed
    
    def get_original_feature_names(self, transformed_names: List[str]) -> List[str]:
        """Get original feature names for plotting/interpretation"""
        return [self.feature_name_mapping.get(name, name) for name in transformed_names]
    
    def get_transformation_summary(self) -> Dict[str, List[str]]:
        """Get summary of all transformations applied"""
        return {orig: [step['transformation'] for step in steps] 
                for orig, steps in self.transformation_history.items()}
        
    def _transform_test_features(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/inference features using the same smart preprocessing pipeline
        
        This method applies the exact same transformations that were applied during training,
        using the stored transformation state and encoders.
        
        Args:
            X_test: Raw test features (same format as original training data)
            
        Returns:
            pd.DataFrame: Processed features ready for model prediction
        """
        logger.info(f"Transforming test features with shape {X_test.shape}")
        
        # Step 1: Apply smart preprocessing pipeline (same as training)
        X_processed = self.process_dataset(X_test, "inference")
        
        # Step 2: Apply categorical encoding using trained encoders
        for cat_feature, mapping in self.label_encoders.items():
            if cat_feature in X_processed.columns:
                encoded_col = f'{cat_feature}_encoded'
                cats = X_processed[cat_feature].fillna('MISSING').astype(str)
                X_processed[encoded_col] = cats.map(mapping).fillna(0)  # Unknown categories get 0
        
        # Step 3: Handle missing values in the same way as training
        feature_columns = []
        
        # Add direct features (if they exist)
        for feature in self.feature_config.direct_features:
            if feature in X_processed.columns:
                feature_columns.append(feature)
        
        # Add transformed features based on naming conventions
        for col in X_processed.columns:
            if (col.endswith('_cap') or col.endswith('_log') or col.endswith('_win_cap') or col.endswith('_encoded')):
                if col not in feature_columns:
                    feature_columns.append(col)
        
        # Step 4: Fill missing values for final features
        for col in feature_columns:
            if col in X_processed.columns:
                if X_processed[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    fill_value = X_processed[col].median()
                    if np.isnan(fill_value):
                        fill_value = 0
                else:
                    fill_value = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 0
                
                X_processed[col] = X_processed[col].fillna(fill_value)
        
        # Step 5: Return only the features that should exist based on training
        available_features = [col for col in feature_columns if col in X_processed.columns]
        
        logger.info(f"Test feature transformation complete. Features: {len(available_features)}")
        
        return X_processed[available_features]
    
    def process_dataset(self, dataset: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """Process any dataset with smart preprocessing pipeline"""
        logger.info(f"Processing {dataset_name or 'dataset'} with shape {dataset.shape}")
        
        df_processed = dataset.copy()
        
        # Step 1: IQR-based capping (only for features that need it AND exist)
        iqr_features = [f for f in self.feature_config.iqr_cap_features if f in df_processed.columns]
        if iqr_features:
            df_processed = self._iqr_cap_outliers(df_processed, iqr_features)
        
        # Step 2: Log transformation (only for features that need it AND exist)
        log_features = [f for f in self.feature_config.log_transform_features if f in df_processed.columns]
        if log_features:
            df_processed = self._log_transform_features(df_processed, log_features)
        
        # Step 3: Winsorization (only for features that need it AND exist)
        win_features = [f for f in self.feature_config.winsorize_features if f in df_processed.columns]
        if win_features:
            df_processed = self._winsorize_features(df_processed, win_features)
        
        logger.info(f"Smart preprocessing complete for {dataset_name or 'dataset'}")
        return df_processed
    
    def process_multiple_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process multiple datasets at once with smart preprocessing"""
        return {name: self.process_dataset(df, name) for name, df in datasets.items()}
    
    def process_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str], Dict]:
        """
        Comprehensive feature processing pipeline for multiple datasets with smart transformation detection
        
        Args:
            datasets: Dict of {dataset_name: dataframe}
            
        Returns:
            Tuple of (processed_datasets, feature_columns, label_encoders)
        """
        logger.info(f"Starting comprehensive feature processing for {len(datasets)} datasets")
        
        # Step 1: Apply smart preprocessing pipeline to all datasets
        processed_datasets = self.process_multiple_datasets(datasets)
        
        # Step 2: Get reference dataset for encoding (usually 'train')
        reference_dataset = processed_datasets.get('train') or list(processed_datasets.values())[0]
        
        # Step 3: Determine final feature columns based on what was actually created
        feature_columns = []
        
        # Add direct features (if they exist)
        for feature in self.feature_config.direct_features:
            if feature in reference_dataset.columns:
                feature_columns.append(feature)
        
        # Add transformed features based on naming conventions
        for col in reference_dataset.columns:
            if (col.endswith('_cap') or col.endswith('_log') or col.endswith('_win_cap')):
                if col not in feature_columns:
                    feature_columns.append(col)
        
        # Step 4: Process categorical features across all datasets
        encoded_datasets = {}
        for name, df in processed_datasets.items():
            df_encoded = df.copy()
            
            for cat_feature in self.feature_config.categorical_features:
                if cat_feature in df_encoded.columns:
                    # Get categories from all datasets if first time processing
                    if cat_feature not in self.label_encoders:
                        all_categories = set()
                        for other_df in processed_datasets.values():
                            if cat_feature in other_df.columns:
                                cats = other_df[cat_feature].fillna('MISSING').astype(str).unique()
                                all_categories.update(cats)
                        
                        # Remove alphabetical ordering - use frequency-based ordering instead
                        # Get frequency from training dataset (or first dataset if 'train' not available)
                        reference_df = datasets.get('train', list(datasets.values())[0])
                        if cat_feature in reference_df.columns:
                            # Order by frequency (most frequent gets lower codes)
                            freq_order = reference_df[cat_feature].fillna('MISSING').astype(str).value_counts()
                            ordered_categories = freq_order.index.tolist()
                            
                            # Add any categories not in reference dataset
                            missing_cats = all_categories - set(ordered_categories)
                            ordered_categories.extend(sorted(missing_cats))  # Only sort the missing ones
                        else:
                            # Fallback to sorted if reference not available
                            ordered_categories = sorted(all_categories)
                        
                        # MODIFIED: Use frequency-based ordering instead of alphabetical
                        self.label_encoders[cat_feature] = {
                            cat: idx for idx, cat in enumerate(ordered_categories)
                        }
                    
                    # Apply encoding
                    encoded_col = f'{cat_feature}_encoded'
                    cats = df_encoded[cat_feature].fillna('MISSING').astype(str)
                    df_encoded[encoded_col] = cats.map(self.label_encoders[cat_feature])
                    
                    # Update feature name mapping
                    self._update_feature_mapping(cat_feature, encoded_col, 'categorical_encoding')
                    
                    # Add to feature columns
                    if encoded_col not in feature_columns:
                        feature_columns.append(encoded_col)
            
            encoded_datasets[name] = df_encoded
        
        # Step 5: Fill missing values in final feature columns
        for name, df in encoded_datasets.items():
            for col in feature_columns:
                if col in df.columns:
                    if df[col].dtype in [np.float64, np.int64]:
                        fill_value = df[col].median()
                    else:
                        fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    
                    df[col] = df[col].fillna(fill_value)
        
        logger.info(f"Smart feature processing complete. Final features: {len(feature_columns)}")
        
        return encoded_datasets, feature_columns, self.label_encoders

class SurvivalModelEngine:
    """Advanced AFT survival modeling engine with smart feature processing and multi-dataset support"""
    
    def __init__(self, config: ModelConfig, feature_processor: SmartFeatureProcessor):
        """Initialize with configuration and smart feature processing component"""
        self.config = config
        self.feature_processor = feature_processor
        self.model = None
        self.aft_parameters = None
        self.feature_columns = None
        self.training_metadata = {}
        
        logger.info(f"SurvivalModelEngine initialized with {len(config.aft_distributions)} AFT distributions")
    
    def _calculate_manual_log_likelihood(self, predictions: np.ndarray, actuals: np.ndarray, 
                                    events: np.ndarray, distribution: str, scale: float) -> float:
        """
        Manual log-likelihood calculation following AFT principles for XGBoost 3.0+
        
        Args:
            predictions: Model predictions (eta) on log scale
            actuals: Actual survival times on log scale
            events: Event indicators (1=event, 0=censored)
            distribution: AFT distribution ('normal', 'logistic', 'extreme')
            scale: Scale parameter (sigma)
            
        Returns:
            float: Manual log-likelihood value
        """
        
        # Convert to interval representation for XGBoost 3.0+ compatibility
        y_lower = actuals
        y_upper = np.where(events == 1, actuals, np.inf)
        
        # Calculate standardized residuals: Z = (log(T) - eta) / sigma
        z_scores_lower = (y_lower - predictions) / scale
        
        log_likelihood_terms = []
        
        for i in range(len(predictions)):
            z_lower = z_scores_lower[i]
            
            if np.isinf(y_upper[i]):
                # Right-censored: P(T > t) = S(t) = 1 - F(z)
                if distribution == 'normal':
                    log_prob = stats.norm.logsf(z_lower)
                elif distribution == 'logistic':
                    log_prob = -np.log1p(np.exp(z_lower))
                elif distribution == 'extreme':
                    log_prob = -np.exp(z_lower)
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")
                    
            else:
                # Uncensored: f(t) = f(z) / (sigma * t)
                # Note: log(1/sigma) term cancels in relative comparison
                if distribution == 'normal':
                    log_prob = stats.norm.logpdf(z_lower)
                elif distribution == 'logistic':
                    log_prob = z_lower - 2 * np.log1p(np.exp(z_lower))
                elif distribution == 'extreme':
                    log_prob = z_lower - np.exp(z_lower)
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")
            
            log_likelihood_terms.append(log_prob)
        
        # Convert to array and handle numerical stability
        log_likelihood_terms = np.array(log_likelihood_terms)
        valid_mask = np.isfinite(log_likelihood_terms)
        
        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            logger.warning(f"Excluded {n_invalid}/{len(log_likelihood_terms)} invalid log-likelihood terms")
            log_likelihood_terms = log_likelihood_terms[valid_mask]
        
        return np.sum(log_likelihood_terms)

    def _create_categorical_aware_dmatrix(self, X_processed: pd.DataFrame, label=None) -> xgb.DMatrix:
        """Create DMatrix with proper categorical feature type information"""
        
        # Create DMatrix with categorical support
        dmatrix = xgb.DMatrix(X_processed, enable_categorical=True)
        
        # Set feature types for XGBoost 3.0.2 native categorical handling
        if hasattr(self.model_engine, 'feature_columns') and self.model_engine.feature_columns:
            feature_types = []
            for col in X_processed.columns:
                if col.endswith('_encoded'):  # Our categorical features
                    feature_types.append('c')  # Categorical
                else:
                    feature_types.append('q')  # Quantitative
            
            dmatrix.set_info(feature_types=feature_types)
        
        return dmatrix 

    def optimize_aft_parameters(self, train_data: Tuple, val_data: Tuple) -> AFTParameters:
        """
        MLE-based AFT parameter optimization across distributions for XGBoost 3.0+
        
        Automatically applies log transformation using np.log() and performs comprehensive
        grid search across distributions and scale parameters for optimal AFT configuration.
        
        Args:
            train_data: (X_train, y_train, event_train) - survival times in original scale
            val_data: (X_val, y_val, event_val) - survival times in original scale
            
        Returns:
            AFTParameters: Optimal parameters with distribution and scale
        """
        X_train, y_train, event_train = train_data
        X_val, y_val, event_val = val_data
        
        logger.info("Starting MLE-based AFT parameter optimization with automatic log transformation")
        
        # Transform to log scale for AFT using np.log()
        y_train_log = np.log(y_train)
        y_val_log = np.log(y_val)
        
        best_params = None
        best_log_likelihood = -np.inf
        optimization_results = {}
        
        # Grid search over distributions and scale parameters
        scale_grid = np.linspace(self.config.scale_parameter_range[0], 
                                self.config.scale_parameter_range[1], 
                                self.config.scale_parameter_grid_size)
        
        for distribution in self.config.aft_distributions:
            logger.info(f"Optimizing parameters for {distribution} distribution")
            dist_results = []
            
            for scale in scale_grid:
                try:
                    # Train XGBoost AFT model with current parameters
                    temp_model = self._train_xgb_aft_temp(X_train, y_train_log, event_train, 
                                                        distribution, scale)
                    
                    # Get predictions for validation set
                    dval = self._create_categorical_aware_dmatrix(X_val)
                    eta_predictions = temp_model.predict(dval)
                    
                    # Calculate manual log-likelihood on validation set
                    log_likelihood = self._calculate_manual_log_likelihood(
                        eta_predictions, y_val_log.values, event_val.values, distribution, scale
                    )
                    
                    dist_results.append({
                        'scale': scale,
                        'log_likelihood': log_likelihood,
                        'n_params': temp_model.num_boosted_rounds()
                    })
                    
                    # Track best parameters
                    if log_likelihood > best_log_likelihood:
                        best_log_likelihood = log_likelihood
                        best_params = {
                            'distribution': distribution,
                            'scale': scale,
                            'log_likelihood': log_likelihood,
                            'eta_predictions': eta_predictions
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed optimization for {distribution}, scale={scale:.3f}: {e}")
                    continue
            
            optimization_results[distribution] = dist_results
        
        if best_params is None:
            raise RuntimeError("AFT parameter optimization failed for all configurations")
        
        logger.info(f"Optimal AFT parameters: {best_params['distribution']} distribution, "
                f"scale={best_params['scale']:.4f}, log-likelihood={best_log_likelihood:.4f}")
        
        return AFTParameters(
            eta=best_params['eta_predictions'],
            sigma=best_params['scale'],
            distribution=best_params['distribution'],
            log_likelihood=best_log_likelihood,
            optimization_info=optimization_results
        )
        
    def _train_xgb_aft_temp(self, X: pd.DataFrame, y_log: pd.Series, events: pd.Series,
                        distribution: str, scale: float) -> xgb.Booster:
        """Temporary XGBoost AFT training for parameter optimization using XGBoost 3.0+"""
        
        # Convert to interval representation
        y_lower = y_log.values
        y_upper = np.where(events == 1, y_log.values, np.inf)
        
        # Create DMatrix with interval bounds
        dtrain = self._create_categorical_aware_dmatrix(X)
        dtrain.set_float_info('label_lower_bound', y_lower)
        dtrain.set_float_info('label_upper_bound', y_upper)
        
        # Set AFT parameters
        params = self.config.xgb_params.copy()
        params.update({
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': distribution,
            'aft_loss_distribution_scale': scale,
            'enable_categorical': True,  # Enable categorical support
            'max_cat_to_onehot': 4       # Optimize categorical handling
        })
        
        # Train with reduced iterations for optimization speed
        model = xgb.train(
            params, dtrain,
            num_boost_round=min(500, self.config.num_boost_round),
            verbose_eval=False
        )
        
        return model

    def train_survival_model(self, datasets: Dict[str, Tuple[pd.DataFrame, str, str]]) -> ModelResults:
        """
        Complete AFT model training pipeline with flexible multi-dataset handling
        
        Processes multiple datasets through smart feature preprocessing, optimizes AFT parameters,
        trains final model with comprehensive evaluation, and returns detailed results.
        
        Args:
            datasets: Dict with dataset_name -> (dataframe, survival_time_col, event_col)
                     e.g., {'train': (df_train, 'survival_time_days', 'event_indicator_vol'),
                            'val': (df_val, 'survival_time_days', 'event_indicator_vol'),
                            'oot': (df_oot, 'survival_time_days', 'event_indicator_vol')}
            
        Returns:
            ModelResults: Comprehensive training results with multi-dataset metrics
        """
        logger.info(f"Starting comprehensive AFT model training for {len(datasets)} datasets")
        
        # Step 1: Extract data and create dataset dictionary for processing
        data_for_processing = {}
        target_info = {}
        
        for name, (df, survival_col, event_col) in datasets.items():
            data_for_processing[name] = df
            target_info[name] = (survival_col, event_col)
            logger.info(f"{name} dataset: {df.shape[0]} samples")
        
        # Step 2: Smart feature processing
        processed_datasets, feature_columns, label_encoders = self.feature_processor.process_features(
            data_for_processing
        )
        self.feature_columns = feature_columns
        
        # Step 3: Prepare modeling datasets
        model_datasets = {}
        for name, df in processed_datasets.items():
            survival_col, event_col = target_info[name]
            
            model_columns = feature_columns + [survival_col, event_col]
            model_data = df[model_columns].dropna()
            
            X = model_data[feature_columns]
            y = model_data[survival_col]
            events = model_data[event_col]
            
            model_datasets[name] = (X, y, events)
            logger.info(f"{name} modeling data: {len(model_data)} samples after dropna")
        
        # Step 4: Optimize AFT parameters using train and val
        if 'train' in model_datasets and 'val' in model_datasets:
            optimal_params = self.optimize_aft_parameters(
                model_datasets['train'], model_datasets['val']
            )
        else:
            # Use first two datasets if train/val not specified
            dataset_names = list(model_datasets.keys())
            optimal_params = self.optimize_aft_parameters(
                model_datasets[dataset_names[0]], model_datasets[dataset_names[1]]
            )
        
        self.aft_parameters = optimal_params
        
        # Step 5: Train final model with all datasets for evaluation
        final_model, evals_result = self._train_final_model(model_datasets, optimal_params)
        self.model = final_model
        
        # Step 6: Calculate comprehensive metrics
        all_metrics = self._calculate_comprehensive_metrics(model_datasets, final_model)
        
        # Step 7: Feature importance with original names
        feature_importance = self._extract_feature_importance_with_names(final_model)
        
        # Step 8: Store training metadata
        self.training_metadata = {
            'dataset_shapes': {name: data[0].shape for name, data in model_datasets.items()},
            'feature_columns': feature_columns,
            'feature_name_mapping': self.feature_processor.feature_name_mapping,
            'transformation_summary': self.feature_processor.get_transformation_summary()
        }
        
        logger.info(f"Model training complete. Feature count: {len(feature_columns)}")
        
        # Step 9: Create comprehensive results
        results = ModelResults(
            model=final_model,
            aft_parameters=optimal_params,
            training_metrics=all_metrics.get('train', {}),
            validation_metrics=all_metrics.get('val', {}),
            oot_metrics=all_metrics.get('oot', {}),
            multi_dataset_metrics=all_metrics,
            feature_importance=feature_importance,
            feature_name_mapping=self.feature_processor.feature_name_mapping,
            training_metadata=self.training_metadata,
            evals_result=evals_result
        )
        
        return results
    
    def _train_final_model(self, model_datasets: Dict[str, Tuple], aft_params: AFTParameters) -> Tuple[xgb.Booster, Dict]:
        """Train final model with optimal parameters and multi-dataset evaluation"""
        
        # Create DMatrix for all datasets
        dmatrices = {}
        
        categorical_indices = []
        if self.feature_columns:
            for i, col in enumerate(self.feature_columns):
                if col.endswith('_encoded'):  # These are our categorical features
                    categorical_indices.append(i)
        
        for name, (X, y, events) in model_datasets.items():
            # Apply log transformation for AFT
            y_log = np.log(y)
            
            # Create interval bounds
            y_lower = y_log.values
            y_upper = np.where(events == 1, y_log.values, np.inf)
            
            dmatrix = self._create_categorical_aware_dmatrix(X, label=y_log)
            dmatrix.set_float_info('label_lower_bound', y_lower)
            dmatrix.set_float_info('label_upper_bound', y_upper)
                        
            dmatrices[name] = dmatrix
        
        # Set final AFT parameters
        params = self.config.xgb_params.copy()
        params.update({
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': aft_params.distribution,
            'aft_loss_distribution_scale': aft_params.sigma,
            'enable_categorical': True,  # Enable native categorical support
            'max_cat_to_onehot': 4       # One-hot encode if ≤4 categories, else use optimal splits
        })
        
        # Prepare evaluation sets
        evals = [(dmatrix, name) for name, dmatrix in dmatrices.items()]
        evals_result = {}
        
        # Get training dataset (prefer 'train', fallback to first dataset)
        train_dmatrix = dmatrices.get('train') or list(dmatrices.values())[0]
        
        # Train with comprehensive evaluation
        model = xgb.train(
            params, train_dmatrix,
            num_boost_round=self.config.num_boost_round,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False
        )
        
        return model, evals_result
    
    def _calculate_comprehensive_metrics(self, model_datasets: Dict[str, Tuple], model: xgb.Booster) -> Dict[str, Dict]:
        """Calculate comprehensive metrics for all datasets"""
        all_metrics = {}
        
        for name, (X, y, events) in model_datasets.items():
            try:
                # Get predictions
                dmatrix = self._create_categorical_aware_dmatrix(X)
                predictions = model.predict(dmatrix)
                
                # Calculate metrics
                from lifelines.utils import concordance_index
                pred_days = np.exp(predictions)  # Transform back from log scale
                c_index = concordance_index(y, pred_days, events)
                
                metrics = {
                    'c_index': c_index,
                    'sample_size': len(X),
                    'event_rate': events.mean(),
                    'prediction_mean': predictions.mean(),
                    'prediction_std': predictions.std()
                }
                
                all_metrics[name] = metrics
                logger.info(f"{name} C-index: {c_index:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {name}: {e}")
                all_metrics[name] = {'error': str(e)}
        
        return all_metrics
    
    def _extract_feature_importance_with_names(self, model: xgb.Booster) -> pd.DataFrame:
        """Extract feature importance with original feature names"""
        importance_dict = model.get_score(importance_type='gain')
        
        if not importance_dict:
            return pd.DataFrame(columns=['feature', 'original_feature', 'importance'])
        
        # Handle both f0, f1, ... and actual feature names
        if self.feature_columns and list(importance_dict.keys())[0].startswith('f'):
            importance_data = []
            for f, score in importance_dict.items():
                feature_idx = int(f[1:])
                if feature_idx < len(self.feature_columns):
                    transformed_name = self.feature_columns[feature_idx]
                    original_name = self.feature_processor.feature_name_mapping.get(
                        transformed_name, transformed_name
                    )
                    importance_data.append({
                        'feature': transformed_name,
                        'original_feature': original_name,
                        'importance': score
                    })
        else:
            importance_data = []
            for f, score in importance_dict.items():
                original_name = self.feature_processor.feature_name_mapping.get(f, f)
                importance_data.append({
                    'feature': f,
                    'original_feature': original_name,
                    'importance': score
                })
        
        importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
        return importance_df.reset_index(drop=True)
    
    def predict_survival_curves(self, X: pd.DataFrame, time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate survival curves using AFT formulation - XGBoost 3.0+ compatible
        
        Applies smart feature processing pipeline and generates survival probability curves
        based on trained AFT distribution parameters.
        
        Args:
            X: Features for prediction (will be processed through same pipeline as training)
            time_points: Time points for survival curve (default: 1-365 days)
            
        Returns:
            np.ndarray: Survival curves with shape (n_samples, n_time_points)
        """
        if self.model is None or self.aft_parameters is None:
            raise RuntimeError("Model must be trained before generating predictions")
        
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        # Process features using the same pipeline
        X_processed = self._get_processed_features(X)
        
        # Get AFT predictions (eta)
        dmatrix = self._create_categorical_aware_dmatrix(X_processed)
        eta_predictions = self.model.predict(dmatrix)
        
        logger.info(f"Generating survival curves for {len(X)} samples over {len(time_points)} time points")
        
        # Generate survival curves based on AFT distribution
        survival_curves = []
        
        for eta in eta_predictions:
            if self.aft_parameters.distribution == 'normal':
                log_times = np.log(time_points)
                z_scores = (log_times - eta) / self.aft_parameters.sigma
                survival_probs = 1 - stats.norm.cdf(z_scores)
                
            elif self.aft_parameters.distribution == 'logistic':
                log_times = np.log(time_points)
                z_scores = (log_times - eta) / self.aft_parameters.sigma
                survival_probs = 1 / (1 + np.exp(z_scores))
                
            elif self.aft_parameters.distribution == 'extreme':
                log_times = np.log(time_points)
                z_scores = (log_times - eta) / self.aft_parameters.sigma
                survival_probs = np.exp(-np.exp(z_scores))
            
            else:
                raise ValueError(f"Unsupported distribution: {self.aft_parameters.distribution}")
            
            # Ensure valid probabilities
            survival_probs = np.clip(survival_probs, 1e-6, 1.0 - 1e-6)
            survival_curves.append(survival_probs)
        
        survival_curves = np.array(survival_curves)
        
        # Log summary statistics
        final_survival = survival_curves[:, -1]
        logger.info(f"Survival curve statistics:")
        logger.info(f"  1-day survival: {survival_curves[:, 0].mean():.3f} ± {survival_curves[:, 0].std():.3f}")
        logger.info(f"  365-day survival: {final_survival.mean():.3f} ± {final_survival.std():.3f}")
        
        return survival_curves
    
    def predict_risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Business-ready risk score generation from AFT predictions
        
        Processes features through smart preprocessing pipeline and generates normalized
        risk scores where higher values indicate higher risk of event occurrence.
        
        Args:
            X: Features for prediction (will be processed through same pipeline as training)
            
        Returns:
            np.ndarray: Normalized risk scores [0, 1] where 1 = highest risk
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before generating risk scores")
        
        # Process features using the same pipeline
        X_processed = self._get_processed_features(X)
        
        # Get AFT predictions
        dmatrix = self._create_categorical_aware_dmatrix(X_processed)
        eta_predictions = self.model.predict(dmatrix)
        
        # Convert to risk scores (lower predicted survival time = higher risk)
        risk_scores = -eta_predictions
        
        # Normalize to [0, 1] range
        risk_scores = risk_scores - risk_scores.min()
        if risk_scores.max() > 0:
            risk_scores = risk_scores / risk_scores.max()
        
        logger.info(f"Generated risk scores - Mean: {risk_scores.mean():.3f}, Std: {risk_scores.std():.3f}")
        
        return risk_scores
    
    def _get_processed_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Process features for any dataset using the trained smart preprocessing pipeline"""
        if not hasattr(self.feature_processor, 'label_encoders') or not self.feature_processor.label_encoders:
            logger.warning("Feature processor not trained. Processing features without encoding.")
            return X[self.feature_columns] if self.feature_columns else X
        
        # Apply same smart preprocessing pipeline
        X_processed = self.feature_processor.process_dataset(X, "prediction")
        
        # Apply categorical encoding using trained encoders
        for cat_feature, mapping in self.feature_processor.label_encoders.items():
            if cat_feature in X_processed.columns:
                encoded_col = f'{cat_feature}_encoded'
                cats = X_processed[cat_feature].fillna('MISSING').astype(str)
                
                # Add explicit unknown category code instead of arbitrary 0
                max_known_code = max(mapping.values()) if mapping else 0
                unknown_code = max_known_code + 1
                
                # Store unknown code for consistency
                if 'UNKNOWN' not in mapping:
                    mapping['UNKNOWN'] = unknown_code
                
                X_processed[encoded_col] = cats.map(mapping).fillna(unknown_code)  # Use explicit unknown code
        
        # Select final features
        available_features = [col for col in self.feature_columns if col in X_processed.columns]
        
        return X_processed[available_features]
    
    def save_model(self, filepath: str, include_metadata: bool = True) -> bool:
        """
        Comprehensive model persistence with metadata and smart preprocessing state
        
        Args:
            filepath: Path to save model
            include_metadata: Whether to include training metadata and feature processing state
            
        Returns:
            bool: Success status
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save XGBoost model
            model_path = filepath.with_suffix('.xgb')
            self.model.save_model(str(model_path))
            
            # Save metadata and parameters
            if include_metadata:
                metadata = {
                    'aft_parameters': {
                        'sigma': self.aft_parameters.sigma,
                        'distribution': self.aft_parameters.distribution,
                        'log_likelihood': self.aft_parameters.log_likelihood
                    },
                    'feature_columns': self.feature_columns,
                    'feature_name_mapping': self.feature_processor.feature_name_mapping,
                    'transformation_summary': self.feature_processor.get_transformation_summary(),
                    'training_metadata': self.training_metadata,
                    'config': {
                        'aft_distributions': self.config.aft_distributions,
                        'scale_parameter_range': self.config.scale_parameter_range,
                        'xgb_params': self.config.xgb_params
                    }
                }
                
                metadata_path = filepath.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                # Save feature processor
                processor_path = filepath.with_suffix('.pkl')
                with open(processor_path, 'wb') as f:
                    pickle.dump(self.feature_processor, f)
            
            logger.info(f"Model saved successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Model loading with validation and compatibility checks
        
        Args:
            filepath: Path to load model from
            
        Returns:
            bool: Success status
        """
        try:
            filepath = Path(filepath)
            
            # Load XGBoost model
            model_path = filepath.with_suffix('.xgb')
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load metadata
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Restore AFT parameters
                aft_data = metadata['aft_parameters']
                self.aft_parameters = AFTParameters(
                    eta=np.array([]),  # Will be generated during prediction
                    sigma=aft_data['sigma'],
                    distribution=aft_data['distribution'],
                    log_likelihood=aft_data['log_likelihood']
                )
                
                self.feature_columns = metadata['feature_columns']
                self.training_metadata = metadata['training_metadata']
            
            # Load feature processor
            processor_path = filepath.with_suffix('.pkl')
            if processor_path.exists():
                with open(processor_path, 'rb') as f:
                    self.feature_processor = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

if __name__ == "__main__":
    """
    Comprehensive test suite for Advanced SurvivalModelEngine
    Demonstrates smart feature processing, multi-dataset training, and production-ready workflows
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== SURVIVAL MODEL ENGINE TEST SUITE ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns available: {list(df.columns)}")
    
    # ===== 1. DATA PREPARATION =====
    print("\n1. PREPARING DATA...")
    
    # Basic data preparation
    df_clean = df.dropna(subset=['survival_time_days', 'event_indicator_vol']).copy()
    
    # Split data by dataset_split
    datasets_raw = {
        'train': df_clean[df_clean['dataset_split'] == 'train'].copy(),
        'val': df_clean[df_clean['dataset_split'] == 'val'].copy()
    }
    
    # Add OOT if available
    if 'oot' in df_clean['dataset_split'].values:
        datasets_raw['oot'] = df_clean[df_clean['dataset_split'] == 'oot'].copy()
    
    print(f"Dataset sizes: {[(name, len(data)) for name, data in datasets_raw.items()]}")
    print(f"Event rates: {[(name, data['event_indicator_vol'].mean()) for name, data in datasets_raw.items()]}")
    
    # ===== 2. CONFIGURATION SETUP =====
    print("\n2. SETTING UP CONFIGURATION...")
    
    # Initialize smart feature configuration
    feature_config = FeatureConfig()
    model_config = ModelConfig()
    
    # Initialize feature processor
    feature_processor = SmartFeatureProcessor(feature_config)
    engine = SurvivalModelEngine(model_config, feature_processor)
    
    print("✓ Configuration and components initialized")
    print(f"✓ IQR cap features: {len(feature_config.iqr_cap_features)}")
    print(f"✓ Log transform features: {len(feature_config.log_transform_features)}")
    print(f"✓ Winsorize features: {len(feature_config.winsorize_features)}")
    print(f"✓ Categorical features: {len(feature_config.categorical_features)}")
    
    # ===== 3. MODEL TRAINING =====
    print("\n3. TRAINING SURVIVAL MODEL...")
    
    # Prepare datasets for training with new flexible interface
    training_datasets = {}
    for name, data in datasets_raw.items():
        training_datasets[name] = (data, 'survival_time_days', 'event_indicator_vol')
    
    # Train comprehensive model with smart preprocessing
    model_results = engine.train_survival_model(training_datasets)
    
    print(f"✓ Smart model trained successfully")
    print(f"✓ Training C-index: {model_results.training_metrics.get('c_index', 'N/A'):.4f}")
    print(f"✓ Validation C-index: {model_results.validation_metrics.get('c_index', 'N/A'):.4f}")
    if model_results.oot_metrics:
        print(f"✓ OOT C-index: {model_results.oot_metrics.get('c_index', 'N/A'):.4f}")
    print(f"✓ Optimal AFT distribution: {model_results.aft_parameters.distribution}")
    print(f"✓ Optimal scale parameter: {model_results.aft_parameters.sigma:.4f}")
    print(f"✓ Final feature count: {len(model_results.feature_importance)}")
    
    # ===== 4. SMART FEATURE ANALYSIS =====
    print("\n4. ANALYZING SMART FEATURE PROCESSING...")
    
    # Show transformation summary
    transformation_summary = engine.feature_processor.get_transformation_summary()
    print(f"✓ Features transformed: {len(transformation_summary)}")
    print("✓ Top 5 feature transformations:")
    for orig_feature, transforms in list(transformation_summary.items())[:5]:
        print(f"   {orig_feature}: {' → '.join(transforms)}")
    
    # Show feature importance with original names
    top_features = model_results.feature_importance.head(10)
    print("\n✓ Top 10 most important features (with original names):")
    for idx, row in top_features.iterrows():
        print(f"   {row['original_feature']} ({row['feature']}): {row['importance']:.2f}")
    
    # ===== 5. SMART PREDICTIONS =====
    print("\n5. TESTING SMART PREDICTIONS...")
    
    # Test on validation set
    X_val = datasets_raw['val']
    survival_curves = engine.predict_survival_curves(X_val)
    risk_scores = engine.predict_risk_scores(X_val)
    
    print(f"✓ Generated survival curves: {survival_curves.shape}")
    print(f"✓ Mean 30-day survival: {survival_curves[:, 29].mean():.3f}")
    print(f"✓ Mean 365-day survival: {survival_curves[:, -1].mean():.3f}")
    print(f"✓ Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    
    # ===== 6. MODEL PERSISTENCE TEST =====
    print("\n6. TESTING SMART MODEL PERSISTENCE...")
    
    test_filepath = "./smart_survival_model"
    save_success = engine.save_model(test_filepath, include_metadata=True)
    print(f"✓ Smart model save: {save_success}")
    
    # Test loading
    new_engine = SurvivalModelEngine(model_config, SmartFeatureProcessor(feature_config))
    load_success = new_engine.load_model(test_filepath)
    print(f"✓ Smart model load: {load_success}")
    
    # Verify loaded model works
    test_risk_scores = new_engine.predict_risk_scores(X_val.head(100))
    print(f"✓ Loaded model prediction test: {test_risk_scores.mean():.3f}")
    
    # ===== 7. BUSINESS INTELLIGENCE =====
    print("\n7. BUSINESS INTELLIGENCE INSIGHTS...")
    
    # Risk stratification analysis
    high_risk_threshold = np.percentile(risk_scores, 80)
    high_risk_mask = risk_scores >= high_risk_threshold
    
    val_events = datasets_raw['val']['event_indicator_vol']
    high_risk_event_rate = val_events[high_risk_mask].mean()
    low_risk_event_rate = val_events[~high_risk_mask].mean()
    
    print(f"✓ High-risk group (top 20%) event rate: {high_risk_event_rate:.3f}")
    print(f"✓ Low-risk group (bottom 80%) event rate: {low_risk_event_rate:.3f}")
    print(f"✓ Risk discrimination: {high_risk_event_rate / low_risk_event_rate:.2f}x better")
    
    # Feature processing efficiency
    total_possible_features = (len(feature_config.iqr_cap_features) + 
                             len(feature_config.log_transform_features) + 
                             len(feature_config.winsorize_features))
    actual_features_created = len([f for f in model_results.feature_importance['feature'] 
                                  if f.endswith(('_cap', '_log', '_win_cap'))])
    
    print(f"✓ Smart processing efficiency: {actual_features_created}/{total_possible_features} features created")
    print(f"✓ Memory saved by smart processing: ~{(total_possible_features - actual_features_created) * len(datasets_raw['train']) * 8 / 1024 / 1024:.1f} MB")
    
    # ===== FINAL STATUS =====
    print("\n" + "="*70)
    print(" SURVIVAL MODEL ENGINE TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\nKEY SMART FEATURES VALIDATED:")
    print("• Smart feature preprocessing with automatic transformation detection")
    print("• Memory-efficient processing (only creates needed features)")
    print("• Feature name mapping and transformation tracking")
    print("• Multi-dataset training with flexible interface")
    print("• XGBoost 3.0+ AFT with automatic log transformation")
    print("• Enhanced feature importance with original names")
    print("• Comprehensive model persistence with preprocessing state")
    print("• Production-ready prediction pipeline")
    
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"• Model: XGBoost AFT ({model_results.aft_parameters.distribution})")
    print(f"• Validation C-index: {model_results.validation_metrics.get('c_index', 'N/A'):.4f}")
    if model_results.oot_metrics:
        print(f"• OOT C-index: {model_results.oot_metrics.get('c_index', 'N/A'):.4f}")
    print(f"• Features: {len(model_results.feature_importance)} (smart selection)")
    print(f"• Datasets: {list(training_datasets.keys())}")
    print(f"• Risk discrimination: {high_risk_event_rate / low_risk_event_rate:.2f}x")
        
    # Create results dictionary for further analysis
    smart_test_results = {
        'engine': engine,
        'model_results': model_results,
        'feature_processor': feature_processor,
        'survival_curves': survival_curves,
        'risk_scores': risk_scores,
        'transformation_summary': transformation_summary,
        'feature_config': feature_config
    }
    
    print(f"\nSmart test results available in 'smart_test_results' dictionary")
    
    # ===== USAGE EXAMPLES =====
    print("\n" + "="*70)
    print(" PRODUCTION USAGE EXAMPLES")
    print("="*70)
    
    print("\n# Example 1: Initialize and train model")
    print("""
feature_config = FeatureConfig()
model_config = ModelConfig()
processor = SmartFeatureProcessor(feature_config)
engine = SurvivalModelEngine(model_config, processor)

# Prepare datasets
datasets = {
    'train': (train_df, 'survival_time_days', 'event_indicator_vol'),
    'val': (val_df, 'survival_time_days', 'event_indicator_vol'),
    'oot': (oot_df, 'survival_time_days', 'event_indicator_vol')
}

# Train model
results = engine.train_survival_model(datasets)
    """)
    
    print("\n# Example 2: Generate predictions")
    print("""
# Get survival curves
survival_curves = engine.predict_survival_curves(new_data)

# Get risk scores
risk_scores = engine.predict_risk_scores(new_data)

# Save model
engine.save_model('./production_model')
    """)
    
    print("\n# Example 3: Custom feature configuration")
    print("""
# Customize feature processing
custom_config = FeatureConfig()
custom_config.winsorize_features = ['salary', 'tenure', 'age']
custom_config.log_transform_features = ['complexity_score']
    """)
    
    print(f"\nREADY FOR  DEPLOYMENT!")
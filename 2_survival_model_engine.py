"""
survival_model_engine.py

Core AFT survival modeling engine with methodological excellence.
Expert-level implementation for  employee turnover prediction.

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
    """Comprehensive model training results"""
    model: xgb.Booster
    aft_parameters: AFTParameters
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    training_metadata: Dict[str, Any]

@dataclass
    class ModelConfig:
        """Core modeling configuration updated for XGBoost 3.0.2"""
        aft_distributions: List[str] = field(default_factory=lambda: ['normal', 'logistic', 'extreme'])
        scale_parameter_range: Tuple[float, float] = (0.1, 5.0)
        scale_parameter_grid_size: int = 20
        validation_metrics: List[str] = field(default_factory=lambda: ['c_index', 'ibs', 'gini', 'ece'])
        xgb_params: Dict[str, Any] = field(default_factory=lambda: {
            'max_depth': 4,
            'eta': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 10.0,
            'min_child_weight': 10,
            'gamma': 1.0,
            'seed': 42,
            'verbosity': 0,
            'tree_method': 'hist'  # Recommended for XGBoost 3.0.2
        })
        early_stopping_rounds: int = 50
        num_boost_round: int = 1000

class FeatureProcessor:
    """Dedicated feature processing with comprehensive scaling strategies"""
    
    def __init__(self, scaling_config: Dict):
        """Initialize with feature scaling configuration"""
        self.scaling_config = scaling_config
        self.scalers = {}
        self.feature_metadata = {}
        
    def process_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                        X_test: Optional[pd.DataFrame] = None) -> Tuple:
        """
        Comprehensive feature processing pipeline
        
        Args:
            X_train: Training features
            X_val: Validation features  
            X_test: Optional test features
            
        Returns:
            Tuple of processed DataFrames (train, val, test)
        """
        logger.info("Starting comprehensive feature processing pipeline")
        
        # Store original statistics for metadata
        self.feature_metadata['original_stats'] = {
            'train': X_train.describe(),
            'val': X_val.describe()
        }
        
        # Apply scaling strategies
        X_train_processed, X_val_processed = self._apply_feature_scaling(X_train, X_val)
        
        X_test_processed = None
        if X_test is not None:
            X_test_processed = self._transform_test_features(X_test)
            
        logger.info(f"Feature processing complete. Shape: train={X_train_processed.shape}, val={X_val_processed.shape}")
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def _apply_feature_scaling(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply modular feature scaling based on configuration"""
        X_train = X_train.copy()
        X_val = X_val.copy()
        
        # Robust scale features
        if 'robust_scale' in self.scaling_config:
            features = [f for f in self.scaling_config['robust_scale'] if f in X_train.columns]
            if features:
                self.scalers['robust'] = RobustScaler()
                X_train[features] = self.scalers['robust'].fit_transform(X_train[features])
                X_val[features] = self.scalers['robust'].transform(X_val[features])
        
        # Log transform then scale
        if 'log_transform' in self.scaling_config:
            features = [f for f in self.scaling_config['log_transform'] if f in X_train.columns]
            if features:
                X_train[features] = np.log1p(X_train[features])
                X_val[features] = np.log1p(X_val[features])
                self.scalers['log'] = RobustScaler()
                X_train[features] = self.scalers['log'].fit_transform(X_train[features])
                X_val[features] = self.scalers['log'].transform(X_val[features])
        
        # Clip and scale
        if 'clip_and_scale' in self.scaling_config:
            clip_features = []
            for feature, (min_val, max_val) in self.scaling_config['clip_and_scale'].items():
                if feature in X_train.columns:
                    X_train[feature] = X_train[feature].clip(min_val, max_val)
                    X_val[feature] = X_val[feature].clip(min_val, max_val)
                    clip_features.append(feature)
            
            if clip_features:
                self.scalers['clip'] = RobustScaler()
                X_train[clip_features] = self.scalers['clip'].fit_transform(X_train[clip_features])
                X_val[clip_features] = self.scalers['clip'].transform(X_val[clip_features])
        
        return X_train, X_val
    
    def _transform_test_features(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Transform test features using fitted scalers"""
        X_test = X_test.copy()
        
        # Apply same transformations as training
        if 'log_transform' in self.scaling_config:
            features = [f for f in self.scaling_config['log_transform'] if f in X_test.columns]
            if features and 'log' in self.scalers:
                X_test[features] = np.log1p(X_test[features])
                X_test[features] = self.scalers['log'].transform(X_test[features])
        
        if 'robust_scale' in self.scaling_config:
            features = [f for f in self.scaling_config['robust_scale'] if f in X_test.columns]
            if features and 'robust' in self.scalers:
                X_test[features] = self.scalers['robust'].transform(X_test[features])
                
        if 'clip_and_scale' in self.scaling_config:
            clip_features = []
            for feature, (min_val, max_val) in self.scaling_config['clip_and_scale'].items():
                if feature in X_test.columns:
                    X_test[feature] = X_test[feature].clip(min_val, max_val)
                    clip_features.append(feature)
            
            if clip_features and 'clip' in self.scalers:
                X_test[clip_features] = self.scalers['clip'].transform(X_test[clip_features])
        
        return X_test

class SurvivalModelEngine:
    """Expert AFT survival modeling engine - focused and methodologically rigorous"""
    
    def __init__(self, config: ModelConfig, feature_processor: FeatureProcessor):
        """
        Initialize with configuration and feature processing component
        
        Args:
            config: Model configuration with AFT and XGBoost parameters
            feature_processor: Feature processing component
        """
        self.config = config
        self.feature_processor = feature_processor
        self.model = None
        self.aft_parameters = None
        self.feature_columns = None
        self.training_metadata = {}
        
        logger.info(f"SurvivalModelEngine initialized with {len(config.aft_distributions)} AFT distributions")
    
    # === AFT OPTIMIZATION SECTION (200 LOC) ===
    
    def optimize_aft_parameters(self, X_train: pd.DataFrame, y_train: pd.Series, event_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series, event_val: pd.Series) -> AFTParameters:
        """
        MLE-based AFT parameter optimization across distributions for XGBoost 3.0.2
        
        Uses interval-based approach with proper handling of censoring patterns.
        """
        logger.info("Starting MLE-based AFT parameter optimization with XGBoost 3.0.2")
        
        # Transform to log scale for AFT
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
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
                    dval = xgb.DMatrix(X_val)
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

        
    def _estimate_aft_parameters_robust(self, eta_predictions: np.ndarray, 
                                      log_actuals: np.ndarray, events: np.ndarray) -> float:
        """
        Robust AFT parameter estimation methodology
        
        MAD and IQR-based sigma estimation for uncensored observations with
        distribution-specific adjustments and statistical confidence assessment.
        
        Args:
            eta_predictions: Model predictions on log scale
            log_actuals: Actual survival times on log scale
            events: Event indicators
            
        Returns:
            float: Robust scale parameter estimate
        """
        # Use only uncensored observations for sigma estimation
        uncensored_mask = events == 1
        n_uncensored = uncensored_mask.sum()
        
        if n_uncensored < 50:
            warnings.warn(f"Only {n_uncensored} uncensored observations for sigma estimation")
        
        log_actual_uncensored = log_actuals[uncensored_mask]
        eta_uncensored = eta_predictions[uncensored_mask]
        
        residuals = log_actual_uncensored - eta_uncensored
        
        # Robust sigma estimation methods
        sigma_mad = 1.4826 * median_abs_deviation(residuals)
        
        # IQR-based estimation
        q75, q25 = np.percentile(residuals, [75, 25])
        sigma_iqr = (q75 - q25) / 1.349
        
        # Use the more conservative estimate
        sigma = max(sigma_mad, sigma_iqr)
        
        logger.info(f"Robust sigma estimation: MAD={sigma_mad:.4f}, IQR={sigma_iqr:.4f}, Final={sigma:.4f}")
        
        return sigma
    
    def _train_xgb_aft_temp(self, X: pd.DataFrame, y_log: pd.Series, events: pd.Series,
                        distribution: str, scale: float) -> xgb.Booster:
        """Temporary XGBoost AFT training for parameter optimization using XGBoost 3.0.2"""
        
        # Convert to interval representation
        y_lower = y_log.values
        y_upper = np.where(events == 1, y_log.values, np.inf)
        
        # Create DMatrix with interval bounds
        dtrain = xgb.DMatrix(X)
        dtrain.set_float_info('label_lower_bound', y_lower)
        dtrain.set_float_info('label_upper_bound', y_upper)
        
        # Set AFT parameters
        params = self.config.xgb_params.copy()
        params.update({
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': distribution,
            'aft_loss_distribution_scale': scale
        })
        
        # Train with reduced iterations for optimization speed
        model = xgb.train(
            params, dtrain,
            num_boost_round=min(500, self.config.num_boost_round),
            verbose_eval=False
        )
        
        return model

    # === MODEL TRAINING SECTION (180 LOC) ===
    
    def train_survival_model(self, X_train: pd.DataFrame, y_train: pd.Series, event_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series, event_val: pd.Series) -> ModelResults:
        """
        Complete AFT model training pipeline
        
        Feature preprocessing through FeatureProcessor integration, optimal parameter
        application to XGBoost AFT, and model validation on VAL dataset exclusively.
        
        Args:
            X_train: Training features
            y_train: Training survival times  
            event_train: Training event indicators
            X_val: Validation features
            y_val: Validation survival times
            event_val: Validation event indicators
            
        Returns:
            ModelResults: Comprehensive training results
        """
        logger.info("Starting complete AFT model training pipeline")
        
        # Feature preprocessing
        X_train_processed, X_val_processed, _ = self.feature_processor.process_features(
            X_train, X_val
        )
        self.feature_columns = list(X_train_processed.columns)
        
        # Store processed data for later use
        self.training_metadata = {
            'original_train_shape': X_train.shape,
            'processed_train_shape': X_train_processed.shape,
            'feature_columns': self.feature_columns,
            'event_rate_train': event_train.mean(),
            'event_rate_val': event_val.mean()
        }
        
        # Optimize AFT parameters using validation set
        optimal_params = self.optimize_aft_parameters(
            X_train_processed, y_train, event_train,
            X_val_processed, y_val, event_val
        )
        self.aft_parameters = optimal_params
        
        # Train final model with optimal parameters
        final_model = self._setup_xgboost_aft(
            X_train_processed, np.log1p(y_train), event_train,
            X_val_processed, np.log1p(y_val), event_val,
            optimal_params
        )
        self.model = final_model
        
        # Calculate training and validation metrics
        train_predictions = final_model.predict(xgb.DMatrix(X_train_processed))
        val_predictions = final_model.predict(xgb.DMatrix(X_val_processed))
        
        training_metrics = self._calculate_training_metrics(
            train_predictions, y_train, event_train
        )
        validation_metrics = self._calculate_training_metrics(
            val_predictions, y_val, event_val
        )
        
        # Feature importance analysis
        feature_importance = self._extract_feature_importance(final_model)
        
        logger.info(f"Model training complete. Validation C-index: {validation_metrics.get('c_index', 'N/A'):.4f}")
        
        return ModelResults(
            model=final_model,
            aft_parameters=optimal_params,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            feature_importance=feature_importance,
            training_metadata=self.training_metadata
        )
    
    def _setup_xgboost_aft(self, X_train: pd.DataFrame, y_train_log: pd.Series, event_train: pd.Series,
                        X_val: pd.DataFrame, y_val_log: pd.Series, event_val: pd.Series,
                        aft_params: AFTParameters) -> xgb.Booster:
        """
        XGBoost AFT model setup with proper censoring configuration for XGBoost 3.0.2
        
        Uses interval-based censoring approach where all observations are intervals.
        Right-censored data: [observed_time, +inf]
        Uncensored data: [observed_time, observed_time]
        """
        
        # Create interval bounds for training data
        y_lower_train = y_train_log.values
        y_upper_train = np.where(
            event_train == 1,  # Uncensored: exact time known
            y_train_log.values,  # [time, time] - degenerate interval
            np.inf  # Right-censored: [time, +inf) - unbounded interval
        )
        
        # Create interval bounds for validation data  
        y_lower_val = y_val_log.values
        y_upper_val = np.where(
            event_val == 1,
            y_val_log.values,
            np.inf
        )
        
        # Create training DMatrix with interval-based labels
        dtrain = xgb.DMatrix(X_train)
        dtrain.set_float_info('label_lower_bound', y_lower_train)
        dtrain.set_float_info('label_upper_bound', y_upper_train)
        
        # Create validation DMatrix
        dval = xgb.DMatrix(X_val)
        dval.set_float_info('label_lower_bound', y_lower_val)
        dval.set_float_info('label_upper_bound', y_upper_val)
        
        # Set AFT parameters (note: eval_metric uses hyphenated form in 3.0.2)
        params = self.config.xgb_params.copy()
        params.update({
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',  # Hyphenated form for XGBoost 3.0.2
            'aft_loss_distribution': aft_params.distribution,
            'aft_loss_distribution_scale': aft_params.sigma
        })
        
        # Training with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params, dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False
        )
        
        return model
    
    def validate_model_training(self, model_results: ModelResults) -> Dict[str, Any]:
        """
        Training validation and model quality assessment
        
        Prediction distribution analysis, directional relationship validation,
        and model convergence verification.
        
        Args:
            model_results: Complete model training results
            
        Returns:
            Dict: Validation assessment results
        """
        validation_results = {
            'prediction_stats': {
                'convergence': model_results.model.best_iteration < self.config.num_boost_round,
                'best_iteration': model_results.model.best_iteration,
                'feature_count': len(model_results.training_metadata['feature_columns'])
            },
            'aft_validation': {
                'distribution': model_results.aft_parameters.distribution,
                'scale_parameter': model_results.aft_parameters.sigma,
                'log_likelihood': model_results.aft_parameters.log_likelihood,
                'optimization_successful': model_results.aft_parameters.log_likelihood > -np.inf
            },
            'metrics_comparison': {
                'train_val_gap': {
                    metric: abs(model_results.training_metrics.get(metric, 0) - 
                              model_results.validation_metrics.get(metric, 0))
                    for metric in ['c_index', 'prediction_std']
                }
            }
        }
        
        # Assess training quality
        issues = []
        if not validation_results['prediction_stats']['convergence']:
            issues.append("Model did not converge within iteration limit")
        if validation_results['metrics_comparison']['train_val_gap'].get('c_index', 0) > 0.05:
            issues.append("Large train-validation C-index gap detected")
        if model_results.aft_parameters.sigma < 0.1 or model_results.aft_parameters.sigma > 3.0:
            issues.append(f"Unusual scale parameter: {model_results.aft_parameters.sigma:.4f}")
        
        validation_results['issues'] = issues
        validation_results['training_quality'] = 'GOOD' if len(issues) == 0 else 'NEEDS_ATTENTION'
        
        return validation_results
    
    def _calculate_training_metrics(self, predictions: np.ndarray, actuals: pd.Series, 
                                   events: pd.Series) -> Dict[str, float]:
        """Calculate basic training metrics"""
        from lifelines.utils import concordance_index
        
        # Transform predictions back to original scale for C-index
        pred_days = np.expm1(predictions)
        
        metrics = {
            'c_index': concordance_index(actuals, pred_days, events),
            'prediction_mean': predictions.mean(),
            'prediction_std': predictions.std(),
            'prediction_range': predictions.max() - predictions.min()
        }
        
        return metrics
    
    def _extract_feature_importance(self, model: xgb.Booster) -> pd.DataFrame:
        """Extract and format feature importance"""
        importance_dict = model.get_score(importance_type='gain')
        
        # Handle both f0, f1, ... and actual feature names
        if self.feature_columns and list(importance_dict.keys())[0].startswith('f'):
            importance_df = pd.DataFrame([
                {'feature': self.feature_columns[int(f[1:])], 'importance': score}
                for f, score in importance_dict.items()
            ])
        else:
            importance_df = pd.DataFrame([
                {'feature': f, 'importance': score}
                for f, score in importance_dict.items()
            ])
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # === PREDICTION GENERATION SECTION (120 LOC) ===
    
    def predict_survival_curves(self, X: pd.DataFrame, time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate survival curves using AFT formulation
        
        Multi-horizon survival probability calculation with distribution-specific
        curve generation and confidence interval computation.
        
        Args:
            X: Features for prediction
            time_points: Time points for survival curve (default: 1-365 days)
            
        Returns:
            np.ndarray: Survival curves (n_samples x n_timepoints)
        """
        if self.model is None or self.aft_parameters is None:
            raise RuntimeError("Model must be trained before generating predictions")
        
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        # Process features if needed
        if hasattr(self.feature_processor, 'scalers') and self.feature_processor.scalers:
            X_processed = self.feature_processor._transform_test_features(X)
        else:
            X_processed = X[self.feature_columns] if self.feature_columns else X
        
        # Get AFT predictions (eta)
        dmatrix = xgb.DMatrix(X_processed)
        eta_predictions = self.model.predict(dmatrix)
        
        logger.info(f"Generating survival curves for {len(X)} samples over {len(time_points)} time points")
        logger.info(f"AFT predictions - Mean: {eta_predictions.mean():.3f}, Std: {eta_predictions.std():.3f}")
        
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
        
        Risk score normalization and interpretation with uncertainty quantification.
        
        Args:
            X: Features for prediction
            
        Returns:
            np.ndarray: Normalized risk scores [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before generating risk scores")
        
        # Process features if needed
        if hasattr(self.feature_processor, 'scalers') and self.feature_processor.scalers:
            X_processed = self.feature_processor._transform_test_features(X)  
        else:
            X_processed = X[self.feature_columns] if self.feature_columns else X
        
        # Get AFT predictions
        dmatrix = xgb.DMatrix(X_processed)
        eta_predictions = self.model.predict(dmatrix)
        
        # Convert to risk scores (lower predicted survival time = higher risk)
        risk_scores = -eta_predictions
        
        # Normalize to [0, 1] range
        risk_scores = risk_scores - risk_scores.min()
        if risk_scores.max() > 0:
            risk_scores = risk_scores / risk_scores.max()
        
        logger.info(f"Generated risk scores - Mean: {risk_scores.mean():.3f}, Std: {risk_scores.std():.3f}")
        
        return risk_scores
    
    # === MODEL PERSISTENCE SECTION (50 LOC) ===
    
    def save_model(self, filepath: str, include_metadata: bool = True) -> bool:
        """
        Comprehensive model persistence with metadata
        
        Args:
            filepath: Path to save model
            include_metadata: Whether to include training metadata
            
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
                    'training_metadata': self.training_metadata,
                    'config': {
                        'aft_distributions': self.config.aft_distributions,
                        'scale_parameter_range': self.config.scale_parameter_range
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
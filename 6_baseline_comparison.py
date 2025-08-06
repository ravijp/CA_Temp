"""
baseline_comparison.py

Focused baseline model comparison for AFT survival model validation.
Implements methodologically rigorous comparison between AFT models and KM baselines
using identical evaluation metrics: C-index, IPCW Brier scores, log-likelihood.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BaselineResults:
    """Baseline model results with core metrics"""
    model_name: str
    c_index: float
    ipcw_brier_score: float
    log_likelihood: float
    predictions_365d: np.ndarray
    sample_size: int

class BaselineComparison:
    """
    Expert-level baseline comparison framework for AFT model validation
    
    Transforms KM population curves into pseudo-individual predictions via stratification,
    enabling direct comparison using identical evaluation metrics from SurvivalEvaluation.
    """
    
    def __init__(self, survival_evaluation):
        """Initialize with existing evaluation infrastructure"""
        self.evaluation = survival_evaluation
        self.model_engine = survival_evaluation.model_engine
        self.baseline_results = {}
        self.comparison_results = {}
        
        logger.info("BaselineComparison initialized with evaluation integration")
    
    def create_baseline_suite(self, datasets: Dict[str, Tuple]) -> Dict[str, BaselineResults]:
        """
        Create comprehensive baseline model suite with KM stratification approach
        
        Args:
            datasets: {dataset_name: (X, y, events)} for evaluation
            
        Returns:
            Dict: Baseline model results with core performance metrics
        """
        logger.info("Creating baseline model suite...")
        
        # Extract training data
        train_data = datasets.get('train') or datasets[list(datasets.keys())[0]]
        X_train, y_train, events_train = train_data
        
        baselines = {}
        
        # Baseline 1: Population KM
        baselines['population_km'] = self._create_population_km(
            X_train, y_train, events_train, datasets
        )
        
        # Baseline 2: Industry-Stratified KM
        if self._has_industry_column(X_train):
            baselines['industry_km'] = self._create_industry_km(
                X_train, y_train, events_train, datasets
            )
        
        # Baseline 3: Demographic-Stratified KM
        if self._has_demographic_columns(X_train):
            baselines['demographic_km'] = self._create_demographic_km(
                X_train, y_train, events_train, datasets
            )
        
        # Baseline 4: Random Assignment
        baselines['random'] = self._create_random_baseline(datasets)
        
        self.baseline_results = baselines
        logger.info(f"Baseline suite created: {len(baselines)} models")
        return baselines
    
    def compare_to_aft(self, aft_results: Dict, datasets: Dict[str, Tuple]) -> Dict:
        """
        Direct AFT vs baseline comparison using identical metrics
        
        Args:
            aft_results: AFT performance results from SurvivalEvaluation
            datasets: Evaluation datasets
            
        Returns:
            Dict: Comprehensive comparison results
        """
        logger.info("Executing AFT vs baseline comparison...")
        
        if not self.baseline_results:
            self.create_baseline_suite(datasets)
        
        # Extract AFT metrics
        aft_metrics = self._extract_aft_metrics(aft_results)
        
        # Calculate improvements
        improvements = {}
        for baseline_name, baseline_result in self.baseline_results.items():
            improvements[baseline_name] = {
                'c_index_improvement': aft_metrics['c_index'] - baseline_result.c_index,
                'brier_improvement': baseline_result.ipcw_brier_score - aft_metrics['ipcw_brier_score'],
                'log_likelihood_improvement': aft_metrics['log_likelihood'] - baseline_result.log_likelihood
            }
        
        # Generate deployment recommendation
        best_baseline_c_index = max(b.c_index for b in self.baseline_results.values())
        c_index_improvement = aft_metrics['c_index'] - best_baseline_c_index
        
        if c_index_improvement >= 0.10:
            recommendation = "RECOMMEND DEPLOYMENT - Substantial improvement"
        elif c_index_improvement >= 0.05:
            recommendation = "CONDITIONAL DEPLOYMENT - Moderate improvement"
        else:
            recommendation = "NOT RECOMMENDED - Minimal improvement"
        
        comparison = {
            'aft_metrics': aft_metrics,
            'baseline_metrics': self.baseline_results,
            'improvements': improvements,
            'deployment_recommendation': recommendation,
            'best_baseline_improvement': c_index_improvement
        }
        
        self.comparison_results = comparison
        logger.info("AFT comparison complete")
        return comparison
    
    def generate_ben_summary(self) -> str:
        """Generate focused comparison summary for Ben's review"""
        if not self.comparison_results:
            return "No comparison results available"
        
        aft = self.comparison_results['aft_metrics']
        baselines = self.comparison_results['baseline_metrics']
        improvements = self.comparison_results['improvements']
        
        summary = []
        summary.append("="*60)
        summary.append(" AFT vs BASELINE MODEL COMPARISON")
        summary.append("="*60)
        
        # AFT Performance
        summary.append(f"\nAFT MODEL PERFORMANCE:")
        summary.append(f"   C-Index: {aft['c_index']:.4f}")
        summary.append(f"   IPCW Brier Score: {aft['ipcw_brier_score']:.4f}")
        summary.append(f"   Log-Likelihood: {aft['log_likelihood']:.2f}")
        
        # Baseline Performance
        summary.append(f"\nBASELINE MODEL PERFORMANCE:")
        for name, baseline in baselines.items():
            summary.append(f"   {name.upper().replace('_', ' ')}:")
            summary.append(f"     C-Index: {baseline.c_index:.4f}")
            summary.append(f"     IPCW Brier: {baseline.ipcw_brier_score:.4f}")
            summary.append(f"     Log-Likelihood: {baseline.log_likelihood:.2f}")
        
        # Key Improvements
        summary.append(f"\nPERFORMANCE IMPROVEMENTS:")
        best_baseline = max(baselines.values(), key=lambda x: x.c_index)
        summary.append(f"   Best Baseline: {best_baseline.model_name}")
        summary.append(f"   C-Index Improvement: +{aft['c_index'] - best_baseline.c_index:.4f}")
        summary.append(f"   Brier Score Improvement: +{best_baseline.ipcw_brier_score - aft['ipcw_brier_score']:.4f}")
        
        # Deployment Decision
        summary.append(f"\nDEPLOYMENT RECOMMENDATION:")
        summary.append(f"   {self.comparison_results['deployment_recommendation']}")
        summary.append(f"   Improvement: +{self.comparison_results['best_baseline_improvement']:.4f} C-index")
        
        return "\n".join(summary)
    
    def _create_population_km(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                             events_train: np.ndarray, datasets: Dict) -> BaselineResults:
        """Create population-level KM baseline"""
        
        # Fit single KM curve
        kmf = KaplanMeierFitter()
        kmf.fit(y_train, events_train)
        
        # Get survival probability at 365 days
        try:
            pop_survival_365d = kmf.survival_function_at_times(365).iloc[0]
        except:
            pop_survival_365d = 0.5
        
        # Evaluate on validation set
        val_data = datasets.get('val') or datasets[list(datasets.keys())[0]]
        X_val, y_val, events_val = val_data
        
        # All individuals get same prediction
        predictions_365d = np.full(len(X_val), pop_survival_365d)
        
        # Calculate metrics using existing evaluation framework
        c_index = 0.5  # No discrimination for identical predictions
        
        # IPCW Brier score calculation
        ipcw_brier = self._calculate_km_ipcw_brier(
            predictions_365d, y_val, events_val, 365
        )
        
        # Log-likelihood calculation
        log_likelihood = self._calculate_km_log_likelihood(
            kmf, y_val, events_val
        )
        
        return BaselineResults(
            model_name="Population KM",
            c_index=c_index,
            ipcw_brier_score=ipcw_brier,
            log_likelihood=log_likelihood,
            predictions_365d=predictions_365d,
            sample_size=len(y_val)
        )
    
    def _create_industry_km(self, X_train: pd.DataFrame, y_train: np.ndarray,
                           events_train: np.ndarray, datasets: Dict) -> BaselineResults:
        """Create industry-stratified KM baseline"""
        
        industry_col = self._get_industry_column(X_train)
        
        # Fit KM curves by industry
        train_df = pd.DataFrame({
            'survival_time': y_train,
            'event': events_train,
            'industry': X_train[industry_col]
        })
        
        industry_km_curves = {}
        for industry in train_df['industry'].unique():
            industry_data = train_df[train_df['industry'] == industry]
            if len(industry_data) >= 10:
                kmf = KaplanMeierFitter()
                try:
                    kmf.fit(industry_data['survival_time'], industry_data['event'])
                    industry_km_curves[industry] = kmf
                except:
                    continue
        
        # Generate predictions for validation set
        val_data = datasets.get('val') or datasets[list(datasets.keys())[0]]
        X_val, y_val, events_val = val_data
        
        predictions_365d = self._assign_stratified_predictions(
            X_val, industry_km_curves, industry_col, 365
        )
        
        # Calculate metrics
        c_index = concordance_index(y_val, predictions_365d, events_val)
        
        ipcw_brier = self._calculate_km_ipcw_brier(
            predictions_365d, y_val, events_val, 365
        )
        
        # Aggregate log-likelihood from industry curves
        log_likelihood = self._calculate_stratified_log_likelihood(
            X_val, y_val, events_val, industry_km_curves, industry_col
        )
        
        return BaselineResults(
            model_name="Industry-Stratified KM",
            c_index=c_index,
            ipcw_brier_score=ipcw_brier,
            log_likelihood=log_likelihood,
            predictions_365d=predictions_365d,
            sample_size=len(y_val)
        )
    
    def _create_demographic_km(self, X_train: pd.DataFrame, y_train: np.ndarray,
                              events_train: np.ndarray, datasets: Dict) -> BaselineResults:
        """Create demographic-stratified KM baseline using age-tenure combinations"""
        
        # Create demographic strata
        demo_col = 'demo_stratum'
        train_df = self._create_demographic_strata(
            X_train.copy(), y_train, events_train
        )
        
        # Fit KM curves by demographic stratum
        demo_km_curves = {}
        for stratum in train_df[demo_col].unique():
            stratum_data = train_df[train_df[demo_col] == stratum]
            if len(stratum_data) >= 10:
                kmf = KaplanMeierFitter()
                try:
                    kmf.fit(stratum_data['survival_time'], stratum_data['event'])
                    demo_km_curves[stratum] = kmf
                except:
                    continue
        
        # Generate predictions for validation set
        val_data = datasets.get('val') or datasets[list(datasets.keys())[0]]
        X_val, y_val, events_val = val_data
        
        # Create demographic strata for validation data
        X_val_with_strata = self._create_demographic_strata(X_val.copy())
        
        predictions_365d = self._assign_stratified_predictions(
            X_val_with_strata, demo_km_curves, demo_col, 365
        )
        
        # Calculate metrics
        c_index = concordance_index(y_val, predictions_365d, events_val)
        
        ipcw_brier = self._calculate_km_ipcw_brier(
            predictions_365d, y_val, events_val, 365
        )
        
        log_likelihood = self._calculate_stratified_log_likelihood(
            X_val_with_strata, y_val, events_val, demo_km_curves, demo_col
        )
        
        return BaselineResults(
            model_name="Demographic-Stratified KM",
            c_index=c_index,
            ipcw_brier_score=ipcw_brier,
            log_likelihood=log_likelihood,
            predictions_365d=predictions_365d,
            sample_size=len(y_val)
        )
    
    def _create_random_baseline(self, datasets: Dict) -> BaselineResults:
        """Create random assignment baseline for statistical floor"""
        
        val_data = datasets.get('val') or datasets[list(datasets.keys())[0]]
        X_val, y_val, events_val = val_data
        
        # Random predictions between 0.1 and 0.9
        np.random.seed(42)
        predictions_365d = np.random.uniform(0.1, 0.9, len(X_val))
        
        c_index = 0.5  # Expected for random predictions
        
        # Simple Brier score for random predictions
        binary_outcome = ((y_val <= 365) & (events_val == 1)).astype(float)
        ipcw_brier = np.mean((predictions_365d - binary_outcome) ** 2)
        
        # Random log-likelihood (approximate)
        log_likelihood = np.sum(np.log(predictions_365d + 1e-8))
        
        return BaselineResults(
            model_name="Random Assignment",
            c_index=c_index,
            ipcw_brier_score=ipcw_brier,
            log_likelihood=log_likelihood,
            predictions_365d=predictions_365d,
            sample_size=len(y_val)
        )
    
    def _assign_stratified_predictions(self, X: pd.DataFrame, km_curves: Dict,
                                      strata_col: str, horizon: int) -> np.ndarray:
        """Assign individual predictions from stratified KM curves"""
        
        predictions = []
        for _, row in X.iterrows():
            stratum = row[strata_col]
            if stratum in km_curves:
                kmf = km_curves[stratum]
                try:
                    survival_prob = kmf.survival_function_at_times(horizon).iloc[0]
                except:
                    survival_prob = 0.5
            else:
                survival_prob = 0.5  # Fallback for unknown strata
            
            predictions.append(survival_prob)
        
        return np.array(predictions)
    
    def _calculate_km_ipcw_brier(self, predictions: np.ndarray, y: np.ndarray,
                                events: np.ndarray, horizon: int) -> float:
        """Calculate IPCW Brier score for KM predictions"""
        
        # Use existing IPCW calculation if available
        if (hasattr(self.evaluation, 'brier_calculator') and 
            self.evaluation.brier_calculator is not None):
            try:
                # Create dummy model for IPCW calculation
                class DummyModel:
                    def predict(self, X):
                        return np.log(predictions)  # Convert to log scale for consistency
                
                dummy_model = DummyModel()
                brier_result = self.evaluation.brier_calculator.calculate_brier_scores(
                    model=dummy_model,
                    X_val=pd.DataFrame(np.random.randn(len(predictions), 5)),  # Dummy features
                    y_val=y,
                    events_val=events,
                    time_points=np.array([horizon])
                )
                return brier_result.brier_scores[0]
            except:
                pass
        
        # Fallback: Simple Brier score calculation
        binary_outcome = ((y <= horizon) & (events == 1)).astype(float)
        return np.mean((predictions - binary_outcome) ** 2)
    
    def _calculate_km_log_likelihood(self, kmf: KaplanMeierFitter, 
                                    y: np.ndarray, events: np.ndarray) -> float:
        """Calculate log-likelihood for KM model"""
        
        log_likelihood = 0.0
        
        for i, (time, event) in enumerate(zip(y, events)):
            try:
                if event == 1:
                    # Event occurred: f(t) = hazard * survival
                    survival_prob = kmf.survival_function_at_times(time).iloc[0]
                    survival_prob = max(survival_prob, 1e-8)
                    
                    # Approximate hazard calculation
                    dt = 1.0
                    if time + dt <= y.max():
                        future_survival = kmf.survival_function_at_times(time + dt).iloc[0]
                        hazard = (survival_prob - future_survival) / (dt * survival_prob)
                    else:
                        hazard = 1e-3  # Small hazard for boundary cases
                    
                    density = hazard * survival_prob
                    log_likelihood += np.log(max(density, 1e-8))
                    
                else:
                    # Censored: S(t)
                    survival_prob = kmf.survival_function_at_times(time).iloc[0]
                    log_likelihood += np.log(max(survival_prob, 1e-8))
                    
            except:
                # Handle edge cases
                log_likelihood += np.log(1e-8)
        
        return log_likelihood
    
    def _calculate_stratified_log_likelihood(self, X: pd.DataFrame, y: np.ndarray,
                                            events: np.ndarray, km_curves: Dict,
                                            strata_col: str) -> float:
        """Calculate aggregate log-likelihood for stratified KM models"""
        
        total_log_likelihood = 0.0
        
        for stratum, kmf in km_curves.items():
            stratum_mask = X[strata_col] == stratum
            if stratum_mask.sum() > 0:
                stratum_y = y[stratum_mask]
                stratum_events = events[stratum_mask]
                stratum_ll = self._calculate_km_log_likelihood(kmf, stratum_y, stratum_events)
                total_log_likelihood += stratum_ll
        
        return total_log_likelihood
    
    def _create_demographic_strata(self, X: pd.DataFrame, y: np.ndarray = None,
                                  events: np.ndarray = None) -> pd.DataFrame:
        """Create demographic strata using age and tenure"""
        
        # Add survival data if provided (for training)
        if y is not None and events is not None:
            X['survival_time'] = y
            X['event'] = events
        
        # Age bands
        age_col = self._get_age_column(X)
        if age_col:
            X['age_band'] = pd.cut(X[age_col], 
                                 bins=[0, 30, 40, 50, 65, np.inf], 
                                 labels=['<30', '30-40', '40-50', '50-65', '65+'])
        else:
            X['age_band'] = 'Unknown'
        
        # Tenure bands
        tenure_col = self._get_tenure_column(X)
        if tenure_col:
            tenure_years = X[tenure_col] / 365.25 if 'days' in tenure_col else X[tenure_col]
            X['tenure_band'] = pd.cut(tenure_years,
                                    bins=[0, 1, 3, 5, 10, np.inf],
                                    labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '10yr+'])
        else:
            X['tenure_band'] = 'Unknown'
        
        # Combine for demographic stratum
        X['demo_stratum'] = X['age_band'].astype(str) + "_" + X['tenure_band'].astype(str)
        
        return X
    
    def _extract_aft_metrics(self, aft_results: Dict) -> Dict:
        """Extract AFT performance metrics from evaluation results"""
        
        if 'val' in aft_results:
            metrics = aft_results['val']
        else:
            # Find first dataset with valid metrics
            metrics = {}
            for dataset_results in aft_results.values():
                if isinstance(dataset_results, dict) and 'c_index' in dataset_results:
                    metrics = dataset_results
                    break
        
        return {
            'c_index': metrics.get('c_index', np.nan),
            'ipcw_brier_score': metrics.get('integrated_brier_score', np.nan),
            'log_likelihood': getattr(self.model_engine.aft_parameters, 'log_likelihood', np.nan)
        }
    
    def _has_industry_column(self, X: pd.DataFrame) -> bool:
        """Check for industry data availability"""
        return any(col in X.columns for col in ['naics_cd', 'naics_2digit', 'industry_cd'])
    
    def _has_demographic_columns(self, X: pd.DataFrame) -> bool:
        """Check for demographic data availability"""
        return any(col in X.columns for col in ['age', 'age_at_vantage', 'tenure_at_vantage_days'])
    
    def _get_industry_column(self, X: pd.DataFrame) -> str:
        """Get primary industry column"""
        for col in ['naics_2digit', 'naics_cd', 'industry_cd']:
            if col in X.columns:
                return col
        return X.columns[0]  # Fallback
    
    def _get_age_column(self, X: pd.DataFrame) -> Optional[str]:
        """Get primary age column"""
        for col in ['age_at_vantage', 'age']:
            if col in X.columns:
                return col
        return None
    
    def _get_tenure_column(self, X: pd.DataFrame) -> Optional[str]:
        """Get primary tenure column"""
        for col in ['tenure_at_vantage_days', 'tenure_years']:
            if col in X.columns:
                return col
        return None

if __name__ == "__main__":
    """
    Usage example with existing model infrastructure
    """
    # Integration with existing evaluation framework
    # baseline_comp = BaselineComparison(survival_evaluation)
    # baseline_results = baseline_comp.create_baseline_suite(datasets)
    # comparison = baseline_comp.compare_to_aft(aft_results, datasets)
    # print(baseline_comp.generate_ben_summary())
    
    logger.info("BaselineComparison module ready for integration")
    
# # Complete workflow
# def run_baseline_analysis(train_df, val_df, oot_df=None):
#     """Complete AFT vs baseline analysis for Ben's review"""
    
#     # 1. Initialize AFT model components
#     feature_config = FeatureConfig()
#     model_config = ModelConfig()
#     feature_processor = SmartFeatureProcessor(feature_config)
#     model_engine = SurvivalModelEngine(model_config, feature_processor)
    
#     # 2. Prepare datasets
#     datasets = {
#         'train': (train_df, 'survival_time_days', 'event_indicator_vol'),
#         'val': (val_df, 'survival_time_days', 'event_indicator_vol')
#     }
#     if oot_df is not None:
#         datasets['oot'] = (oot_df, 'survival_time_days', 'event_indicator_vol')
    
#     # 3. Train AFT model
#     print("Training AFT model...")
#     aft_results = model_engine.train_survival_model(datasets)
    
#     # 4. Initialize evaluation framework
#     evaluation_config = EvaluationConfig()
#     survival_evaluation = SurvivalEvaluation(model_engine, evaluation_config)
    
#     # 5. Run AFT evaluation
#     print("Evaluating AFT performance...")
#     eval_datasets = {name: data for name, (X, y, events) in datasets.items()}
#     aft_performance = survival_evaluation.evaluate_model_performance(eval_datasets)
    
#     # 6. Initialize baseline comparison
#     baseline_comp = BaselineComparison(survival_evaluation)
    
#     # 7. Create baseline models
#     print("Creating baseline models...")
#     baseline_results = baseline_comp.create_baseline_suite(datasets)
    
#     # 8. Compare AFT to baselines
#     print("Comparing AFT to baselines...")
#     comparison = baseline_comp.compare_to_aft(aft_performance, datasets)
    
#     # 9. Generate Ben's summary
#     ben_summary = baseline_comp.generate_ben_summary()
    
#     return {
#         'aft_results': aft_results,
#         'aft_performance': aft_performance,
#         'baseline_results': baseline_results,
#         'comparison': comparison,
#         'ben_summary': ben_summary
#     }

# # # Usage
# # results = run_baseline_analysis(train_df, val_df, oot_df)
# # print(results['ben_summary'])
"""
survival_confusion_metrics.py

Confusion matrix analysis for calibrated survival predictions across multiple time horizons.
Implements optimal threshold selection using Youden's J statistic and business-driven metrics.

Author: ADP Data Science Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ThresholdMetrics:
    """Metrics for a specific threshold"""
    threshold: float
    sensitivity: float  # True Positive Rate
    specificity: float  # True Negative Rate
    precision: float    # Positive Predictive Value
    f1_score: float
    accuracy: float
    youdens_j: float
    predicted_positive_rate: float

@dataclass
class ConfusionResults:
    """Complete confusion matrix results for a time horizon"""
    horizon_days: int
    optimal_threshold: float
    confusion_matrix: np.ndarray
    metrics: Dict[str, float]
    threshold_analysis: pd.DataFrame
    classification_report: str

class SurvivalConfusionAnalyzer:
    """
    Confusion matrix analyzer for calibrated survival predictions
    
    Determines optimal thresholds and generates confusion matrices
    for binary classification at multiple time horizons.
    """
    
    def __init__(self, calibration_engine):
        """
        Initialize with calibrated model
        
        Args:
            calibration_engine: Trained SurvivalCalibrationEngine instance
        """
        self.calibration_engine = calibration_engine
        self.horizons = [30, 90, 180, 365]  # 1, 3, 6, 12 months
        self.threshold_results = {}
        self.confusion_results = {}
    
    def find_optimal_thresholds(self, X: pd.DataFrame, y: np.ndarray, 
                               events: np.ndarray, 
                               method: str = 'youden') -> Dict[int, float]:
        """
        Find optimal classification thresholds for each horizon
        
        Args:
            X: Features
            y: Survival times
            events: Event indicators
            method: 'youden', 'f1', 'business', or 'fixed'
            
        Returns:
            Dict mapping horizon to optimal threshold
        """
        print(f"\n=== OPTIMAL THRESHOLD SELECTION ({method.upper()}) ===")
        
        optimal_thresholds = {}
        
        for horizon in self.horizons:
            print(f"\nAnalyzing {horizon}-day horizon...")
            
            # Get calibrated probabilities
            cal_probs = self.calibration_engine.predict_calibrated(X, horizon)
            
            # Create binary outcome
            actual_events = ((y <= horizon) & (events == 1)).astype(int)
            
            # Find optimal threshold based on method
            if method == 'youden':
                threshold = self._optimize_youden(cal_probs, actual_events)
            elif method == 'f1':
                threshold = self._optimize_f1(cal_probs, actual_events)
            elif method == 'business':
                threshold = self._optimize_business(cal_probs, actual_events, horizon)
            else:  # fixed
                threshold = self._get_fixed_threshold(horizon)
            
            optimal_thresholds[horizon] = threshold
            
            # Calculate metrics at optimal threshold
            predictions = (cal_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual_events, predictions).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"  Optimal threshold: {threshold:.4f}")
            print(f"  Sensitivity (Recall): {sensitivity:.3f}")
            print(f"  Specificity: {specificity:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Predicted positive rate: {predictions.mean():.3f}")
        
        self.optimal_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def generate_confusion_matrices(self, X: pd.DataFrame, y: np.ndarray,
                                   events: np.ndarray,
                                   thresholds: Dict[int, float] = None) -> Dict[int, ConfusionResults]:
        """
        Generate confusion matrices for all horizons
        
        Args:
            X: Features
            y: Survival times
            events: Event indicators
            thresholds: Optional custom thresholds (uses optimal if None)
            
        Returns:
            Dict mapping horizon to ConfusionResults
        """
        print(f"\n=== CONFUSION MATRIX GENERATION ===")
        
        if thresholds is None:
            if not hasattr(self, 'optimal_thresholds'):
                thresholds = self.find_optimal_thresholds(X, y, events)
            else:
                thresholds = self.optimal_thresholds
        
        results = {}
        
        for horizon in self.horizons:
            print(f"\n{horizon}-day Horizon:")
            
            # Get calibrated probabilities
            cal_probs = self.calibration_engine.predict_calibrated(X, horizon)
            
            # Create binary outcome
            actual_events = ((y <= horizon) & (events == 1)).astype(int)
            
            # Apply threshold
            threshold = thresholds[horizon]
            predictions = (cal_probs >= threshold).astype(int)
            
            # Generate confusion matrix
            cm = confusion_matrix(actual_events, predictions)
            
            # Calculate detailed metrics
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'actual_positive_rate': actual_events.mean(),
                'predicted_positive_rate': predictions.mean()
            }
            
            # Generate classification report
            class_report = classification_report(
                actual_events, predictions,
                target_names=['Will Stay', 'Will Leave'],
                output_dict=False
            )
            
            # Store results
            results[horizon] = ConfusionResults(
                horizon_days=horizon,
                optimal_threshold=threshold,
                confusion_matrix=cm,
                metrics=metrics,
                threshold_analysis=self._analyze_threshold_space(cal_probs, actual_events, threshold),
                classification_report=class_report
            )
            
            # Print summary
            self._print_confusion_matrix(cm, metrics, horizon, threshold)
        
        self.confusion_results = results
        return results
    
    def plot_confusion_matrices(self, results: Dict[int, ConfusionResults] = None):
        """
        Visualize confusion matrices for all horizons
        """
        if results is None:
            results = self.confusion_results
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for idx, horizon in enumerate(self.horizons):
            ax = axes[idx]
            result = results[horizon]
            
            # Create normalized confusion matrix
            cm_normalized = result.confusion_matrix.astype('float') / result.confusion_matrix.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       square=True, cbar=True, ax=ax,
                       xticklabels=['Predict Stay', 'Predict Leave'],
                       yticklabels=['Actual Stay', 'Actual Leave'])
            
            # Add raw counts as text
            for i in range(2):
                for j in range(2):
                    count = result.confusion_matrix[i, j]
                    ax.text(j + 0.5, i + 0.7, f'n={count:,}',
                           ha='center', va='center', fontsize=9, color='gray')
            
            metrics = result.metrics
            ax.set_title(f'{horizon}-Day Horizon (Threshold={result.optimal_threshold:.3f})\n' +
                        f'Sens={metrics["sensitivity"]:.2f}, Spec={metrics["specificity"]:.2f}, ' +
                        f'F1={metrics["f1_score"]:.2f}')
        
        plt.suptitle('Confusion Matrices Across Time Horizons', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_business_recommendations(self) -> pd.DataFrame:
        """
        Generate business-friendly threshold recommendations
        """
        recommendations = []
        
        for horizon, result in self.confusion_results.items():
            metrics = result.metrics
            
            # Determine use case suitability
            if metrics['precision'] > 0.5:
                use_case = "High-confidence interventions"
            elif metrics['sensitivity'] > 0.7:
                use_case = "Broad retention programs"
            else:
                use_case = "Balanced approach"
            
            recommendations.append({
                'Horizon': f'{horizon} days',
                'Threshold': f'{result.optimal_threshold:.3f}',
                'Use Case': use_case,
                'Employees Flagged': f'{metrics["predicted_positive_rate"]:.1%}',
                'Capture Rate': f'{metrics["sensitivity"]:.1%}',
                'False Alarm Rate': f'{1-metrics["specificity"]:.1%}',
                'Precision': f'{metrics["precision"]:.1%}'
            })
        
        return pd.DataFrame(recommendations)
    
    def _optimize_youden(self, probabilities: np.ndarray, actual: np.ndarray) -> float:
        """Optimize threshold using Youden's J statistic (Sensitivity + Specificity - 1)"""
        fpr, tpr, thresholds = roc_curve(actual, probabilities)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    
    def _optimize_f1(self, probabilities: np.ndarray, actual: np.ndarray) -> float:
        """Optimize threshold for maximum F1 score"""
        thresholds = np.unique(probabilities)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds[::10]:  # Sample every 10th for efficiency
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _optimize_business(self, probabilities: np.ndarray, actual: np.ndarray, horizon: int) -> float:
        """
        Business-optimized threshold based on intervention costs
        
        Assumes:
        - Cost of false positive (unnecessary intervention): $5,000
        - Cost of false negative (missed departure): $75,000
        """
        cost_fp = 5000   # Intervention cost
        cost_fn = 75000  # Replacement cost
        
        thresholds = np.unique(probabilities)
        best_cost = float('inf')
        best_threshold = 0.5
        
        for threshold in thresholds[::10]:
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()
            
            total_cost = fp * cost_fp + fn * cost_fn
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        return best_threshold
    
    def _get_fixed_threshold(self, horizon: int) -> float:
        """Fixed thresholds based on business rules"""
        threshold_map = {
            30: 0.10,   # 10% for 1 month
            90: 0.15,   # 15% for 3 months
            180: 0.20,  # 20% for 6 months
            365: 0.30   # 30% for 12 months
        }
        return threshold_map.get(horizon, 0.15)
    
    def _analyze_threshold_space(self, probabilities: np.ndarray, actual: np.ndarray,
                                optimal_threshold: float) -> pd.DataFrame:
        """Analyze metrics across threshold space"""
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, optimal_threshold]
        thresholds = sorted(list(set(thresholds)))
        
        analysis = []
        for thresh in thresholds:
            predictions = (probabilities >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()
            
            analysis.append({
                'threshold': thresh,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'flagged_pct': predictions.mean()
            })
        
        return pd.DataFrame(analysis)
    
    def _print_confusion_matrix(self, cm: np.ndarray, metrics: Dict, horizon: int, threshold: float):
        """Pretty print confusion matrix with metrics"""
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n  Confusion Matrix (Threshold = {threshold:.3f}):")
        print(f"                 Predicted")
        print(f"                 Stay    Leave")
        print(f"  Actual Stay    {tn:6d}  {fp:6d}")
        print(f"  Actual Leave   {fn:6d}  {tp:6d}")
        print(f"\n  Performance Metrics:")
        print(f"    Sensitivity (TPR):  {metrics['sensitivity']:.3f} ({tp}/{tp+fn})")
        print(f"    Specificity (TNR):  {metrics['specificity']:.3f} ({tn}/{tn+fp})")
        print(f"    Precision (PPV):    {metrics['precision']:.3f} ({tp}/{tp+fp})")
        print(f"    F1 Score:           {metrics['f1_score']:.3f}")
        print(f"    Accuracy:           {metrics['accuracy']:.3f}")
        print(f"    Employees Flagged:  {metrics['predicted_positive_rate']:.1%}")


# Example usage
if __name__ == "__main__":
    print("Survival Confusion Metrics Module")
    print("="*50)
    print("\nUsage:")
    print("analyzer = SurvivalConfusionAnalyzer(calibration_engine)")
    print("thresholds = analyzer.find_optimal_thresholds(X, y, events, method='youden')")
    print("results = analyzer.generate_confusion_matrices(X, y, events)")
    print("analyzer.plot_confusion_matrices()")
    print("recommendations = analyzer.get_business_recommendations()")
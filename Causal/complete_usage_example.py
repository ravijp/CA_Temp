"""
complete_usage_example.py

Complete example showing how to use the causal inference and driver analysis
modules with your actual data and trained model.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Your existing modules
from survival_model_engine import SurvivalModelEngine, FeatureConfig, SmartFeatureProcessor

# New modules
from causal_inference_concise import CausalInterventionAnalyzer
from driver_analysis_concise import IndividualDriverAnalyzer
from business_intelligence_integrated import BusinessIntelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Complete workflow example using actual data structure
    """
    print("="*80)
    print("COMPLETE ADP TURNOVER ANALYSIS WORKFLOW")
    print("="*80)
    
    # ========================================
    # 1. LOAD YOUR TRAINED MODEL
    # ========================================
    print("\n1. LOADING TRAINED MODEL")
    print("-" * 50)
    
    # Assuming you have already trained your model
    # If loading from disk:
    # engine = SurvivalModelEngine(model_config, feature_processor)
    # engine.load_model('./path_to_saved_model')
    
    # Or if you have it in memory from training:
    # engine = your_trained_engine
    
    # For this example, I'll show the structure:
    feature_config = FeatureConfig()
    feature_processor = SmartFeatureProcessor(feature_config)
    
    # Assume 'engine' is your trained SurvivalModelEngine
    # engine = ... (your trained model)
    
    print("✓ Model loaded with smart feature processor")
    print(f"✓ AFT distribution: {engine.aft_parameters.distribution}")
    print(f"✓ Scale parameter: {engine.aft_parameters.sigma:.4f}")
    
    # ========================================
    # 2. LOAD RAW VALIDATION DATA
    # ========================================
    print("\n2. LOADING RAW DATA")
    print("-" * 50)
    
    # Load your actual raw data (before any preprocessing)
    # This should be the same format as what you used for training
    # datasets_raw = {
    #     'val': df[df['dataset_split'] == 'val'].copy()
    # }
    
    # For demonstration:
    val_data_raw = datasets_raw['val'].copy()
    
    print(f"✓ Validation data loaded: {len(val_data_raw)} employees")
    print(f"✓ Features available: {len(val_data_raw.columns)} columns")
    
    # ========================================
    # 3. POPULATION-LEVEL CAUSAL ANALYSIS
    # ========================================
    print("\n3. POPULATION-LEVEL CAUSAL ANALYSIS")
    print("-" * 50)
    
    # Initialize causal analyzer
    causal_analyzer = CausalInterventionAnalyzer(
        model_engine=engine,
        n_bootstrap=100,  # Increase for production
        confidence_level=0.95
    )
    
    # Sample for faster analysis (remove sampling for full analysis)
    sample_size = min(1000, len(val_data_raw))
    val_sample = val_data_raw.sample(n=sample_size, random_state=42)
    
    print(f"Analyzing {len(val_sample)} employees...")
    
    # Estimate salary intervention effect
    print("\n3a. Salary Increase Intervention (15%):")
    salary_effect = causal_analyzer.estimate_salary_intervention(
        val_sample, 
        increase_pct=0.15,
        horizon=365
    )
    
    print(f"   ATE: {salary_effect.ate:.2%} risk reduction")
    print(f"   95% CI: ({salary_effect.ate_ci_lower:.2%}, {salary_effect.ate_ci_upper:.2%})")
    print(f"   P-value: {salary_effect.p_value:.4f}")
    print(f"   Significant: {salary_effect.significant}")
    print(f"   {salary_effect.responders_pct:.0f}% of employees would benefit")
    
    # Estimate promotion intervention effect
    print("\n3b. Promotion Intervention:")
    promotion_effect = causal_analyzer.estimate_promotion_intervention(
        val_sample,
        horizon=365
    )
    
    print(f"   ATE: {promotion_effect.ate:.2%} risk reduction")
    print(f"   95% CI: ({promotion_effect.ate_ci_lower:.2%}, {promotion_effect.ate_ci_upper:.2%})")
    print(f"   P-value: {promotion_effect.p_value:.4f}")
    print(f"   Significant: {promotion_effect.significant}")
    print(f"   {promotion_effect.responders_pct:.0f}% of employees would benefit")
    
    # ========================================
    # 4. HIGH-RISK EMPLOYEE IDENTIFICATION
    # ========================================
    print("\n4. HIGH-RISK EMPLOYEE IDENTIFICATION")
    print("-" * 50)
    
    # Get risk scores for all employees
    risk_scores = engine.predict_risk_scores(val_data_raw)
    
    # Identify high-risk employees (top tercile)
    high_risk_threshold = np.percentile(risk_scores, 67)
    high_risk_mask = risk_scores >= high_risk_threshold
    high_risk_employees = val_data_raw[high_risk_mask]
    
    print(f"✓ High-risk threshold: {high_risk_threshold:.2%}")
    print(f"✓ High-risk employees: {high_risk_mask.sum()} ({high_risk_mask.mean():.1%})")
    
    # ========================================
    # 5. INDIVIDUAL DRIVER ANALYSIS
    # ========================================
    print("\n5. INDIVIDUAL DRIVER ANALYSIS (Top 3 High-Risk)")
    print("-" * 50)
    
    # Initialize driver analyzer
    driver_analyzer = IndividualDriverAnalyzer(
        model_engine=engine,
        causal_analyzer=causal_analyzer
    )
    
    # Analyze top 3 high-risk employees
    top_risk_indices = np.argsort(risk_scores)[-3:]
    
    for idx in top_risk_indices:
        employee = val_data_raw.iloc[idx]
        employee_id = employee.name if hasattr(employee, 'name') else f"Employee_{idx}"
        
        print(f"\n   Employee: {employee_id}")
        print(f"   Risk Score: {risk_scores[idx]:.2%}")
        
        # Analyze drivers
        analysis = driver_analyzer.analyze_employee(employee, n_drivers=5)
        
        print(f"   Risk Category: {analysis.risk_category}")
        
        print(f"   Top Risk Factors:")
        for factor in analysis.top_risk_factors[:3]:
            print(f"     • {factor['display_name']}: {factor['value']}")
            if factor['modifiable']:
                print(f"       (Modifiable - intervention possible)")
        
        print(f"   Recommended Interventions:")
        for rec in analysis.intervention_recommendations[:2]:
            print(f"     • {rec['intervention']}: {rec['expected_risk_reduction']}")
            print(f"       Confidence: {rec['confidence']}, Priority: {rec['priority']}")
    
    # ========================================
    # 6. BUSINESS INTELLIGENCE INTEGRATION
    # ========================================
    print("\n6. INTEGRATED BUSINESS INTELLIGENCE")
    print("-" * 50)
    
    # Initialize integrated BI module
    bi = BusinessIntelligence(model_engine=engine)
    
    # Analyze high-risk cohort
    cohort_analysis = bi.analyze_high_risk_cohort(val_data_raw, risk_threshold=0.67)
    
    print(f"High-Risk Cohort Analysis:")
    print(f"   Size: {cohort_analysis['cohort_size']} employees")
    print(f"   Percentage: {cohort_analysis['percentage_of_population']:.1f}%")
    
    print(f"\n   Intervention Effectiveness:")
    for intervention, effect in cohort_analysis['intervention_effectiveness'].items():
        print(f"     {effect['name']}: {effect['ate']} reduction")
        print(f"       Responders: {effect['responders_pct']}")
    
    print(f"\n   Common Risk Factors:")
    for factor in cohort_analysis['common_risk_factors'][:3]:
        print(f"     • {factor['factor']}: {factor['frequency_pct']:.0f}% of cohort")
    
    print(f"\n   Priority Actions:")
    for action in cohort_analysis['priority_actions'][:2]:
        print(f"     • {action['action']}")
        print(f"       Expected retention: {action['expected_retention']} employees")
        print(f"       ROI estimate: {action['roi_estimate']}")
    
    # ========================================
    # 7. EXECUTIVE SUMMARY FOR ADP
    # ========================================
    print("\n7. EXECUTIVE SUMMARY FOR ADP")
    print("="*80)
    
    print("\nKEY FINDINGS:")
    print(f"• {high_risk_mask.sum()} employees identified as high risk ({high_risk_mask.mean():.1%})")
    
    if salary_effect.significant:
        print(f"• 15% salary increase would reduce turnover by {salary_effect.ate:.1%} (p={salary_effect.p_value:.3f})")
        print(f"  - {salary_effect.responders_pct:.0f}% of employees would benefit")
    
    if promotion_effect.significant:
        print(f"• Promotion would reduce turnover by {promotion_effect.ate:.1%} (p={promotion_effect.p_value:.3f})")
        print(f"  - {promotion_effect.responders_pct:.0f}% of employees would benefit")
    
    # Determine best intervention
    if salary_effect.ate > promotion_effect.ate and salary_effect.significant:
        print("\nRECOMMENDATION: Prioritize salary increases for high-risk employees")
        prevented_turnover = int(high_risk_mask.sum() * salary_effect.ate)
        print(f"   Expected impact: Prevent ~{prevented_turnover} turnovers")
    elif promotion_effect.significant:
        print("\nRECOMMENDATION: Prioritize promotions for high-risk employees")
        prevented_turnover = int(high_risk_mask.sum() * promotion_effect.ate)
        print(f"   Expected impact: Prevent ~{prevented_turnover} turnovers")
    else:
        print("\nRECOMMENDATION: Further investigation needed - interventions show limited effect")
    
    print("\nUSE CASE FOR ADP CLIENTS:")
    print("• Real-time risk scoring with explanations")
    print("• Evidence-based intervention recommendations")
    print("• Individual-level actionable insights")
    print("• Statistical confidence in recommendations")
    
    # ========================================
    # 8. SAVE RESULTS FOR REPORTING
    # ========================================
    print("\n8. SAVING RESULTS")
    print("-" * 50)
    
    # Create results dictionary
    results = {
        'population_analysis': {
            'sample_size': len(val_sample),
            'salary_effect': {
                'ate': float(salary_effect.ate),
                'ci_lower': float(salary_effect.ate_ci_lower),
                'ci_upper': float(salary_effect.ate_ci_upper),
                'p_value': float(salary_effect.p_value),
                'significant': bool(salary_effect.significant),
                'responders_pct': float(salary_effect.responders_pct)
            },
            'promotion_effect': {
                'ate': float(promotion_effect.ate),
                'ci_lower': float(promotion_effect.ate_ci_lower),
                'ci_upper': float(promotion_effect.ate_ci_upper),
                'p_value': float(promotion_effect.p_value),
                'significant': bool(promotion_effect.significant),
                'responders_pct': float(promotion_effect.responders_pct)
            }
        },
        'high_risk_analysis': {
            'n_high_risk': int(high_risk_mask.sum()),
            'pct_high_risk': float(high_risk_mask.mean()),
            'threshold': float(high_risk_threshold)
        },
        'recommendations': cohort_analysis['priority_actions']
    }
    
    # Save to JSON
    import json
    with open('intervention_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to intervention_analysis_results.json")
    
    # ========================================
    # 9. EXAMPLE UI OUTPUT
    # ========================================
    print("\n9. EXAMPLE UI OUTPUT FORMAT")
    print("-" * 50)
    
    # Get one high-risk employee for UI example
    example_employee = high_risk_employees.iloc[0]
    ui_data = bi.analyze_individual(example_employee)
    
    print("UI Display Data Structure:")
    print(f"   Employee ID: {ui_data.get('employee_id', 'Unknown')}")
    print(f"   Risk Summary: {ui_data['risk_summary']}")
    print(f"   Top Risk Factors: {ui_data['top_risk_factors'][:2]}")
    print(f"   Recommendations: {ui_data['recommendations'][:2]}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Note: You need to have your trained model 'engine' and raw data 'datasets_raw' available
    
    # Example of how to set up:
    """
    # 1. Load your trained model
    feature_config = FeatureConfig()
    feature_processor = SmartFeatureProcessor(feature_config)
    engine = SurvivalModelEngine(model_config, feature_processor)
    engine.load_model('./your_saved_model_path')
    
    # 2. Load your raw data
    df = pd.read_csv('your_data.csv')  # or however you load it
    datasets_raw = {
        'val': df[df['dataset_split'] == 'val'].copy()
    }
    
    # 3. Run the analysis
    results = main()
    """
    
    print("Ready to run with your actual model and data!")
    print("\nReplace the mock objects with:")
    print("  - engine: Your trained SurvivalModelEngine")
    print("  - datasets_raw: Your actual raw validation data")

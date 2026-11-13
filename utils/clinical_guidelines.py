"""
Cardio Vue AI — Clinical Decision Support Tools
ASCVD Risk Score Calculator and Guideline-Based Recommendations
"""

import numpy as np
import math

def ascvd_risk_score(age, sex, race, total_chol, hdl_chol, systolic_bp, 
                     treated_hypertension, diabetes, smoker):
    """
    Compute 10-year ASCVD risk using the Pooled Cohort Equations.
    
    Parameters:
        age: years (40-79)
        sex: 'M' or 'F'
        race: 'White' or 'African American'
        total_chol: mg/dL
        hdl_chol: mg/dL
        systolic_bp: mmHg
        treated_hypertension: bool
        diabetes: bool
        smoker: bool
    
    Returns risk percentage (0-100)
    """
    # Validate age range
    if age < 40 or age > 79:
        return None  # ASCVD equation not validated outside 40-79
    
    # Coefficients from ACC/AHA guidelines
    if race.lower() == 'african american':
        if sex == 'M':
            coeff = {
                'beta_age': -0.321670, 'beta_tc': -0.402662, 'beta_hdl': -0.676495,
                'beta_sbp': -0.144514, 'beta_sbp_treated': -0.164110, 'beta_smoker': -0.263852,
                'beta_diabetes': -0.385087, 'beta_intercept': 4.112491,
                'mean_risk': 0.9055, 'sigma': 0.7999
            }
        else:  # Female
            coeff = {
                'beta_age': -0.031418, 'beta_tc': -0.273963, 'beta_hdl': -0.687775,
                'beta_sbp': -0.117014, 'beta_sbp_treated': -0.124443, 'beta_smoker': -0.354279,
                'beta_diabetes': -0.285082, 'beta_intercept': 3.248779,
                'mean_risk': 0.6736, 'sigma': 0.8112
            }
    else:  # White (including other)
        if sex == 'M':
            coeff = {
                'beta_age': 1.71691, 'beta_tc': 0.32548, 'beta_hdl': -0.35928,
                'beta_sbp': 0.25837, 'beta_sbp_treated': 0.70549, 'beta_smoker': 0.58053,
                'beta_diabetes': 0.65772, 'beta_intercept': -7.97618,
                'mean_risk': 0.9055, 'sigma': 0.7999
            }
        else:  # Female
            coeff = {
                'beta_age': 2.32889, 'beta_tc': 0.14427, 'beta_hdl': -0.33570,
                'beta_sbp': 0.31909, 'beta_sbp_treated': 0.67949, 'beta_smoker': 0.60429,
                'beta_diabetes': 0.64878, 'beta_intercept': -10.00574,
                'mean_risk': 0.6736, 'sigma': 0.8112
            }
    
    # Transform variables
    ln_age = math.log(age)
    ln_tc = math.log(total_chol)
    ln_hdl = math.log(hdl_chol)
    
    if treated_hypertension:
        bp_term = coeff['beta_sbp_treated'] * (systolic_bp / 10)
    else:
        bp_term = coeff['beta_sbp'] * (systolic_bp / 10)
    
    # Compute linear predictor
    lp = (coeff['beta_age'] * ln_age +
          coeff['beta_tc'] * ln_tc +
          coeff['beta_hdl'] * ln_hdl +
          bp_term +
          coeff['beta_smoker'] * (1 if smoker else 0) +
          coeff['beta_diabetes'] * (1 if diabetes else 0) +
          coeff['beta_intercept'])
    
    # Survival function at 10 years
    surv = np.exp(-np.exp(lp) * coeff['mean_risk'])
    risk = 1 - surv
    return risk * 100

def get_ascvd_recommendation(risk_percent, age, diabetes=False, smoking=False, hypertension=False):
    """
    Provide guideline-based recommendations based on ASCVD risk.
    """
    if risk_percent is None:
        return "ASCVD risk score only available for ages 40-79. For younger patients, focus on lifestyle modifications."
    
    if risk_percent < 5:
        return "✅ Low risk (<5%). Lifestyle modifications recommended. Reassess in 4-6 years."
    elif risk_percent < 7.5:
        return "⚠️ Borderline risk (5-7.5%). Consider lifestyle changes; discuss statin therapy if risk factors persist."
    elif risk_percent < 20:
        return "🔴 Intermediate risk (7.5-20%). Moderate-intensity statin therapy is recommended. Consider aspirin if high risk."
    else:
        return "🚨 High risk (≥20%). High-intensity statin therapy is recommended. Consider aspirin 81mg daily. Urgent lifestyle intervention needed."

def get_lifestyle_recommendations(risk_level):
    """Get lifestyle recommendations based on risk level."""
    recommendations = {
        'Low': [
            "Continue healthy habits",
            "150 minutes of moderate exercise weekly",
            "Maintain healthy weight",
            "Balanced diet with fruits and vegetables"
        ],
        'Moderate': [
            "Increase physical activity to 150-300 minutes/week",
            "Adopt Mediterranean diet",
            "Limit sodium to <2300mg/day",
            "Consider stress management techniques"
        ],
        'High': [
            "Urgent: Schedule cardiology consultation",
            "Start supervised exercise program",
            "Strict dietary changes - consult nutritionist",
            "Monitor BP and glucose daily"
        ],
        'Critical': [
            "IMMEDIATE: Seek cardiology care",
            "Emergency evaluation if symptoms present",
            "Daily health monitoring required",
            "Medication adherence is critical"
        ]
    }
    return recommendations.get(risk_level, recommendations['Moderate'])

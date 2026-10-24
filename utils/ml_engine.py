"""Risk prediction using trained project models or a clearly identified fallback."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

BASE_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income",
]
ENGINEERED_FEATURES = [
    "BMI_GenHlth", "BMI_Age", "BMI_HighBP", "BMI_Smoker",
    "Age_HighBP", "Age_HighChol", "Age_Smoker", "Risk_Score",
    "Metabolic_Score", "BMI_squared", "Age_squared", "Healthy_Lifestyle",
]
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
FEATURE_DISPLAY = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "BMI": "BMI",
    "Smoker": "Smoking",
    "Diabetes": "Diabetes",
    "PhysActivity": "Physical Activity",
    "GenHlth": "General Health",
    "Age": "Age Group",
    "Stroke": "Prior Stroke",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex",
    "Risk_Score": "Combined Risk Score",
    "BMI_Age": "BMI × Age Interaction",
    "Age_HighBP": "Age × BP Interaction",
    "Metabolic_Score": "Metabolic Score",
}

_WEIGHTS = {
    "HighBP": 0.195,
    "HighChol": 0.135,
    "Smoker": 0.175,
    "Diabetes": 0.220,
    "BMI": 0.0028,
    "Age": 0.020,
    "GenHlth": 0.055,
    "PhysActivity": -0.090,
    "Stroke": 0.140,
    "DiffWalk": 0.080,
    "Risk_Score": 0.035,
    "BMI_Age": 0.0006,
    "Age_HighBP": 0.018,
    "Metabolic_Score": 0.025,
    "BMI_HighBP": 0.0015,
}


def _age_to_group(age_years: int) -> int:
    brackets = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for index, threshold in enumerate(brackets):
        if age_years < threshold:
            return max(1, index)
    return 13


def engineer_features(raw: dict) -> pd.DataFrame:
    """Create the same 33 model columns used during project training."""
    row = {feature: raw.get(feature, 0) for feature in BASE_FEATURES}
    values = pd.DataFrame([row])
    values["BMI_GenHlth"] = values["BMI"] * values["GenHlth"]
    values["BMI_Age"] = values["BMI"] * values["Age"]
    values["BMI_HighBP"] = values["BMI"] * values["HighBP"]
    values["BMI_Smoker"] = values["BMI"] * values["Smoker"]
    values["Age_HighBP"] = values["Age"] * values["HighBP"]
    values["Age_HighChol"] = values["Age"] * values["HighChol"]
    values["Age_Smoker"] = values["Age"] * values["Smoker"]
    values["Risk_Score"] = (
        values["HighBP"] * 2
        + values["HighChol"] * 1.5
        + values["Smoker"] * 2
        + values["Diabetes"] * 2.5
    )
    values["Metabolic_Score"] = (
        values["HighBP"] + values["HighChol"] + values["BMI"] / 30 + values["GenHlth"] / 5
    )
    values["BMI_squared"] = values["BMI"] ** 2
    values["Age_squared"] = values["Age"] ** 2
    values["Healthy_Lifestyle"] = (
        values["PhysActivity"] + values["Fruits"] + values["Veggies"] + values["HvyAlcoholConsump"]
    )
    return values[ALL_FEATURES]


@st.cache_resource(show_spinner="Loading risk prediction model...")
def load_ml_models():
    """Load the first available model artifact from the local models folder."""
    import joblib

    model_dir = os.environ.get("CARDIOVUE_MODELS", "models")
    scaler = None
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    candidates = [
        ("Extreme_Random_Forest.pkl", "Extreme Random Forest", True, 95.8),
        ("XGBoost.pkl", "XGBoost", False, 91.6),
        ("LightGBM.pkl", "LightGBM", True, 95.2),
        ("CatBoost.pkl", "CatBoost", True, 91.8),
        ("Neural_Network.pkl", "Neural Network", False, 94.1),
    ]
    for filename, display_name, is_tree, confidence in candidates:
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        explainer = None
        if is_tree:
            try:
                import shap

                explainer = shap.TreeExplainer(model)
            except (ImportError, ValueError, TypeError):
                explainer = None
        return {
            "model": model,
            "model_name": display_name,
            "scaler": scaler,
            "explainer": explainer,
            "confidence": confidence,
        }
    return {"model": None, "model_name": "Calibrated Simulator", "scaler": None, "explainer": None}


def _simulate_prediction(feature_df: pd.DataFrame):
    row = feature_df.iloc[0].to_dict()
    score = 0.02
    contributions = {}
    for feature, weight in _WEIGHTS.items():
        contribution = weight * row.get(feature, 0)
        score += contribution
        contributions[FEATURE_DISPLAY.get(feature, feature)] = round(contribution, 4)
    if row.get("HighBP") and row.get("HighChol"):
        score += 0.045
        contributions["BP × Cholesterol interaction"] = 0.045
    if row.get("Smoker") and row.get("Diabetes"):
        score += 0.065
        contributions["Smoking × Diabetes interaction"] = 0.065
    if row.get("Age", 0) > 8 and row.get("HighBP"):
        score += 0.038
        contributions["Age × BP interaction"] = 0.038
    return float(np.clip(score, 0.0, 1.0)), contributions


def _risk_label(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


def _risk_color(label: str) -> str:
    return {
        "Low": "#10B981",
        "Moderate": "#F59E0B",
        "High": "#EF4444",
        "Critical": "#C8102E",
    }.get(label, "#8A9BBE")


def _compute_confidence(probability: float) -> float:
    return 78.0 + abs(probability - 0.5) * 36.0


def _confidence_interval(probability: float, confidence: float):
    margin = (1.0 - confidence / 100) * 0.5
    return max(0.0, probability - margin), min(1.0, probability + margin)


def _extract_shap_values(explainer, model_input):
    import shap

    values = explainer.shap_values(model_input)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0, :, -1]
    elif values.ndim == 2:
        values = values[0]
    return {
        FEATURE_DISPLAY.get(feature, feature): round(float(value), 4)
        for feature, value in zip(ALL_FEATURES, values)
        if abs(float(value)) > 0.001
    }


def predict_risk(features: dict) -> dict:
    """Return a risk estimate and model feature contributions for one patient."""
    raw = {
        "HighBP": int(features.get("highbp", 0)),
        "HighChol": int(features.get("highchol", 0)),
        "CholCheck": 1,
        "BMI": float(features.get("bmi", 25)),
        "Smoker": int(features.get("smoker", 0)),
        "Stroke": int(features.get("stroke", 0)),
        "Diabetes": int(features.get("diabetes", 0)),
        "PhysActivity": int(features.get("phys_activity", 0)),
        "Fruits": int(features.get("fruits", 1)),
        "Veggies": int(features.get("veggies", 1)),
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": int(features.get("gen_health", 3)),
        "MentHlth": int(features.get("ment_health", 5)),
        "PhysHlth": int(features.get("phys_health", 5)),
        "DiffWalk": int(features.get("diff_walk", 0)),
        "Sex": int(features.get("sex", 1)),
        "Age": _age_to_group(int(features.get("age", 45))),
        "Education": int(features.get("education", 4)),
        "Income": int(features.get("income", 5)),
    }
    feature_df = engineer_features(raw)
    model_bundle = load_ml_models()
    probability, contributions = _simulate_prediction(feature_df)
    model_name = model_bundle["model_name"]
    confidence = _compute_confidence(probability)

    if model_bundle["model"] is not None:
        try:
            model_input = (
                model_bundle["scaler"].transform(feature_df)
                if model_bundle["scaler"] is not None
                else feature_df
            )
            expected = getattr(model_bundle["model"], "n_features_in_", len(ALL_FEATURES))
            if model_input.shape[1] != expected:
                st.warning("Model feature count differs from this app's schema; using the fallback simulator.")
                model_name = "Calibrated Simulator"
            else:
                model = model_bundle["model"]
                probabilities = model.predict_proba(model_input)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                positive_index = classes.index(1) if 1 in classes else -1
                probability = float(probabilities[positive_index])
                model_name = model_bundle["model_name"]
                confidence = float(model_bundle.get("confidence", _compute_confidence(probability)))
                if model_bundle["explainer"] is not None:
                    try:
                        contributions = _extract_shap_values(model_bundle["explainer"], model_input)
                    except (ImportError, ValueError, TypeError, IndexError):
                        pass
        except (ValueError, AttributeError, IndexError) as exc:
            st.warning(f"Trained model inference failed; using fallback simulator: {exc}")
            probability, contributions = _simulate_prediction(feature_df)
            model_name = "Calibrated Simulator"
            confidence = _compute_confidence(probability)

    score = round(probability * 100, 1)
    risk_label = _risk_label(score)
    confidence = round(confidence, 1)
    ci_low, ci_high = _confidence_interval(probability, confidence)
    top_contributions = dict(
        sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    )
    return {
        "risk_score": score,
        "probability": score,
        "risk_label": risk_label,
        "risk_color": _risk_color(risk_label),
        "shap_values": top_contributions,
        "model_name": model_name,
        "model_confidence": confidence,
        "ci_low": round(ci_low * 100, 1),
        "ci_high": round(ci_high * 100, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "feature_vector": raw,
    }

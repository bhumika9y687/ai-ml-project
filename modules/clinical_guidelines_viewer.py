# modules/clinical_guidelines_viewer.py
"""
Clinical Guidelines Reference - ACC/AHA/ESC Guidelines
"""

import streamlit as st
import pandas as pd

def show_clinical_guidelines():
    """Display clinical guidelines for cardiovascular care"""
    
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-size: 2rem;">📋</div>
            <div>
                <h1 style="font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; margin: 0;">Clinical Guidelines</h1>
                <p style="color: var(--t2); margin: 4px 0 0;">Evidence-based recommendations from ACC/AHA/ESC</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["🏥 Blood Pressure", "🩸 Cholesterol", "💊 Diabetes", "🏃 Lifestyle", "📊 Risk Assessment"])
    
    with tabs[0]:
        show_bp_guidelines()
    
    with tabs[1]:
        show_cholesterol_guidelines()
    
    with tabs[2]:
        show_diabetes_guidelines()
    
    with tabs[3]:
        show_lifestyle_guidelines()
    
    with tabs[4]:
        show_risk_guidelines()

def show_bp_guidelines():
    st.markdown("### 2017 ACC/AHA Blood Pressure Guidelines")
    
    bp_data = {
        "Category": ["Normal", "Elevated", "Stage 1 HTN", "Stage 2 HTN", "Hypertensive Crisis"],
        "Systolic": ["<120", "120-129", "130-139", "≥140", ">180"],
        "Diastolic": ["<80", "<80", "80-89", "≥90", ">120"],
        "Recommendation": [
            "Maintain healthy lifestyle",
            "Lifestyle modification",
            "Lifestyle + consider medication if high risk",
            "Lifestyle + medication",
            "Urgent medical attention"
        ]
    }
    
    df = pd.DataFrame(bp_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.info("💡 **Key Takeaway**: New guidelines lowered threshold for hypertension to 130/80 mmHg to enable earlier intervention.")

def show_cholesterol_guidelines():
    st.markdown("### 2018 ACC/AHA Cholesterol Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### LDL Cholesterol Goals")
        ldl_data = {
            "Risk Category": ["Low (<5%)", "Borderline (5-7.5%)", "Intermediate (7.5-20%)", "High (>20%)"],
            "LDL Goal": ["<160 mg/dL", "<130 mg/dL", "<100 mg/dL", "<70 mg/dL"],
            "Statin Intensity": ["None", "Low", "Moderate", "High"]
        }
        st.dataframe(pd.DataFrame(ldl_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Statin Therapy Indications")
        st.markdown("""
        - **Primary prevention**: Age 40-75 with LDL 70-189 mg/dL AND 10-year ASCVD risk ≥7.5%
        - **Secondary prevention**: Known ASCVD (any LDL level)
        - **Diabetes**: Age 40-75 with LDL 70-189 mg/dL (moderate-intensity)
        - **Very high LDL**: LDL ≥190 mg/dL (high-intensity)
        """)
    
    st.warning("⚠️ **Note**: Statin therapy decisions should be individualized based on patient preferences and risk-benefit assessment.")

def show_diabetes_guidelines():
    st.markdown("### 2024 ADA Diabetes Standards of Care")
    
    st.markdown("#### Glycemic Targets")
    
    targets = {
        "Parameter": ["HbA1c", "Fasting Glucose", "Postprandial Glucose", "Time in Range (TIR)"],
        "Target": ["<7.0%", "80-130 mg/dL", "<180 mg/dL", ">70%"],
        "Comments": [
            "Individualize for older adults",
            "Before meals",
            "1-2 hours after meals",
            "70-180 mg/dL"
        ]
    }
    
    st.dataframe(pd.DataFrame(targets), use_container_width=True, hide_index=True)
    
    st.markdown("#### Cardiovascular Risk Management in Diabetes")
    
    cv_risks = {
        "Condition": ["ASCVD", "Heart Failure", "CKD", "High CV Risk"],
        "Recommended Agent": ["GLP-1 RA or SGLT2i", "SGLT2i", "SGLT2i or GLP-1 RA", "GLP-1 RA or SGLT2i"],
        "Evidence Level": ["A", "A", "A", "A"]
    }
    
    st.dataframe(pd.DataFrame(cv_risks), use_container_width=True, hide_index=True)

def show_lifestyle_guidelines():
    st.markdown("### 2021 AHA/ACC Lifestyle Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🥗 Diet")
        st.success("**DASH Diet**")
        st.write("- Fruits: 4-5 servings/day")
        st.write("- Vegetables: 4-5 servings/day")
        st.write("- Whole grains: 6-8 servings/day")
        st.write("- Low-fat dairy: 2-3 servings/day")
        st.write("- Lean meat/fish: <6 oz/day")
        
        st.warning("**Limit**")
        st.write("- Sodium: <2300 mg/day (1500 mg if high BP)")
        st.write("- Saturated fat: <6% of calories")
        st.write("- Added sugars: <10% of calories")
    
    with col2:
        st.markdown("#### 🏃 Physical Activity")
        st.success("**AHA Recommendations**")
        st.write("- **Aerobic**: 150 min/week moderate OR 75 min/week vigorous")
        st.write("- **Strength**: 2 days/week resistance training")
        st.write("- **Flexibility**: Daily stretching")
        
        st.info("💡 **Start slow**: Even 10-minute walks count! Gradually increase duration and intensity.")
    
    st.markdown("#### 🚭 Smoking Cessation")
    st.error("**Impact**: Smoking cessation reduces CVD risk by 50% within 1 year")
    st.write("**Resources**: Nicotine replacement therapy, counseling, medications (varenicline, bupropion)")

def show_risk_guidelines():
    st.markdown("### Risk Assessment Guidelines")
    
    st.markdown("#### ASCVD Risk Estimator Plus")
    st.write("**Validated for**: Adults 40-79 years without known ASCVD")
    
    risk_categories = {
        "Risk Level": ["Low", "Borderline", "Intermediate", "High"],
        "10-Year Risk": ["<5%", "5-7.5%", "7.5-20%", ">20%"],
        "Recommended Action": [
            "Lifestyle modification, reassess in 4-6 years",
            "Lifestyle changes, consider statin",
            "Moderate-intensity statin",
            "High-intensity statin, consider aspirin"
        ]
    }
    
    st.dataframe(pd.DataFrame(risk_categories), use_container_width=True, hide_index=True)
    
    st.markdown("#### When to Consider Additional Testing")
    st.markdown("""
    - **Coronary Artery Calcium (CAC) Score**: Adults 40-75 with borderline/intermediate risk
    - **Ankle-Brachial Index (ABI)**: For peripheral artery disease screening
    - **High-sensitivity CRP**: For risk refinement in intermediate-risk patients
    - **Lipoprotein(a)**: Family history of premature ASCVD
    """)
    
    st.caption("Source: 2019 ACC/AHA Guidelines on the Primary Prevention of Cardiovascular Disease")

def show_medication_reference():
    """Medication reference tool"""
    
    st.markdown("### 💊 Cardiovascular Medications Reference")
    
    meds = {
        "Medication Class": ["Statins", "Beta-blockers", "ACE Inhibitors", "ARBs", "Calcium Channel Blockers", "SGLT2 Inhibitors", "GLP-1 Agonists"],
        "Common Drugs": ["Atorvastatin, Rosuvastatin", "Metoprolol, Carvedilol", "Lisinopril, Ramipril", "Losartan, Valsartan", "Amlodipine, Diltiazem", "Empagliflozin, Dapagliflozin", "Semaglutide, Liraglutide"],
        "Primary Use": ["LDL reduction", "Heart rate, BP", "BP, HF", "BP, HF", "BP, angina", "DM, HF, CKD", "DM, CV risk"],
        "Key Side Effects": ["Myalgia, LFT elevation", "Fatigue, bradycardia", "Cough, hyperkalemia", "Hyperkalemia", "Edema, headache", "UTI, dehydration", "GI symptoms"]
    }
    
    df = pd.DataFrame(meds)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.caption("⚠️ **Medical Disclaimer**: This is for reference only. All medication decisions must be made by a qualified healthcare provider.")
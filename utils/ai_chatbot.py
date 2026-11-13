"""
Cardio Vue AI — Intelligent Health Chatbot
Uses Google Gemini 1.5 Flash (free tier) for personalized cardiovascular advice.
Falls back to rule-based responses if API key not configured.
"""

import re
import os
import streamlit as st

# Try Gemini (free), then fall back to rule-based
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


SYSTEM_PROMPT = """You are Cardio Vue AI, an intelligent cardiovascular health assistant integrated
into a medical platform. You provide personalized, evidence-based guidance about heart health.

PATIENT CONTEXT:
{patient_context}

GUIDELINES:
- Be warm, empathetic, and clear. Use plain language, not medical jargon.
- Always personalize responses to the patient's specific data above.
- For HIGH or CRITICAL risk patients, be more urgent about recommendations.
- Keep responses concise (3-5 sentences max for simple questions).
- NEVER diagnose or prescribe. Always encourage doctor consultation for medical decisions.
- If someone mentions chest pain, breathlessness, or emergency symptoms: immediately direct to emergency services.
- Base lifestyle advice on ACC/AHA cardiovascular guidelines.
- When relevant, quantify the impact (e.g., "quitting smoking reduces your risk by ~18%").
- Always sign off as "— Cardio Vue AI 🫀"

FORMAT: Use simple markdown. Bullet points for lists. No headers for short responses."""

EMERGENCY_KEYWORDS = [
    'chest pain', 'chest pressure', 'breathless', "can't breathe", 'cannot breathe',
    'heart attack', 'stroke', 'faint', 'dizzy', 'severe', 'collapse', 'emergency',
    'numb arm', 'jaw pain', 'sweating heavily', 'palpitation', 'irregular heartbeat'
]

HEALTH_TOPICS = {
    'diet': [
        "mediterranean diet", "what to eat", "food", "diet", "nutrition",
        "sodium", "salt", "cholesterol food", "omega-3", "fiber"
    ],
    'exercise': [
        "exercise", "workout", "physical activity", "walking", "cardio",
        "how much exercise", "gym", "jogging", "swimming"
    ],
    'medication': [
        "medication", "medicine", "aspirin", "statin", "beta blocker",
        "blood pressure medication", "metformin", "drugs"
    ],
    'stress': [
        "stress", "anxiety", "mental health", "sleep", "relax",
        "meditation", "breathing", "mindfulness"
    ],
    'risk': [
        "risk score", "what does my score mean", "risk factors",
        "how to reduce", "probability", "chance of heart disease"
    ],
    'symptoms': [
        "symptom", "feeling", "tired", "fatigue", "shortness",
        "swollen", "weight gain", "headache"
    ]
}

FALLBACK_RESPONSES = {
    'diet': """For your heart health, here's what matters most:

- **Mediterranean diet**: Olive oil, fish (salmon, mackerel), legumes, nuts, vegetables
- **Reduce sodium** to <2,300 mg/day (aim for 1,500 mg with high BP)
- **Increase fiber**: 25–35g/day — oats, lentils, berries are excellent
- **Omega-3 foods**: Fatty fish 2x/week, walnuts, flaxseeds
- **Limit**: Processed meats, fried foods, sugary drinks, trans fats

Studies show the Mediterranean diet reduces cardiovascular events by ~30%.

— Cardio Vue AI 🫀""",

    'exercise': """Exercise is one of the most powerful interventions for heart health:

- **Target**: 150 min/week of moderate activity, OR 75 min/week of vigorous
- **Best types**: Brisk walking, cycling, swimming, dancing
- **Start slow** if inactive: Begin with 10-min walks, gradually increase
- **Add strength training**: 2x/week improves metabolic health
- **Monitor heart rate**: Aim for 50–70% of max HR (220 − your age) during moderate exercise

Regular exercise can reduce your cardiovascular risk by 12–15%.

— Cardio Vue AI 🫀""",

    'stress': """Chronic stress is a significant cardiovascular risk factor:

- **Deep breathing**: The 4-7-8 technique (inhale 4s, hold 7s, exhale 8s) activates the parasympathetic system
- **Mindfulness meditation**: Even 10 min/day reduces blood pressure by ~5 mmHg
- **Sleep**: 7–9 hours is essential — sleep deprivation increases heart disease risk by 48%
- **Social connection**: Strong relationships are protective against cardiovascular disease
- **Yoga**: Combines breathing, movement, and mindfulness — clinical evidence for BP reduction

Would you like more detail on any of these techniques?

— Cardio Vue AI 🫀""",

    'risk': """Your risk score reflects the output of our ensemble ML model (XGBoost + LightGBM + Stacking):

- **0–25%**: Low risk — maintain healthy habits
- **25–50%**: Moderate risk — lifestyle improvements can significantly reduce risk
- **50–75%**: High risk — medical consultation and active management recommended
- **75–100%**: Critical — urgent medical evaluation needed

The key modifiable factors in your score include blood pressure, cholesterol, smoking status, BMI, and physical activity. Addressing these can measurably reduce your score.

— Cardio Vue AI 🫀""",

    'medication': """I can share general information, but medication decisions should always involve your doctor:

- **Statins** (e.g., atorvastatin): Reduce LDL cholesterol and cardiovascular events by 25–35%
- **Aspirin (75mg)**: Often prescribed for high-risk patients to prevent clot formation
- **Beta-blockers**: Help control heart rate and blood pressure
- **ACE inhibitors/ARBs**: First-line for blood pressure management with diabetes

Please discuss your specific medication needs with Dr. Priya Sharma at your next consultation.

— Cardio Vue AI 🫀""",

    'default': """I'm here to help you understand your cardiovascular health better. I can assist with:

- 🥗 **Diet & nutrition** for heart health
- 🏃 **Exercise recommendations** tailored to your risk level
- 😴 **Stress management & sleep** guidance
- 💊 **General medication** information (not prescriptions)
- 📊 **Understanding your risk score** and what drives it
- 🎯 **Lifestyle interventions** with quantified impact

What would you like to know? I'll personalize my advice based on your health profile.

— Cardio Vue AI 🫀"""
}


def _build_patient_context(patient_data: dict, latest_record: dict = None) -> str:
    parts = [
        f"Name: {patient_data.get('name', 'Patient')}",
        f"Age: {patient_data.get('age', 'unknown')} years",
    ]
    if latest_record:
        parts += [
            f"Current Risk Score: {latest_record.get('risk_score', 'N/A')}% ({latest_record.get('risk_label', 'N/A')})",
            f"Blood Pressure: {latest_record.get('bp_systolic', '--')}/{latest_record.get('bp_diastolic', '--')} mmHg",
            f"Cholesterol: {latest_record.get('cholesterol', 'N/A')} mg/dL",
            f"BMI: {latest_record.get('bmi', 'N/A')}",
            f"Smoker: {'Yes' if latest_record.get('smoker') else 'No'}",
            f"Diabetes: {'Yes' if latest_record.get('diabetes') else 'No'}",
            f"High BP: {'Yes' if latest_record.get('highbp') else 'No'}",
            f"High Cholesterol: {'Yes' if latest_record.get('highchol') else 'No'}",
            f"Physical Activity: {'Active' if latest_record.get('phys_activity') else 'Sedentary'}",
        ]
    return "\n".join(parts)


def _classify_topic(message: str) -> str:
    msg_lower = message.lower()
    for topic, keywords in HEALTH_TOPICS.items():
        if any(kw in msg_lower for kw in keywords):
            return topic
    return 'default'


def _is_emergency(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in EMERGENCY_KEYWORDS)


def get_ai_response(
    message: str,
    chat_history: list,
    patient_data: dict,
    latest_record: dict = None,
    api_key: str = None
) -> str:
    """
    Get AI response for a health question.
    Tries Gemini API first, falls back to rule-based if unavailable.
    """
    # Emergency check — always first
    if _is_emergency(message):
        return (
            "🚨 **This sounds like it could be a medical emergency.**\n\n"
            "**Please call emergency services immediately (112 in India / 911 in US)**\n\n"
            "Symptoms like chest pain, breathlessness, jaw pain, or arm numbness can indicate "
            "a heart attack or stroke. Do not wait — every minute matters. "
            "Call emergency services or have someone take you to the nearest emergency room NOW.\n\n"
            "— Cardio Vue AI 🫀"
        )

    # Try Gemini API
    if GEMINI_AVAILABLE and (api_key or os.environ.get("GEMINI_API_KEY")):
        try:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            patient_context = _build_patient_context(patient_data, latest_record)
            system = SYSTEM_PROMPT.format(patient_context=patient_context)

            # Build conversation history for context
            history = []
            for msg in chat_history[-6:]:  # Last 3 exchanges
                if msg['role'] == 'user':
                    history.append({'role': 'user', 'parts': [msg['content']]})
                elif msg['role'] == 'assistant':
                    history.append({'role': 'model', 'parts': [msg['content']]})

            chat = model.start_chat(history=history)
            response = chat.send_message(f"{system}\n\nUser question: {message}")
            return response.text

        except Exception as e:
            # Fall through to rule-based
            pass

    # Rule-based fallback
    topic = _classify_topic(message)
    base_response = FALLBACK_RESPONSES.get(topic, FALLBACK_RESPONSES['default'])

    # Personalize if we have patient data
    if latest_record:
        risk_label = latest_record.get('risk_label', 'Moderate')
        risk_score = latest_record.get('risk_score', 50)
        name = patient_data.get('name', 'there').split()[0]

        if topic == 'risk':
            base_response = f"Hi {name}, your current risk score is **{risk_score}%** ({risk_label}).\n\n" + base_response

        if risk_label in ['High', 'Critical'] and topic in ['exercise', 'diet']:
            base_response += (
                f"\n\n⚠️ Given your **{risk_label.lower()} risk profile ({risk_score}%)**, "
                "please consult Dr. Kishan before starting any new exercise program."
            )

    return base_response


def get_quick_insights(patient_data: dict, latest_record: dict) -> list:
    """
    Generate 4 personalized insight cards for the dashboard.
    """
    insights = []
    if not latest_record:
        return []

    score = latest_record.get('risk_score', 50)
    smoker = latest_record.get('smoker', 0)
    phys = latest_record.get('phys_activity', 0)
    bmi = latest_record.get('bmi', 25)
    highbp = latest_record.get('highbp', 0)
    highchol = latest_record.get('highchol', 0)

    if smoker:
        insights.append({
            'icon': '🚭', 'title': 'Quit Smoking',
            'body': 'Smoking is your #1 modifiable risk factor. Cessation reduces heart disease risk by ~50% within 1 year.',
            'impact': '↓ Risk ~18%', 'priority': 'high'
        })

    if not phys:
        insights.append({
            'icon': '🏃', 'title': 'Start Moving',
            'body': 'You\'re currently sedentary. Just 30 min brisk walking 5x/week can reduce your cardiovascular risk by 12%.',
            'impact': '↓ Risk ~12%', 'priority': 'high'
        })

    if bmi > 28:
        insights.append({
            'icon': '⚖️', 'title': 'Weight Management',
            'body': f'Your BMI ({bmi}) is above optimal. Reducing by 5% reduces BP, cholesterol, and diabetes risk simultaneously.',
            'impact': '↓ Risk ~8%', 'priority': 'medium'
        })

    if highbp:
        insights.append({
            'icon': '💊', 'title': 'BP Control',
            'body': 'High BP is present. The DASH diet + medication (if prescribed) can reduce systolic BP by 10–20 mmHg.',
            'impact': '↓ Risk ~10%', 'priority': 'medium'
        })

    if highchol:
        insights.append({
            'icon': '🥗', 'title': 'Cholesterol Diet',
            'body': 'Increase omega-3 fatty acids (salmon, walnuts), soluble fiber (oats, lentils), and plant sterols.',
            'impact': '↓ LDL ~15%', 'priority': 'medium'
        })

    insights.append({
        'icon': '😴', 'title': 'Optimize Sleep',
        'body': 'Aim for 7-9 hours. Sleep deprivation increases inflammatory markers and cardiovascular risk by up to 48%.',
        'impact': '↓ Risk ~6%', 'priority': 'low'
    })

    return insights[:4]

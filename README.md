# Cardio Vue AI

Cardio Vue AI is a cardiovascular disease risk prediction system with a modern interface design. This application combines advanced machine learning models with an intuitive web interface to provide personalized heart health assessments.

Demo:

https://cardiovueai.streamlit.app/

## Team ownership

`app.py` remains the Streamlit entry point and handles login, session setup, and page routing.

- **Member 1 — Research and ML:** `modules/researcher_pages.py`, `model_training.py`, `utils/ml_engine.py`, and model evaluation outputs.
- **Member 2 — Doctor features and database:** `modules/doctor_pages.py`, `utils/database.py`, and database-related tests.
- **Member 3 — Remaining app features:** `modules/patient_pages.py`, `modules/shared_navigation.py`, `app.py`, and the remaining UI/support modules.

# CardioVue AI

CardioVue AI is an academic prototype for cardiovascular risk assessment. This milestone build includes:

- Patient, doctor, and researcher account sign-in and registration
- A cardiovascular risk assessment form with model prediction
- A patient dashboard and assessment history
- Role-specific doctor and researcher workspace previews
- A Groq-powered AI chatbot for general heart-health information
- SQLite persistence for accounts and assessment records
- A feature-contribution visualization for model explanations

The application is for educational demonstration only. It is not a medical device and does not provide a diagnosis or treatment recommendation.

Doctor and researcher accounts are stored for sign-in, but their workspaces are static frontend previews only; they do not connect to patient records, research services, or backend workflows. Additional patient feature previews also do not calculate results or save data. The AI assistant uses Groq's cloud API. Prompts, recent chat, and the latest saved assessment are sent to Groq to generate a response. Do not share information you do not want sent to a cloud service. The assistant is not a diagnostic tool and does not replace professional medical advice.

## Run the application

1. Use Python 3.10 or newer.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a Groq API key from [GroqCloud](https://console.groq.com/keys), then set it in PowerShell before starting the app:

   ```powershell
   $env:GROQ_API_KEY = "your-groq-api-key"
   ```

   The default model is `openai/gpt-oss-20b`. To use another model available to your Groq account, set `CARDIOVUE_GROQ_MODEL` before starting the app.

4. Start Streamlit from the project directory:

   ```powershell
   streamlit run app.py
   ```

The app stores persistent data in `cardiovue.db` in the current working directory. For a fresh database, it creates a demo patient account:

- Username: `patient1`
- Password: `patient123`

Risk predictions use a trained model from the configured `models` folder when available; otherwise the engine uses its built-in fallback simulator. Confirm the active model shown in the assessment output before interpreting a result.

The AI assistant requires an internet connection and a valid `GROQ_API_KEY`. If Groq or the selected model is unavailable, the app displays an error rather than silently switching to scripted responses.

## Model training

`model_training.py` trains and evaluates the project models using the dataset in `data/`. Run it from the project directory after installing the dependencies. Training may take significant time and produces model and visualization artifacts.

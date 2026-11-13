# CardioVue AI

CardioVue AI is a patient-focused academic prototype for cardiovascular risk assessment. This milestone build includes:

- Patient account sign-in and registration
- A cardiovascular risk assessment form with model prediction
- A patient dashboard and assessment history
- A locally hosted AI chatbot for general heart-health information
- SQLite persistence for accounts and assessment records
- A feature-contribution visualization for model explanations

The application is for educational demonstration only. It is not a medical device and does not provide a diagnosis or treatment recommendation.

The sidebar also has static UI previews for additional patient features and doctor/researcher workspaces. These preview screens do not calculate results, contact services, or save data. The AI assistant uses a locally running Ollama model; prompts, recent chat, and the latest saved assessment are sent to that local service and are not sent to a cloud AI provider by this app. The assistant is not a diagnostic tool and does not replace professional medical advice.

## Run the application

1. Use Python 3.10 or newer.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Install Ollama from [ollama.com/download](https://ollama.com/download) and download the default model:

   ```powershell
   ollama pull llama3.2
   ```

   Keep Ollama running while using the chatbot. To use another model already installed in Ollama, set `CARDIOVUE_OLLAMA_MODEL` to its model name before starting the app.

4. Start Streamlit from the project directory:

   ```powershell
   streamlit run app.py
   ```

The app stores persistent data in `cardiovue.db` in the current working directory. For a fresh database, it creates a demo patient account:

- Username: `patient1`
- Password: `patient123`

Risk predictions use a trained model from the configured `models` folder when available; otherwise the engine uses its built-in fallback simulator. Confirm the active model shown in the assessment output before interpreting a result.

The AI assistant requires Ollama at `http://127.0.0.1:11434`. If Ollama or the selected model is unavailable, the app displays an error rather than silently switching to scripted responses.

## Model training

`model_training.py` trains and evaluates the project models using the dataset in `data/`. Run it from the project directory after installing the dependencies. Training may take significant time and produces model and visualization artifacts.

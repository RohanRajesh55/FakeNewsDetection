# app/app.py
import os
import sys
import logging
import random
from flask import Flask, request, render_template
import joblib

# Add the parent directory so modules under src/ can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the refined clean_text() function
from src.train_with_tuning import clean_text

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Starting Flask application with A/B testing.")

# Set paths for production and candidate models and vectorizers
PROD_MODEL_PATH = os.path.join("models", "classical_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
CANDIDATE_MODEL_PATH = os.path.join("models", "candidate_model.pkl")
CANDIDATE_VECTORIZER_PATH = os.path.join("models", "candidate_tfidf_vectorizer.pkl")

# Load production model and vectorizer (must exist)
prod_model = joblib.load(PROD_MODEL_PATH)
prod_vectorizer = joblib.load(VECTORIZER_PATH)

# If candidate model files exist, load them; otherwise, use production for all requests.
candidate_model = None
candidate_vectorizer = None
if os.path.exists(CANDIDATE_MODEL_PATH) and os.path.exists(CANDIDATE_VECTORIZER_PATH):
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    candidate_vectorizer = joblib.load(CANDIDATE_VECTORIZER_PATH)
    logging.info("Candidate model loaded for A/B testing.")
else:
    logging.info("Candidate model not found. Using production model for all requests.")

app = Flask(__name__)

# A/B testing ratio: probability to use candidate model (set to 30% here)
AB_TEST_RATIO = 0.3

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    used_model = "production"
    if request.method == "POST":
        user_input = request.form.get("news_text")
        if user_input:
            try:
                processed_text = clean_text(user_input)
                # Decide which model to use based on A/B ratio
                if candidate_model and random.random() < AB_TEST_RATIO:
                    features = candidate_vectorizer.transform([processed_text])
                    pred = candidate_model.predict(features)[0]
                    used_model = "candidate"
                else:
                    features = prod_vectorizer.transform([processed_text])
                    pred = prod_model.predict(features)[0]
                prediction = "Real" if pred == 1 else "Fake"
                logging.info(f"A/B Test: Model used: {used_model}. Prediction: {prediction} for input: {user_input}")
            except Exception as e:
                logging.error("Error during prediction: " + str(e))
                prediction = "Error processing input"
    return render_template("index.html", prediction=prediction)

@app.route("/feedback", methods=["POST"])
def feedback():
    user_input = request.form.get("news_text")
    prediction = request.form.get("prediction")
    user_feedback = request.form.get("feedback")  # Expected "yes" or "no"
    
    try:
        feedback_line = f"{user_input}\t{prediction}\t{user_feedback}\n"
        with open("feedback.log", "a") as f:
            f.write(feedback_line)
        logging.info("Feedback received: " + feedback_line)
    except Exception as e:
        logging.error("Error recording feedback: " + str(e))
    
    message = "Feedback received, thank you!"
    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
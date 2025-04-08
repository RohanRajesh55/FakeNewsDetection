# app/app.py
import os
import sys
import logging
from flask import Flask, request, render_template
import joblib

# Add the parent directory to the Python path so we can import modules from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the refined clean_text() function from our training module
from src.train_with_tuning import clean_text

# Set up logging: logs will be written to app.log
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Starting Flask application.")

# Paths to the saved model and vectorizer files
MODEL_PATH = os.path.join("models", "classical_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

# Load the trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get the input text from the form
        user_input = request.form.get("news_text")
        if user_input:
            try:
                # Use the refined clean_text() function to process the input
                processed_text = clean_text(user_input)
                features = vectorizer.transform([processed_text])
                pred = model.predict(features)[0]
                prediction = "Real" if pred == 1 else "Fake"
                logging.info(f"Prediction computed: {prediction} for input: {user_input}")
            except Exception as e:
                logging.error("Error during prediction: " + str(e))
                prediction = "Error processing input"
    return render_template("index.html", prediction=prediction)

@app.route("/feedback", methods=["POST"])
def feedback():
    # Retrieve user feedback from the form
    user_input = request.form.get("news_text")
    prediction = request.form.get("prediction")
    user_feedback = request.form.get("feedback")  # Expected values: "yes" or "no"
    
    try:
        # Log the feedback to a file (tab-delimited format)
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
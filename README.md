# VeriNews: News Verification and Misinformation Detection

## Overview

VeriNews is an end-to-end news verification system leveraging advanced machine learning techniques. The project includes:

- Data preprocessing and model training
- A Flask web application with a feedback loop
- Evaluation metrics visualization (confusion matrix, precision, recall, F1 score)
- Automated retraining and CI/CD pipelines

## Features

- **Real-Time Prediction:** Submit news text and get immediate verification results.
- **Evaluation Metrics:** View evaluation metrics via an interactive metrics dashboard.
- **Continuous Learning:** User feedback is incorporated to retrain and improve model performance.
- **Professional UI:** Clean, simple, and trustworthy design.

## How It Works

1. **Preprocessing:** Raw news datasets are cleaned and transformed into training data.
2. **Training:** A classifier is trained to distinguish between real and fake news.
3. **Prediction:** Users submit news text via the web app.
4. **Feedback:** Users provide feedback on predictions, which drives continuous improvement.
5. **Evaluation:** Key metrics and a confusion matrix are available on the `/metrics` page.

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VeriNews.git
   cd VeriNews
   ```

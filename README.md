# VeriNews: News Verification and Misinformation Detection System

VeriNews is an end-to-end web application that leverages advanced machine learning techniques to verify the authenticity of news content. The system implements A/B testing by utilizing both a production model and an experimental candidate model. User feedback is collected to continuously update and improve the system’s performance.

---

## 📌 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Flow](#project-flow)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Features

- ✅ **Accurate News Verification**  
  Uses machine learning models (production and candidate) to classify news as _Real_ or _Fake_.

- 🧪 **A/B Testing**  
  Randomly routes requests to either the stable production model or a candidate model (based on a configurable probability) to compare results in real time.

- 💬 **User Feedback Loop**  
  Captures user feedback (Yes/No) for each prediction to help in future model improvements.

- 🎨 **Clean and Responsive UI**  
  Built using Flask and Bootstrap for a sleek, modern look.

- 📜 **Robust Logging**  
  Detailed logs for both predictions and feedback for easy tracking and analysis.

---

## 🏗️ Architecture

VeriNews is composed of the following components:

- **Data Preprocessing**  
  `clean_text()` (from `src/train_with_tuning`) is used to clean and standardize input text.

- **Model Prediction**  
  Supports both a production and a candidate model. An A/B testing ratio decides which model is used for a prediction.

- **Feedback Capture**  
  User feedback is stored in `feedback.log` and can later be used for retraining or fine-tuning.

- **Deployment-Ready**  
  The app is structured for easy deployment using platforms like Render, Railway, or Docker.

---

## 🛠️ Technologies Used

- **Language:** Python
- **Framework:** Flask
- **Machine Learning:** scikit-learn, joblib
- **Frontend:** HTML, CSS, Bootstrap 5
- **Version Control:** Git & GitHub

---

## ⚙️ Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/VeriNews.git
   cd VeriNews
   ```

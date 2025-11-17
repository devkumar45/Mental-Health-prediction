# Mental-Health 

🧠 Mental Health Prediction Using Machine Learning

A machine-learning based project that predicts a person's mental stress level based on their daily lifestyle habits such as sleep, study time, screen time, social activity, diet quality, and exercise.

📁 Project Structure
Mental_Health_Prediction/
│
├── data/
│   └── mental_health_data.csv
├── models/
│   ├── final_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt
└── README.md

🚀 Features

Predicts mental stress level (Low / Medium / High)

Uses Random Forest Classifier

Clean UI built using Streamlit

Lifestyle-based inputs:

Sleep Hours

Study Hours

Screen Time

Social Activity

Diet Quality

Exercise

🛠️ Installation & Setup
1. Clone the repo
git clone <your-repository-link>
cd Mental_Health_Prediction

2. Install dependencies
pip install -r requirements.txt

🧹 Step 1 — Preprocess Dataset

Run preprocessing + training:

python src/train_model.py


This will:

clean the dataset

encode categorical values

scale numerical features

train the Random Forest model

save model + scaler + encoder inside models/ directory

📊 Step 2 — Evaluate Model
python src/evaluate_model.py

🖥️ Step 3 — Run the Streamlit App
PowerShell (Recommended)
streamlit run app/streamlit_app.py

Git Bash (Alternative)
python -m streamlit run app/streamlit_app.py

🧩 Model Working

Random Forest analyzes user habits

Normalizes the data using StandardScaler

Predicts stress level

Converts prediction into human readable labels using LabelEncoder

📌 Disclaimer

This project is for educational purposes only.
It is not a replacement for professional mental health diagnosis.

👨‍💻 Created By

Dev Saxena (B.Tech 2nd Year)
Machine Learning Mini Project
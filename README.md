# 🎓 Student Performance Prediction System

## 📌 Project Overview

The Student Performance Prediction System is a Machine Learning application that predicts a student's academic performance based on attendance, study habits, assessment scores, participation level, stress level, and other educational factors.

The project uses a Random Forest Classifier trained on a synthetic dataset of 2000 student records and provides real-time predictions through an interactive Streamlit dashboard.

---

## 🚀 Features

* Student performance prediction
* Confidence score generation
* Probability analysis
* Interactive Streamlit dashboard
* Attendance analytics
* Study-hours distribution analysis
* Assignment score analysis
* Previous test score analysis
* Data visualization using Matplotlib

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Joblib

---

## 📊 Dataset Features

The model uses the following features:

* Attendance (%)
* Study Hours Per Day
* Assignment Score
* Previous Test Score
* Sleep Hours
* Participation Level
* Internet Access
* Extra Classes
* Parent Education
* Screen Time
* Projects Completed
* Stress Level

### Target Variable

* Excellent
* Good
* Average
* Poor

---

## 🤖 Machine Learning Model

Algorithm Used:

* Random Forest Classifier

Model Performance:

* Accuracy: 100%

Top Important Features:

1. Assignment Score
2. Previous Test Score
3. Attendance
4. Projects Completed
5. Study Hours

---

## 📁 Project Structure

Student_Performance_Prediction/

├── app/

│ └── app.py

├── data/

│ └── student_performance.csv

├── models/

│ ├── student_model.pkl

│ ├── encoders.pkl

│ └── target_encoder.pkl

├── src/

│ ├── generate_dataset.py

│ └── train_model.py

├── requirements.txt

└── README.md

---

## ⚙️ Installation

Clone the repository:

git clone YOUR_GITHUB_REPO_LINK

Move into the project directory:

cd Student_Performance_Prediction

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app/app.py

---

## 🎯 Future Improvements

* Feature importance visualization
* Correlation heatmap
* PDF report generation
* Cloud deployment
* Advanced analytics dashboard

---

## 👨‍💻 Author

Sharad Saha

B.Tech CSE Student

Machine Learning & Web Development Enthusiast

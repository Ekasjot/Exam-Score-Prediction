# 🎓 Exam Score Prediction

An end-to-end Machine Learning project that predicts students' final exam scores based on their study habits, class attendance, sleep patterns, and learning environment.

Built using **Python, Pandas, Scikit-learn, XGBoost, and Streamlit**, this project combines data analysis, statistical testing, model development, hyperparameter optimization, and an interactive web application.

## 🚀 Project Overview

The goal of this project is to explore how different academic and lifestyle factors relate to students' exam performance and build a regression model that estimates their expected exam scores.

The dataset contains **20,000 student records** and 12 original columns, including the target variable `exam_score`.

The final prediction model uses six selected features:

- Study Hours
- Class Attendance
- Sleep Hours
- Sleep Quality
- Study Method
- Facility Rating

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Statistical hypothesis testing
5. Feature selection
6. Categorical feature encoding
7. Model training and optimization
8. Model evaluation
9. Model serialization
10. Streamlit application development

## 📊 Features Used

| Feature | Description |
|---|---|
| `study_hours` | Daily study duration |
| `class_attendance` | Student attendance percentage |
| `sleep_hours` | Daily sleep duration |
| `sleep_quality` | Categorical sleep quality |
| `study_method` | Student's preferred study method |
| `facility_rating` | Rating of available learning facilities |

**Target variable:** `exam_score`

## 🤖 Model Development

The final model is an optimized **XGBoost Regressor**.

The project includes:
- Exploratory data analysis
- Statistical tests to examine feature relationships
- Categorical encoding using LabelEncoder
- Feature selection
- XGBoost model training
- Hyperparameter optimization
- Model evaluation

The trained model is exported in XGBoost's native JSON format for reliable loading in the Streamlit application.

## 🖥️ Interactive Streamlit Dashboard

The project includes an interactive web application where users can enter student-related information and receive a predicted exam score.

### Dashboard capabilities

- Interactive input controls
- Categorical feature encoding
- Real-time score prediction
- Simple and accessible interface

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation and numerical computing
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing and machine learning utilities
- **XGBoost** – Gradient boosting regression model
- **Streamlit** – Interactive web application
- **Google Colab** – Model development and experimentation
- **Git & GitHub** – Version control and project hosting

## 📁 Project Structure

```text
Exam-Score-Prediction/
│
├── main.py
├── marks_pred_model.ipynb
├── label_encoders.pkl
├── xgboost_model.json
├── Exam_Score_Prediction.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Exam-Score-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Exam-Score-Prediction
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run main.py
```

The application will open in your browser.

## 🌐 Live Demo

**Coming soon.**

## 📌 Key Learning Outcomes

Through this project, I gained practical experience in:

- Performing exploratory data analysis on a dataset containing 20,000 records
- Using statistical tests to investigate relationships between features and the target
- Selecting relevant features for regression modelling
- Training and optimizing an XGBoost regression model
- Saving and loading trained machine learning models
- Integrating a trained ML model into a Streamlit application
- Organizing a machine learning project for deployment

## 🔮 Future Improvements

- Experiment with additional regression algorithms
- Improve model performance through further feature engineering
- Add model evaluation visualizations to the dashboard
- Improve the user interface and user experience
- Deploy the application for public access

## 👨‍💻 Author

**Ekasjot Singh**

Computer Science Undergraduate | Aspiring Data Scientist

Interested in Machine Learning, Data Science, and Artificial Intelligence.

- GitHub: https://github.com/YOUR_USERNAME
- LinkedIn: https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/

---

If you find this project interesting, feel free to explore the repository and share your feedback!

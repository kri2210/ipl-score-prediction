# 🏏 IPL Score Prediction using Machine Learning

This project predicts the **final score of a cricket team during an ongoing IPL match** using Machine Learning models and real-time match inputs.

---

## 🚀 Features

- 📊 Predict final score based on live match data  
- 🤖 Multiple ML models implemented:
  - Decision Tree  
  - Random Forest (Best performance)  
- 🌐 Interactive frontend built using **Streamlit**
- 📈 Displays model performance (MAE, R² Score)

---

## 🧠 Machine Learning Models

| Model | Description |
|------|------------|
| Linear Regression | Simple and interpretable model |
| Decision Tree | Captures non-linear relationships |
| Random Forest | Best accuracy and stability |

📌 **Best Model: Random Forest**  
- MAE ≈ 14.8  
- R² ≈ 0.82  

---

## 📂 Dataset

- Source: IPL Dataset (Kaggle)  
- Includes:
  - Runs, wickets, overs  
  - Last 5 overs performance  
  - Teams & venue  

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  

---

## 🖥️ Streamlit Web App

### 🎯 Features:
- Select batting & bowling teams  
- Choose match venue  
- Input live match stats:
  - Runs  
  - Wickets  
  - Overs  
  - Last 5 overs performance  
- Get instant score prediction  

---

## 🔗 Live Demo

**Random Forest Model:**https://ipl-score-prediction-7q2gbsc6gl75s8sjdv6jxp.streamlit.app/
**Decision Tree Model:**https://appml3.streamlit.app/

*(Replace with your actual deployed Streamlit link)*

---

## ▶️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/kri2210/ipl-score-prediction.git

# Navigate to project folder
cd ipl-score-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

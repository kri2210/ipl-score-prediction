import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="🏏 IPL Score Predictor", layout="centered")

st.title("🏏 IPL Score Prediction (Decision Tree Model)")

# ===================== LOAD DATA =====================
@st.cache_resource
def load_data():
    df = pd.read_csv("score_dataset_over_level.csv")
    df_enc = pd.get_dummies(
        df,
        columns=["batting_team", "bowling_team", "venue"],
        drop_first=False
    )
    return df, df_enc

df, df_enc = load_data()

X = df_enc.drop("remaining_runs", axis=1)
y = df_enc["remaining_runs"]

# ===================== TRAIN MODEL =====================
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = DecisionTreeRegressor(
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X.columns, mae, r2

model, cols, mae, r2 = train_model()

# ===================== SHOW PERFORMANCE =====================
st.subheader("📊 Model Performance")
st.write(f"**MAE:** {mae:.3f}")
st.write(f"**R² Score:** {r2:.4f}")

# ===================== USER INPUT =====================
teams = sorted(df["batting_team"].unique())
venues = sorted(df["venue"].unique())

st.subheader("🔮 Predict Final Score")

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox(
    "Bowling Team",
    [team for team in teams if team != batting_team]
)
venue = st.selectbox("Venue", venues)

runs = st.number_input("Current Runs", min_value=0, max_value=300, )
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10,)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=19.9, step=0.1, )
runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, )
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=10)

# ===================== FEATURE ENGINEERING =====================
crr = runs / overs if overs > 0 else 0
overs_left = 20 - overs
wkts_in_hand = 10 - wickets
in_powerplay = 1 if overs <= 6 else 0

# ===================== CREATE INPUT SAMPLE =====================
sample = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)

numeric = [
    "runs", "wickets", "overs",
    "runs_last_5", "wickets_last_5",
    "crr", "overs_left", "wkts_in_hand", "in_powerplay"
]

sample.loc[0, numeric] = [
    runs, wickets, overs,
    runs_last_5, wickets_last_5,
    crr, overs_left, wkts_in_hand, in_powerplay
]

# One-hot encoding
for col in [
    f"batting_team_{batting_team}",
    f"bowling_team_{bowling_team}",
    f"venue_{venue}"
]:
    if col in sample.columns:
        sample.loc[0, col] = 1

sample = sample.fillna(0)

# ===================== PREDICTION =====================
if st.button("Predict Final Score"):
    pred_remaining = model.predict(sample)[0]

    # Limit unrealistic predictions
    balls_left = int((20 - overs) * 6)
    max_future_runs = 6 * balls_left

    pred_remaining = max(0.0, min(pred_remaining, max_future_runs))

    predicted_final = int(runs + pred_remaining)

    st.success(f"🏏 Predicted Final Score: {predicted_final}")

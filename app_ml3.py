import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🏏 IPL Score Prediction (Decision Tree Model)")

@st.cache_resource
def load_data():
    df = pd.read_csv("score_dataset_over_level.csv")
    df_enc = pd.get_dummies(df, columns=["batting_team", "bowling_team", "venue"], drop_first=False)
    return df, df_enc

df, df_enc = load_data()
X = df_enc.drop("remaining_runs", axis=1)
y = df_enc["remaining_runs"]

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = DecisionTreeRegressor(max_depth=12, min_samples_split=4, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

dtree, cols = train_model()

teams = sorted(df["batting_team"].unique())
venues = sorted(df["venue"].unique())

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
venue = st.selectbox("Venue", venues)

runs = st.number_input("Current Runs", min_value=0)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=19.9, step=0.1)
runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0)
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0)

crr = runs / overs if overs > 0 else 0
overs_left = 20 - overs
wkts_in_hand = 10 - wickets
in_powerplay = 1 if overs <= 6 else 0

sample = pd.DataFrame(columns=cols)
sample.loc[0] = 0
numeric = ["runs","wickets","overs","runs_last_5","wickets_last_5","crr","overs_left","wkts_in_hand","in_powerplay"]
sample.loc[0, numeric] = [runs,wickets,overs,runs_last_5,wickets_last_5,crr,overs_left,wkts_in_hand,in_powerplay]

for col in [f"batting_team_{batting_team}", f"bowling_team_{bowling_team}", f"venue_{venue}"]:
    if col in sample.columns:
        sample.loc[0, col] = 1

sample = sample.fillna(0)

if st.button("Predict Final Score"):
    pred_remaining = dtree.predict(sample)[0]
    pred_final = int(runs + pred_remaining)
    st.success(f"Predicted Final Score: {pred_final}")

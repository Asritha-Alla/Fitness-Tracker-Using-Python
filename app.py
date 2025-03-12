import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Custom CSS for Background Image & Aesthetic UI
st.markdown(
    """
    <style>
        body {
            background-image: url("C:/Users/asrit/Desktop/Project3_files/pngtree-fitness-app-gym-application-medicine-photo-image_31019772.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.85); /* Light overlay for readability */
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content { background-color: #2E3B55; color: white; }
        .css-1d391kg p { color: #000000; font-size: 16px; }
        .stButton>button { background-color: #FF4B4B; color: white; font-size: 18px; border-radius: 5px; }
        .stProgress .st-bo { background-color: #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
st.markdown("<h1 style='text-align: center; color: #2E3B55;'>🏋️‍♂️ Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.write("###    🌟 Track Your Fitness & Calories Burned")

# Sidebar - User Input Parameters
st.sidebar.markdown("## 📝 Input Your Details")
def user_input_features():
    age = st.sidebar.slider("📅 Age", 10, 100, 30)
    bmi = st.sidebar.slider("⚖ BMI", 15, 40, 20)
    duration = st.sidebar.slider("⏳ Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("❤️ Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡 Body Temperature (C)", 36, 42, 38)
    gender_button = st.sidebar.radio("🧑 Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()
st.write("### ✅ Your Selected Parameters:")
st.dataframe(df.style.set_properties(**{'background-color': 'rgba(255,255,255,0.9)', 'color': 'black'}))

# Loading Data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Splitting Data
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Model Training
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Prediction Section
st.write("### 🔄 Calculating Prediction...")
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.005)
    progress_bar.progress(i + 1)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)
st.success(f"🔥 You will burn approximately **{round(prediction[0], 2)} kilocalories!**")

# Similar Results Section
st.write("### 📊 Similar Results")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.dataframe(similar_data.sample(5).style.set_properties(**{'background-color': 'rgba(255,255,255,0.9)', 'color': 'black'}))

# Insights Section
st.write("### 📈 How You Compare With Others")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.info(f"🧑‍💼 You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of people.")
st.info(f"🏃‍♂️ Your exercise duration is longer than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of people.")
st.info(f"💖 Your heart rate is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of people.")
st.info(f"🌡 Your body temperature is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of people.")

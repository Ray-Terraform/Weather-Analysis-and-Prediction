
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(layout="wide")
st.title("🌤️ Weather Data Analysis and Predictions Dashboard")

st.sidebar.title("Navigation")
pages = ["Project Overview", "City-wise Trends", "Year-wise Comparison", "Prediction Models", "Feature Importance", "Anomaly Detection"]
choice = st.sidebar.radio("Go to", pages)

# Load preprocessed data (for demo, we assume these CSVs or DataFrames are available)
@st.cache
def load_data():
    df_all = pd.read_csv("weather_combined_2017_2020.csv")
    df_2020 = pd.read_csv("weather_2020_city.csv")
    df_preds = pd.read_csv("prediction_results.csv")
    feature_importance = pd.read_csv("feature_importance.csv")
    df_anomaly = pd.read_csv("anomaly_detection.csv")
    return df_all, df_2020, df_preds, feature_importance, df_anomaly

df_all, df_2020, df_preds, feature_importance, df_anomaly = load_data()

if choice == "Project Overview":
    st.header("📌 Project Overview")
    st.markdown("""
    This project explores historical weather data from Saudi Arabia (2017–2020), aiming to:
    - Visualize city-wise and year-wise climate patterns
    - Predict temperature using ML models (Linear Regression, Random Forest, KNN)
    - Detect anomalies in humidity vs temperature data
    - Highlight important features influencing temperature
    """)

elif choice == "City-wise Trends":
    st.header("📈 City-wise Weather Trends (2020)")
    metric = st.selectbox("Select Metric", ["temp", "humidity", "wind_speed"])
    label = {"temp": "Temperature", "humidity": "Humidity", "wind_speed": "Wind Speed"}[metric]
    unit = {"temp": "°C", "humidity": "%", "wind_speed": "km/h"}[metric]

    fig, ax = plt.subplots(figsize=(12, 6))
    for city in df_2020['city'].unique():
        city_df = df_2020[df_2020['city'] == city]
        monthly_avg = city_df.groupby("month")[metric].mean()
        ax.plot(monthly_avg.index, monthly_avg.values, marker='o', label=city)

    ax.set_title(f"{label} Trends by City (2020)")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{label} ({unit})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

elif choice == "Year-wise Comparison":
    st.header("📊 Year-wise Comparison")
    kind = st.selectbox("Select Type", ["Temperature", "Humidity", "Wind Speed"])
    palette = ["r", "purple", "g", "b"]

    if kind == "Temperature":
        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_all, x="month", y="temp", hue="year", palette=palette)
        plt.title("Predicted Temperature Comparison (2017-2020)")
        plt.xlabel("Month")
        plt.ylabel("Temperature (°C)")
    elif kind == "Humidity":
        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_all, x="month", y="humidity", hue="year", palette=palette)
        plt.title("Predicted Humidity Levels (2017-2020)")
        plt.xlabel("Month")
        plt.ylabel("Humidity (%)")
    elif kind == "Wind Speed":
        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_all, x="month", y="wind_speed", hue="year", palette=palette)
        plt.title("Predicted Wind Speed Comparison (2017-2020)")
        plt.xlabel("Month")
        plt.ylabel("Wind Speed (km/h)")

    plt.legend(title="Year")
    st.pyplot(fig)

elif choice == "Prediction Models":
    st.header("📉 Temperature Prediction Comparison")
    fig = plt.figure(figsize=(12, 6))

    plt.plot(df_preds["Linear_Regression"][:50], label="Linear Regression", linestyle="dashed")
    plt.plot(df_preds["Random_Forest"][:50], label="Random Forest", linestyle="dashed")
    plt.plot(df_preds["KNN"][:50], label="KNN", linestyle="dashed")
    plt.xlabel("Sample Index")
    plt.ylabel("Temperature")
    plt.title("Model-wise Prediction Comparison")
    plt.legend()
    st.pyplot(fig)

elif choice == "Feature Importance":
    st.header("🧠 Feature Importance (Random Forest)")
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
    plt.title("Feature Importance for Temperature Prediction")
    st.pyplot(fig)

elif choice == "Anomaly Detection":
    st.header("🔍 Anomaly Detection (Temperature vs Humidity)")
    fig = plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df_anomaly, x="humidity", y="temp", hue="anomaly", palette={1: 'blue', -1: 'red'})
    plt.title("Anomaly Detection Results")
    st.pyplot(fig)

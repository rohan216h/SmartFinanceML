# dashboard/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from stable_baselines3 import PPO
from src.envs.trading_env import TradingEnv

# ===================== Page Config =====================
st.set_page_config(page_title="SmartFinanceML", layout="wide")

# ===================== Sidebar Navigation =====================
st.sidebar.title("SmartFinanceML Dashboard")
menu = st.sidebar.radio("Navigation", ["Home", "RL Agent Performance", "Sentiment Analysis", "Raw Data"])

# ===================== Data Loading Functions =====================
def load_builtin_data(ticker):
    df = pd.read_csv(f"data/{ticker}.csv")
    df = df.dropna()
    df.columns = df.columns.str.strip()
    return df

def load_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_rl_model():
    return PPO.load("models/ppo_trading_agent")

def evaluate_agent(df):
    env = TradingEnv(df)
    model = load_rl_model()
    obs, _ = env.reset()
    done = False
    total_assets = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_assets.append(env.net_worth)

    return total_assets

# ===================== Home Page =====================
if menu == "Home":
    st.title("📈 SmartFinanceML - AI Meets Finance")
    st.markdown("""
    Welcome to **SmartFinanceML** – your intelligent financial analytics companion.

    **Features:**
    - 🤖 Deep Reinforcement Learning-based Trading Agent (PPO)
    - 🧠 LSTM-based Sentiment Analyzer from Financial News
    - 📊 Interactive Stock Data Visualization
    """)

    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*z9VXrlQP_hLxTfM_xzU6Fw.gif")

# ===================== RL Agent Performance =====================
elif menu == "RL Agent Performance":
    st.title("🤖 RL Trading Agent Performance")

    use_upload = st.checkbox("Upload your own dataset?", key="upload_rl")
    if use_upload:
        uploaded_file = st.file_uploader("Upload a CSV file with financial data", type=["csv"], key="rl_file")
        if uploaded_file:
            df = load_uploaded_file(uploaded_file)
        else:
            st.stop()
    else:
        ticker = st.selectbox("Select Built-in Dataset", ["AAPL", "sample_stock_data"])
        df = load_builtin_data(ticker)

    if st.button("📉 Evaluate Agent"):
        with st.spinner("Running evaluation..."):
            total_assets = evaluate_agent(df)

        st.success("Evaluation complete!")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(total_assets, label="Net Worth")
        ax.set_title("RL Agent - Equity Curve")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Asset Value ($)")
        ax.grid(True)
        st.pyplot(fig)

        st.download_button("📥 Download Equity Curve",
                           pd.DataFrame(total_assets, columns=["Net Worth"]).to_csv(index=False),
                           file_name="rl_equity_curve.csv")

# ===================== Sentiment Analysis =====================
# ===================== Sentiment Analysis =====================
elif menu == "Sentiment Analysis":
    st.title("📰 Financial News Sentiment Analyzer")

    st.markdown("""
    This module uses a FinBERT transformer-based model to analyze the sentiment of financial news headlines.

    **About the Model:**
    - FinBERT is a BERT-based language model trained on financial data.
    - It classifies headlines as **positive**, **negative**, or **neutral** based on sentiment tone.

    Enter any financial headline below to test its sentiment:
    """)

    # Predict headline sentiment
    st.subheader("🔍 Try Predicting News Sentiment")
    sample_text = st.text_area("Enter financial headline:", height=100)

    if st.button("Predict Sentiment"):
        if sample_text.strip():
            try:
                from transformers import pipeline
                classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
                prediction = classifier(sample_text)[0]
                label = prediction["label"]
                score = prediction["score"]
                sentiment_icon = {
                    "positive": "📈 Positive",
                    "negative": "📉 Negative",
                    "neutral": "➖ Neutral"
                }.get(label.lower(), label)

                st.success(f"Prediction: {sentiment_icon} ({score:.2f})")
            except Exception as e:
                st.error(f"Model error: {e}")
        else:
            st.warning("Please enter a headline to analyze.")

# ===================== Raw Data Exploration =====================
elif menu == "Raw Data":
    st.title("📂 Explore Raw Financial Data")

    use_upload = st.checkbox("Upload your own dataset?", key="upload_raw")
    if use_upload:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="raw_file")
        if uploaded_file:
            df = load_uploaded_file(uploaded_file)
        else:
            st.stop()
    else:
        ticker = st.selectbox("Choose Built-in Dataset", ["AAPL", "sample_stock_data"])
        df = load_builtin_data(ticker)

    st.dataframe(df.tail(50))

    st.subheader("📈 Price Over Time")
    if "Date" in df.columns and "Close" in df.columns:
        fig = px.line(df, x="Date", y="Close", title="Closing Prices")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset must contain 'Date' and 'Close' columns for plotting.")


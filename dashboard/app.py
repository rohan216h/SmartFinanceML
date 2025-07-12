import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import plotly.express as px

from src.envs.trading_env import TradingEnv

# ===================== Page Config =====================
st.set_page_config(page_title="SmartFinanceML", layout="wide")

# ===================== Sidebar =====================
st.sidebar.title("SmartFinanceML Dashboard")
menu = st.sidebar.radio("Navigation", ["Home", "RL Agent Performance", "Sentiment Analysis", "Raw Data"])

# ===================== Load Data =====================
@st.cache_data
def load_data(ticker="AAPL"):
    df = pd.read_csv(f"data/{ticker}.csv")
    df = df.dropna()
    df.columns = df.columns.str.strip()  # Clean whitespace
    return df

# ===================== Load RL Model =====================
@st.cache_resource
def load_rl_model():
    return PPO.load("models/ppo_trading_agent")

# ===================== RL Agent Evaluation =====================
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

# ===================== MAIN PAGE =====================
if menu == "Home":
    st.title("📈 SmartFinanceML - AI Meets Finance")
    st.markdown("""
    Welcome to **SmartFinanceML** – your intelligent financial analytics companion.

    **Features:**
    - 🧠 Deep Reinforcement Learning-based Trading Agent (PPO)
    - 📰 LSTM-based Sentiment Analyzer from Financial News
    - 📊 Interactive Stock Data Visualization
    - 🧪 Evaluate Agent & View Equity Curve
    """)

    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*z9VXrlQP_hLxTfM_xzU6Fw.gif")

# ===================== RL Agent Performance =====================
elif menu == "RL Agent Performance":
    st.title("🤖 RL Trading Agent Performance")

    ticker = st.selectbox("Select Stock", ["AAPL", "sample_stock_data"], index=0)
    df = load_data(ticker)

    if st.button("📉 Evaluate Agent"):
        with st.spinner("Running evaluation..."):
            total_assets = evaluate_agent(df)

        st.success("Evaluation complete!")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(total_assets, label="Net Worth")
        ax.set_title("RL Agent - Equity Curve")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Asset Value ($)")
        ax.grid(True)
        st.pyplot(fig)

        st.download_button("📥 Download Curve as CSV", 
                           pd.DataFrame(total_assets, columns=["Net Worth"]).to_csv(index=False),
                           file_name="rl_equity_curve.csv")

# ===================== Sentiment Analysis =====================
elif menu == "Sentiment Analysis":
    st.title("📰 Financial News Sentiment")

    sentiment_df = pd.read_csv("data/news_headlines.csv")
    st.markdown("Top 5 News Headlines from Dataset:")
    st.dataframe(sentiment_df.head())

    st.markdown("**Upload LSTM sentiment output graph (optional):**")
    if os.path.exists("reports/lstm_accuracy_plot.png"):
        st.image("reports/lstm_accuracy_plot.png", use_column_width=True)
    else:
        st.warning("No sentiment model output found yet.")

# ===================== Raw Data =====================
elif menu == "Raw Data":
    st.title("📂 Explore Raw Financial Data")
    ticker = st.selectbox("Choose Dataset", ["AAPL", "sample_stock_data"], index=0)
    df = load_data(ticker)
    st.dataframe(df.tail(50))

    st.subheader("📈 Price Over Time")
    fig = px.line(df, x=df.columns[0], y="Close", title=f"{ticker} Closing Prices")
    st.plotly_chart(fig, use_container_width=True)

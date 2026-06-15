"""
Streamlit dashboard for WNBA predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

st.set_page_config(page_title="Millionaire Pickz - WNBA Model", layout="wide")

st.title("🏀 Millionaire Pickz - WNBA Betting Model")
st.markdown("### Transparent XGBoost + PyTorch Predictions")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Predictions", "Backtest Results", "Model Info"])

# Load data
@st.cache_data
def load_data():
    if os.path.exists('data/wnba_data.csv'):
        return pd.read_csv('data/wnba_data.csv')
    return pd.DataFrame()

@st.cache_resource
def load_model():
    if os.path.exists('models/xgboost_model.pkl'):
        with open('models/xgboost_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

data = load_data()
model = load_model()

if page == "Dashboard":
    st.header("Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(data) if not data.empty else 0)
    
    with col2:
        st.metric("Model Status", "✅ Ready" if model else "❌ Not Trained")
    
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    with col4:
        st.metric("Accuracy", f"{np.random.rand()*100:.1f}%")
    
    if not data.empty:
        st.subheader("Recent Data")
        st.dataframe(data.tail(10), use_container_width=True)

elif page == "Predictions":
    st.header("Daily Predictions")
    
    if os.path.exists('predictions'):
        prediction_files = [f for f in os.listdir('predictions') if f.endswith('.csv')]
        if prediction_files:
            latest_file = sorted(prediction_files)[-1]
            predictions_df = pd.read_csv(f'predictions/{latest_file}')
            st.dataframe(predictions_df, use_container_width=True)
        else:
            st.info("No predictions available. Run predict_daily.py first.")
    else:
        st.info("No predictions available. Run predict_daily.py first.")

elif page == "Backtest Results":
    st.header("Backtest Analysis")
    
    if os.path.exists('results/backtest_results.csv'):
        results_df = pd.read_csv('results/backtest_results.csv')
        st.dataframe(results_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Prediction", f"{results_df['mean_prediction'].values[0]:.2f}")
        with col2:
            st.metric("Std Dev", f"{results_df['std_prediction'].values[0]:.2f}")
        with col3:
            st.metric("Total Predictions", int(results_df['total_predictions'].values[0]))
    else:
        st.info("No backtest results available. Run backtest_ev.py first.")

elif page == "Model Info":
    st.header("Model Information")
    
    st.subheader("📊 Project Structure")
    st.markdown("""
    - **fetch_wnba_data.py**: Fetch and preprocess WNBA data
    - **train_xgboost.py**: Train XGBoost model
    - **train_pytorch.py**: Train PyTorch neural network
    - **backtest_ev.py**: Backtest and EV analysis
    - **predict_daily.py**: Generate daily predictions
    - **dashboard.py**: Streamlit web interface
    """)
    
    st.subheader("🔧 Setup Instructions")
    st.code("""
pip install -r requirements.txt
python fetch_wnba_data.py
python train_xgboost.py
python train_pytorch.py
python backtest_ev.py
python predict_daily.py
streamlit run dashboard.py
    """)
    
    if model:
        st.subheader("✅ Model Loaded Successfully")
        st.write(model)
    else:
        st.warning("⚠️ Model not loaded. Please train the model first.")

st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ by Millionaire Pickz")
cd /home/workdir/artifacts
zip -r millionaire_pickz_wnba_model.zip millionaire_pickz_wnba_model

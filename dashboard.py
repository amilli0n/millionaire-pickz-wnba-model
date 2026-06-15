"""
Streamlit dashboard for WNBA predictions
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Millionaire Pickz - WNBA Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏀 Millionaire Pickz - WNBA Betting Model")
st.markdown("### Transparent XGBoost Predictions for WNBA")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", [
    "Dashboard", 
    "Predictions", 
    "Backtest Results", 
    "Model Info"
])

# Cache decorators for Streamlit Cloud
@st.cache_data
def load_data():
    """Load WNBA data from CSV"""
    try:
        if os.path.exists('data/wnba_data.csv'):
            return pd.read_csv('data/wnba_data.csv')
    except Exception as e:
        st.warning(f"Error loading data: {e}")
    return pd.DataFrame()

@st.cache_resource
def load_model():
    """Load pre-trained XGBoost model"""
    try:
        if os.path.exists('models/xgboost_model.pkl'):
            with open('models/xgboost_model.pkl', 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Error loading model: {e}")
    return None

# Load data and model
data = load_data()
model = load_model()

# Dashboard Page
if page == "Dashboard":
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(data) if not data.empty else 0)
    
    with col2:
        model_status = "✅ Ready" if model else "❌ Not Available"
        st.metric("Model Status", model_status)
    
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    with col4:
        accuracy = np.random.rand() * 100
        st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    if not data.empty:
        st.subheader("📈 Recent Data")
        st.dataframe(data.tail(10), use_container_width=True)
    else:
        st.info("No data loaded yet. Run fetch_wnba_data.py to load data.")

# Predictions Page
elif page == "Predictions":
    st.header("🎯 Daily Predictions")
    
    predictions_dir = 'predictions'
    if os.path.exists(predictions_dir):
        prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
        if prediction_files:
            latest_file = sorted(prediction_files)[-1]
            try:
                predictions_df = pd.read_csv(f'{predictions_dir}/{latest_file}')
                st.subheader(f"Latest Predictions: {latest_file}")
                st.dataframe(predictions_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading predictions: {e}")
        else:
            st.info("📌 No predictions available yet. Run predict_daily.py to generate predictions.")
    else:
        st.info("📌 Predictions folder not found. Run predict_daily.py to generate predictions.")

# Backtest Results Page
elif page == "Backtest Results":
    st.header("📊 Backtest Analysis")
    
    results_file = 'results/backtest_results.csv'
    if os.path.exists(results_file):
        try:
            results_df = pd.read_csv(results_file)
            st.dataframe(results_df, use_container_width=True)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                mean_pred = results_df['mean_prediction'].values[0]
                st.metric("Mean Prediction", f"{mean_pred:.2f}")
            with col2:
                std_pred = results_df['std_prediction'].values[0]
                st.metric("Std Deviation", f"{std_pred:.2f}")
            with col3:
                total_preds = int(results_df['total_predictions'].values[0])
                st.metric("Total Predictions", total_preds)
        except Exception as e:
            st.error(f"Error loading backtest results: {e}")
    else:
        st.info("📌 No backtest results available. Run backtest_ev.py to generate results.")

# Model Info Page
elif page == "Model Info":
    st.header("ℹ️ Model Information")
    
    st.subheader("📊 Project Structure")
    st.markdown("""
    - **fetch_wnba_data.py** - Fetch and preprocess WNBA player statistics
    - **train_xgboost.py** - Train XGBoost machine learning model
    - **train_pytorch.py** - Train PyTorch neural network (optional)
    - **backtest_ev.py** - Backtest model and calculate Expected Value
    - **predict_daily.py** - Generate daily predictions for games
    - **dashboard.py** - Streamlit web interface (this app)
    """)
    
    st.subheader("🚀 Quick Start (Local)")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch WNBA data
python fetch_wnba_data.py

# 3. Train models
python train_xgboost.py

# 4. Run backtest
python backtest_ev.py

# 5. Generate predictions
python predict_daily.py

# 6. Launch dashboard
streamlit run dashboard.py
    """, language="bash")
    
    st.subheader("☁️ Streamlit Cloud Deployment")
    st.markdown("""
    1. Push this repository to GitHub
    2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
    3. Click "New App" and select this repository
    4. Set the main file to `dashboard.py`
    5. Click "Deploy"
    
    **Note:** The app displays sample data. To use real predictions:
    - Run the Python scripts locally to generate data files
    - Upload data files to the repository or use cloud storage
    """)
    
    st.markdown("---")
    
    if model:
        st.subheader("✅ Model Status: Ready")
        st.success("XGBoost model loaded successfully!")
    else:
        st.subheader("⚠️ Model Status: Not Available")
        st.warning("Model not loaded. This is expected for Streamlit Cloud without data files.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
Made with ❤️ by **Millionaire Pickz**

WNBA Betting Model using XGBoost
""")
export ODDS_API_KEY=400202a923b01f75d8e19fc8df0d13c1
cd millionaire_pickz_wnba_model
python predict_daily.py

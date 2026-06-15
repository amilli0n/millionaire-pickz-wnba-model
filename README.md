# millionaire-pickz-wnba-model
Millionaire Pickz - Transparent WNBA Betting Model (XGBoost + PyTorch)
cd millionaire_pickz_wnba_model
pip install -r requirements.txt
python train_xgboost.py
python backtest_ev.py
cd /home/workdir/artifacts/millionaire_pickz_wnba_model
pip install -r requirements.txt
python train_xgboost.py
python train_pytorch.py
python predict_daily.py
streamlit run dashboard.py
cd /home/workdir/artifacts/millionaire_pickz_wnba_model

# 1. Fetch real-ish data
python fetch_wnba_data.py

# 2. Train models (with better features)
python train_xgboost.py
python train_pytorch.py

# 3. Backtest
python backtest_ev.py

# 4. Daily predictions
python predict_daily.py

# 5. Launch the Web App (Dashboard)
streamlit run dashboard.py
cd /home/workdir/artifacts
zip -r millionaire_pickz_wnba_model.zip millionaire_pickz_wnba_model

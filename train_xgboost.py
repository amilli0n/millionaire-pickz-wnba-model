"""
Train XGBoost model for WNBA predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

def train_xgboost_model():
    """
    Train XGBoost model on WNBA data
    """
    print("Training XGBoost model...")
    
    # Load data
    if not os.path.exists('data/wnba_data.csv'):
        print("Data file not found. Please run fetch_wnba_data.py first.")
        return None
    
    data = pd.read_csv('data/wnba_data.csv')
    
    # TODO: Feature engineering
    X = data.drop(['date', 'team', 'player'], axis=1, errors='ignore')
    y = data.get('points', np.random.rand(len(data)))  # Target variable
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"XGBoost Train Score: {train_score:.4f}")
    print(f"XGBoost Test Score: {test_score:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/xgboost_model.pkl")
    
    return model

if __name__ == "__main__":
    train_xgboost_model()
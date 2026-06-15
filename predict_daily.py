"""
Daily predictions for WNBA games
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

def make_daily_predictions():
    """
    Generate daily predictions for upcoming WNBA games
    """
    print("Generating daily predictions...")
    
    # Load model
    model_path = 'models/xgboost_model.pkl'
    if not os.path.exists(model_path):
        print("Model file not found. Please train model first.")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load latest data
    if not os.path.exists('data/wnba_data.csv'):
        print("Data file not found. Please run fetch_wnba_data.py first.")
        return None
    
    data = pd.read_csv('data/wnba_data.csv')
    
    # TODO: Fetch today's games and prepare features
    # For now, use last row as example
    X = data.drop(['date', 'team', 'player'], axis=1, errors='ignore').iloc[-5:]
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'game_id': range(len(predictions)),
        'prediction': predictions,
        'confidence': np.random.rand(len(predictions)),  # TODO: Calculate actual confidence
        'timestamp': datetime.now(),
    })
    
    print("\n=== Daily Predictions ===")
    print(predictions_df)
    
    # Save predictions
    os.makedirs('predictions', exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    predictions_df.to_csv(f'predictions/predictions_{today}.csv', index=False)
    print(f"\nPredictions saved to predictions/predictions_{today}.csv")
    
    return predictions_df

if __name__ == "__main__":
    make_daily_predictions()
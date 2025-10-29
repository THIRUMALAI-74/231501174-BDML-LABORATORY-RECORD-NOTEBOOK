"""
Data Preprocessing Script - Uses existing src modules
Run: python run_preprocessing.py
"""

import pandas as pd
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
import joblib
import os

def main():
    print("="*80)
    print("DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*80)
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('data/raw/train.csv')
    print(f"   ✓ Shape: {df.shape}")
    
    # Initialize preprocessor
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, target_col='SalePrice', fit=True)
    
    # Feature engineering
    print("\n3. Engineering features...")
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    
    print(f"\n   ✓ Final shape: {X_engineered.shape}")
    print(f"   ✓ Created features: {len(engineer.created_features)}")
    
    # Save processed data
    print("\n4. Saving processed data...")
    X_engineered.to_csv('data/processed/X_train_processed.csv', index=False)
    y.to_csv('data/processed/y_train.csv', index=False)
    print("   ✓ Saved to data/processed/")
    
    # Save preprocessor objects
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    print("   ✓ Saved preprocessor objects")
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("NEXT STEP: python run_training.py")
    print("="*80)

if __name__ == "__main__":
    main()
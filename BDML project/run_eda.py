"""
Quick EDA Script - Uses existing src/preprocessing.py
Run: python run_eda.py
"""

import pandas as pd
from src.preprocessing import DataPreprocessor

def main():
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📊 Loading dataset...")
    df = pd.read_csv('data/raw/train.csv')
    print(f"✓ Dataset shape: {df.shape}")
    
    # Use existing preprocessor for analysis
    preprocessor = DataPreprocessor()
    preprocessor.initial_analysis(df)
    
    print("\n" + "="*80)
    print("✓ EDA Complete!")
    print("NEXT STEP: python run_preprocessing.py")
    print("="*80)

if __name__ == "__main__":
    main()
"""
Model Training Script - Uses existing src/model.py
Run: python run_training.py
"""

import pandas as pd
from src.model import HousePriceModel
from sklearn.model_selection import train_test_split

def main():
    print("="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    # Load processed data
    print("\n1. Loading processed data...")
    X = pd.read_csv('data/processed/X_train_processed.csv')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()
    print(f"   ✓ Features: {X.shape}")
    print(f"   ✓ Target: {y.shape}")
    
    # Train-test split
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Train: {X_train.shape}")
    print(f"   ✓ Test: {X_test.shape}")
    
    # Initialize and train models
    print("\n3. Training models...")
    model_trainer = HousePriceModel()
    model_trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n4. Model Comparison:")
    results_df = model_trainer.get_results_dataframe()
    print(results_df[['Test R²', 'Test RMSE', 'Test MAE']].to_string())
    
    # Cross-validation on best model
    print(f"\n5. Cross-validating best model ({model_trainer.best_model_name})...")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)])
    cv_results = model_trainer.cross_validate_model(
        model_trainer.best_model, X_full, y_full, cv=5
    )
    
    # Feature importance
    print("\n6. Top 20 Feature Importances:")
    feature_importance = model_trainer.get_feature_importance(X_train, top_n=20)
    
    # Save model
    print("\n7. Saving model...")
    model_trainer.save_model('models/trained_model.pkl')
    
    # Save feature names for prediction
    with open('models/model_features.txt', 'w') as f:
        for feature in X_train.columns:
            f.write(f"{feature}\n")
    print("   ✓ Saved feature names to models/model_features.txt")
    
    # Final summary
    best_metrics = results_df.loc[model_trainer.best_model_name]
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"""
BEST MODEL: {model_trainer.best_model_name}

PERFORMANCE:
  • Test R²: {best_metrics['Test R²']:.4f}
  • Test RMSE: ${best_metrics['Test RMSE']:,.2f}
  • Test MAE: ${best_metrics['Test MAE']:,.2f}
  • Test MAPE: {best_metrics['Test MAPE']:.2f}%

MODEL SAVED: models/trained_model.pkl

NEXT STEP: Start the API
  → uvicorn api.main:app --reload
  → Open http://localhost:8000
    """)
    print("="*80)

if __name__ == "__main__":
    main()
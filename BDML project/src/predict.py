"""
Prediction Module
Handles prediction logic and model inference.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Union
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    """
    House price predictor for inference.
    """
    
    def __init__(self, model_path='models/trained_model.pkl'):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = None
    
    def load_model(self):
        """
        Load the trained model.
        """
        try:
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def prepare_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            DataFrame ready for prediction
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # Add any missing features with default values
        # This should match your training feature set
        
        return df
    
    def predict_single(self, input_data: Dict) -> float:
        """
        Predict price for a single house.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Predicted price
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare input
        X = self.prepare_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction
    
    def predict_batch(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices for multiple houses.
        
        Args:
            input_df: DataFrame with feature values
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions = self.model.predict(input_df)
        
        return predictions
    
    def predict_with_confidence(self, input_data: Dict, 
                               confidence_level: float = 0.95) -> Dict:
        """
        Predict price with confidence interval.
        
        Args:
            input_data: Dictionary with feature values
            confidence_level: Confidence level for interval
            
        Returns:
            Dictionary with prediction and confidence interval
        """
        prediction = self.predict_single(input_data)
        
        # Simple confidence interval based on percentage
        # In production, use proper prediction intervals
        margin = prediction * 0.10  # ±10%
        
        result = {
            'predicted_price': float(prediction),
            'confidence_interval': {
                'lower': float(prediction - margin),
                'upper': float(prediction + margin)
            },
            'confidence_level': confidence_level
        }
        
        return result
    
    def get_feature_contribution(self, input_data: Dict, 
                                top_n: int = 10) -> Dict:
        """
        Get feature contributions to prediction (for tree-based models).
        
        Args:
            input_data: Dictionary with feature values
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature contributions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        X = self.prepare_input(input_data)
        
        # This is simplified - in production, use SHAP values
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            contributions = {}
            
            for i, col in enumerate(X.columns):
                contributions[col] = float(importances[i] * X[col].values[0])
            
            # Sort and get top N
            sorted_contrib = dict(sorted(contributions.items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True)[:top_n])
            
            return sorted_contrib
        else:
            return {}


def predict_price(input_data: Dict, model_path: str = 'models/trained_model.pkl') -> Dict:
    """
    Convenience function to predict house price.
    
    Args:
        input_data: Dictionary with feature values
        model_path: Path to trained model
        
    Returns:
        Dictionary with prediction results
    """
    predictor = HousePricePredictor(model_path)
    predictor.load_model()
    
    result = predictor.predict_with_confidence(input_data)
    
    return result


if __name__ == "__main__":
    print("Prediction Module")
    print("Import this module for making predictions")
    
    # Example usage:
    # predictor = HousePricePredictor()
    # predictor.load_model()
    # result = predictor.predict_single({'GrLivArea': 1500, 'OverallQual': 7, ...})
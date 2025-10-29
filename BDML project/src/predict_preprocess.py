"""
Prediction Preprocessing Module
Prepares single prediction inputs to match training feature format.
"""

import pandas as pd
import numpy as np
import joblib
import os


def preprocess_for_prediction(input_dict, model_features_path='models/model_features.txt'):
    """
    Preprocess a single prediction input to match training features.
    
    Args:
        input_dict: Dictionary with user input
        model_features_path: Path to file containing model feature names
        
    Returns:
        DataFrame ready for model prediction
    """
    # Create base DataFrame with API inputs
    base_data = {
        'OverallQual': input_dict.get('overall_qual'),
        'GrLivArea': input_dict.get('gr_liv_area'),
        'GarageArea': input_dict.get('garage_area'),
        'TotalBsmtSF': input_dict.get('total_bsmt_sf'),
        '1stFlrSF': input_dict.get('first_flr_sf'),
        'YearBuilt': input_dict.get('year_built'),
        'YearRemodAdd': input_dict.get('year_remod_add'),
        'FullBath': input_dict.get('full_bath'),
        'BedroomAbvGr': input_dict.get('bedroom_abv_gr'),
        'KitchenAbvGr': input_dict.get('kitchen_abv_gr'),
        'TotRmsAbvGrd': input_dict.get('totrms_abv_grd'),
        'Fireplaces': input_dict.get('fireplaces'),
        'GarageCars': input_dict.get('garage_cars'),
        'LotArea': input_dict.get('lot_area', 8000),
        'OverallCond': input_dict.get('overall_cond', 5),
    }
    
    # Add derived/default features commonly used in training
    # These are typical defaults - adjust based on your actual training
    base_data.update({
        '2ndFlrSF': 0,
        'LowQualFinSF': 0,
        'BsmtFullBath': 0,
        'BsmtHalfBath': 0,
        'HalfBath': 0,
        'BsmtFinSF1': base_data['TotalBsmtSF'],
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 0,
        'GarageYrBlt': base_data['YearBuilt'],
        'MasVnrArea': 0,
        'WoodDeckSF': 0,
        'OpenPorchSF': 0,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'PoolArea': 0,
        'MiscVal': 0,
        'YrSold': 2024,
        'MoSold': 6,
    })
    
    # Create DataFrame
    df = pd.DataFrame([base_data])
    
    # Load model features if available
    if os.path.exists(model_features_path):
        with open(model_features_path, 'r') as f:
            model_features = [line.strip() for line in f.readlines()]
        
        # Create DataFrame with all model features, default to 0
        result_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Fill in values we have
        for col in df.columns:
            if col in result_df.columns:
                result_df[col] = df[col].values[0]
        
        return result_df
    else:
        # If no feature list, return what we have
        return df


def get_simple_prediction_input(input_dict):
    """
    Simplified version that works with basic inputs.
    Use this if full preprocessing causes issues.
    
    Args:
        input_dict: Dictionary with user input
        
    Returns:
        DataFrame with basic features
    """
    data = {
        'OverallQual': [input_dict.get('overall_qual')],
        'GrLivArea': [input_dict.get('gr_liv_area')],
        'GarageArea': [input_dict.get('garage_area')],
        'TotalBsmtSF': [input_dict.get('total_bsmt_sf')],
        '1stFlrSF': [input_dict.get('first_flr_sf')],
        'YearBuilt': [input_dict.get('year_built')],
        'YearRemodAdd': [input_dict.get('year_remod_add')],
        'FullBath': [input_dict.get('full_bath')],
        'BedroomAbvGr': [input_dict.get('bedroom_abv_gr')],
        'KitchenAbvGr': [input_dict.get('kitchen_abv_gr')],
        'TotRmsAbvGrd': [input_dict.get('totrms_abv_grd')],
        'Fireplaces': [input_dict.get('fireplaces')],
        'GarageCars': [input_dict.get('garage_cars')],
        'LotArea': [input_dict.get('lot_area', 8000)],
        'OverallCond': [input_dict.get('overall_cond', 5)],
    }
    
    return pd.DataFrame(data)
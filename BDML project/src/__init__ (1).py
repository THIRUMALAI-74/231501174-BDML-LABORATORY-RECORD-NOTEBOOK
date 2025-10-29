"""
House Price Prediction System
Core modules for data preprocessing, feature engineering, and modeling.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import preprocessing
from . import feature_engineering
from . import model
from . import predict

__all__ = [
    "preprocessing",
    "feature_engineering",
    "model",
    "predict"
]
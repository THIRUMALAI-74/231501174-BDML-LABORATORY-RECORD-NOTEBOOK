"""
Test Preprocessing Module
Tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'GrLivArea': [1500, 2000, np.nan, 1800],
        'OverallQual': [7, 8, 6, 9],
        'YearBuilt': [2000, 1995, 2010, 2005],
        'Neighborhood': ['NAmes', 'Edwards', 'NAmes', 'BrkSide'],
        'SalePrice': [200000, 250000, 180000, 300000]
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.scaler is not None
        assert isinstance(preprocessor.label_encoders, dict)
        assert isinstance(preprocessor.numerical_features, list)
        assert isinstance(preprocessor.categorical_features, list)
    
    def test_identify_feature_types(self, preprocessor, sample_data):
        """Test feature type identification."""
        preprocessor.identify_feature_types(sample_data)
        
        assert 'GrLivArea' in preprocessor.numerical_features
        assert 'OverallQual' in preprocessor.numerical_features
        assert 'Neighborhood' in preprocessor.categorical_features
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        preprocessor.identify_feature_types(sample_data)
        df_clean = preprocessor.handle_missing_values(sample_data)
        
        # Check no missing values remain
        assert df_clean.isnull().sum().sum() == 0
        
        # Check that median was used for numerical features
        assert df_clean['GrLivArea'].iloc[2] > 0
    
    def test_detect_outliers(self, preprocessor, sample_data):
        """Test outlier detection."""
        outliers = preprocessor.detect_outliers(sample_data, 'GrLivArea', method='iqr')
        
        assert isinstance(outliers, pd.Series)
        assert len(outliers) == len(sample_data)
        assert outliers.dtype == bool
    
    def test_handle_outliers(self, preprocessor, sample_data):
        """Test outlier handling."""
        preprocessor.identify_feature_types(sample_data)
        
        # Add extreme outlier
        sample_data.loc[0, 'GrLivArea'] = 10000
        
        df_clean = preprocessor.handle_outliers(sample_data, ['GrLivArea'])
        
        # Check outlier was capped
        assert df_clean['GrLivArea'].max() < 10000
    
    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical encoding."""
        preprocessor.identify_feature_types(sample_data)
        df_encoded = preprocessor.encode_categorical_features(sample_data)
        
        # Check that categorical features are encoded
        assert 'Neighborhood' not in df_encoded.columns or df_encoded['Neighborhood'].dtype != 'object'


class TestDataValidation:
    """Test data validation functions."""
    
    def test_missing_value_count(self, sample_data):
        """Test missing value counting."""
        missing_count = sample_data.isnull().sum().sum()
        assert missing_count == 1  # One NaN in GrLivArea
    
    def test_data_types(self, sample_data):
        """Test data type validation."""
        assert sample_data['OverallQual'].dtype in [np.int64, int]
        assert sample_data['Neighborhood'].dtype == object
    
    def test_value_ranges(self, sample_data):
        """Test value range validation."""
        assert (sample_data['OverallQual'] >= 1).all()
        assert (sample_data['OverallQual'] <= 10).all()


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""
    
    def test_full_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline."""
        X, y = preprocessor.preprocess_pipeline(sample_data, fit=True)
        
        # Check output shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check no missing values
        assert X.isnull().sum().sum() == 0
        
        # Check target is numeric
        assert pd.api.types.is_numeric_dtype(y)
    
    def test_pipeline_without_target(self, preprocessor, sample_data):
        """Test pipeline with data without target."""
        sample_data_no_target = sample_data.drop('SalePrice', axis=1)
        X, y = preprocessor.preprocess_pipeline(sample_data_no_target, target_col='SalePrice', fit=True)
        
        assert X is not None
        assert y is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
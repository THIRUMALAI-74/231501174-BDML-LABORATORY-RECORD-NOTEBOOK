"""
Data Preprocessing Module
Handles missing values, outliers, and data cleaning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for house price prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_features = []
        self.categorical_features = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def initial_analysis(self, df: pd.DataFrame) -> None:
        """
        Perform initial data analysis.
        
        Args:
            df: Input DataFrame
        """
        print("\n" + "="*50)
        print("INITIAL DATA ANALYSIS")
        print("="*50)
        print(f"\nDataset Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        print("\n--- Data Types ---")
        print(df.dtypes.value_counts())
        
        print("\n--- Missing Values Summary ---")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.head(20))
        else:
            print("No missing values found!")
        
        print("\n--- Basic Statistics ---")
        print(df.describe())
    
    def identify_feature_types(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> None:
        """
        Identify numerical and categorical features.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
        """
        # Separate features from target
        features = [col for col in df.columns if col != target_col]
        
        # Numerical features
        self.numerical_features = df[features].select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        # Categorical features
        self.categorical_features = df[features].select_dtypes(
            include=['object']
        ).columns.tolist()
        
        print(f"\n✓ Identified {len(self.numerical_features)} numerical features")
        print(f"✓ Identified {len(self.categorical_features)} categorical features")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using domain-specific strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        df_clean = df.copy()
        
        print("\n" + "="*50)
        print("HANDLING MISSING VALUES")
        print("="*50)
        
        # Numerical features: Median imputation
        for col in self.numerical_features:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"✓ {col}: Filled with median ({median_val:.2f})")
        
        # Categorical features: Mode imputation or 'None' for meaningful missing
        for col in self.categorical_features:
            if df_clean[col].isnull().sum() > 0:
                # Features where missing means "None"
                if col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                          'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                          'PoolQC', 'Fence', 'MiscFeature']:
                    df_clean[col].fillna('None', inplace=True)
                    print(f"✓ {col}: Filled with 'None'")
                else:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"✓ {col}: Filled with mode ({mode_val})")
        
        print(f"\n✓ Missing values handled. Remaining: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, column: str, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame
            column: Column name to check
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Handle outliers by capping at percentiles.
        
        Args:
            df: Input DataFrame
            columns: List of columns to process (if None, use all numerical)
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = self.numerical_features
        
        print("\n" + "="*50)
        print("HANDLING OUTLIERS")
        print("="*50)
        
        for col in columns:
            if col in df_clean.columns:
                # Calculate percentiles
                lower_percentile = df_clean[col].quantile(0.01)
                upper_percentile = df_clean[col].quantile(0.99)
                
                # Count outliers
                outliers_lower = (df_clean[col] < lower_percentile).sum()
                outliers_upper = (df_clean[col] > upper_percentile).sum()
                
                if outliers_lower + outliers_upper > 0:
                    # Cap outliers
                    df_clean[col] = df_clean[col].clip(
                        lower=lower_percentile,
                        upper=upper_percentile
                    )
                    print(f"✓ {col}: Capped {outliers_lower + outliers_upper} outliers")
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding and One-Hot Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for testing)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*50)
        
        # Ordinal features (with natural ordering) - use Label Encoding
        ordinal_features = {
            'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
        }
        
        for feature, order in ordinal_features.items():
            if feature in df_encoded.columns:
                if fit:
                    df_encoded[feature] = df_encoded[feature].apply(
                        lambda x: order.index(x) if x in order else 0
                    )
                print(f"✓ {feature}: Label encoded")
        
        # Nominal features (no ordering) - use One-Hot Encoding for low cardinality
        nominal_features = [col for col in self.categorical_features 
                          if col not in ordinal_features.keys()]
        
        for feature in nominal_features:
            if feature in df_encoded.columns:
                unique_count = df_encoded[feature].nunique()
                
                # One-hot encode if cardinality < 10
                if unique_count < 10:
                    dummies = pd.get_dummies(df_encoded[feature], 
                                            prefix=feature, 
                                            drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(feature, axis=1, inplace=True)
                    print(f"✓ {feature}: One-hot encoded ({unique_count} categories)")
                else:
                    # Label encode high cardinality features
                    if fit:
                        le = LabelEncoder()
                        df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                        self.label_encoders[feature] = le
                    else:
                        if feature in self.label_encoders:
                            df_encoded[feature] = self.label_encoders[feature].transform(
                                df_encoded[feature].astype(str)
                            )
                    print(f"✓ {feature}: Label encoded ({unique_count} categories)")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, 
                      columns: List[str] = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (if None, scale all numerical)
            fit: Whether to fit scaler (True for training, False for testing)
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = [col for col in self.numerical_features if col in df.columns]
        
        print("\n" + "="*50)
        print("SCALING FEATURES")
        print("="*50)
        
        if fit:
            df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
            print(f"✓ Fitted and transformed {len(columns)} features")
        else:
            df_scaled[columns] = self.scaler.transform(df_scaled[columns])
            print(f"✓ Transformed {len(columns)} features")
        
        return df_scaled
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                           target_col: str = 'SalePrice',
                           fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            fit: Whether to fit preprocessors
            
        Returns:
            Tuple of (processed features, target)
        """
        print("\n" + "="*50)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*50)
        
        # Initial analysis
        if fit:
            self.initial_analysis(df)
            self.identify_feature_types(df, target_col)
        
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(target_col, axis=1)
            y = df[target_col]
        else:
            X = df.copy()
            y = None
        
        # Step 1: Handle missing values
        X = self.handle_missing_values(X)
        
        # Step 2: Handle outliers (only for numerical features)
        X = self.handle_outliers(X)
        
        # Step 3: Encode categorical features
        X = self.encode_categorical_features(X, fit=fit)
        
        # Step 4: Scale features (excluding target)
        # Note: Scaling is typically done after train-test split
        # X = self.scale_features(X, fit=fit)
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Final shape: {X.shape}")
        print(f"Features: {X.columns.tolist()[:10]}... (showing first 10)")
        
        return X, y


def preprocess_data(train_path: str, test_path: str = None):
    """
    Convenience function to preprocess training and test data.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV (optional)
        
    Returns:
        Preprocessed training and test data
    """
    preprocessor = DataPreprocessor()
    
    # Load and preprocess training data
    train_df = preprocessor.load_data(train_path)
    X_train, y_train = preprocessor.preprocess_pipeline(train_df, fit=True)
    
    # Preprocess test data if provided
    if test_path:
        test_df = preprocessor.load_data(test_path)
        X_test, y_test = preprocessor.preprocess_pipeline(test_df, fit=False)
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("Import this module to use preprocessing functions")
    
    # Example:
    # preprocessor = DataPreprocessor()
    # df = preprocessor.load_data('data/raw/train.csv')
    # X, y = preprocessor.preprocess_pipeline(df)
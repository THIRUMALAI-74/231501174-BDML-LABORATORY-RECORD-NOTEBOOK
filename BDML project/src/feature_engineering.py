"""
Feature Engineering Module
Creates new features from existing ones to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating derived features.
    """
    
    def __init__(self):
        self.created_features = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        df_new = df.copy()
        
        print("\n--- Creating Temporal Features ---")
        
        # House age
        if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
            df_new['HouseAge'] = df_new['YrSold'] - df_new['YearBuilt']
            self.created_features.append('HouseAge')
            print("✓ HouseAge = YrSold - YearBuilt")
        
        # Years since remodel
        if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
            df_new['YearsSinceRemod'] = df_new['YrSold'] - df_new['YearRemodAdd']
            self.created_features.append('YearsSinceRemod')
            print("✓ YearsSinceRemod = YrSold - YearRemodAdd")
        
        # Is new house (< 5 years old)
        if 'HouseAge' in df_new.columns:
            df_new['IsNewHouse'] = (df_new['HouseAge'] <= 5).astype(int)
            self.created_features.append('IsNewHouse')
            print("✓ IsNewHouse = 1 if HouseAge <= 5")
        
        # Was remodeled (different from year built)
        if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
            df_new['WasRemodeled'] = (df_new['YearRemodAdd'] != df_new['YearBuilt']).astype(int)
            self.created_features.append('WasRemodeled')
            print("✓ WasRemodeled = 1 if remodel year differs from built year")
        
        # Garage age
        if 'GarageYrBlt' in df.columns and 'YrSold' in df.columns:
            df_new['GarageAge'] = df_new['YrSold'] - df_new['GarageYrBlt']
            df_new['GarageAge'] = df_new['GarageAge'].fillna(0)
            self.created_features.append('GarageAge')
            print("✓ GarageAge = YrSold - GarageYrBlt")
        
        return df_new
    
    def create_area_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create area-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with area features
        """
        df_new = df.copy()
        
        print("\n--- Creating Area Features ---")
        
        # Total square footage
        area_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        if all(col in df.columns for col in area_cols):
            df_new['TotalSF'] = df_new[area_cols].sum(axis=1)
            self.created_features.append('TotalSF')
            print("✓ TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF")
        
        # Total bathrooms
        if 'FullBath' in df.columns and 'HalfBath' in df.columns:
            df_new['TotalBath'] = df_new['FullBath'] + 0.5 * df_new['HalfBath']
            self.created_features.append('TotalBath')
            print("✓ TotalBath = FullBath + 0.5 * HalfBath")
        
        if 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
            df_new['TotalBath'] = (df_new.get('TotalBath', 0) + 
                                   df_new['BsmtFullBath'] + 
                                   0.5 * df_new['BsmtHalfBath'])
            print("✓ Added basement bathrooms to TotalBath")
        
        # Total porch area
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        if all(col in df.columns for col in porch_cols):
            df_new['TotalPorchSF'] = df_new[porch_cols].sum(axis=1)
            self.created_features.append('TotalPorchSF')
            print("✓ TotalPorchSF = Sum of all porch areas")
        
        # Living area per room
        if 'GrLivArea' in df.columns and 'TotRmsAbvGrd' in df.columns:
            df_new['AreaPerRoom'] = df_new['GrLivArea'] / (df_new['TotRmsAbvGrd'] + 1)
            self.created_features.append('AreaPerRoom')
            print("✓ AreaPerRoom = GrLivArea / TotRmsAbvGrd")
        
        # Basement percentage
        if 'TotalBsmtSF' in df.columns and 'TotalSF' in df_new.columns:
            df_new['BsmtPercent'] = df_new['TotalBsmtSF'] / (df_new['TotalSF'] + 1)
            self.created_features.append('BsmtPercent')
            print("✓ BsmtPercent = TotalBsmtSF / TotalSF")
        
        # Garage to house area ratio
        if 'GarageArea' in df.columns and 'GrLivArea' in df.columns:
            df_new['GarageAreaRatio'] = df_new['GarageArea'] / (df_new['GrLivArea'] + 1)
            self.created_features.append('GarageAreaRatio')
            print("✓ GarageAreaRatio = GarageArea / GrLivArea")
        
        return df_new
    
    def create_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create quality-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with quality features
        """
        df_new = df.copy()
        
        print("\n--- Creating Quality Features ---")
        
        # Overall quality * condition
        if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
            df_new['QualityCondScore'] = df_new['OverallQual'] * df_new['OverallCond']
            self.created_features.append('QualityCondScore')
            print("✓ QualityCondScore = OverallQual × OverallCond")
        
        # Total quality score (combining multiple quality metrics)
        quality_cols = ['ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC', 'GarageQual']
        available_quality = [col for col in quality_cols if col in df.columns]
        
        if available_quality:
            # Convert to numeric if not already
            for col in available_quality:
                if df_new[col].dtype == 'object':
                    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
                    df_new[col] = df_new[col].map(quality_map).fillna(0)
            
            df_new['TotalQualityScore'] = df_new[available_quality].sum(axis=1)
            self.created_features.append('TotalQualityScore')
            print(f"✓ TotalQualityScore = Sum of {len(available_quality)} quality metrics")
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        
        print("\n--- Creating Interaction Features ---")
        
        # Living area × overall quality
        if 'GrLivArea' in df.columns and 'OverallQual' in df.columns:
            df_new['AreaQuality'] = df_new['GrLivArea'] * df_new['OverallQual']
            self.created_features.append('AreaQuality')
            print("✓ AreaQuality = GrLivArea × OverallQual")
        
        # Garage area × garage quality
        if 'GarageArea' in df.columns and 'GarageQual' in df.columns:
            if df_new['GarageQual'].dtype == 'object':
                quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
                df_new['GarageQual'] = df_new['GarageQual'].map(quality_map).fillna(0)
            df_new['GarageQualityArea'] = df_new['GarageArea'] * df_new['GarageQual']
            self.created_features.append('GarageQualityArea')
            print("✓ GarageQualityArea = GarageArea × GarageQual")
        
        # Total SF × Overall Quality
        if 'TotalSF' in df_new.columns and 'OverallQual' in df.columns:
            df_new['TotalSFQuality'] = df_new['TotalSF'] * df_new['OverallQual']
            self.created_features.append('TotalSFQuality')
            print("✓ TotalSFQuality = TotalSF × OverallQual")
        
        return df_new
    
    def create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with binary features
        """
        df_new = df.copy()
        
        print("\n--- Creating Binary Features ---")
        
        # Has pool
        if 'PoolArea' in df.columns:
            df_new['HasPool'] = (df_new['PoolArea'] > 0).astype(int)
            self.created_features.append('HasPool')
            print("✓ HasPool = 1 if PoolArea > 0")
        
        # Has garage
        if 'GarageArea' in df.columns:
            df_new['HasGarage'] = (df_new['GarageArea'] > 0).astype(int)
            self.created_features.append('HasGarage')
            print("✓ HasGarage = 1 if GarageArea > 0")
        
        # Has basement
        if 'TotalBsmtSF' in df.columns:
            df_new['HasBsmt'] = (df_new['TotalBsmtSF'] > 0).astype(int)
            self.created_features.append('HasBsmt')
            print("✓ HasBsmt = 1 if TotalBsmtSF > 0")
        
        # Has fireplace
        if 'Fireplaces' in df.columns:
            df_new['HasFireplace'] = (df_new['Fireplaces'] > 0).astype(int)
            self.created_features.append('HasFireplace')
            print("✓ HasFireplace = 1 if Fireplaces > 0")
        
        # Has second floor
        if '2ndFlrSF' in df.columns:
            df_new['Has2ndFloor'] = (df_new['2ndFlrSF'] > 0).astype(int)
            self.created_features.append('Has2ndFloor')
            print("✓ Has2ndFloor = 1 if 2ndFlrSF > 0")
        
        # Has wood deck
        if 'WoodDeckSF' in df.columns:
            df_new['HasWoodDeck'] = (df_new['WoodDeckSF'] > 0).astype(int)
            self.created_features.append('HasWoodDeck')
            print("✓ HasWoodDeck = 1 if WoodDeckSF > 0")
        
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                   columns: List[str] = None) -> pd.DataFrame:
        """
        Create polynomial features (squares, cubes) for important variables.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create polynomial features for
            
        Returns:
            DataFrame with polynomial features
        """
        df_new = df.copy()
        
        if columns is None:
            columns = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'LotArea']
        
        print("\n--- Creating Polynomial Features ---")
        
        for col in columns:
            if col in df.columns:
                # Square
                df_new[f'{col}_Squared'] = df_new[col] ** 2
                self.created_features.append(f'{col}_Squared')
                print(f"✓ {col}_Squared = {col}²")
        
        return df_new
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df_engineered = df.copy()
        
        # Create all feature types
        df_engineered = self.create_temporal_features(df_engineered)
        df_engineered = self.create_area_features(df_engineered)
        df_engineered = self.create_quality_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        df_engineered = self.create_binary_features(df_engineered)
        # df_engineered = self.create_polynomial_features(df_engineered)
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*50)
        print(f"Original features: {df.shape[1]}")
        print(f"New features created: {len(self.created_features)}")
        print(f"Total features: {df_engineered.shape[1]}")
        print(f"\nCreated features: {self.created_features[:10]}...")
        
        return df_engineered
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names.
        
        Returns:
            List of feature names
        """
        return self.created_features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer all features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Import this module to use feature engineering functions")
    
    # Example:
    # engineer = FeatureEngineer()
    # df_engineered = engineer.create_all_features(df)
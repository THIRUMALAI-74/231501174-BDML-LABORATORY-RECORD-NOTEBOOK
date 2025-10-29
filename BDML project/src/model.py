"""
Model Training Module
Trains and evaluates multiple machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class HousePriceModel:
    """
    House price prediction model trainer and evaluator.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Initialize all models to be trained.
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }
        
        print("\n✓ Initialized 8 models")
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_train, X_test: Feature sets
            y_train, y_test: Target sets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Test MAE': mean_absolute_error(y_test, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train R²': r2_score(y_train, y_train_pred),
            'Test R²': r2_score(y_test, y_test_pred),
            'Train MAPE': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
            'Test MAPE': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        }
        
        return metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models.
        
        Args:
            X_train, X_test: Feature sets
            y_train, y_test: Target sets
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            self.results[name] = metrics
            
            # Print results
            print(f"✓ Training complete")
            print(f"  Test R²: {metrics['Test R²']:.4f}")
            print(f"  Test RMSE: ${metrics['Test RMSE']:,.2f}")
            print(f"  Test MAE: ${metrics['Test MAE']:,.2f}")
            print(f"  Test MAPE: {metrics['Test MAPE']:.2f}%")
        
        # Find best model
        self.best_model_name = max(self.results.keys(), 
                                   key=lambda x: self.results[x]['Test R²'])
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*50)
        print(f"BEST MODEL: {self.best_model_name}")
        print("="*50)
    
    def get_results_dataframe(self):
        """
        Get results as a formatted DataFrame.
        
        Returns:
            DataFrame with all model results
        """
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)
        return df_results
    
    def cross_validate_model(self, model, X, y, cv=5):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        print(f"\n--- Cross-Validation ({cv} folds) ---")
        
        # R² scores
        r2_scores = cross_val_score(model, X, y, cv=cv, 
                                     scoring='r2', n_jobs=-1)
        
        # Negative MAE (sklearn returns negative)
        mae_scores = -cross_val_score(model, X, y, cv=cv, 
                                       scoring='neg_mean_absolute_error', n_jobs=-1)
        
        # Negative RMSE
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, 
                                                scoring='neg_mean_squared_error', n_jobs=-1))
        
        print(f"R² Score: {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
        print(f"MAE: ${mae_scores.mean():,.2f} (+/- ${mae_scores.std():,.2f})")
        print(f"RMSE: ${rmse_scores.mean():,.2f} (+/- ${rmse_scores.std():,.2f})")
        
        return {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std()
        }
    
    def tune_random_forest(self, X_train, y_train, method='grid'):
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random'
            
        Returns:
            Best estimator
        """
        print("\n" + "="*50)
        print(f"TUNING RANDOM FOREST ({method.upper()} SEARCH)")
        print("="*50)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(rf, param_grid, cv=5, 
                                 scoring='r2', n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(rf, param_grid, n_iter=50, cv=5,
                                       scoring='r2', n_jobs=-1, verbose=1,
                                       random_state=42)
        
        search.fit(X_train, y_train)
        
        print(f"\n✓ Best parameters: {search.best_params_}")
        print(f"✓ Best R² score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def tune_xgboost(self, X_train, y_train, method='random'):
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random'
            
        Returns:
            Best estimator
        """
        print("\n" + "="*50)
        print(f"TUNING XGBOOST ({method.upper()} SEARCH)")
        print("="*50)
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5, 1],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        }
        
        xgb = XGBRegressor(random_state=42, verbosity=0)
        
        if method == 'grid':
            # Use smaller grid for grid search
            param_grid = {
                'n_estimators': [100, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7],
                'subsample': [0.8, 1.0]
            }
            search = GridSearchCV(xgb, param_grid, cv=5,
                                 scoring='r2', n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(xgb, param_distributions, n_iter=50, cv=5,
                                       scoring='r2', n_jobs=-1, verbose=1,
                                       random_state=42)
        
        search.fit(X_train, y_train)
        
        print(f"\n✓ Best parameters: {search.best_params_}")
        print(f"✓ Best R² score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def tune_lightgbm(self, X_train, y_train, method='random'):
        """
        Tune LightGBM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random'
            
        Returns:
            Best estimator
        """
        print("\n" + "="*50)
        print(f"TUNING LIGHTGBM ({method.upper()} SEARCH)")
        print("="*50)
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, -1],
            'num_leaves': [31, 50, 70, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        }
        
        lgbm = LGBMRegressor(random_state=42, verbose=-1)
        
        if method == 'grid':
            param_grid = {
                'n_estimators': [100, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7],
                'num_leaves': [31, 50]
            }
            search = GridSearchCV(lgbm, param_grid, cv=5,
                                 scoring='r2', n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(lgbm, param_distributions, n_iter=50, cv=5,
                                       scoring='r2', n_jobs=-1, verbose=1,
                                       random_state=42)
        
        search.fit(X_train, y_train)
        
        print(f"\n✓ Best parameters: {search.best_params_}")
        print(f"✓ Best R² score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def save_model(self, filepath='models/trained_model.pkl'):
        """
        Save the best model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.best_model is None:
            print("✗ No model trained yet!")
            return
        
        joblib.dump(self.best_model, filepath)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/trained_model.pkl'):
        """
        Load a saved model.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        self.best_model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return self.best_model
    
    def get_feature_importance(self, X, top_n=20):
        """
        Get feature importance from the best model.
        
        Args:
            X: Feature DataFrame
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.best_model is None:
            print("✗ No model trained yet!")
            return None
        
        # Check if model has feature_importances_
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            print(f"\n--- Top {top_n} Feature Importances ---")
            print(feature_importance_df.to_string(index=False))
            
            return feature_importance_df
        else:
            print("✗ Model does not have feature_importances_ attribute")
            return None
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions array
        """
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        return self.best_model.predict(X)


def train_model_pipeline(X_train, X_test, y_train, y_test, tune=False):
    """
    Complete model training pipeline.
    
    Args:
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
        tune: Whether to perform hyperparameter tuning
        
    Returns:
        Trained HousePriceModel instance
    """
    model_trainer = HousePriceModel()
    
    # Train all models
    model_trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(model_trainer.get_results_dataframe())
    
    # Cross-validation on best model
    print(f"\nPerforming cross-validation on {model_trainer.best_model_name}...")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    model_trainer.cross_validate_model(model_trainer.best_model, X_full, y_full)
    
    # Hyperparameter tuning if requested
    if tune:
        if 'XGBoost' in model_trainer.best_model_name:
            tuned_model = model_trainer.tune_xgboost(X_train, y_train)
            model_trainer.best_model = tuned_model
        elif 'LightGBM' in model_trainer.best_model_name:
            tuned_model = model_trainer.tune_lightgbm(X_train, y_train)
            model_trainer.best_model = tuned_model
        elif 'Random Forest' in model_trainer.best_model_name:
            tuned_model = model_trainer.tune_random_forest(X_train, y_train)
            model_trainer.best_model = tuned_model
        
        # Re-evaluate tuned model
        print("\n--- Evaluating Tuned Model ---")
        metrics = model_trainer.evaluate_model(
            model_trainer.best_model, X_train, X_test, y_train, y_test
        )
        print(f"Tuned Test R²: {metrics['Test R²']:.4f}")
        print(f"Tuned Test RMSE: ${metrics['Test RMSE']:,.2f}")
        print(f"Tuned Test MAE: ${metrics['Test MAE']:,.2f}")
    
    # Feature importance
    model_trainer.get_feature_importance(X_train)
    
    return model_trainer


if __name__ == "__main__":
    print("Model Training Module")
    print("Import this module to train house price prediction models")
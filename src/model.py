"""
XGBoost Fraud Detection Model Training & Persistence
====================================================
Handles model training, evaluation, and persistence for binary classification tasks.

This script is now flexible, allowing any CSV data file and target column to be used.

Authors: Ash Dehghan Ph.D and Cristian Perera
Date: November 2025
"""

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from typing import Tuple, Optional
import os
import sys

# Define the default column name for the target variable (label)
TARGET_COLUMN = "Class" 
# Default ratio for the specific credit card dataset (173 non-fraud per 1 fraud)
DEFAULT_POS_WEIGHT = 173


class FraudDetectionModel:
    """
    A reusable model class for XGBoost, specializing in binary classification 
    with extreme class imbalance handling. It retains the original name for 
    compatibility with other modules.
    """
    
    def __init__(self, target_column: str = TARGET_COLUMN, scale_pos_weight: float = DEFAULT_POS_WEIGHT):
        """
        Initialize the XGBoost model with optimal hyperparameters.
        
        Args:
            target_column: The name of the target variable column in the data.
            scale_pos_weight: The ratio to weight the positive class (critical for imbalance).
        """
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Dynamically set
            eval_metric="auc",
            tree_method="hist",
            random_state=42
        )
        self.target_column = target_column
        
    def load_and_split_data(
        self, 
        data_file: str, 
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load data from a CSV file and perform temporal train/test split.
        
        Args:
            data_file: Full path to the CSV file containing features and the target.
            train_ratio: Percentage of data for training (default 0.8 = 80% train, 20% test)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Data file path is now dynamic
        df = pd.read_csv(data_file) 
        print(f"Loaded {len(df):,} total rows | {df[self.target_column].sum()} positive cases")
        
        # Calculate train size from percentage
        train_size = int(len(df) * train_ratio)
        print(f"Split: {train_ratio:.0%} train ({train_size:,} rows) | {1-train_ratio:.0%} test ({len(df)-train_size:,} rows)")
        
        # Temporal split (no shuffle - simulates real deployment)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]
        
        # Use the dynamic target column name
        X_train, y_train = train.drop(self.target_column, axis=1), train[self.target_column]
        X_test, y_test = test.drop(self.target_column, axis=1), test[self.target_column]
        
        return X_train, y_train, X_test, y_test
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: int = 10
    ) -> dict:
        """
        Train the XGBoost model with evaluation monitoring.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for monitoring only)
            y_test: Test labels (for monitoring only)
            verbose: Print eval metrics every N iterations
            
        Returns:
            Dictionary with training metrics
        """
        print("Training XGBoost classification model...")
        
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=verbose
        )
        
        # Calculate performance metrics
        train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        
        metrics = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfitting_gap': train_auc - test_auc
        }
        
        self._print_overfitting_check(metrics)
        
        return metrics
    
    def _print_overfitting_check(self, metrics: dict) -> None:
        """Print formatted overfitting analysis."""
        print("\n" + "=" * 60)
        print("OVERFITTING CHECK")
        print("=" * 60)
        print(f"Train AUC: {metrics['train_auc']:.4f}")
        print(f"Test AUC:  {metrics['test_auc']:.4f}")
        print(f"Gap:       {metrics['overfitting_gap']:.4f}")
        
        gap = metrics['overfitting_gap']
        if gap < 0.01:
            print("✅ Minimal overfitting - model generalizes well!")
        elif gap < 0.02:
            print("⚠️  Slight overfitting - still acceptable")
        else:
            print("❌ Significant overfitting detected")
        print("=" * 60 + "\n")
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Full path where model will be saved (e.g., 'models/my_model.pkl')
        """
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load pre-trained model from disk.
        
        Args:
            filepath: Full path to saved model file
        """
        self.model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
    
    def predict_proba(self, X: pd.DataFrame) -> float:
        """
        Get classification probability for input features.
        
        Args:
            X: Features (single row or multiple rows)
            
        Returns:
            Probability of the positive class (0-1 scale)
        """
        return self.model.predict_proba(X)[:, 1]


def train_and_save_model(
    data_file: str,
    target_column: str = TARGET_COLUMN,
    scale_pos_weight: float = DEFAULT_POS_WEIGHT,
    model_path: str = "models/xgboost_fraud_model.pkl" # Default for the original purpose
) -> FraudDetectionModel:
    """
    Convenience function to train and save model in one call, accepting any data file.
    
    Args:
        data_file: Path to the training data CSV file.
        target_column: The name of the target variable column in the data.
        scale_pos_weight: The weight applied to the positive class.
        model_path: Where to save the trained model.
        
    Returns:
        Trained FraudDetectionModel instance
    """
    # Initialize the model with flexible parameters
    model = FraudDetectionModel(target_column=target_column, scale_pos_weight=scale_pos_weight)
    
    # Load and split the data based on the provided file path
    X_train, y_train, X_test, y_test = model.load_and_split_data(data_file)
    
    # Train and save
    model.train(X_train, y_train, X_test, y_test)
    model.save(model_path)
    
    return model


if __name__ == "__main__":
    # The default execution logic, now using a standard project path for the default data
    
    # --- DYNAMIC PATH RESOLUTION ---
    # Determine the script's directory and construct the path to the expected default data file.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'creditcard.csv')
    
    # --- CHECK ARGUMENTS FOR CUSTOM DATA FILE ---
    # Allow the user to specify a data file path when running from the command line
    custom_data_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FILE
    
    # --- TRAINING EXECUTION ---
    print("🚀 Starting generic XGBoost model training...\n")
    print(f"Using data file: {custom_data_file}")
    
    try:
        # Use the specific parameters for the credit card fraud dataset
        model = train_and_save_model(
            data_file=custom_data_file,
            target_column=TARGET_COLUMN,
            scale_pos_weight=DEFAULT_POS_WEIGHT,
            model_path=os.path.join(SCRIPT_DIR, '..', 'models', 'xgboost_fraud_model.pkl')
        )
        print("\n✅ Training complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Data file not found at {custom_data_file}")
        print("Please ensure the CSV file exists or pass its path as a command-line argument.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during training: {e}")
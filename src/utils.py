"""
Utility Functions for Fraud Detection Pipeline
==============================================
Helper functions for data loading, preprocessing, and visualization.

Author: Cristian Perera
Date: November 2025
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import os


def load_creditcard_data(
    data_path: str = "data/creditcard.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Args:
        data_path: Path to creditcard.csv
        verbose: Print loading info
        
    Returns:
        DataFrame with transaction data
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    
    df = pd.read_csv(data_path)
    
    if verbose:
        fraud_count = df['Class'].sum()
        fraud_rate = fraud_count / len(df) * 100
        print(f"✅ Loaded {len(df):,} transactions")
        print(f"   • Fraud cases: {fraud_count} ({fraud_rate:.3f}%)")
        print(f"   • Safe cases: {len(df) - fraud_count:,}")
        print(f"   • Class imbalance ratio: {(len(df) - fraud_count) / fraud_count:.0f}:1")
    
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    train_size: int = 227845,
    target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Perform temporal train/test split (no shuffling).
    
    This simulates real-world deployment where the model predicts
    future transactions based on historical data.
    
    Args:
        df: Full dataset
        train_size: Number of samples for training
        target_col: Name of the target column
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    X_test = test.drop(target_col, axis=1)
    y_test = test[target_col]
    
    print(f"✅ Train/Test split: {len(X_train):,} / {len(X_test):,}")
    print(f"   • Train frauds: {y_train.sum()}")
    print(f"   • Test frauds: {y_test.sum()}")
    
    return X_train, y_train, X_test, y_test


def get_random_test_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_fraud: int = 1,
    n_safe: int = 1,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample random fraud and safe transactions from test set.
    
    Args:
        X_test: Test features
        y_test: Test labels
        n_fraud: Number of fraud cases to sample
        n_safe: Number of safe cases to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (fraud_cases, safe_cases) DataFrames
    """
    fraud_cases = X_test[y_test == 1].sample(n=n_fraud, random_state=random_state)
    safe_cases = X_test[y_test == 0].sample(n=n_safe, random_state=random_state)
    
    return fraud_cases, safe_cases


def load_api_key(env_var: str = "GOOGLE_AI_API_KEY") -> str:
    """
    Load API key from environment variable.
    
    Args:
        env_var: Name of environment variable containing API key
        
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found
    """
    api_key = os.getenv(env_var)
    
    if api_key is None:
        raise ValueError(
            f"API key not found. Set {env_var} environment variable:\n"
            f"  export {env_var}='your-key-here'  # Linux/Mac\n"
            f"  set {env_var}=your-key-here       # Windows CMD\n"
            f"  $env:{env_var}='your-key-here'    # Windows PowerShell"
        )
    
    return api_key


def format_transaction_summary(transaction: pd.Series) -> str:
    """
    Create a human-readable summary of a transaction.
    
    Args:
        transaction: Transaction feature vector
        
    Returns:
        Formatted string with key transaction details
    """
    return f"""
Transaction Summary:
  Amount: ${transaction['Amount']:.2f}
  Time: {transaction['Time']:.0f}s ({transaction['Time']//3600:.0f}h into dataset)
  V14: {transaction['V14']:.2f}
  V17: {transaction['V17']:.2f}
    """.strip()


def print_model_performance(
    train_auc: float,
    test_auc: float,
    benchmark_auc: float = 0.98
) -> None:
    """
    Print formatted model performance report.
    
    Args:
        train_auc: Training set AUC
        test_auc: Test set AUC
        benchmark_auc: Industry benchmark for comparison
    """
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Train AUC:     {train_auc:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Overfit Gap:   {train_auc - test_auc:.4f}")
    print(f"Benchmark:     {benchmark_auc:.4f} (industry production-ready threshold)")
    
    if test_auc >= benchmark_auc:
        print("✅ PRODUCTION-READY: Exceeds industry benchmark!")
    else:
        print(f"⚠️  Below benchmark by {benchmark_auc - test_auc:.4f}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    """Demo: Load data and show statistics."""
    df = load_creditcard_data()
    X_train, y_train, X_test, y_test = temporal_train_test_split(df)
    
    print("\nSampling test cases...")
    fraud_cases, safe_cases = get_random_test_cases(X_test, y_test)
    
    print("\nFraud case details:")
    print(format_transaction_summary(fraud_cases.iloc[0]))
    
    print("\nSafe case details:")
    print(format_transaction_summary(safe_cases.iloc[0]))
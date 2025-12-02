"""
Unit Tests for Utility Functions
=================================
Tests data loading, splitting, formatting, and helper functions.

Authors: Ash Dehghan Ph.D and Cristian Perera
Date: November 2025
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    load_creditcard_data,
    temporal_train_test_split,
    get_random_test_cases,
    load_api_key,
    format_transaction_summary,
    print_model_performance
)


class TestLoadCreditcardData:
    """Test data loading functionality."""
    
    def test_load_data_success(self, mock_csv_path, capsys):
        """Test successful data loading with verbose output."""
        df = load_creditcard_data(mock_csv_path, verbose=True)
        
        assert isinstance(df, pd.DataFrame)
        assert 'Class' in df.columns
        assert len(df) == 100
        
        captured = capsys.readouterr()
        assert "Loaded" in captured.out
        assert "transactions" in captured.out
    
    def test_load_data_silent(self, mock_csv_path, capsys):
        """Test data loading without verbose output."""
        df = load_creditcard_data(mock_csv_path, verbose=False)
        
        assert isinstance(df, pd.DataFrame)
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_creditcard_data("nonexistent/path.csv")


class TestTemporalTrainTestSplit:
    """Test train/test splitting functionality."""
    
    def test_split_default_params(self, sample_dataframe, capsys):
        """Test temporal split with default parameters."""
        X_train, y_train, X_test, y_test = temporal_train_test_split(
            sample_dataframe,
            train_size=80
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check Class column removed from features
        assert 'Class' not in X_train.columns
        assert 'Class' not in X_test.columns
        
        captured = capsys.readouterr()
        assert "Train/Test split" in captured.out
    
    def test_split_custom_target(self, sample_dataframe):
        """Test split with custom target column name."""
        df = sample_dataframe.rename(columns={'Class': 'Fraud'})
        
        X_train, y_train, X_test, y_test = temporal_train_test_split(
            df,
            train_size=70,
            target_col='Fraud'
        )
        
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert 'Fraud' not in X_train.columns
    
    def test_split_preserves_order(self, sample_dataframe):
        """Test that temporal order is preserved (no shuffling)."""
        sample_dataframe['Time'] = range(len(sample_dataframe))
        
        X_train, y_train, X_test, y_test = temporal_train_test_split(
            sample_dataframe,
            train_size=80
        )
        
        # First train sample should be earlier than first test sample
        assert X_train['Time'].iloc[0] < X_test['Time'].iloc[0]
        # Last train sample should be earlier than last test sample
        assert X_train['Time'].iloc[-1] < X_test['Time'].iloc[-1]


class TestGetRandomTestCases:
    """Test random sampling functionality."""
    
    def test_sample_fraud_and_safe(self, sample_dataframe):
        """Test sampling fraud and safe cases."""
        X = sample_dataframe.drop('Class', axis=1)
        y = sample_dataframe['Class']
        
        fraud_cases, safe_cases = get_random_test_cases(X, y, n_fraud=1, n_safe=1)
        
        assert len(fraud_cases) == 1
        assert len(safe_cases) == 1
        assert isinstance(fraud_cases, pd.DataFrame)
        assert isinstance(safe_cases, pd.DataFrame)
    
    def test_sample_multiple_cases(self, sample_dataframe):
        """Test sampling multiple cases."""
        # Ensure we have enough fraud cases
        sample_dataframe.loc[0:5, 'Class'] = 1
        
        X = sample_dataframe.drop('Class', axis=1)
        y = sample_dataframe['Class']
        
        fraud_cases, safe_cases = get_random_test_cases(
            X, y, 
            n_fraud=3, 
            n_safe=5
        )
        
        assert len(fraud_cases) == 3
        assert len(safe_cases) == 5
    
    def test_sample_with_random_state(self, sample_dataframe):
        """Test reproducible sampling with random_state."""
        sample_dataframe.loc[0:5, 'Class'] = 1
        X = sample_dataframe.drop('Class', axis=1)
        y = sample_dataframe['Class']
        
        fraud1, safe1 = get_random_test_cases(X, y, random_state=42)
        fraud2, safe2 = get_random_test_cases(X, y, random_state=42)
        
        # Should get same samples with same random state
        pd.testing.assert_frame_equal(fraud1, fraud2)
        pd.testing.assert_frame_equal(safe1, safe2)


class TestLoadApiKey:
    """Test API key loading from environment."""
    
    @patch('utils.os.getenv')
    def test_load_api_key_success(self, mock_getenv):
        """Test successful API key loading."""
        mock_getenv.return_value = "test-api-key-123"
        
        key = load_api_key()
        
        assert key == "test-api-key-123"
        mock_getenv.assert_called_once_with("OPENROUTER_API_KEY")
    
    @patch('utils.os.getenv')
    def test_load_api_key_custom_var(self, mock_getenv):
        """Test loading from custom environment variable."""
        mock_getenv.return_value = "custom-key"
        
        key = load_api_key(env_var="CUSTOM_API_KEY")
        
        assert key == "custom-key"
        mock_getenv.assert_called_once_with("CUSTOM_API_KEY")
    
    @patch('utils.os.getenv')
    def test_load_api_key_not_found(self, mock_getenv):
        """Test error when API key is not found."""
        mock_getenv.return_value = None
        
        with pytest.raises(ValueError, match="API key not found"):
            load_api_key()


class TestFormatTransactionSummary:
    """Test transaction formatting."""
    
    def test_format_summary(self, sample_transaction):
        """Test transaction summary formatting."""
        transaction = pd.Series(sample_transaction)
        summary = format_transaction_summary(transaction)
        
        assert "Transaction Summary:" in summary
        assert "Amount:" in summary
        assert "Time:" in summary
        assert "V14:" in summary
        assert "V17:" in summary
        assert "$" in summary  # Currency symbol
    
    def test_format_includes_all_fields(self, fraud_transaction):
        """Test that all expected fields are in summary."""
        transaction = pd.Series(fraud_transaction)
        summary = format_transaction_summary(transaction)
        
        # Check specific values are formatted correctly
        assert f"${fraud_transaction['Amount']:.2f}" in summary
        assert f"{fraud_transaction['V14']:.2f}" in summary


class TestPrintModelPerformance:
    """Test model performance reporting."""
    
    def test_print_performance_above_benchmark(self, capsys):
        """Test performance report when exceeding benchmark."""
        print_model_performance(
            train_auc=0.9900,
            test_auc=0.9886,
            benchmark_auc=0.98
        )
        
        captured = capsys.readouterr()
        assert "MODEL PERFORMANCE REPORT" in captured.out
        assert "0.9900" in captured.out
        assert "0.9886" in captured.out
        assert "PRODUCTION-READY" in captured.out
    
    def test_print_performance_below_benchmark(self, capsys):
        """Test performance report when below benchmark."""
        print_model_performance(
            train_auc=0.9700,
            test_auc=0.9650,
            benchmark_auc=0.98
        )
        
        captured = capsys.readouterr()
        assert "Below benchmark" in captured.out
        assert "0.9650" in captured.out
    
    def test_print_performance_custom_benchmark(self, capsys):
        """Test with custom benchmark value."""
        print_model_performance(
            train_auc=0.9500,
            test_auc=0.9400,
            benchmark_auc=0.95
        )
        
        captured = capsys.readouterr()
        assert "0.9500" in captured.out
        assert "Benchmark:" in captured.out
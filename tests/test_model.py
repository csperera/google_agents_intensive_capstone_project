"""
Unit Tests for FraudDetectionModel
===================================
Tests model initialization, training, saving, loading, and predictions.

Authors: Ash Dehghan Ph.D and Cristian Perera
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import FraudDetectionModel, train_and_save_model


class TestFraudDetectionModel:
    """Test suite for FraudDetectionModel class."""
    
    def test_model_initialization(self):
        """Test that model initializes with correct hyperparameters."""
        model = FraudDetectionModel()
        
        assert model.model is not None
        assert model.model.n_estimators == 200
        assert model.model.max_depth == 6
        assert model.model.learning_rate == 0.05
        assert model.model.scale_pos_weight == 173
        assert model.model.random_state == 42
    
    def test_load_and_split_data(self, mock_csv_path):
        """Test data loading and temporal splitting."""
        model = FraudDetectionModel()
        X_train, y_train, X_test, y_test = model.load_and_split_data(
            mock_csv_path, 
            train_size=80
        )
        
        # Check splits
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check no Class column in features
        assert 'Class' not in X_train.columns
        assert 'Class' not in X_test.columns
        
        # Check target values are binary
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
    
    def test_predict_proba_shape(self, mock_xgboost_model, sample_dataframe):
        """Test that predict_proba returns correct shape."""
        model = FraudDetectionModel()
        model.model = mock_xgboost_model
        
        X = sample_dataframe.drop('Class', axis=1)
        predictions = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_model_save_and_load(self, temp_model_path, mock_xgboost_model):
        """Test model persistence (save and load)."""
        model = FraudDetectionModel()
        model.model = mock_xgboost_model
        
        # Save model
        model.save(str(temp_model_path))
        assert temp_model_path.exists()
        
        # Load model
        new_model = FraudDetectionModel()
        new_model.load(str(temp_model_path))
        assert new_model.model is not None
    
    def test_overfitting_check_minimal(self, capsys):
        """Test overfitting check with minimal gap."""
        model = FraudDetectionModel()
        metrics = {
            'train_auc': 0.9900,
            'test_auc': 0.9895,
            'overfitting_gap': 0.0005
        }
        
        model._print_overfitting_check(metrics)
        captured = capsys.readouterr()
        
        assert "Minimal overfitting" in captured.out
        assert "0.9900" in captured.out
        assert "0.9895" in captured.out
    
    def test_overfitting_check_significant(self, capsys):
        """Test overfitting check with significant gap."""
        model = FraudDetectionModel()
        metrics = {
            'train_auc': 1.0000,
            'test_auc': 0.9500,
            'overfitting_gap': 0.0500
        }
        
        model._print_overfitting_check(metrics)
        captured = capsys.readouterr()
        
        assert "Significant overfitting" in captured.out
    
    def test_train_returns_metrics(self, mock_csv_path, mock_xgboost_model):
        """Test that training returns proper metrics dictionary."""
        model = FraudDetectionModel()
        model.model = mock_xgboost_model
        
        X_train, y_train, X_test, y_test = model.load_and_split_data(
            mock_csv_path,
            train_size=80
        )
        
        # Mock the fit and predict_proba methods
        model.model.fit = Mock()
        model.model.predict_proba = Mock(side_effect=[
            np.column_stack([np.ones(80) * 0.1, np.ones(80) * 0.9]),  # train
            np.column_stack([np.ones(20) * 0.1, np.ones(20) * 0.9])   # test
        ])
        
        metrics = model.train(X_train, y_train, X_test, y_test, verbose=0)
        
        assert 'train_auc' in metrics
        assert 'test_auc' in metrics
        assert 'overfitting_gap' in metrics
        assert isinstance(metrics['train_auc'], float)
        assert 0 <= metrics['train_auc'] <= 1
        assert 0 <= metrics['test_auc'] <= 1


class TestTrainAndSaveModel:
    """Test the convenience function for training and saving."""
    
    @patch('model.FraudDetectionModel')
    def test_train_and_save_model_flow(self, mock_model_class, mock_csv_path, tmp_path):
        """Test the complete train and save workflow."""
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance
        
        # Mock methods
        mock_instance.load_and_split_data = Mock(return_value=(
            pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
        ))
        mock_instance.train = Mock(return_value={'train_auc': 0.99, 'test_auc': 0.98})
        mock_instance.save = Mock()
        
        # Call function
        result = train_and_save_model(
            data_path=mock_csv_path,
            model_path=str(tmp_path / "model.pkl")
        )
        
        # Verify calls
        mock_instance.load_and_split_data.assert_called_once()
        mock_instance.train.assert_called_once()
        mock_instance.save.assert_called_once()


# Integration test (optional - can be slow)
@pytest.mark.slow
def test_full_training_pipeline_small_dataset(mock_csv_path, tmp_path):
    """
    Integration test: Train on small dataset and verify it completes.
    Mark as 'slow' so it can be skipped in fast test runs.
    """
    model = FraudDetectionModel()
    model.model.n_estimators = 10  # Reduce for speed
    
    X_train, y_train, X_test, y_test = model.load_and_split_data(
        mock_csv_path,
        train_size=80
    )
    
    metrics = model.train(X_train, y_train, X_test, y_test, verbose=0)
    
    assert metrics['train_auc'] > 0
    assert metrics['test_auc'] > 0
    
    # Save and reload
    model_path = tmp_path / "integration_model.pkl"
    model.save(str(model_path))
    
    new_model = FraudDetectionModel()
    new_model.load(str(model_path))
    
    predictions = new_model.predict_proba(X_test)
    assert len(predictions) == len(X_test)
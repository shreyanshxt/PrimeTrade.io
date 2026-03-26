#!/usr/bin/env python3
"""
Unit tests for MLOps Task 0 batch job.
Run with: pytest test_run.py -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import yaml

import run


class TestLoadConfig(unittest.TestCase):
    """Test configuration loading and validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "config.yaml"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_config(self):
        """Test loading a valid config file."""
        config_data = {'seed': 42, 'window': 5, 'version': 'v1'}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = run.load_config(str(self.config_file))
        self.assertEqual(config['seed'], 42)
        self.assertEqual(config['window'], 5)
        self.assertEqual(config['version'], 'v1')

    def test_missing_config_file(self):
        """Test error when config file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            run.load_config(str(self.config_file))

    def test_missing_required_field_seed(self):
        """Test error when seed field is missing."""
        config_data = {'window': 5, 'version': 'v1'}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            run.load_config(str(self.config_file))
        self.assertIn('seed', str(ctx.exception))

    def test_missing_required_field_window(self):
        """Test error when window field is missing."""
        config_data = {'seed': 42, 'version': 'v1'}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            run.load_config(str(self.config_file))
        self.assertIn('window', str(ctx.exception))

    def test_missing_required_field_version(self):
        """Test error when version field is missing."""
        config_data = {'seed': 42, 'window': 5}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            run.load_config(str(self.config_file))
        self.assertIn('version', str(ctx.exception))

    def test_invalid_seed_type(self):
        """Test error when seed is not an integer."""
        config_data = {'seed': '42', 'window': 5, 'version': 'v1'}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            run.load_config(str(self.config_file))
        self.assertIn('seed', str(ctx.exception))

    def test_invalid_window_type(self):
        """Test error when window is not an integer."""
        config_data = {'seed': 42, 'window': '5', 'version': 'v1'}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            run.load_config(str(self.config_file))
        self.assertIn('window', str(ctx.exception))

    def test_invalid_window_value(self):
        """Test error when window is zero or negative."""
        config_data = {'seed': 42, 'window': 0, 'version': 'v1'}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            run.load_config(str(self.config_file))
        self.assertIn('window', str(ctx.exception))


class TestLoadData(unittest.TestCase):
    """Test data loading and validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_file = Path(self.temp_dir.name) / "data.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_data(self):
        """Test loading valid OHLCV data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        df.to_csv(self.data_file, index=False)

        loaded_df = run.load_data(str(self.data_file))
        self.assertEqual(len(loaded_df), 100)
        self.assertIn('close', loaded_df.columns)

    def test_missing_data_file(self):
        """Test error when data file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            run.load_data(str(self.data_file))

    def test_invalid_csv_format(self):
        """Test error with invalid CSV format."""
        with open(self.data_file, 'w') as f:
            f.write("this is not valid csv\n")
            f.write("broken,data,here\n")
            f.write("1,2,3,4,5,6,7,8,9,10\n")

        with self.assertRaises(ValueError):
            run.load_data(str(self.data_file))

    def test_empty_csv(self):
        """Test error with empty CSV file."""
        with open(self.data_file, 'w') as f:
            f.write("timestamp,open,high,low,close,volume\n")

        with self.assertRaises(ValueError) as ctx:
            run.load_data(str(self.data_file))
        self.assertIn('empty', str(ctx.exception).lower())

    def test_missing_close_column(self):
        """Test error when 'close' column is missing."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        df.to_csv(self.data_file, index=False)

        with self.assertRaises(ValueError) as ctx:
            run.load_data(str(self.data_file))
        self.assertIn('close', str(ctx.exception))


class TestComputeSignal(unittest.TestCase):
    """Test signal computation logic."""

    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.arange(100) + np.random.rand(100) * 5,  # Increasing trend
            'volume': np.random.randint(1000000, 5000000, 100)
        })

    def test_signal_computation(self):
        """Test basic signal computation."""
        signal, rows_processed, signal_rate = run.compute_signal(
            self.df.copy(), window=5, seed=42
        )
        
        self.assertEqual(len(signal), 100)
        self.assertEqual(rows_processed, 100)
        self.assertGreaterEqual(signal_rate, 0.0)
        self.assertLessEqual(signal_rate, 1.0)

    def test_signal_values(self):
        """Test that signal values are only 0 or 1."""
        signal, _, _ = run.compute_signal(
            self.df.copy(), window=5, seed=42
        )
        
        unique_values = set(signal)
        self.assertTrue(unique_values.issubset({0, 1}))

    def test_rolling_mean_calculation(self):
        """Test rolling mean is computed correctly."""
        df = self.df.copy()
        signal, _, _ = run.compute_signal(df, window=5, seed=42)
        
        # Check rolling mean exists and is reasonable
        self.assertIn('rolling_mean', df.columns)
        self.assertFalse(df['rolling_mean'].isna().all())

    def test_different_windows(self):
        """Test signal computation with different window sizes."""
        for window in [1, 3, 5, 10]:
            signal, rows_processed, signal_rate = run.compute_signal(
                self.df.copy(), window=window, seed=42
            )
            self.assertEqual(len(signal), 100)
            self.assertEqual(rows_processed, 100)

    def test_determinism(self):
        """Test that signal computation is deterministic."""
        df1 = self.df.copy()
        df2 = self.df.copy()
        
        signal1, _, rate1 = run.compute_signal(df1, window=5, seed=42)
        signal2, _, rate2 = run.compute_signal(df2, window=5, seed=42)
        
        np.testing.assert_array_equal(signal1, signal2)
        self.assertEqual(rate1, rate2)


class TestWriteMetrics(unittest.TestCase):
    """Test metrics output."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metrics_file = Path(self.temp_dir.name) / "metrics.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_success_metrics(self):
        """Test writing success metrics."""
        metrics = {
            'version': 'v1',
            'rows_processed': 10000,
            'metric': 'signal_rate',
            'value': 0.5,
            'latency_ms': 100,
            'seed': 42,
            'status': 'success'
        }
        
        run.write_metrics(metrics, str(self.metrics_file))
        
        self.assertTrue(self.metrics_file.exists())
        
        with open(self.metrics_file, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded, metrics)

    def test_write_error_metrics(self):
        """Test writing error metrics."""
        metrics = {
            'version': 'v1',
            'status': 'error',
            'error_message': 'Test error'
        }
        
        run.write_metrics(metrics, str(self.metrics_file))
        
        with open(self.metrics_file, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['status'], 'error')
        self.assertEqual(loaded['error_message'], 'Test error')

    def test_metrics_json_format(self):
        """Test that metrics are valid JSON."""
        metrics = {
            'version': 'v1',
            'rows_processed': 10000,
            'metric': 'signal_rate',
            'value': 0.6465,
            'latency_ms': 34,
            'seed': 42,
            'status': 'success'
        }
        
        run.write_metrics(metrics, str(self.metrics_file))
        
        # Try to parse as JSON
        with open(self.metrics_file, 'r') as f:
            json.load(f)  # Will raise if invalid


class TestIntegration(unittest.TestCase):
    """Integration tests for full job execution."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=1000, freq='h'),
            'open': np.random.rand(1000) * 100 + 100,
            'high': np.random.rand(1000) * 100 + 101,
            'low': np.random.rand(1000) * 100 + 99,
            'close': np.cumsum(np.random.randn(1000) * 0.5 + 0.1) + 100,
            'volume': np.random.randint(1000000, 5000000, 1000)
        })
        self.data_file = self.temp_path / "data.csv"
        df.to_csv(self.data_file, index=False)
        
        # Create test config
        config_data = {'seed': 42, 'window': 5, 'version': 'v1'}
        self.config_file = self.temp_path / "config.yaml"
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        self.metrics_file = self.temp_path / "metrics.json"
        self.log_file = self.temp_path / "run.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_execution(self):
        """Test full job execution with all components."""
        # Load config
        config = run.load_config(str(self.config_file))
        self.assertIsNotNone(config)
        
        # Load data
        df = run.load_data(str(self.data_file))
        self.assertEqual(len(df), 1000)
        
        # Compute signal
        signal, rows_processed, signal_rate = run.compute_signal(
            df, config['window'], config['seed']
        )
        self.assertEqual(rows_processed, 1000)
        
        # Create metrics
        metrics = {
            'version': config['version'],
            'rows_processed': rows_processed,
            'metric': 'signal_rate',
            'value': round(signal_rate, 4),
            'latency_ms': 100,
            'seed': config['seed'],
            'status': 'success'
        }
        
        # Write metrics
        run.write_metrics(metrics, str(self.metrics_file))
        
        # Verify metrics file exists and is valid
        self.assertTrue(self.metrics_file.exists())
        with open(self.metrics_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        self.assertEqual(loaded_metrics['status'], 'success')
        self.assertEqual(loaded_metrics['rows_processed'], 1000)


if __name__ == '__main__':
    unittest.main()

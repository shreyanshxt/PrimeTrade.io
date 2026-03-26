#!/usr/bin/env python3
"""
MLOps Task 0: Batch Signal Generation Job
Demonstrates reproducibility, observability, and deployment readiness.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def write_metrics(metrics: Dict[str, Any], output_file: str) -> None:
    """Write metrics to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_config(config_file: str) -> Dict[str, Any]:
    """
    load and validate configuration from YAML.
    args:
        config_file: Path to config.yaml
    returns:
        Validated config dictionary
    raises:
        FileNotFoundError: if config file doesn't exist
        ValueError: if required fields are missing
        yaml.YAMLError: if YAML is malformed
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"config file not found: {config_file}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML format: {e}")

    if not isinstance(config, dict):
        raise ValueError("config must be a YAML dictionary")

    # Validate required fields
    required_fields = ['seed', 'window', 'version']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"missing required config fields: {missing_fields}")

    # Validate field types
    if not isinstance(config['seed'], int):
        raise ValueError("'seed' must be an integer")
    if not isinstance(config['window'], int) or config['window'] < 1:
        raise ValueError("'window' must be a positive integer")
    if not isinstance(config['version'], str):
        raise ValueError("'version' must be a string")

    return config


def load_data(input_file: str) -> pd.DataFrame:
    """
    load and validate CSV data.
    args:
        input_file: Path to data.csv
    returns:
        validated DataFrame with 'close' column
    raises:
        FileNotFoundError: if input file doesn't exist
        ValueError: if file is empty, invalid format, or missing 'close' column
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except pd.errors.ParserError as e:
        raise ValueError(f"invalid CSV format: {e}")
    except Exception as e:
        raise ValueError(f"error reading CSV: {e}")

    if df.empty:
        raise ValueError("input CSV is empty")

    if 'close' not in df.columns:
        raise ValueError(
            f"missing required 'close' column. available columns: {list(df.columns)}"
        )

    return df


def compute_signal(df: pd.DataFrame, window: int, seed: int) -> tuple:
    """
    compute rolling mean and binary signal.
    args:
        df: DataFrame with 'close' column
        window: rolling window size
        seed: random seed for reproducibility
    returns:
    tuple of (signal array, rows_processed, signal_rate)
    """
    np.random.seed(seed)
    df['rolling_mean'] = df['close'].rolling(window=window, min_periods=1).mean()
    df['signal'] = (df['close'] > df['rolling_mean']).astype(int)

    rows_processed = len(df)
    signal_rate = df['signal'].mean()
    print("signal_rate",signal_rate)
    return df['signal'].values, rows_processed, signal_rate


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='MLOps Task 0: Batch Signal Generation'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (OHLCV data)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output metrics.json'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        required=True,
        help='Path to log file'
    )

    args = parser.parse_args()

    # Initialize logging
    logger = setup_logging(args.log_file)
    logger.info("="*60)
    logger.info("MLOps Task 0: Batch Signal Generation Job")
    logger.info(f"Started at {datetime.now().isoformat()}")
    logger.info("="*60)

    start_time = time.time()
    metrics = {
        'version': None,
        'status': 'error',
        'error_message': None
    }

    try:
        # Load and validate config
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
        logger.info(
            f"Config loaded: seed={config['seed']}, "
            f"window={config['window']}, version={config['version']}"
        )

        # Load and validate data
        logger.info(f"Loading data from {args.input}")
        df = load_data(args.input)
        logger.info(f"Data loaded successfully: {len(df)} rows")

        # Set seed for reproducibility
        np.random.seed(config['seed'])

        # Compute rolling mean and signal
        logger.info(
            f"Computing rolling mean (window={config['window']}) "
            f"and signal generation"
        )
        signal, rows_processed, signal_rate = compute_signal(
            df,
            config['window'],
            config['seed']
        )
        logger.info(
            f"Signal generation complete: "
            f"rows_processed={rows_processed}, "
            f"signal_rate={signal_rate:.4f}"
        )

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Build success metrics
        metrics = {
            'version': config['version'],
            'rows_processed': rows_processed,
            'metric': 'signal_rate',
            'value': round(signal_rate, 4),
            'latency_ms': latency_ms,
            'seed': config['seed'],
            'status': 'success'
        }

        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        logger.info(f"Total latency: {latency_ms}ms")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        metrics['error_message'] = str(e)
        metrics['version'] = 'v1'  # Set default version for error response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        metrics['error_message'] = str(e)
        metrics['version'] = 'v1'  # Set default version for error response

    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        metrics['error_message'] = f"{type(e).__name__}: {e}"
        metrics['version'] = 'v1'  # Set default version for error response

    finally:
        # Write metrics (both success and error cases)
        try:
            write_metrics(metrics, args.output)
            logger.info(f"Metrics written to {args.output}")
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
            sys.exit(1)

        # Print metrics to stdout
        print("\n" + "="*60)
        print("FINAL METRICS")
        print("="*60)
        print(json.dumps(metrics, indent=2))
        print("="*60 + "\n")

        logger.info(f"Job completed at {datetime.now().isoformat()}")
        logger.info("="*60)

        # Exit with appropriate code
        if metrics['status'] == 'success':
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()

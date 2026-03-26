# MLOps Task 0: Batch Signal Generation Job

> A production ready MLOps batch job demonstrating **reproducibility**, **observability**, and **deployment readiness**  processing OHLCV financial data to generate rolling mean based binary trading signals.

---

## Table of Contents

- [Overview](#overview)
- [What It Does](#what-it-does)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Run](#installation--run)
- [Configuration](#configuration)
- [Input Data Format](#input-data-format)
- [Output: metrics.json](#output-metricsjson)
- [Output: run.log](#output-runlog)
- [Signal Logic Explained](#signal-logic-explained)
- [Error Handling](#error-handling)
- [Docker Execution](#docker-execution)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Complete Solution Summary](#complete-solution-summary)

---

## Overview

This project mirrors real world trading-signal pipeline work with three core principles:

| Principle | Implementation |
|---|---|
| **Reproducibility** | Deterministic runs via fixed `seed` in `config.yaml` |
| **Observability** | Structured JSON metrics + timestamped log file |
| **Deployment Readiness** | Dockerized execution with no hardcoded paths |

---

## What It Does

```
data.csv ──► load_data() ──► compute_signal() ──► metrics.json
config.yaml ──► load_config() ──►                ──► run.log
```

1. Loads and validates `config.yaml` (seed, window, version)
2. Reads 10,000-row OHLCV data from `data.csv`
3. Computes **rolling mean** on the `close` price with configurable window
4. Generates **binary signals**: `1` if `close > rolling_mean`, else `0`
5. Writes structured metrics to `metrics.json`
6. Logs all operations to `run.log`

---

## Project Structure

```
files/
├── run.py              # Main batch job — signal computation & metrics output
├── test_run.py         # 22 unit + integration tests
├── config.yaml         # Job configuration (seed, window, version)
├── data.csv            # OHLCV market data (10,000 rows)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── metrics.json        # Sample output — structured metrics
├── run.log             # Sample output — execution log
├── ci-cd.yml           # GitHub Actions CI/CD pipeline
└── README.md           # This file
```

### Key File Responsibilities

| File | Role |
|---|---|
| `run.py` | Entry point — argparse CLI, config loading, data loading, signal generation, metrics writing |
| `config.yaml` | Controls `seed` (reproducibility), `window` (rolling mean), `version` (job ID) |
| `data.csv` | Input OHLCV dataset with `close` column required |
| `metrics.json` | Machine-readable output with `signal_rate`, `rows_processed`, `latency_ms` |
| `run.log` | Human-readable execution trace (DEBUG to file, INFO to console) |
| `test_run.py` | Full test suite covering happy path, edge cases, and error scenarios |
| `Dockerfile` | Containerizes the job for portable, reproducible deployment |

---

## Requirements

- Python **3.9+** (tested on Python 3.14)
- `numpy >= 1.20.0`
- `pandas >= 1.3.0`
- `PyYAML >= 5.4.0`
- Docker *(for containerized execution)*

---

## Installation & Run

### 1. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the Job

```bash
python3 run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

### CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--input` | ✅ | Path to OHLCV CSV file |
| `--config` | ✅ | Path to YAML configuration file |
| `--output` | ✅ | Path to write output `metrics.json` |
| `--log-file` | ✅ | Path to write execution log |

### 3. Check Results

```bash
cat metrics.json   # View output metrics
cat run.log        # View execution log
```

---

## Configuration

**`config.yaml`:**

```yaml
seed: 42       # Random seed — ensures deterministic output
window: 5      # Rolling window size for mean calculation
version: "v1"  # Job version tag (appears in metrics output)
```

**Validation rules:**
- `seed` must be an integer
- `window` must be a positive integer
- `version` must be a string
- All three fields are required

---

## Input Data Format

**`data.csv`** — Required columns:

| Column | Description |
|---|---|
| `timestamp` | Datetime of the candle |
| `open` | Opening price |
| `high` | High price |
| `low` | Low price |
| `close` | **Closing price — required for signal calculation** |
| `volume` | Trading volume |

The job validates:
- File exists and is readable
- CSV is properly formatted (not empty, not malformed)
- `close` column is present

---

## Output: metrics.json

### Success

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.6465,
  "latency_ms": 34,
  "seed": 42,
  "status": "success"
}
```

### Field Reference

| Field | Source | Description |
|---|---|---|
| `version` | `config.yaml` | Job version identifier |
| `rows_processed` | `len(df)` | Total rows from CSV |
| `metric` | Hardcoded | Always `"signal_rate"` |
| `value` | `df['signal'].mean()` | Fraction of rows where `close > rolling_mean` |
| `latency_ms` | `time.time()` delta | Wall-clock job duration in milliseconds |
| `seed` | `config.yaml` | Seed used for reproducibility |
| `status` | Exception handling | `"success"` or `"error"` |

### Error Output

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

---

## Output: run.log

```
2026-03-26 16:08:53 - INFO - ============================================================
2026-03-26 16:08:53 - INFO - MLOps Task 0: Batch Signal Generation Job
2026-03-26 16:08:53 - INFO - Started at 2026-03-26T16:08:53.670667
2026-03-26 16:08:53 - INFO - Config loaded: seed=42, window=5, version=v1
2026-03-26 16:08:53 - INFO - Data loaded successfully: 10000 rows
2026-03-26 16:08:53 - INFO - Signal generation complete: rows_processed=10000, signal_rate=0.6465
2026-03-26 16:08:53 - INFO - Total latency: 16ms
2026-03-26 16:08:53 - INFO - Metrics written to metrics.json
2026-03-26 16:08:53 - INFO - Job completed at 2026-03-26T16:08:53.687911
```

- **File handler**: DEBUG level (all internal details)
- **Console handler**: INFO level (key milestones only)

---

## Signal Logic Explained

```python
# Step 1: Rolling mean with min_periods=1 (no NaN rows)
df['rolling_mean'] = df['close'].rolling(window=5, min_periods=1).mean()

# Step 2: Binary signal — 1 if bullish (close above mean), 0 if not
df['signal'] = (df['close'] > df['rolling_mean']).astype(int)

# Step 3: Signal rate — fraction of bullish signals
signal_rate = df['signal'].mean()   # e.g. 0.6465 = 64.65% bullish
```

**Why `min_periods=1`?** The first `window-1` rows don't have enough data for a full window. `min_periods=1` uses whatever rows are available instead of producing `NaN`, ensuring all rows are included in signal generation.

---

## Error Handling

| Error Case | Behaviour |
|---|---|
| Missing input file | `FileNotFoundError` logged → error metrics written → exit code 1 |
| Missing config file | `FileNotFoundError` logged → error metrics written → exit code 1 |
| Invalid CSV format | `ValueError` logged → error metrics written → exit code 1 |
| Empty CSV | `ValueError` logged → error metrics written → exit code 1 |
| Missing `close` column | `ValueError` logged → error metrics written → exit code 1 |
| Malformed YAML | `ValueError` logged → error metrics written → exit code 1 |
| Missing config fields | `ValueError` logged → error metrics written → exit code 1 |
| Unexpected exceptions | Exception type + message logged → error metrics written → exit code 1 |

> **Guarantee**: `metrics.json` and `run.log` are **always** written via `try/finally`, even on failure.

---

## Docker Execution

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

### Run with Custom Arguments

```bash
docker run --rm mlops-task python3 run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

### Extract Output Files

```bash
docker run --rm -v $(pwd)/output:/app/output mlops-task \
  cp metrics.json run.log /app/output/
```

Exit code `0` = success, non-zero = failure.

---

## Testing

### Run All Tests

```bash
python3 test_run.py
```

**22 tests** covering:
- Config loading (valid, missing fields, wrong types, malformed YAML)
- Data loading (valid CSV, empty CSV, missing `close` column, bad format)
- Signal computation (correctness, edge cases, all-zero/all-one signals)
- Metrics writing
- Full integration (end-to-end execution)

### Expected Output

```
Ran 22 tests in 0.146s
OK
```

> **Note:** Requires `pandas >= 2.0`. Frequency aliases must be lowercase (`'h'` not `'H'`).

---

## Reproducibility

Running the job twice with the same `config.yaml` and `data.csv` always produces identical `metrics.json`:

```bash
python3 run.py --input data.csv --config config.yaml --output m1.json --log-file l1.log
python3 run.py --input data.csv --config config.yaml --output m2.json --log-file l2.log
diff m1.json m2.json   # No output = identical
```

Determinism is guaranteed by:
- Fixed `seed` in config → `np.random.seed(seed)` is set
- Rolling mean is purely deterministic (no randomness in core logic)
- Same input always produces same output

---

## Complete Solution Summary

### Architecture

```
CLI Args ──► argparse
                │
                ├──► load_config(config.yaml)   # Validates: seed, window, version
                │
                ├──► load_data(data.csv)         # Validates: exists, has 'close', non-empty
                │
                ├──► compute_signal(df, window, seed)
                │         │
                │         ├── rolling_mean = close.rolling(window).mean()
                │         ├── signal = (close > rolling_mean).astype(int)
                │         └── signal_rate = signal.mean()
                │
                ├──► Build metrics dict
                │
                ├──► write_metrics() ──► metrics.json
                │
                └──► Logging ──► run.log + stdout
```

### Design Decisions

1. **`try/finally` for metrics writing** — ensures output is always written, even on error
2. **`min_periods=1`** — avoids NaN signals for first `window-1` rows
3. **Dual logging** — DEBUG to file (full trace), INFO to console (clean output)
4. **Argparse CLI** — no hardcoded paths, fully portable
5. **Config validation** — fails fast with clear error messages before processing begins
6. **Exit codes** — `0` on success, `1` on any error (CI/CD friendly)

### Performance

| Metric | Value |
|---|---|
| Throughput | ~10,000 rows in 7–50ms |
| Memory | ~50–100MB for 10,000 OHLCV rows |
| Test coverage | 22 tests, all passing |

---

*Built for MLOps Task 0 — demonstrating production-grade batch ML pipeline design.*

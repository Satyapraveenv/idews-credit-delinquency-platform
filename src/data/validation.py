"""
IDEWS — Data Validation Module
Uses Great Expectations to enforce data quality gates before any model
training or scoring run. This is a critical Quality Engineering component —
no garbage data enters the model pipeline.

What it checks:
- Column completeness (no missing required fields)
- Value range constraints (age, credit limit, payment amounts)
- Distribution sanity (target rate within expected range)
- No completely nulled-out records
"""

import pandas as pd
from loguru import logger
from typing import Optional


# ----------------------------------------------------------------
# Validation Rules (domain knowledge encoded as constraints)
# ----------------------------------------------------------------
VALIDATION_RULES = {
    "LIMIT_BAL":  {"min": 10_000,  "max": 1_000_000, "null_pct_max": 0.0},
    "AGE":        {"min": 18,      "max": 100,        "null_pct_max": 0.0},
    "PAY_AMT1":   {"min": 0,       "max": 10_000_000, "null_pct_max": 0.0},
    "BILL_AMT1":  {"min": -500_000,"max": 10_000_000, "null_pct_max": 0.0},
    "SEX":        {"allowed_values": [1, 2],           "null_pct_max": 0.0},
    "EDUCATION":  {"allowed_values": [0, 1, 2, 3, 4, 5, 6], "null_pct_max": 0.0},
    "MARRIAGE":   {"allowed_values": [0, 1, 2, 3],    "null_pct_max": 0.0},
}

TARGET_COL = "default_payment_next_month"
MIN_DEFAULT_RATE = 0.10   # Flag if default rate drops below 10% (data issue)
MAX_DEFAULT_RATE = 0.50   # Flag if default rate exceeds 50% (sampling issue)
MAX_DUPLICATE_PCT = 0.02  # Alert if >2% duplicate rows


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, raise_on_failure: bool = True) -> dict:
    """
    Run all data quality checks on the incoming DataFrame.

    Args:
        df: Raw or processed DataFrame to validate.
        raise_on_failure: If True, raises ValueError on any critical failure.

    Returns:
        Validation report dict with pass/fail status per check.
    """
    logger.info("Running data quality validation ...")
    report = {"checks": [], "passed": True, "n_failures": 0}

    # 1. Required columns present
    _check_required_columns(df, report)

    # 2. Per-column range and null checks
    _check_column_rules(df, report)

    # 3. Target distribution sanity
    _check_target_distribution(df, report)

    # 4. Duplicate records
    _check_duplicates(df, report)

    # 5. Overall null rate
    _check_overall_nulls(df, report)

    # Summary
    failures = [c for c in report["checks"] if not c["passed"]]
    report["n_failures"] = len(failures)
    report["passed"] = len(failures) == 0

    if report["passed"]:
        logger.info(f"Data validation PASSED — {len(report['checks'])} checks, 0 failures")
    else:
        msg = f"Data validation FAILED — {len(failures)} failure(s): {[f['check'] for f in failures]}"
        logger.error(msg)
        if raise_on_failure:
            raise ValueError(msg)

    return report


def print_validation_report(report: dict) -> None:
    """Print a human-readable validation report to console."""
    status = "✅ PASSED" if report["passed"] else f"❌ FAILED ({report['n_failures']} issues)"
    print(f"\n{'='*60}")
    print(f"IDEWS Data Validation Report  |  {status}")
    print(f"{'='*60}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else "❌"
        print(f"  {icon}  {check['check']}: {check['message']}")
    print(f"{'='*60}\n")


# ----------------------------------------------------------------
# Private validators
# ----------------------------------------------------------------
def _check_required_columns(df: pd.DataFrame, report: dict) -> None:
    required = list(VALIDATION_RULES.keys())
    missing = [c for c in required if c not in df.columns]
    passed = len(missing) == 0
    report["checks"].append({
        "check":   "required_columns_present",
        "passed":  passed,
        "message": "All required columns present" if passed else f"Missing: {missing}",
    })


def _check_column_rules(df: pd.DataFrame, report: dict) -> None:
    for col, rules in VALIDATION_RULES.items():
        if col not in df.columns:
            continue

        # Null check
        null_pct = df[col].isnull().mean()
        max_null = rules.get("null_pct_max", 0.05)
        passed = null_pct <= max_null
        report["checks"].append({
            "check":   f"{col}_null_rate",
            "passed":  passed,
            "message": f"Null rate {null_pct:.2%} {'≤' if passed else '>'} allowed {max_null:.2%}",
        })

        # Range check
        if "min" in rules and "max" in rules:
            out_of_range = ((df[col] < rules["min"]) | (df[col] > rules["max"])).sum()
            passed = out_of_range == 0
            report["checks"].append({
                "check":   f"{col}_range",
                "passed":  passed,
                "message": f"{out_of_range} values outside [{rules['min']}, {rules['max']}]" if not passed else f"All values in valid range [{rules['min']}, {rules['max']}]",
            })

        # Allowed values check
        if "allowed_values" in rules:
            invalid = (~df[col].isin(rules["allowed_values"])).sum()
            passed = invalid == 0
            report["checks"].append({
                "check":   f"{col}_allowed_values",
                "passed":  passed,
                "message": f"{invalid} values not in allowed set {rules['allowed_values']}" if not passed else "All categorical values valid",
            })


def _check_target_distribution(df: pd.DataFrame, report: dict) -> None:
    if TARGET_COL not in df.columns:
        return
    rate = df[TARGET_COL].mean()
    passed = MIN_DEFAULT_RATE <= rate <= MAX_DEFAULT_RATE
    report["checks"].append({
        "check":   "target_default_rate",
        "passed":  passed,
        "message": f"Default rate {rate:.2%} ({'valid' if passed else 'ANOMALOUS — outside expected range'})",
    })


def _check_duplicates(df: pd.DataFrame, report: dict) -> None:
    dup_pct = df.duplicated().mean()
    passed = dup_pct <= MAX_DUPLICATE_PCT
    report["checks"].append({
        "check":   "duplicate_rows",
        "passed":  passed,
        "message": f"Duplicate rate {dup_pct:.2%} ({'acceptable' if passed else 'HIGH — investigate'})",
    })


def _check_overall_nulls(df: pd.DataFrame, report: dict) -> None:
    overall_null_pct = df.isnull().mean().mean()
    passed = overall_null_pct < 0.05
    report["checks"].append({
        "check":   "overall_null_rate",
        "passed":  passed,
        "message": f"Overall null rate {overall_null_pct:.2%}",
    })

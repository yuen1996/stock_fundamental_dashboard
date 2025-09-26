"""Utilities for fetching dividend yields across the application."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

try:  # Robust imports when running within Streamlit or scripts
    from utils import io_helpers
except Exception:  # pragma: no cover - fallback for legacy entrypoints
    import io_helpers  # type: ignore

try:
    from utils import calculations as _calc
except Exception:  # pragma: no cover - fallback if utils package not used
    import calculations as _calc  # type: ignore

_DIVIDEND_LABELS = [
    "Dividend Yield (%)",
    "Distribution/Dividend Yield (%)",
    "Distribution Yield (%)",
    "Dividend Yield",
    "DY (%)",
    "DY",
]

_PRICE_COLUMNS = [
    "CurrentPrice",
    "SharePrice",
    "Price",
    "Annual Price per Share (RM)",
]


def _normalise_name(name: str) -> str:
    return str(name or "").strip().upper()


def _annual_only(stock_df: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty:
        return stock_df.copy()
    if "IsQuarter" in stock_df.columns:
        annual = stock_df[stock_df["IsQuarter"] != True].copy()
    else:
        annual = stock_df.copy()
    if "Year" in annual.columns:
        annual["Year"] = pd.to_numeric(annual["Year"], errors="coerce")
        annual = annual.sort_values("Year")
    return annual


def _quarterly_only(stock_df: pd.DataFrame) -> pd.DataFrame:
    if "IsQuarter" not in stock_df.columns:
        return pd.DataFrame(columns=stock_df.columns)
    qtr = stock_df[stock_df["IsQuarter"] == True].copy()
    if "Year" in qtr.columns:
        qtr["Year"] = pd.to_numeric(qtr["Year"], errors="coerce")
    if "Quarter" in qtr.columns:
        qtr["Quarter"] = pd.to_numeric(qtr["Quarter"], errors="coerce")
    return qtr


def _last_numeric(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _price_fallback(stock_df: pd.DataFrame) -> Optional[float]:
    for col in _PRICE_COLUMNS:
        if col in stock_df.columns:
            val = _last_numeric(stock_df[col])
            if val is not None:
                return val
    return None


def _find_column(df: pd.DataFrame, labels: list[str]) -> Optional[str]:
    if df.empty:
        return None
    import re

    norm = {
        re.sub(r"[^0-9a-z]+", "", str(col).lower()): col for col in df.columns
    }
    for label in labels:
        key = re.sub(r"[^0-9a-z]+", "", str(label).lower())
        if key in norm:
            return norm[key]
    for label in labels:
        key = re.sub(r"[^0-9a-z]+", "", str(label).lower())
        if not key:
            continue
        for norm_key, col in norm.items():
            if key in norm_key:
                return col
    return None


def _latest_fy_value(sum_df: pd.DataFrame, metric: str) -> Optional[float]:
    if sum_df is None or sum_df.empty:
        return None
    import re

    fy_cols = [
        col
        for col in sum_df.columns
        if isinstance(col, str) and re.match(r"^(FY\s*)?\d{4}$", col.strip(), flags=re.I)
    ]
    if not fy_cols:
        return None
    metric_rows = sum_df[sum_df["Metric"] == metric]
    if metric_rows.empty:
        return None
    for col in reversed(fy_cols):
        s = pd.to_numeric(metric_rows[col], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[0])
    return None


def _ttm_value(sum_df: pd.DataFrame, metric: str) -> Optional[float]:
    if sum_df is None or sum_df.empty:
        return None
    ttm_col = next(
        (
            col
            for col in reversed(sum_df.columns)
            if isinstance(col, str) and col.upper().startswith("TTM")
        ),
        None,
    )
    if not ttm_col:
        return None
    metric_rows = sum_df[sum_df["Metric"] == metric]
    if metric_rows.empty:
        return None
    s = pd.to_numeric(metric_rows[ttm_col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[0])


def _data_version_token() -> str:
    """Return a token that changes whenever the fundamentals dataset updates."""
    path = getattr(io_helpers, "DATA_FILE", None)
    if isinstance(path, str):
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return "missing"
        except Exception:
            return "unknown"
        else:
            return f"{stat.st_mtime_ns}-{stat.st_size}"
    return "unknown"


@lru_cache(maxsize=256)
def _latest_dividend_yield_cached(
    norm_name: str, data_version: str
) -> Optional[float]:
    df = io_helpers.load_data()
    if df is None or df.empty or "Name" not in df.columns:
        return None

    stock_rows = df[df["Name"].astype(str).str.upper() == norm_name]
    if stock_rows.empty:
        return None

    bucket = None
    if "IndustryBucket" in stock_rows.columns:
        bucket_series = stock_rows["IndustryBucket"].dropna()
        if not bucket_series.empty:
            bucket = str(bucket_series.iloc[0])

    annual = _annual_only(stock_rows)
    quarterly = _quarterly_only(stock_rows)
    price = _price_fallback(stock_rows)

    try:
        sum_df = _calc.build_summary_table(
            annual_df=annual,
            quarterly_df=quarterly,
            bucket=bucket or "General",
            include_ttm=True,
            price_fallback=price,
        )
    except Exception:
        sum_df = None

    if isinstance(sum_df, pd.DataFrame) and not sum_df.empty:
        for label in (
            "Dividend Yield (%)",
            "Distribution Yield (%)",
            "Distribution/Dividend Yield (%)",
        ):
            ttm_val = _ttm_value(sum_df, label)
            if ttm_val is not None:
                return ttm_val
        for label in (
            "Dividend Yield (%)",
            "Distribution Yield (%)",
            "Distribution/Dividend Yield (%)",
        ):
            fy_val = _latest_fy_value(sum_df, label)
            if fy_val is not None:
                return fy_val

    # Direct fallback to raw columns if summary is unavailable
    col = _find_column(annual, _DIVIDEND_LABELS) or _find_column(
        stock_rows, _DIVIDEND_LABELS
    )
    if col:
        val = _last_numeric(stock_rows[col])
        if val is not None:
            return val

    return None


def get_latest_dividend_yield(
    name: str, *, data_version: Optional[str] = None
) -> Optional[float]:
    """Return the latest dividend yield percentage for *name* if available."""
    norm_name = _normalise_name(name)
    if not norm_name:
        return None

    version = data_version or _data_version_token()
    return _latest_dividend_yield_cached(norm_name, version)


def dividend_data_version() -> str:
    """Expose the current dividend data version for caching call sites."""
    return _data_version_token()
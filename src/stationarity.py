"""Stationarity diagnostics: Augmented Dickey-Fuller test and rolling statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFResult:
    """Container for an Augmented Dickey-Fuller test result."""
    name: str
    statistic: float
    pvalue: float
    n_lags: int
    n_obs: int
    crit_values: dict = field(default_factory=dict)
    regression: str = "c"

    @property
    def is_stationary(self) -> bool:
        """Reject the unit-root null (non-stationary) at the 5% level -> stationary."""
        return self.pvalue < 0.05

    @property
    def order_of_integration(self) -> str:
        """Coarse label used in the report."""
        return "I(0) estacionaria" if self.is_stationary else "I(1)/no estacionaria"


def adf_test(series: pd.Series, name: str | None = None,
             regression: str = "c") -> ADFResult:
    """Run the ADF test on a single series and return a structured result.

    ``regression='c'`` tests against a constant-mean alternative (H0: unit root).
    """
    s = pd.Series(series).dropna()
    stat, pval, nlags, nobs, crit, _ = adfuller(s, regression=regression, autolag="AIC")
    return ADFResult(
        name=name or (series.name if hasattr(series, "name") else "series"),
        statistic=float(stat), pvalue=float(pval), n_lags=int(nlags),
        n_obs=int(nobs), crit_values={k: float(v) for k, v in crit.items()},
        regression=regression,
    )


def adf_table(df: pd.DataFrame, columns: list[str],
              names: dict | None = None) -> pd.DataFrame:
    """Run ADF on several columns and return a tidy summary DataFrame."""
    rows = []
    for col in columns:
        r = adf_test(df[col], name=col)
        rows.append({
            "Variable": col,
            "Nombre": (names or {}).get(col, ""),
            "ADF_stat": round(r.statistic, 4),
            "p_value": round(r.pvalue, 5),
            "Estacionaria (5%)": r.is_stationary,
            "Orden": r.order_of_integration,
        })
    return pd.DataFrame(rows)


def rolling_stats(series: pd.Series, window: int = 50) -> pd.DataFrame:
    """Return rolling mean and rolling variance over ``window`` records."""
    s = pd.Series(series)
    return pd.DataFrame({
        "rolling_mean": s.rolling(window).mean(),
        "rolling_var": s.rolling(window).var(),
    }, index=s.index)


def classify_drift_vs_randomwalk(series: pd.Series, window: int = 50) -> dict:
    """Heuristic to distinguish a Random Walk from a Random Walk *with Drift*.

    A drift shows a persistent, roughly linear trend in the rolling mean (large,
    consistent-sign slope). A pure random walk wanders with no systematic direction,
    so the average first difference is ~0 relative to its dispersion.
    """
    s = pd.Series(series).dropna()
    diffs = s.diff().dropna()
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std())
    # slope of the rolling mean via simple linear fit on the time order
    rmean = s.rolling(window).mean().dropna()
    x = np.arange(len(rmean))
    slope = float(np.polyfit(x, rmean.values, 1)[0]) if len(rmean) > 1 else 0.0
    # signal-to-noise of the drift term
    drift_snr = abs(mean_diff) / std_diff if std_diff > 0 else 0.0
    verdict = "Random Walk con Drift" if drift_snr > 0.05 else "Random Walk puro"
    return {
        "mean_first_diff": mean_diff,
        "std_first_diff": std_diff,
        "drift_snr": drift_snr,
        "rolling_mean_slope": slope,
        "verdict": verdict,
    }

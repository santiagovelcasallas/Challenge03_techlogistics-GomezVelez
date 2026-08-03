"""Data loading and synthetic time-index utilities.

The delivered CSVs contain NO timestamp column, so for any time-series analysis we
construct a *synthetic* hourly ``DatetimeIndex`` (documented assumption). All loaders
optionally attach that index so downstream ARIMA/ADF/FFT code sees a proper time axis.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Fixed anchor date for the synthetic time axis (documented assumption).
SYNTHETIC_START = "2023-01-01 00:00:00"
SYNTHETIC_FREQ = "h"  # hourly

# Human-readable names taken from the variable dictionary. Used in plots/reports.
AGRO_NAMES = {
    "Agro_1": "Humedad del suelo",
    "Agro_2": "Evapotranspiración",
    "Agro_3": "Humedad Relativa (RH)",
    "Agro_4": "Radiación PAR",
    "Agro_5": "NDVI",
    "Agro_6": "Biomasa",
    "Agro_7": "Índice biótico 3",
    "Agro_8": "Textura de suelo",
    "Agro_9": "Viento (comp.)",
    "Agro_10": "Varianza del viento",
}
ENER_NAMES = {
    "Ener_1": "Demanda",
    "Ener_2": "Precio spot",
    "Ener_3": "Temperatura",
    "Ener_4": "Generación Eólica",
    "Ener_5": "Costo del Gas",
    "Ener_6": "Emisiones CO2",
    "Ener_7": "Factor macro 3",
    "Ener_8": "Frecuencia",
    "Ener_9": "Voltaje",
    "Ener_10": "Factor de Potencia",
}


def make_synthetic_index(n: int, start: str = SYNTHETIC_START,
                         freq: str = SYNTHETIC_FREQ) -> pd.DatetimeIndex:
    """Return a synthetic hourly DatetimeIndex of length ``n``.

    There is no timestamp in the source data; we assume equally-spaced hourly samples
    starting at a fixed anchor date so the analysis is fully reproducible.
    """
    return pd.date_range(start=start, periods=n, freq=freq)


def load_dataset(path: str | Path, add_time_index: bool = True) -> pd.DataFrame:
    """Load one CSV and optionally attach the synthetic hourly time index."""
    df = pd.read_csv(path)
    if add_time_index:
        df.index = make_synthetic_index(len(df))
        df.index.name = "timestamp"
    return df


def load_pair(data_dir: str | Path, prefix: str,
              add_time_index: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the ``{prefix}_clean.csv`` and ``{prefix}_noise.csv`` pair.

    Returns ``(clean, noise)``. Although the brief assumed only ``*_noise`` files were
    delivered, the real ``*_clean`` ground truth IS present, so we use it directly as
    the reference signal (corrected assumption, documented in the report).
    """
    data_dir = Path(data_dir)
    clean = load_dataset(data_dir / f"{prefix}_clean.csv", add_time_index)
    noise = load_dataset(data_dir / f"{prefix}_noise.csv", add_time_index)
    return clean, noise


def value_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return the ordered list of measurement columns (``{prefix}_1..{prefix}_10``)."""
    return [c for c in df.columns if c.startswith(f"{prefix}_")]


def set_seeds(seed: int = 42) -> None:
    """Fix numpy's RNG seed for reproducibility."""
    np.random.seed(seed)

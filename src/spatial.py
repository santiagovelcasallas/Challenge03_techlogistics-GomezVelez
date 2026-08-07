"""Autocorrelación espacial: I de Moran global con test de permutación.

La I de Moran mide si valores parecidos están cerca en el espacio, usando pesos de
k vecinos más cercanos. Se interpreta como una correlación:
    I > 0  -> clustering (valores parecidos juntos)
    I ~ 0  -> distribución aleatoria (sin patrón espacial)
    I < 0  -> dispersión tipo tablero de ajedrez
Su valor esperado bajo aleatoriedad es E[I] = -1/(n-1) (tiende a 0 al crecer n).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _knn_index(lat, lon, k: int) -> np.ndarray:
    """Índices de los k vecinos más cercanos de cada punto (excluye el propio punto)."""
    X = np.c_[np.asarray(lat, float), np.asarray(lon, float)]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def _moran_from_idx(values: np.ndarray, idx: np.ndarray) -> float:
    """I de Moran dado el vector de valores y la matriz de vecinos (pesos binarios)."""
    n, k = idx.shape
    z = values - values.mean()
    num = np.sum(z[:, None] * z[idx])   # sum_i z_i * sum_j z_{vecino(i,j)}
    W = n * k                            # cada fila aporta k pesos unitarios
    denom = np.sum(z ** 2)
    return float((n / W) * (num / denom))


def morans_i(lat, lon, values, k: int = 8) -> tuple[float, float]:
    """I de Moran global con k vecinos. Devuelve (I, E[I]) con E[I] = -1/(n-1)."""
    val = np.asarray(values, float)
    idx = _knn_index(lat, lon, k)
    return _moran_from_idx(val, idx), -1.0 / (len(val) - 1)


def morans_i_by_k(lat, lon, values, ks=(4, 8, 15, 25, 50)) -> dict:
    """Tabla de sensibilidad: I de Moran para distintos números de vecinos k."""
    val = np.asarray(values, float)
    return {k: _moran_from_idx(val, _knn_index(lat, lon, k)) for k in ks}


def morans_i_perm(lat, lon, values, k: int = 8, n_perm: int = 999,
                  seed: int = 42) -> dict:
    """I de Moran + p-value por permutación (barajando las etiquetas, fija la geometría).

    p-value pseudo de dos colas (estilo PySAL): fracción de permutaciones cuya desviación
    respecto a E[I] iguala o supera la observada, con corrección (+1)/(n_perm+1).
    """
    rng = np.random.default_rng(seed)
    val = np.asarray(values, float)
    idx = _knn_index(lat, lon, k)
    E_I = -1.0 / (len(val) - 1)
    I_obs = _moran_from_idx(val, idx)
    perms = np.array([_moran_from_idx(rng.permutation(val), idx) for _ in range(n_perm)])
    ge = int(np.sum(np.abs(perms - E_I) >= np.abs(I_obs - E_I)))
    p = (ge + 1) / (n_perm + 1)
    return {"I": I_obs, "E_I": E_I, "p_perm": p, "n_perm": n_perm,
            "perm_mean": float(perms.mean()), "perm_std": float(perms.std())}

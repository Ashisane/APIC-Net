"""
Φ Tensor — Feature Extraction

Extracts features from each 4×4 Φ matrix AND raw residual magnitudes,
then aggregates over a sliding window (mean + std).

Features per timestep:
  6 from Φ matrix: off_diag_std, frobenius_norm, trace, effective_rank,
                   condition_number, max_off_diag
  4 from raw residuals: |r1|, |r2|, |r3|, |r4|
  Total: 10 per timestep

Window aggregation: mean + std → 20-dim MLP input.
"""
import numpy as np


def extract_features_single(Phi, residuals=None):
    """
    Extract scalar features from a single 4×4 Φ matrix + optional raw residuals.

    Args:
        Phi: (4, 4) matrix.
        residuals: optional (4,) raw residual values.

    Returns:
        features: (10,) array if residuals provided, (6,) otherwise.
    """
    n = Phi.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = Phi[mask]

    # Singular values for rank/condition
    sv = np.linalg.svd(Phi, compute_uv=False)
    sv_sum = sv.sum() + 1e-10
    sv_sq_sum = (sv ** 2).sum() + 1e-10

    phi_features = np.array([
        np.std(off_diag),                       # 0: off_diag_std (primary)
        np.linalg.norm(Phi, 'fro'),             # 1: frobenius_norm
        np.trace(Phi),                          # 2: trace
        (sv_sum ** 2) / sv_sq_sum,              # 3: effective_rank
        sv[0] / (sv[-1] + 1e-10),              # 4: condition_number
        np.max(np.abs(off_diag)),               # 5: max_off_diag
    ])

    if residuals is not None:
        raw_features = np.abs(residuals)  # |r1|, |r2|, |r3|, |r4|
        return np.concatenate([phi_features, raw_features])
    return phi_features


FEATURE_NAMES = [
    "off_diag_std", "frobenius_norm", "trace",
    "effective_rank", "condition_number", "max_off_diag",
    "abs_r1", "abs_r2", "abs_r3", "abs_r4",
]
N_FEATURES = len(FEATURE_NAMES)  # 10


def extract_features_batch(Phi_series, residuals_series):
    """
    Extract features for an entire time series of Φ matrices + residuals.

    Args:
        Phi_series: (T, 4, 4) — Φ matrices over time.
        residuals_series: (T, 4) — raw residuals over time.

    Returns:
        features: (T, 10) — feature time series.
    """
    T = Phi_series.shape[0]
    features = np.zeros((T, N_FEATURES))

    for t in range(T):
        features[t] = extract_features_single(Phi_series[t], residuals_series[t])

    return features


def aggregate_window(features, window_size=20):
    """
    Aggregate features over a sliding window using mean + std.

    Args:
        features: (T, 10) — per-timestep features.
        window_size: number of timesteps per window.

    Returns:
        aggregated: (N_windows, 20) — [mean_f1..f10, std_f1..f10]
        window_indices: (N_windows,) — starting index of each window.
    """
    T = features.shape[0]
    if T < window_size:
        return np.zeros((0, N_FEATURES * 2)), np.array([], dtype=int)

    n_windows = T - window_size + 1
    aggregated = np.zeros((n_windows, N_FEATURES * 2))
    indices = np.arange(n_windows)

    for i in range(n_windows):
        window = features[i:i + window_size]
        aggregated[i, :N_FEATURES] = window.mean(axis=0)
        aggregated[i, N_FEATURES:] = window.std(axis=0)

    return aggregated, indices

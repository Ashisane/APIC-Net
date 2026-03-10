"""
Φ Tensor — Gradient and Matrix Computation

Computes the 4×4 Φ matrix from residuals using magnitude-weighted
gradient cosine similarity.

Φ_ij = |r_i| · |r_j| · cos(∇r_i, ∇r_j) · sign(r_i · r_j)
"""
import numpy as np
from .residuals import (
    OMEGA_STAR, DT, H, D_VAL, F_VAL,
    COL_OMEGA, COL_DELTA, COL_P, COL_PSTAR, COL_OMEGA_C,
    N_RESIDUALS, compute_residuals_vectorized,
)


# State vector indices for gradient computation:
# [omega, delta, p, p_star, omega_C]
# We only use 5 state dimensions that the residuals depend on.
N_STATE = 5
STATE_OMEGA   = 0
STATE_DELTA   = 1
STATE_P       = 2
STATE_PSTAR   = 3
STATE_OMEGA_C = 4


def compute_gradients_analytical(data):
    """
    Compute analytical gradients ∂r_i/∂state for all timesteps.

    The state vector is [omega, delta, p, p_star, omega_C] at time t.
    Gradients are computed w.r.t. the values at time t (the "current" state),
    since residuals are functions of state(t) and state(t+1).

    Args:
        data: (N_steps, 8) raw signal array for one VSM.

    Returns:
        gradients: (N_steps-1, N_RESIDUALS, N_STATE) gradient tensor.
    """
    n = data.shape[0] - 1

    omega_t   = data[:-1, COL_OMEGA]
    omega_C_t = data[:-1, COL_OMEGA_C]

    grads = np.zeros((n, N_RESIDUALS, N_STATE))

    # ── ∇r1 (swing equation) ──
    # r1 = 2H*(ω(t+1)-ω(t))/dt - [(p*-p)·ω* + D·(ω*-ω) + F·(ωC-ω)]
    # ∂r1/∂ω(t) = -2H/dt + D + F
    # ∂r1/∂p(t) = ω*
    # ∂r1/∂p*(t) = -ω*
    # ∂r1/∂ωC(t) = -F
    grads[:, 0, STATE_OMEGA]   = -2 * H / DT + D_VAL + F_VAL
    grads[:, 0, STATE_P]       = OMEGA_STAR
    grads[:, 0, STATE_PSTAR]   = -OMEGA_STAR
    grads[:, 0, STATE_OMEGA_C] = -F_VAL

    # ── ∇r2 (angle-frequency) ──
    # r2 = (δ(t+1)-δ(t))/dt - ω(t)
    # ∂r2/∂δ(t) = -1/dt
    # ∂r2/∂ω(t) = -1
    grads[:, 1, STATE_DELTA] = -1.0 / DT
    grads[:, 1, STATE_OMEGA] = -1.0

    # ── ∇r3 (power balance) ──
    # r3 = (p*-p) - [2H·dω/dt - D·(ω*-ω) - F·(ωC-ω)] / ω*
    # ∂r3/∂ω(t) = -[2H/dt + D + F] / ω*  (negated from dω = ω(t+1)-ω(t))
    #   Wait, more carefully:
    #   expected_dp = [2H·(ω(t+1)-ω(t))/dt - D·(ω*-ω(t)) - F·(ωC(t)-ω(t))] / ω*
    #   ∂expected_dp/∂ω(t) = [-2H/dt - D - F] / ω*  ... actually:
    #     ∂/∂ω(t) of [2H*(ω(t+1)-ω(t))/dt] = -2H/dt
    #     ∂/∂ω(t) of [-D*(ω*-ω(t))] = D
    #     ∂/∂ω(t) of [-F*(ωC(t)-ω(t))] = F
    #   So ∂expected_dp/∂ω(t) = (-2H/dt + D + F) / ω*
    #   r3 = (p*-p) - expected_dp, so ∂r3/∂ω(t) = -∂expected_dp/∂ω(t)
    grads[:, 2, STATE_OMEGA]   = -((-2 * H / DT) + D_VAL + F_VAL) / OMEGA_STAR
    grads[:, 2, STATE_P]       = -1.0
    grads[:, 2, STATE_PSTAR]   = 1.0
    grads[:, 2, STATE_OMEGA_C] = F_VAL / OMEGA_STAR

    # ── ∇r4 (COI feedback) ──
    # r4 = ωC(t) - ω(t)
    # ∂r4/∂ωC(t) = 1
    # ∂r4/∂ω(t) = -1
    grads[:, 3, STATE_OMEGA_C] = 1.0
    grads[:, 3, STATE_OMEGA]   = -1.0

    return grads


def compute_phi_matrix_batch(residuals, gradients):
    """
    Compute Φ matrices for a batch of timesteps (fully vectorized).

    Args:
        residuals: (N, 4) — residuals at each timestep.
        gradients: (N, 4, 5) — gradient tensor.

    Returns:
        Phi: (N, 4, 4) — Φ matrix at each timestep.
    """
    N = residuals.shape[0]
    n_res = residuals.shape[1]

    # Gradient norms: (N, 4)
    grad_norms = np.linalg.norm(gradients, axis=2) + 1e-10

    # Normalize gradients: (N, 4, 5)
    grad_normed = gradients / grad_norms[:, :, np.newaxis]

    # Cosine similarity matrix: (N, 4, 4)
    # cos_sim[n, i, j] = dot(grad_normed[n,i], grad_normed[n,j])
    cos_sim = np.einsum('nik,njk->nij', grad_normed, grad_normed)

    # Sign term: sign(r_i * r_j) → (N, 4, 4)
    # Outer product of signs
    r_signs = np.sign(residuals)  # (N, 4)
    sign_term = r_signs[:, :, np.newaxis] * r_signs[:, np.newaxis, :]  # (N, 4, 4)

    # Magnitude product: |r_i| * |r_j| → (N, 4, 4)
    r_abs = np.abs(residuals)  # (N, 4)
    mag_product = r_abs[:, :, np.newaxis] * r_abs[:, np.newaxis, :]  # (N, 4, 4)

    # Φ = |r_i|·|r_j|·cos(∇r_i, ∇r_j)·sign(r_i·r_j)
    Phi = mag_product * cos_sim * sign_term

    return Phi


def compute_phi_for_scenario(data):
    """
    Compute Φ matrix time series for a single VSM scenario.

    Args:
        data: (N_steps, 8) raw signal array.

    Returns:
        Phi: (N_steps-1, 4, 4) — Φ matrix at each timestep.
        residuals: (N_steps-1, 4) — raw residuals.
    """
    residuals = compute_residuals_vectorized(data)
    gradients = compute_gradients_analytical(data)
    Phi = compute_phi_matrix_batch(residuals, gradients)
    return Phi, residuals

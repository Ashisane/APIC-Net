import torch
import torch.nn as nn
import numpy as np

# Physics constants — must match vsm_simulator.py exactly
OMEGA_STAR = 2 * np.pi * 50.0
DT = 0.005

class PILSTM(nn.Module):
    """
    Physics-Informed LSTM Baseline.
    Input: [omega, p, v_dc, omega_C_received, p_star, v_ref]
    Output: [omega_hat(t+1), delta_hat(t+1)]
    """
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=2):
        super(PILSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # We only care about the prediction for the next timestep
        # which comes from the last hidden state of the sequence
        last_out = lstm_out[:, -1, :] 
        out = self.fc(last_out) # (batch_size, output_size)
        return out

class PILoss(nn.Module):
    def __init__(self, lambda_physics=0.6):
        super(PILoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
        
    def compute_loss(self, s_hat, s_true, state_prev, mean, std):
        """Compute combined data + physics loss.
        
        Args:
            s_hat: predicted [omega, delta] at t+1, normalized. Shape (B, 2)
            s_true: true [omega, delta] at t+1, normalized. Shape (B, 2)
            state_prev: full state at t, normalized. Shape (B, 8)
                Cols: 0:omega, 1:delta, 2:p, 3:p_star, 4:v_ref, 5:v_dc, 6:omega_C, 7:i_f
            mean, std: per-feature normalization stats. Shape (8,)
        """
        L_data = self.mse(s_hat, s_true)
        
        # Physics constants — match vsm_simulator.py exactly
        H, D_val, F_val = 5.0, 25.0, 10.0
        
        # Unscale predicted omega to physical units
        omega_mean, omega_std = mean[0], std[0]
        omega_hat_phys = s_hat[:, 0] * omega_std + omega_mean
        
        # Unscale state_prev features needed for the swing equation
        omega_prev_phys = state_prev[:, 0] * std[0] + mean[0]
        p_prev_phys = state_prev[:, 2] * std[2] + mean[2]
        p_star_prev_phys = state_prev[:, 3] * std[3] + mean[3]
        omega_C_prev_phys = state_prev[:, 6] * std[6] + mean[6]
        
        # Euler step matching vsm_simulator.py swing equation
        d_omega = (1.0 / (2.0 * H)) * (
            (p_star_prev_phys - p_prev_phys) * OMEGA_STAR
            + D_val * (OMEGA_STAR - omega_prev_phys)
            + F_val * (omega_C_prev_phys - omega_prev_phys)
        )
        omega_phys_expected = omega_prev_phys + DT * d_omega
        
        # Re-normalize to match L_data's scale
        omega_phys_expected_norm = (omega_phys_expected - omega_mean) / omega_std
        L_phys = self.mse(s_hat[:, 0], omega_phys_expected_norm)
        
        L_total = L_data + self.lambda_physics * L_phys
        return L_total, L_data, L_phys

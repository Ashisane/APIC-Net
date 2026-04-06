# APIC-Net
**Author**: Utkarsh Tyagi

## Overview

**APIC-Net** is a research project investigating cyberattack detection for Virtual Synchronous Machines (VSMs) in power grids, simulated on the IEEE 39-bus (New England) system.

The project evaluates detection methods ranging from simple physics residuals to the Φ Tensor cross-constraint framework, using a federated learning architecture where each VSM is a separate client.

---

## Key Findings (Current State)

| Method | Overall AUC | freq | coi | power | voltage |
|---|---|---|---|---|---|
| Pure r_phys | 0.632 | 0.998 | 0.500 | 0.500 | 0.500 |
| r4-only (threshold) | 0.931 | 0.891 | 0.931 | 0.919 | 0.976 |
| r4 + r_phys (combined) | 0.959 | 0.997 | 0.931 | 0.920 | 0.976 |

*Results on V2 dataset (calibrated amplitudes). No training required for r4 or r_phys.*

**r_phys** (swing-equation residual) is structurally blind to COI, power, and voltage attacks — it can only see freq attacks where `ω` is directly corrupted.

**r4** (`|ω_C_received − ω_local|`) detects all 4 attack types with AUC > 0.89 using a single threshold.

---

## Project Structure

```
APIC-Net/
├── simulation/
│   ├── vsm_simulator.py        # Euler-integrated VSM swing equation
│   ├── attack_generator.py     # Attack injection (freq, coi, power, voltage)
│   └── generate_dataset.py     # Full dataset pipeline with validation
│
├── configs/
│   └── simulation_config.py   # All hyperparameters (DT, N_STEPS, amplitudes, etc.)
│
├── models/
│   ├── pi_lstm/               # Physics-Informed LSTM (federated)
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── federated.py
│   │   └── pure_physics_baseline.py
│   └── phi_tensor/            # Φ Tensor cross-constraint detector
│       ├── residuals.py       # r1–r4 residual computation
│       ├── phi_tensor.py      # Φ matrix + analytical gradients
│       ├── features.py        # 20-dim feature extraction
│       ├── signal_check.py    # Gate 2: feature separation
│       ├── r4_baseline.py     # r4-only + combined score detector
│       ├── dataset_audit.py   # Dataset validity audit (SNR, correlation)
│       └── train.py           # MLP detector with FedAvg
│
├── data/                      # Per-VSM .npz files (vsm_0 through vsm_5)
├── results/                   # JSON results + plots
└── reports/                   # Numbered research reports
```

---

## Dataset (V2)

Generated with calibrated attack amplitudes to ensure realistic signal-to-noise ratios.

| Attack | Injection | Amplitude | SNR | Detection Rate |
|---|---|---|---|---|
| freq | ω sensor spoof | 1–5 rad/s (unchanged) | 1.78× | 67% |
| coi | ω_C bias (outgoing) | calibrated | 1.63× | 83% |
| power | p* injection | 0.285–0.76 rad/s | 2.69× | 79% |
| voltage | v_ref tamper | 1–5 rad/s (unchanged) | 16.35× | 94% |

> Voltage attacks are intentionally high-SNR: nonlinear controller amplification causes inherently large omega deviations. Documented as a high-severity attack class.

**Generate dataset:**
```bash
python -m simulation.generate_dataset
```

**Validate/audit:**
```bash
python -m models.phi_tensor.dataset_audit
```

---

## Baselines

**Pure physics residual (r_phys):**
```bash
python -m models.pi_lstm.pure_physics_baseline
```

**r4-only + combined detector:**
```bash
python -m models.phi_tensor.r4_baseline
```

---

## Reports

| Report | Description |
|---|---|
| `report_01_dataset.md` | Initial dataset generation |
| `report_02d_closure.md` | PI-LSTM diagnosis and closure |
| `report_02e_physics_baseline.md` | Pure physics residual evaluation |
| `report_03_phi_tensor.md` | Φ Tensor implementation + spec conflicts |
| `report_03b_decision_gate.md` | r4 decision gate (r4 dominates) |
| `report_03c_dataset_audit.md` | V1 dataset validity audit (mixed verdict) |
| `report_04_dataset_v2.md` | V2 dataset regeneration + full audit |
| `report_04b_coi_diagnosis.md` | COI correlation diagnosis + fix decision |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, PyTorch, scikit-learn, SciPy, matplotlib, pandapower

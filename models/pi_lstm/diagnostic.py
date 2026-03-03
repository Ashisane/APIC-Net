"""
Phase 1 Diagnostic for PI-LSTM (TASK 02D)

Computes:
  1A: r_data and r_phys separately for normal vs each attack type
  1B: Inside vs outside attack window prediction error
  1C: r_phys during attack vs outside attack window

Uses the best available checkpoint. Saves tables + plots.

Usage: python -m models.pi_lstm.diagnostic
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .dataset import get_dataloaders
from .model import PILSTM, PILoss, OMEGA_STAR, DT


CONFIG = {
    "input_sequence_length": 20,
    "hidden_units": 64,
    "recurrent_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "local_epochs": 2,
    "num_clients": 6,
    "lambda_physics": 0.6,
}

ATTACK_TYPE_NAMES = {0: "normal", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}
H, D_VAL, F_VAL = 5.0, 25.0, 10.0


def compute_separate_scores(model, dataloader, device):
    """Compute r_data and r_phys separately for every sample."""
    model.eval()
    
    mean_t = torch.tensor(dataloader.dataset.mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(dataloader.dataset.std, dtype=torch.float32, device=device)
    
    all_r_data = []
    all_r_phys = []
    all_labels = []
    all_atk_types = []
    
    with torch.no_grad():
        for x, y, state_prev, label, atk_type in dataloader:
            x = x.to(device)
            y = y.to(device)
            state_prev = state_prev.to(device)
            
            s_hat = model(x)
            
            # r_data: L2 norm of prediction error (normalized scale)
            r_data = torch.norm(s_hat - y, p=2, dim=1)
            
            # r_phys: physics residual
            omega_prev = state_prev[:, 0] * std_t[0] + mean_t[0]
            p_prev = state_prev[:, 2] * std_t[2] + mean_t[2]
            p_star_prev = state_prev[:, 3] * std_t[3] + mean_t[3]
            omega_C_prev = state_prev[:, 6] * std_t[6] + mean_t[6]
            
            d_omega = (1.0 / (2.0 * H)) * (
                (p_star_prev - p_prev) * OMEGA_STAR
                + D_VAL * (OMEGA_STAR - omega_prev)
                + F_VAL * (omega_C_prev - omega_prev)
            )
            omega_phys = omega_prev + DT * d_omega
            omega_phys_norm = (omega_phys - mean_t[0]) / std_t[0]
            r_phys = torch.abs(s_hat[:, 0] - omega_phys_norm)
            
            all_r_data.append(r_data.cpu().numpy())
            all_r_phys.append(r_phys.cpu().numpy())
            all_labels.append(label.numpy())
            all_atk_types.extend(atk_type)
    
    return (
        np.concatenate(all_r_data),
        np.concatenate(all_r_phys),
        np.concatenate(all_labels),
        np.array(all_atk_types),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(results_dir, "pi_lstm_global.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    global_weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Aggregate scores across all clients
    all_r_data = []
    all_r_phys = []
    all_labels = []
    all_atk_types = []
    
    for vsm_id in range(CONFIG["num_clients"]):
        print(f"Processing VSM {vsm_id}...")
        _, _, test_ld = get_dataloaders(
            data_dir, vsm_id, CONFIG["batch_size"], CONFIG["input_sequence_length"]
        )
        
        model = PILSTM(input_size=6, hidden_size=64, num_layers=2, output_size=2).to(device)
        model.load_state_dict(global_weights)
        
        r_data, r_phys, labels, atk_types = compute_separate_scores(model, test_ld, device)
        all_r_data.append(r_data)
        all_r_phys.append(r_phys)
        all_labels.append(labels)
        all_atk_types.append(atk_types)
    
    r_data = np.concatenate(all_r_data)
    r_phys = np.concatenate(all_r_phys)
    labels = np.concatenate(all_labels)
    atk_types = np.concatenate(all_atk_types)
    r_total = 0.5 * r_data + 0.5 * r_phys
    
    print(f"\nTotal test samples: {len(r_data)}")
    print(f"  Normal: {(labels == 0).sum()}")
    print(f"  Attack: {(labels == 1).sum()}")
    
    # ═══════════════════════════════════════
    # 1A: Score Distribution Table
    # ═══════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 1A: SCORE DISTRIBUTION TABLE")
    print("="*80)
    
    table_1a = {}
    header = f"{'Sample Type':<14} {'r_data mean':>11} {'r_data std':>10} {'r_phys mean':>11} {'r_phys std':>10} {'r_total mean':>12}"
    print(header)
    print("-" * len(header))
    
    # Normal
    nm = labels == 0
    row = {
        "r_data_mean": float(r_data[nm].mean()),
        "r_data_std": float(r_data[nm].std()),
        "r_phys_mean": float(r_phys[nm].mean()),
        "r_phys_std": float(r_phys[nm].std()),
        "r_total_mean": float(r_total[nm].mean()),
        "n": int(nm.sum()),
    }
    table_1a["normal"] = row
    print(f"{'Normal':<14} {row['r_data_mean']:>11.4f} {row['r_data_std']:>10.4f} {row['r_phys_mean']:>11.4f} {row['r_phys_std']:>10.4f} {row['r_total_mean']:>12.4f}  (n={row['n']})")
    
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        mask = atk_types == code
        if mask.sum() == 0:
            continue
        row = {
            "r_data_mean": float(r_data[mask].mean()),
            "r_data_std": float(r_data[mask].std()),
            "r_phys_mean": float(r_phys[mask].mean()),
            "r_phys_std": float(r_phys[mask].std()),
            "r_total_mean": float(r_total[mask].mean()),
            "n": int(mask.sum()),
        }
        table_1a[name] = row
        sep = row["r_total_mean"] - table_1a["normal"]["r_total_mean"]
        indicator = "✅" if sep > 0 else "❌ INVERTED"
        print(f"{name:<14} {row['r_data_mean']:>11.4f} {row['r_data_std']:>10.4f} {row['r_phys_mean']:>11.4f} {row['r_phys_std']:>10.4f} {row['r_total_mean']:>12.4f}  (n={row['n']}) {indicator}")
    
    # ═══════════════════════════════════════
    # 1B: Predictability — inside vs outside attack window
    # ═══════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 1B: PREDICTABILITY (inside vs outside attack window)")
    print("="*80)
    
    table_1b = {}
    # For attack scenarios (labels has attack somewhere), compare:
    # - timesteps INSIDE the attack window (label=1, atk_type>0)
    # - timesteps OUTSIDE the attack window in SAME scenarios (label=0, but scenario has attacks)
    # We approximate: "outside" = all label=0 samples, "inside" = label=1 per attack type
    
    outside_r_data = float(r_data[labels == 0].mean())
    print(f"{'Attack Type':<14} {'r_data INSIDE':>14} {'r_data OUTSIDE':>15} {'Diff':>8} {'Verdict'}")
    print("-" * 70)
    
    inverted_count = 0
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        mask = atk_types == code
        if mask.sum() == 0:
            continue
        inside = float(r_data[mask].mean())
        diff = inside - outside_r_data
        verdict = "HARDER (good)" if diff > 0 else "EASIER (inverted)"
        if diff < 0:
            inverted_count += 1
        table_1b[name] = {
            "r_data_inside": inside,
            "r_data_outside": outside_r_data,
            "diff": diff,
        }
        print(f"{name:<14} {inside:>14.4f} {outside_r_data:>15.4f} {diff:>+8.4f} {verdict}")
    
    print(f"\n→ {inverted_count}/4 attack types easier to predict than normal")
    if inverted_count >= 3:
        print("→ CONFIRMED: Attacks are more predictable than normal. Architectural issue.")
    
    # ═══════════════════════════════════════
    # 1C: Physics Residual Audit
    # ═══════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 1C: PHYSICS RESIDUAL AUDIT")
    print("="*80)
    
    table_1c = {}
    normal_r_phys = float(r_phys[labels == 0].mean())
    print(f"{'Attack Type':<14} {'r_phys ATTACK':>14} {'r_phys NORMAL':>14} {'Ratio':>8} {'Physics detects?'}")
    print("-" * 70)
    
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        mask = atk_types == code
        if mask.sum() == 0:
            continue
        attack_rp = float(r_phys[mask].mean())
        ratio = attack_rp / normal_r_phys if normal_r_phys > 0 else 0
        detects = "YES ✅" if ratio > 1.5 else "NO ❌ (same as normal)"
        table_1c[name] = {
            "r_phys_attack": attack_rp,
            "r_phys_normal": normal_r_phys,
            "ratio": ratio,
        }
        print(f"{name:<14} {attack_rp:>14.4f} {normal_r_phys:>14.4f} {ratio:>8.2f}x {detects}")
    
    # ═══════════════════════════════════════
    # Save diagnostic tables
    # ═══════════════════════════════════════
    diagnostic = {
        "phase_1a_score_distributions": table_1a,
        "phase_1b_predictability": table_1b,
        "phase_1c_physics_audit": table_1c,
        "summary": {
            "inverted_attack_types": inverted_count,
            "attacks_more_predictable_than_normal": inverted_count >= 3,
        }
    }
    
    diag_path = os.path.join(results_dir, "diagnostic_tables.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostic, f, indent=4)
    print(f"\nDiagnostic tables saved to {diag_path}")
    
    # ═══════════════════════════════════════
    # Score Distribution Plots
    # ═══════════════════════════════════════
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Score Distributions: Normal vs Attack (r_total = 0.5·r_data + 0.5·r_phys)", fontsize=14)
    
    normal_scores = r_total[labels == 0]
    
    for idx, (code, name) in enumerate([(1, "freq"), (2, "coi"), (3, "power"), (4, "voltage")]):
        ax = axes[idx // 2][idx % 2]
        mask = atk_types == code
        attack_scores = r_total[mask]
        
        bins = np.linspace(0, max(normal_scores.max(), attack_scores.max()) * 0.8, 80)
        ax.hist(normal_scores, bins=bins, alpha=0.6, label=f"Normal (n={len(normal_scores)})", density=True, color="steelblue")
        if len(attack_scores) > 0:
            ax.hist(attack_scores, bins=bins, alpha=0.6, label=f"{name} (n={len(attack_scores)})", density=True, color="crimson")
        
        ax.axvline(normal_scores.mean(), color="steelblue", linestyle="--", linewidth=1.5, label=f"Normal mean={normal_scores.mean():.3f}")
        if len(attack_scores) > 0:
            ax.axvline(attack_scores.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"{name} mean={attack_scores.mean():.3f}")
        
        ax.set_title(f"{name.upper()} Attack vs Normal")
        ax.set_xlabel("Anomaly Score (r_total)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "score_distributions.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Score distribution plot saved to {plot_path}")
    
    # Separate r_data vs r_phys plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("r_data vs r_phys Breakdown by Attack Type", fontsize=14)
    
    for idx, (code, name) in enumerate([(1, "freq"), (2, "coi"), (3, "power"), (4, "voltage")]):
        ax = axes2[idx // 2][idx % 2]
        mask = atk_types == code
        
        categories = ["Normal\nr_data", f"{name}\nr_data", "Normal\nr_phys", f"{name}\nr_phys"]
        values = [
            r_data[labels == 0].mean(),
            r_data[mask].mean() if mask.sum() > 0 else 0,
            r_phys[labels == 0].mean(),
            r_phys[mask].mean() if mask.sum() > 0 else 0,
        ]
        colors = ["steelblue", "crimson", "lightsteelblue", "lightsalmon"]
        
        bars = ax.bar(categories, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", 
                    ha="center", va="bottom", fontsize=9)
        
        ax.set_title(f"{name.upper()}")
        ax.set_ylabel("Mean Score")
    
    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, "score_distributions_per_attack.png")
    fig2.savefig(plot_path2, dpi=150)
    print(f"Per-attack plot saved to {plot_path2}")
    
    plt.close('all')
    print("\n✓ Phase 1 diagnostic complete.")


if __name__ == "__main__":
    main()

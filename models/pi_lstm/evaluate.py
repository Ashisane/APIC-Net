"""
Evaluation script for PI-LSTM.
Extracts threshold from validation set.
Computes metrics on test set per-attack breakdown. 
"""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from .model import OMEGA_STAR, DT
import time

def evaluate_inference_anomaly_scores(model, loss_fn, dataloader, device):
    """
    Computes r_data and r_phys scores across all timesteps.
    Returns scores and labels and attack types.
    """
    model.eval()
    all_r_total = []
    all_labels = []
    all_attack_types = []
    
    omega_star = OMEGA_STAR
    dt = DT
    H, D_val, F_val = 5.0, 25.0, 10.0
    
    inference_time = 0.0
    num_samples = 0
    
    # Extract scalers for physics
    mean_t = torch.tensor(dataloader.dataset.mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(dataloader.dataset.std, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for i, (x, y, state_prev, label, atk_type) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            state_prev = state_prev.to(device)
            
            start_t = time.time()
            s_hat = model(x)
            
            # compute r_data (on normalized scale)
            r_data = torch.norm(s_hat - y, p=2, dim=1)
            
            # compute r_phys
            # Unscale state features for physics
            omega_prev_phys = state_prev[:, 0] * std_t[0] + mean_t[0]
            p_prev_phys = state_prev[:, 2] * std_t[2] + mean_t[2]
            p_star_prev_phys = state_prev[:, 3] * std_t[3] + mean_t[3]
            omega_C_prev_phys = state_prev[:, 6] * std_t[6] + mean_t[6]
            
            # physical prediction (Euler step)
            d_omega = (1.0 / (2.0 * H)) * (
                (p_star_prev_phys - p_prev_phys) * omega_star
                + D_val * (omega_star - omega_prev_phys)
                + F_val * (omega_C_prev_phys - omega_prev_phys)
            )
            omega_phys_expected = omega_prev_phys + dt * d_omega
            
            # Normalize expected omega to compute r_phys on same scale as r_data
            omega_phys_expected_norm = (omega_phys_expected - mean_t[0]) / std_t[0]
            r_phys = torch.abs(s_hat[:, 0] - omega_phys_expected_norm)
            
            # r_total = 0.5 * r_data + 0.5 * r_phys
            r_total = 0.5 * r_data + 0.5 * r_phys
            
            inference_time += time.time() - start_t
            num_samples += x.size(0)
            
            all_r_total.append(r_total.cpu().numpy())
            all_labels.append(label.numpy())
            all_attack_types.extend(atk_type)
            
    all_r_total = np.concatenate(all_r_total)
    all_labels = np.concatenate(all_labels)
    all_attack_types = np.array(all_attack_types)
    
    avg_inf_time = (inference_time / num_samples) * 1000.0 if num_samples > 0 else 0.0
    
    return all_r_total, all_labels, all_attack_types, avg_inf_time

def evaluate_client(client, val_loader, test_loader, device):
    """
    Evaluates one client's performance.
    """
    # 1. Validation (normal only) to set threshold
    val_scores, val_labels, _, _ = evaluate_inference_anomaly_scores(client.model, client.loss_fn, val_loader, device)
    
    normal_val_scores = val_scores[val_labels == 0]
    tau = np.percentile(normal_val_scores, 95)
    
    # 2. Test
    test_scores, test_labels, test_atk_types, inf_time = evaluate_inference_anomaly_scores(client.model, client.loss_fn, test_loader, device)
    
    preds = (test_scores > tau).astype(int)
    
    metrics = {
        "overall": {
            "precision": precision_score(test_labels, preds, zero_division=0),
            "recall": recall_score(test_labels, preds, zero_division=0),
            "f1": f1_score(test_labels, preds, zero_division=0),
            "auc": roc_auc_score(test_labels, test_scores) if len(np.unique(test_labels)) > 1 else 0.0,
            "inference_time_ms": inf_time,
            "tau": float(tau)
        },
        "by_attack": {}
    }
    
    # Map integer attack type codes to human-readable names
    ATTACK_TYPE_NAMES = {0: "none", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}
    
    # Calculate metrics grouped by attack type
    for atype in np.unique(test_atk_types):
        if atype == 0:  # skip normal ("none")
            continue
            
        mask = (test_atk_types == atype) | (test_labels == 0)
        sub_labels = test_labels[mask]
        sub_preds = preds[mask]
        sub_scores = test_scores[mask]
        
        atype_name = ATTACK_TYPE_NAMES.get(int(atype), str(atype))
        
        if len(np.unique(sub_labels)) > 1:
            metrics["by_attack"][atype_name] = {
                "precision": precision_score(sub_labels, sub_preds, zero_division=0),
                "recall": recall_score(sub_labels, sub_preds, zero_division=0),
                "f1": f1_score(sub_labels, sub_preds, zero_division=0),
                "auc": roc_auc_score(sub_labels, sub_scores)
            }
        else:
            metrics["by_attack"][atype_name] = {"precision": 0, "recall": 0, "f1": 0, "auc": 0}
            
    return metrics, test_scores, test_labels

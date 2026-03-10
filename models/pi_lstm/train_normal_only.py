"""
Phase 2: Normal-only training for PI-LSTM (TASK 02D)

Standard anomaly detection: train ONLY on normal timesteps so the model
learns what "normal" looks like. Attacks become out-of-distribution.

Usage: python -m models.pi_lstm.train_normal_only
"""
import os
import json
import torch
import numpy as np

from .dataset import get_dataloaders
from .model import PILSTM, PILoss
from .evaluate import evaluate_client
import time
import copy
from torch.optim import Adam


CONFIG = {
    "input_sequence_length": 20,
    "hidden_units": 64,
    "recurrent_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "fl_rounds": 50,
    "local_epochs": 2,
    "num_clients": 6,
    "lambda_physics": 0.6,
}

EVAL_EVERY = 10


class NormalOnlyFLClient:
    """FL Client that trains ONLY on normal (label=0) samples."""
    
    def __init__(self, client_id, train_loader, val_loader, config, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.model = PILSTM(
            input_size=6,
            hidden_size=config["hidden_units"],
            num_layers=config["recurrent_layers"],
            output_size=2
        ).to(device)
        
        self.loss_fn = PILoss(lambda_physics=config["lambda_physics"]).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=config["learning_rate"])
        self.local_epochs = config["local_epochs"]
    
    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def train(self):
        """Train on normal-only samples using per-sample loss masking."""
        self.model.train()
        train_loss = 0.0
        n_batches = 0
        n_normal = 0
        n_skipped = 0
        
        mean_t = torch.tensor(self.train_loader.dataset.mean, dtype=torch.float32, device=self.device)
        std_t = torch.tensor(self.train_loader.dataset.std, dtype=torch.float32, device=self.device)
        
        mse_none = torch.nn.MSELoss(reduction='none')
        
        start_time = time.time()
        for epoch in range(self.local_epochs):
            for x, y, state_prev, label, _ in self.train_loader:
                # Build normal mask
                normal_mask = (label == 0)
                n_norm = normal_mask.sum().item()
                
                if n_norm == 0:
                    n_skipped += 1
                    continue
                
                # Filter to normal samples only
                x_n = x[normal_mask].to(self.device)
                y_n = y[normal_mask].to(self.device)
                sp_n = state_prev[normal_mask].to(self.device)
                
                self.optimizer.zero_grad()
                s_hat = self.model(x_n)
                
                L_total, _, _ = self.loss_fn.compute_loss(s_hat, y_n, sp_n, mean=mean_t, std=std_t)
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += L_total.item()
                n_batches += 1
                n_normal += n_norm
        
        train_time = time.time() - start_time
        return train_loss / max(n_batches, 1), train_time, n_normal, n_skipped


class FLServerNormalOnly:
    """FL Server for normal-only training."""
    
    def __init__(self, config, device):
        self.global_model = PILSTM(
            input_size=6,
            hidden_size=config["hidden_units"],
            num_layers=config["recurrent_layers"],
            output_size=2
        ).to(device)
        self.device = device
        self.clients = []
    
    def add_client(self, client):
        self.clients.append(client)
    
    def federated_averaging(self, client_weights_list):
        n_clients = len(client_weights_list)
        avg_weights = copy.deepcopy(client_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, n_clients):
                avg_weights[key] += client_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], n_clients)
        self.global_model.load_state_dict(avg_weights)
        return avg_weights
    
    def train_round(self):
        global_weights = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        
        client_weights = []
        round_loss = 0.0
        total_normal = 0
        total_skipped = 0
        
        for client in self.clients:
            client.set_weights(global_weights)
            loss, t_time, n_norm, n_skip = client.train()
            client_weights.append(client.get_weights())
            round_loss += loss
            total_normal += n_norm
            total_skipped += n_skip
        
        self.federated_averaging(client_weights)
        return round_loss / len(self.clients), total_normal, total_skipped


def evaluate_and_save(server, val_loaders, test_loaders, device, round_losses, results_dir, tag=""):
    """Evaluate all clients and save results."""
    global_weights = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
    overall_metrics = []
    
    for i, client in enumerate(server.clients):
        client.set_weights(global_weights)
        metrics, _, _ = evaluate_client(client, val_loaders[i], test_loaders[i], device)
        overall_metrics.append(metrics)
        print(f"  VSM {i} F1: {metrics['overall']['f1']:.4f}, AUC: {metrics['overall']['auc']:.4f}")
    
    avg_metrics = {
        "overall": {
            "precision": float(np.mean([m['overall']['precision'] for m in overall_metrics])),
            "recall": float(np.mean([m['overall']['recall'] for m in overall_metrics])),
            "f1": float(np.mean([m['overall']['f1'] for m in overall_metrics])),
            "auc": float(np.mean([m['overall']['auc'] for m in overall_metrics])),
            "inference_time_ms": float(np.mean([m['overall']['inference_time_ms'] for m in overall_metrics])),
        },
        "by_attack": {},
        "training_loss_per_round": round_losses,
    }
    
    all_attack_keys = set()
    for m in overall_metrics:
        all_attack_keys.update(m["by_attack"].keys())
    for atype in all_attack_keys:
        vals = [m["by_attack"][atype] for m in overall_metrics if atype in m["by_attack"]]
        if vals:
            avg_metrics["by_attack"][atype] = {
                "precision": float(np.mean([v["precision"] for v in vals])),
                "recall": float(np.mean([v["recall"] for v in vals])),
                "f1": float(np.mean([v["f1"] for v in vals])),
                "auc": float(np.mean([v["auc"] for v in vals])),
            }
    
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "pi_lstm_normal_only_results.json")
    with open(results_file, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    torch.save(server.global_model.state_dict(), os.path.join(results_dir, "pi_lstm_normal_only_global.pt"))
    
    f1 = avg_metrics['overall']['f1']
    auc = avg_metrics['overall']['auc']
    print(f"  >> {tag}Overall F1: {f1:.4f}, AUC: {auc:.4f} — saved to {results_file}")
    return avg_metrics


def main():
    device = torch.device("cpu")  # CPU is faster for this tiny model
    print(f"Using device: {device}")
    print("MODE: Normal-only training (training ONLY on label=0 samples)")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    
    server = FLServerNormalOnly(CONFIG, device)
    
    print("Initializing clients and loading data...")
    val_loaders = []
    test_loaders = []
    
    for vsm_id in range(CONFIG["num_clients"]):
        train_ld, val_ld, test_ld = get_dataloaders(
            data_dir, vsm_id, CONFIG["batch_size"], CONFIG["input_sequence_length"]
        )
        val_loaders.append(val_ld)
        test_loaders.append(test_ld)
        
        client = NormalOnlyFLClient(vsm_id, train_ld, val_ld, CONFIG, device)
        server.add_client(client)
    
    print(f"Starting Normal-Only FL Training for {CONFIG['fl_rounds']} rounds (eval every {EVAL_EVERY})...")
    round_losses = []
    
    for rnd in range(CONFIG["fl_rounds"]):
        avg_loss, n_normal, n_skipped = server.train_round()
        round_losses.append(avg_loss)
        print(f"[Round {rnd+1}/{CONFIG['fl_rounds']}] Avg Loss: {avg_loss:.6f} "
              f"(normal samples: {n_normal}, skipped batches: {n_skipped})")
        
        if (rnd + 1) % EVAL_EVERY == 0:
            print(f"--- Checkpoint at round {rnd+1} ---")
            evaluate_and_save(server, val_loaders, test_loaders, device,
                              round_losses, results_dir, tag=f"Round {rnd+1}: ")
    
    if CONFIG["fl_rounds"] % EVAL_EVERY != 0:
        print("--- Final evaluation ---")
        evaluate_and_save(server, val_loaders, test_loaders, device,
                          round_losses, results_dir, tag="Final: ")
    
    print("Done!")


if __name__ == "__main__":
    main()

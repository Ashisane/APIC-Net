"""
Federated learning implementation for PI-LSTM.
Strict isolation: 6 clients train their own models, sever computes FedAvg.
"""
import copy
import torch
from torch.optim import Adam
from .model import PILSTM, PILoss
import time

class FLClient:
    def __init__(self, client_id: int, train_loader, val_loader, config, device):
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
        self.model.train()
        train_loss = 0.0
        n_batches = 0
        
        # Cache normalization tensors (constant per client)
        mean_t = torch.tensor(self.train_loader.dataset.mean, dtype=torch.float32, device=self.device)
        std_t = torch.tensor(self.train_loader.dataset.std, dtype=torch.float32, device=self.device)
        
        start_time = time.time()
        for epoch in range(self.local_epochs):
            for x, y, state_prev, label, _ in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                state_prev = state_prev.to(self.device)
                
                self.optimizer.zero_grad()
                s_hat = self.model(x)
                
                L_total, L_data, L_phys = self.loss_fn.compute_loss(s_hat, y, state_prev, mean=mean_t, std=std_t)
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += L_total.item()
                n_batches += 1
                
        train_time = time.time() - start_time
        return train_loss / n_batches if n_batches > 0 else 0.0, train_time

class FLServer:
    def __init__(self, config, device):
        self.global_model = PILSTM(
            input_size=6,
            hidden_size=config["hidden_units"],
            num_layers=config["recurrent_layers"],
            output_size=2
        ).to(device)
        self.device = device
        self.clients = []
        
    def add_client(self, client: FLClient):
        self.clients.append(client)
        
    def federated_averaging(self, client_weights_list):
        """ FedAvg: weighted average (uniform here since data sizes are equal) """
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
        total_time = 0.0
        
        for client in self.clients:
            client.set_weights(global_weights)
            loss, t_time = client.train()
            client_weights.append(client.get_weights())
            round_loss += loss
            total_time += t_time
            
        self.federated_averaging(client_weights)
        return round_loss / len(self.clients), total_time / len(self.clients)

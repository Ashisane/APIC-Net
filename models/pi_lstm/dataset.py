"""
Dataset module for PI-LSTM baseline.
Implements sliding window loading of VSM data with strict per-VSM isolation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VSMDataset(Dataset):
    """
    Sliding window dataset for a single VSM.
    Inputs: [omega, p, v_dc, omega_C_received, p_star, v_ref]
    Targets: [omega(t+1), delta(t+1)]
    """
    def __init__(self, data_file: str, meta_file: str, split: str = "train", window_size: int = 20, stride: int = 1):
        super().__init__()
        self.window_size = window_size
        
        # Load data
        data = np.load(data_file)
        self.recording = data["data"]           # (N_SCENARIOS, N_STEPS, 8)
        self.labels = data["labels"]            # (N_SCENARIOS, N_STEPS)
        self.attack_types_arr = data["attack_types"] # (N_SCENARIOS, N_STEPS)
        
        # Load metadata
        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)["scenarios"]
            
        # Splitting indices
        n_scenarios = self.recording.shape[0]
        n_train = int(n_scenarios * 0.7)
        n_val = int(n_scenarios * 0.15)
        
        # Fit scaler ONLY on train data to prevent data leakage
        train_data = self.recording[0:n_train].reshape(-1, self.recording.shape[2])
        self.mean = np.mean(train_data, axis=0)
        self.std = np.std(train_data, axis=0)
        # Avoid div by zero
        self.std[self.std == 0] = 1.0
        
        # Slice data for this split
        if split == "train":
            s_idx, e_idx = 0, n_train
        elif split == "val":
            s_idx, e_idx = n_train, n_train + n_val
        elif split == "test":
            s_idx, e_idx = n_train + n_val, n_scenarios
        else:
            raise ValueError(f"Unknown split: {split}")
            
        self.recording = self.recording[s_idx:e_idx]
        self.labels = self.labels[s_idx:e_idx]
        self.attack_types_arr = self.attack_types_arr[s_idx:e_idx]
        self.scenarios = self.metadata[s_idx:e_idx]
        
        # Normalize the reporting data using the train statistics
        self.recording = (self.recording - self.mean) / self.std
        
        # Feature indices
        self.input_cols = [0, 2, 5, 6, 3, 4]
        self.target_cols = [0, 1]  # omega, delta
        
        self.samples = []
        n_steps = self.recording.shape[1]
        
        for scnt in range(self.recording.shape[0]):
            for t in range(0, n_steps - window_size, stride):
                self.samples.append((scnt, t))
                
        # To tensor
        self.recording_tensor = torch.tensor(self.recording, dtype=torch.float32)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        scnt, t = self.samples[idx]
        
        # Input sliding window: t to t+window_size-1
        x = self.recording_tensor[scnt, t:t+self.window_size, self.input_cols]
        
        # Target: t+window_size (the next timestep)
        y = self.recording_tensor[scnt, t+self.window_size, self.target_cols]
        
        # Also return full raw data for the window needed by physics loss
        state_prev = self.recording_tensor[scnt, t+self.window_size-1]
        
        # label at target timestep
        label = self.labels_tensor[scnt, t+self.window_size]
        atk_type = self.attack_types_arr[scnt, t+self.window_size]
        
        return x, y, state_prev, label, atk_type

def get_dataloaders(data_dir: str, vsm_id: int, batch_size: int = 32, window_size: int = 20):
    """
    Get (train, val, test) dataloaders for a specific VSM.
    """
    data_file = os.path.join(data_dir, f"vsm_{vsm_id}.npz")
    meta_file = os.path.join(data_dir, "metadata.json")
    
    train_ds = VSMDataset(data_file, meta_file, split="train", window_size=window_size)
    val_ds = VSMDataset(data_file, meta_file, split="val", window_size=window_size)
    test_ds = VSMDataset(data_file, meta_file, split="test", window_size=window_size)
    
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_ld, val_ld, test_ld

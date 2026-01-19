import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# --- 1. Dataset ---
class StepwiseDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Target: d_ratio
        val = float(row['d_ratio']) - 1.0
        
        # Condition: k_left
        k = int(row['k_left'])
        
        return {
            'x': torch.tensor([val], dtype=torch.float32), # [1]
            'k': torch.tensor(k, dtype=torch.long)         # Scalar
        }

# --- 2. Model (Small MLP) ---
class StepwiseDiffusionNet(nn.Module):
    def __init__(self, hidden_dim=128, n_steps=1000):
        super().__init__()
        
        # Embeddings
        self.time_embed = nn.Embedding(n_steps, hidden_dim)
        self.k_embed = nn.Embedding(20, hidden_dim)
        
        # Input Projection
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # MLP Backbone (ResNet Style)
        self.block1 = self._make_block(hidden_dim)
        self.block2 = self._make_block(hidden_dim)
        self.block3 = self._make_block(hidden_dim)
        
        # Output
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.SiLU()

    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, t, k):
        # x: [batch, 1]
        
        # 1. combination
        h = self.input_proj(x)
        t_emb = self.time_embed(t)
        k_emb = self.k_embed(k)
        
        cond = t_emb + k_emb
        
        # 2. ResNet Blocks
        h = h + cond
        h = h + self.block1(self.act(h))
        
        h = h + cond
        h = h + self.block2(self.act(h))
        
        h = h + cond
        h = h + self.block3(self.act(h))
        

        return self.out(self.act(h))

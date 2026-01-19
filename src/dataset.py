import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RedEnvelopeDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.max_people = 10
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        ratios = row[[f'ratio_{i}' for i in range(self.max_people)]].values.astype(np.float32)
        
        mask = row[[f'mask_{i}' for i in range(self.max_people)]].values.astype(np.float32)
        
        n_people = int(row['num_people'])
        
        total_amount = np.log1p(float(row['total_amount'])) # log(x+1)
        
        return {
            'x_0': torch.tensor(ratios),           # [10]
            'mask': torch.tensor(mask),            # [10]
            'n_people': torch.tensor(n_people),    # Scalar
            'total_amount': torch.tensor(total_amount, dtype=torch.float32) # Scalar

        }

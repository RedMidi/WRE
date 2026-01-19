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
        
        # 1. 获取 Target (Ratio) - 模型要拟合的目标
        # 提取 ratio_0 到 ratio_9
        ratios = row[[f'ratio_{i}' for i in range(self.max_people)]].values.astype(np.float32)
        
        # 2. 获取 Mask - 哪里是填充的
        mask = row[[f'mask_{i}' for i in range(self.max_people)]].values.astype(np.float32)
        
        # 3. 获取条件 (Conditions)
        # 条件 A: 人数 (归一化或者直接用 Embedding，这里我们用 Embedding，传入整数即可)
        n_people = int(row['num_people'])
        
        # 条件 B: 总金额 (取对数，平滑数值范围)
        total_amount = np.log1p(float(row['total_amount'])) # log(x+1)
        
        return {
            'x_0': torch.tensor(ratios),           # [10]
            'mask': torch.tensor(mask),            # [10]
            'n_people': torch.tensor(n_people),    # Scalar
            'total_amount': torch.tensor(total_amount, dtype=torch.float32) # Scalar
        }
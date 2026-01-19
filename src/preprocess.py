import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def preprocess_data(input_path, output_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    
    rename_map = {
        'source_file': 'session_id', 'image_id': 'session_id',
        'money': 'amount', 'amount': 'amount',
        'order_id': 'order', 'id': 'order'
    }
    df = df.rename(columns=rename_map)
    
    df = df.sort_values(by=['session_id', 'order'])
    
    processed_rows = []
    
    grouped = df.groupby('session_id')
    
    print("Processing sequential logic...")
    for session_id, group in tqdm(grouped):
        amounts = group['amount'].values.astype(float)
        n_total = len(amounts)
        
        if n_total < 2:
            continue
            
        current_balance = amounts.sum()
        
        for i in range(n_total - 1):
            grab_amount = amounts[i]
            k_left = n_total - i
            
            current_avg = current_balance / k_left
            
            if current_avg > 0:
                d_ratio = grab_amount / current_avg
            else:
                d_ratio = 0
            
            processed_rows.append({
                'd_ratio': d_ratio,
                'k_left': k_left
            })
            
            current_balance -= grab_amount
            
    result_df = pd.DataFrame(processed_rows)
    print(f"Generated {len(result_df)} training samples.")
    
    mean_val = result_df['d_ratio'].mean()
    print(f"Stats: Mean d_ratio = {mean_val:.4f} (Expected ~1.0)")
    
    result_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_data(
        os.path.join(BASE_DIR, 'data', 'output.csv'),
        os.path.join(BASE_DIR, 'data', 'data_stepwise.csv')

    )

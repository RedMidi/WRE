import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def simulate_timestamps(input_path, output_path):
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    if 'source_file' in df.columns:
        grp_col = 'source_file'
    elif 'image_id' in df.columns:
        grp_col = 'image_id'
    else:
        grp_col = 'session_id'
        
    if 'order_id' in df.columns:
        df = df.sort_values(by=[grp_col, 'order_id'])
    elif 'id' in df.columns:
        df = df.sort_values(by=[grp_col, 'id'])

    print("Simulating timestamps...")
    
    all_arrival_times = []
    all_inter_arrivals = []
    
    grouped = df.groupby(grp_col)
    
    for _, group in tqdm(grouped):
        n_people = len(group)
        
        gaps = np.random.uniform(0.5, 2.0, size=n_people)
        
        arrival_times = np.cumsum(gaps)
        
        all_arrival_times.extend(arrival_times)
        all_inter_arrivals.extend(gaps)
        
    df['sim_arrival_time'] = all_arrival_times
    df['sim_inter_arrival'] = all_inter_arrivals
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simulate_timestamps(
        os.path.join(BASE_DIR, 'data', 'output.csv'),
        os.path.join(BASE_DIR, 'data', 'data_with_time.csv')

    )

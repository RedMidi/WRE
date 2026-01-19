import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def simulate_timestamps(input_path, output_path):
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 兼容列名
    if 'source_file' in df.columns:
        grp_col = 'source_file'
    elif 'image_id' in df.columns:
        grp_col = 'image_id'
    else:
        # 如果是 session_id
        grp_col = 'session_id'
        
    # 确保按抢包顺序排序
    if 'order_id' in df.columns:
        df = df.sort_values(by=[grp_col, 'order_id'])
    elif 'id' in df.columns:
        df = df.sort_values(by=[grp_col, 'id'])

    print("⏳ Simulating timestamps (Uniform 0.5s - 2.0s)...")
    
    # 容器
    all_arrival_times = []
    all_inter_arrivals = []
    
    # 分组处理
    grouped = df.groupby(grp_col)
    
    for _, group in tqdm(grouped):
        n_people = len(group)
        
        # 1. 生成间隔 (Inter-arrival times)
        # 第一个人的时间通常不是 0，而是从发出到抢到的延迟
        # 这里假设第一个人也是 0.5-2.0s 抢到
        gaps = np.random.uniform(0.5, 2.0, size=n_people)
        
        # 2. 计算绝对时间 (Arrival times)
        # t1 = gap1, t2 = gap1 + gap2 ...
        arrival_times = np.cumsum(gaps)
        
        all_arrival_times.extend(arrival_times)
        all_inter_arrivals.extend(gaps)
        
    # 将模拟结果写入 DataFrame
    df['sim_arrival_time'] = all_arrival_times
    df['sim_inter_arrival'] = all_inter_arrivals
    
    # 保存
    df.to_csv(output_path, index=False)
    print(f"✅ Data with simulated time saved to: {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simulate_timestamps(
        os.path.join(BASE_DIR, 'data', 'output.csv'),
        os.path.join(BASE_DIR, 'data', 'data_with_time.csv')
    )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, kstest
import os

def analyze_poisson_process():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_with_time.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results_temporal')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH)
    
    gaps = df['sim_inter_arrival'].values
    
    mean_gap = np.mean(gaps)
    lambda_hat = 1.0 / mean_gap
    print(f"Estimated Lambda (Rate): {lambda_hat:.4f} grabs/second")
    print(f"Mean Inter-arrival Time: {mean_gap:.4f} seconds")
    
    loc, scale = expon.fit(gaps)
    
    plt.figure(figsize=(10, 6))
    
    count, bins, ignored = plt.hist(gaps, bins=50, density=True, alpha=0.6, color='blue', label='Simulated Data')
    
    x = np.linspace(min(gaps), max(gaps), 100)
    pdf = expon.pdf(x, loc=loc, scale=scale)
    plt.plot(x, pdf, 'r-', lw=2, label=f'Fitted Exponential (scale={scale:.2f})')
    
    plt.title("Inter-arrival Time Distribution (Poisson Check)")
    plt.xlabel("Time Gap (seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "poisson_check.png")
    plt.savefig(save_path)
    print(f"📊 Poisson plot saved to: {save_path}")
    
    ks_stat, p_value = kstest(gaps, 'expon', args=(loc, scale))
    print(f"\n=== KS Test Results ===")
    print(f"Statistic: {ks_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Reject H0. Data is likely NOT strictly Exponential.")
        print("(This is expected because we simulated Uniform(0.5, 2.0), not Exponential!)")
    else:
        print("Result: Cannot reject H0. Fits Exponential well.")

if __name__ == "__main__":

    analyze_poisson_process()

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
from tqdm import tqdm
from train_stepwise import DiffusionSchedules
from model_stepwise import StepwiseDiffusionNet

def simulate_red_envelope(model, diff_utils, total_money, n_people, device):
    model.eval()
    remaining_money = total_money
    results = []
    
    for k in range(n_people, 1, -1):
        k_tensor = torch.tensor([k], device=device).long()
        
        x = torch.randn(1, 1, device=device) # Start with noise
        
        for t in reversed(range(1000)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise = model(x, t_tensor, k_tensor)
            
            # DDPM Update
            beta = diff_utils.betas[t]
            alpha = diff_utils.alphas[t]
            alpha_bar = diff_utils.alphas_cumprod[t]
            
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = coef1 * (x - coef2 * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        
        d_ratio = x.item() + 1.0 
        
        current_avg = remaining_money / k
        grab_amount = d_ratio * current_avg
        
        grab_amount = max(0.01, grab_amount)
        max_allowed = remaining_money - 0.01 * (k - 1)
        grab_amount = min(grab_amount, max_allowed)
        
        results.append(grab_amount)
        remaining_money -= grab_amount
        
    results.append(remaining_money)
    
    return results

def run_evaluation():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stepwise_model.pth')
    
    RAW_CSV = os.path.join(BASE_DIR, 'data', 'output.csv')
    df_real = pd.read_csv(RAW_CSV)
    
    real_ratios = []
    print("Loading real data...")
    df_real = df_real.rename(columns={'money': 'amount', 'amount': 'amount'})
    for session_id, group in df_real.groupby('source_file' if 'source_file' in df_real.columns else 'image_id'):
        if len(group) == 10:
            amounts = group['amount'].values.astype(float)
            total = amounts.sum()
            if total > 0:
                real_ratios.extend(amounts / total)

    diff_utils = DiffusionSchedules(device=DEVICE)
    model = StepwiseDiffusionNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    print("Generating 500 simulated sessions (10 people)...")
    fake_ratios = []
    for _ in tqdm(range(500)):
        amounts = simulate_red_envelope(model, diff_utils, 100.0, 10, DEVICE)
        amounts = np.array(amounts)
        ratios = amounts / amounts.sum()
        fake_ratios.extend(ratios)

    print("Performing KS Test...")
    ks_stat, p_value = ks_2samp(real_ratios, fake_ratios)
    print(f"KS Statistic: {ks_stat:.4f}, P-Value: {p_value:.4f}")

    plt.figure(figsize=(10,6))
    plt.hist(real_ratios, bins=50, density=True, alpha=0.5, color='blue', label='Real WeChat Data')
    plt.hist(fake_ratios, bins=50, density=True, alpha=0.5, color='orange', label='Diffusion Simulation')
    
    plt.title(f"Final Validation: Real vs. Diffusion (KS p={p_value:.3f})")
    plt.xlabel("Money Ratio")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = os.path.join(BASE_DIR, "results", "final_comparison.png")
    plt.savefig(save_path)

if __name__ == "__main__":

    run_evaluation()

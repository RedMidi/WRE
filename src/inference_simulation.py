import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
from tqdm import tqdm
from train_stepwise import DiffusionSchedules # 引用工具类
from model_stepwise import StepwiseDiffusionNet

def simulate_red_envelope(model, diff_utils, total_money, n_people, device):
    """
    模拟一局完整的抢红包过程
    """
    model.eval()
    remaining_money = total_money
    results = []
    
    # 比如 10个人，k 从 10 倒数到 2
    # 倒数第1个人直接拿剩下的，不用预测
    for k in range(n_people, 1, -1):
        # 1. 准备条件
        k_tensor = torch.tensor([k], device=device).long()
        
        # 2. Diffusion 采样 (生成 x)
        # x 代表 normalized d_ratio
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
        
        # 3. 还原数值
        # 也就是反归一化: x = (d_ratio - 1.0)
        d_ratio = x.item() + 1.0 
        
        # 4. 计算本轮金额
        current_avg = remaining_money / k
        grab_amount = d_ratio * current_avg
        
        # --- 微信逻辑修正 (Hard Constraints) ---
        # 必须大于 0.01
        grab_amount = max(0.01, grab_amount)
        # 必须给后面的人留够 0.01 * (k-1)
        max_allowed = remaining_money - 0.01 * (k - 1)
        grab_amount = min(grab_amount, max_allowed)
        
        # 记录
        results.append(grab_amount)
        remaining_money -= grab_amount
        
    # 最后一个人拿走所有
    results.append(remaining_money)
    
    return results

def run_evaluation():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stepwise_model.pth')
    
    # 1. 读取真实数据 (为了拿到真实的 Ratio 分布)
    RAW_CSV = os.path.join(BASE_DIR, 'data', 'output.csv')
    df_real = pd.read_csv(RAW_CSV)
    
    # 我们只关心 10人局的数据来做对比 (控制变量法)
    # 假设 session_id 分组后长度为 10 的
    real_ratios = []
    print("Loading real data...")
    # 简单的逻辑：直接算 money / total
    # 这里的 total 需要按 session 求和
    # 我们可以复用 preprocess 的逻辑，或者直接粗暴一点
    # 为了简单，我们假设 output.csv 里有 money 列
    
    # 重新计算一下真实数据的 ratio
    df_real = df_real.rename(columns={'money': 'amount', 'amount': 'amount'}) # 兼容列名
    for session_id, group in df_real.groupby('source_file' if 'source_file' in df_real.columns else 'image_id'):
        if len(group) == 10: # 只取10人局对比
            amounts = group['amount'].values.astype(float)
            total = amounts.sum()
            if total > 0:
                real_ratios.extend(amounts / total)

    # 2. 生成模拟数据
    diff_utils = DiffusionSchedules(device=DEVICE)
    model = StepwiseDiffusionNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    print("Generating 500 simulated sessions (10 people)...")
    fake_ratios = []
    for _ in tqdm(range(500)):
        # 模拟：10人分100块
        amounts = simulate_red_envelope(model, diff_utils, 100.0, 10, DEVICE)
        amounts = np.array(amounts)
        ratios = amounts / amounts.sum()
        fake_ratios.extend(ratios)

    # 3. 终极对比图 & KS Test
    print("Performing KS Test...")
    ks_stat, p_value = ks_2samp(real_ratios, fake_ratios)
    print(f"KS Statistic: {ks_stat:.4f}, P-Value: {p_value:.4f}")

    plt.figure(figsize=(10,6))
    # 绘制真实数据 (蓝色)
    plt.hist(real_ratios, bins=50, density=True, alpha=0.5, color='blue', label='Real WeChat Data')
    # 绘制生成数据 (橙色)
    plt.hist(fake_ratios, bins=50, density=True, alpha=0.5, color='orange', label='Diffusion Simulation')
    
    plt.title(f"Final Validation: Real vs. Diffusion (KS p={p_value:.3f})")
    plt.xlabel("Money Ratio")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = os.path.join(BASE_DIR, "results", "final_comparison.png")
    plt.savefig(save_path)
    print(f"🏆 Final plot saved to: {save_path}")

if __name__ == "__main__":
    run_evaluation()
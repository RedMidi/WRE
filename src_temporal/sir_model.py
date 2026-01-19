import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import os

def sir_deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def fit_sir_cumulative(t, beta, gamma):
    N = 10.0 
    I0, R0 = 1.0, 0.0 
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    ret = odeint(sir_deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return R

def run_sir_analysis():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_with_time.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results_temporal')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Reading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    possible_cols = ['session_id', 'source_file', 'image_id', 'file_name']
    grp_col = None
    for col in possible_cols:
        if col in df.columns:
            grp_col = col
            break
    
    all_time_points = []
    all_grab_counts = []
    
    grouped = df.groupby(grp_col)
    
    valid_sessions = 0
    
    for _, group in grouped:
        if len(group) == 10:
            times = group['sim_arrival_time'].values
            all_time_points.extend(times)
            all_grab_counts.extend(np.arange(1, 11))
            valid_sessions += 1

    x_data = np.array(all_time_points)
    y_data = np.array(all_grab_counts)
    
    sort_idx = np.argsort(x_data)
    x_data_sorted = x_data[sort_idx]
    y_data_sorted = y_data[sort_idx]

    print(f"Fitting SIR model on {valid_sessions} sessions...")
    
    p0 = [0.5, 0.1] 
    bounds = ([0, 0], [10, 10])
    
    popt, pcov = curve_fit(fit_sir_cumulative, x_data_sorted, y_data_sorted, p0=p0, bounds=bounds)
    beta_est, gamma_est = popt
    
    print(f"\n=== SIR Fitting Results ===")
    print(f"Beta (Infection Rate): {beta_est:.4f}")
    print(f"Gamma (Recovery Rate): {gamma_est:.4f}")
    print(f"R0 (Reproduction Num): {beta_est/gamma_est:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data_sorted, y_data_sorted, s=1, alpha=0.05, color='blue', label='Real Data (Simulated Time)')
    
    t_eval = np.linspace(0, max(x_data), 100)
    fitted_curve = fit_sir_cumulative(t_eval, beta_est, gamma_est)
    plt.plot(t_eval, fitted_curve, 'r-', lw=3, label=f'SIR Model Fit')
    
    plt.title(f"SIR Model Fitting (Beta={beta_est:.2f}, Gamma={gamma_est:.2f})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cumulative Grabs")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "sir_fitting.png")
    plt.savefig(save_path)

if __name__ == "__main__":

    run_sir_analysis()

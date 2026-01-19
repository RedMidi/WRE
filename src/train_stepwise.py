import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

class DiffusionSchedules:
    def __init__(self, n_steps=1000, device='cpu'):
        self.n_steps = n_steps
        self.betas = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

from model_stepwise import StepwiseDataset, StepwiseDiffusionNet

def train():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'data_stepwise.csv')
    SAVE_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Config
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 1e-3
    
    # Init
    dataset = StepwiseDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    diff_utils = DiffusionSchedules(device=DEVICE)
    model = StepwiseDiffusionNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = torch.nn.MSELoss()
    
    print("Start Sequential Training...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in loader:
            x_0 = batch['x'].to(DEVICE) # [batch, 1]
            k = batch['k'].to(DEVICE)
            
            t = torch.randint(0, 1000, (x_0.shape[0],), device=DEVICE).long()
            x_t, noise = diff_utils.q_sample(x_0, t)
            
            pred_noise = model(x_t, t, k)
            loss = mse(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f}")
            
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "stepwise_model.pth"))
    
    plt.plot(loss_history)
    plt.savefig(os.path.join(BASE_DIR, "stepwise_loss.png"))
    print("Done.")

if __name__ == "__main__":

    train()

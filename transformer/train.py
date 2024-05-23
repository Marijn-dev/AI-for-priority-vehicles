
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt


from dataset import TrajectoryDataset
from model import TrajectoryAttentionModel

# Hyperparameters and options
param_grid = {
    'learning_rate': [0.001, 0.0005],
    'batch_size': [16, 32],
    'num_layers': [2, 4],
    'heads': [4, 8],
    'dropout': [0.1, 0.2]
}
hyperparameter_sets = list(ParameterGrid(param_grid))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise(data, noise_level=0.01):
    noise = noise_level * torch.randn(data.size())
    return data + noise

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for obs_traj, pred_traj in dataloader:
        obs_traj, pred_traj = obs_traj.to(device), pred_traj.to(device)
        obs_traj = add_noise(obs_traj)  # Noise injection
        optimizer.zero_grad()
        output = model(obs_traj, pred_traj.size(1))
        loss = criterion(output, pred_traj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def plot_trajectories(true_data, pred_data, title='Comparison of Trajectories'):
    """Plots the comparison of true and predicted trajectories for selected vehicles."""
    plt.figure(figsize=(10, 5))
    # 假设 true_data 和 pred_data 已经被过滤为只包含选定车辆的数据
    for true_traj, pred_traj in zip(true_data, pred_data):
        plt.plot(true_traj[:, 0], true_traj[:, 1], 'bo-', label='Ground Truth')
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'ro-', label='Predicted')
        plt.legend()
        plt.title(title)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()


def evaluate_and_plot(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for obs_traj, pred_traj in dataloader:
        obs_traj, pred_traj = obs_traj.to(device), pred_traj.to(device)
        output = model(obs_traj, pred_traj.size(1))
        loss = criterion(output, pred_traj)
        total_loss += loss.item()

        # Convert tensors to numpy for plotting
        true_coords = pred_traj.cpu().detach().numpy()
        pred_coords = output.cpu().detach().numpy()
        plot_trajectories(true_coords, pred_coords)  # Plot for each batch

    return total_loss / len(dataloader)


# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for obs_traj, pred_traj in dataloader:
#             obs_traj, pred_traj = obs_traj.to(device), pred_traj.to(device)
#             output = model(obs_traj, pred_traj.size(1))
#             loss = criterion(output, pred_traj)
#             total_loss += loss.item()
#     return total_loss / len(dataloader)

def main():
    # Load the data
    directory = r'C:\5ARIP10\AI-for-priority-vehicles\data'
    dataset = TrajectoryDataset(directory)
    criterion = nn.MSELoss()

    # best_model = None
    # best_loss = float('inf')
    for params in hyperparameter_sets:
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        model = TrajectoryAttentionModel(
            embed_size=128,
            num_layers=params['num_layers'],
            heads=params['heads'],
            forward_expansion=4,
            dropout=params['dropout'],
            device=device,
            max_length=dataset.obs_len + dataset.pred_len
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        for epoch in range(100):  # Training for a fixed number of epochs
            train_loss = train(model, dataloader, criterion, optimizer, device)
            print(f'Epoch {epoch}, Train Loss: {train_loss}')

        val_loss = evaluate_and_plot(model, dataloader, criterion, device)
        print(f'Validation Loss: {val_loss} with params {params}')

        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model = model

    # torch.save(best_model.state_dict(), 'best_trajectory_model.pth')
    print('Training completed.')

if __name__ == "__main__":
    main()

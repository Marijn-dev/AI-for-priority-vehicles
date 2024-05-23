
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch

class TrajectoryDataset(Dataset):
    def __init__(self, directory, obs_len=5, pred_len=5):
        """
        Args:
            directory (str): file path
            obs_len (int): 
            pred_len (int):
        """
        self.data = []
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.load_files(directory)

    def load_files(self, directory):
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                self.process_file(df)

    def process_file(self, df):
        # 
        df = df.sort_values(by=['Actor ID', 'Time'])  #
        for actor_id in df['Actor ID'].unique():
            actor_data = df[df['Actor ID'] == actor_id]
            # 
            num_windows = len(actor_data) - (self.obs_len + self.pred_len) + 1
            for i in range(num_windows):
                window = actor_data.iloc[i:i+self.obs_len+self.pred_len]
                obs = window[['X', 'Y']].values[:self.obs_len]
                pred = window[['X', 'Y']].values[self.obs_len:]
                self.data.append((obs, pred))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, pred = self.data[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32)

def get_dataloader(directory, batch_size=32):
    dataset = TrajectoryDataset(directory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    # 
    dataloader = get_dataloader(r'C:\5ARIP10\AI-for-priority-vehicles\data')
    for obs_traj, pred_traj in dataloader:
        print(f'Observation: {obs_traj.shape}, Prediction: {pred_traj.shape}')

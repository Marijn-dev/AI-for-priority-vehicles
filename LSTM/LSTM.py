import torch.utils
import torch.utils.data
from datapreprocessing import create_data

import csv
import os
import pandas as pd
import numpy as np
import torch
import torch
import torch.nn as nn


import torch
import torch.nn as nn


##### LOAD IN DATA ####
cwd = os.getcwd()
file1 = cwd + '/LSTM/relative_coordinates_T20_1.csv'
past_timesegments = 5
future_timesegments = 1
batch_size = 2

X, y = create_data(file1,past_timesegments,future_timesegments)

# make dataloader from numpy arrays
X_tensor, y_tensor = torch.from_numpy(X), torch.from_numpy(y)
X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)

# get test inputs to run through model
batch = iter(train_loader)
X, y = next(batch)



# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cell = nn.RNN(input_size, hidden_size,  num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Forward pass through the RNN layer
        out, hidden = self.rnn_cell(x, hidden)
        # Reshape the output to fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Define input, hidden, and output sizes
input_size = 2  # Size of input vectors (x and y coordinates)
hidden_size = 20  # Size of hidden state (hyperparameter)
output_size = 2  # Size of output vectors (x and y coordinates)
num_layers = 2 # amount of layers (hyperparameter)

# Create an instance of the RNN model
rnn_model = SimpleRNN(input_size, hidden_size, output_size, num_layers)


hidden = rnn_model.init_hidden(batch_size)
output, _ = rnn_model(X, hidden)
print("Output shape:", output.shape)
print(output[-1:, :]) # --> Last time step)


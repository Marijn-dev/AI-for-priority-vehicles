import torch.utils
import torch.utils.data

import csv
import os
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
from datapreprocessing import create_data

def prediction(input):
    # Define the LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments=4):
            super(SimpleLSTM, self).__init__()
            self.future_timesegments = future_timesegments
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.LSTM_cell = nn.LSTM(input_size, hidden_size,  num_layers, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            # self.fc2 = nn.Linear(hidden_size, 50)
            self.fc3 = nn.Linear(hidden_size,output_size)


        def forward(self, x, hidden):
            # Forward pass through the RNN layer
            out, hidden = self.LSTM_cell(x, hidden)
            # Reshape the output to fit into the fully connected layer
            # out = out.contiguous().view(-1, self.hidden_size) many to one
            out = out[:, -self.future_timesegments:, :] # --> Last time step 
            # out = F.relu(self.fc1(out))
            out = F.relu(self.fc1(out))
            # out = F.sigmoid(self.fc2(out))
            out = self.fc3(out)
            # print(out.shape)
            return out, hidden

        def init_hidden(self, x):
            # Initialize hidden state with zeros
            return (torch.zeros(self.num_layers, x.size(0), self.hidden_size),torch.zeros(self.num_layers, x.size(0), self.hidden_size))

    # # Define input, hidden, and output sizes
    # input_size = 2  # Size of input vectors (x and y coordinates)
    # hidden_size = 500  # Size of hidden state (hyperparameter)
    # output_size = 2  # Size of output vectors (x and y coordinates)
    # num_layers = 2 # amount of layers (hyperparameter)
    # future_timesegments = 4
    # Create an instance of the LSTM model
    # LSTM_model = SimpleLSTM(input_size, hidden_size, output_size, num_layers, future_timesegments)
    LSTM_model = torch.load('FINAL_MODELS/LSTM_PAST5_FUTURE4_H500_L2RELU_V2.pt') # Save
    hidden = LSTM_model.init_hidden(input)
    future_pred, _ = LSTM_model(input,hidden)
    return future_pred

# Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments=4):
        super(SimpleLSTM, self).__init__()
        self.future_timesegments = future_timesegments
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM_cell = nn.LSTM(input_size, hidden_size,  num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 50)
        self.fc3 = nn.Linear(hidden_size,output_size)


    def forward(self, x, hidden):
        # Forward pass through the RNN layer
        out, hidden = self.LSTM_cell(x, hidden)
        # Reshape the output to fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_size) many to one
        out = out[:, -self.future_timesegments:, :] # --> Last time step 
        # out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        # out = F.sigmoid(self.fc2(out))
        out = self.fc3(out)
        # print(out.shape)
        return out, hidden

    def init_hidden(self, x):
        # Initialize hidden state with zeros
        return (torch.zeros(self.num_layers, x.size(0), self.hidden_size),torch.zeros(self.num_layers, x.size(0), self.hidden_size))



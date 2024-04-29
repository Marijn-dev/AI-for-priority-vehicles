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
batch_size = 1

X, y = create_data(file1,past_timesegments,future_timesegments)

# make dataloader from numpy arrays
X_tensor, y_tensor = torch.from_numpy(X), torch.from_numpy(y)
X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)

# get test inputs to run through model

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments):
        super(SimpleRNN, self).__init__()
        self.future_timesegments = future_timesegments
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cell = nn.RNN(input_size, hidden_size,  num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Forward pass through the RNN layer
        out, hidden = self.rnn_cell(x, hidden)
        # Reshape the output to fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_size) many to one
        out = out[:, -future_timesegments:, :] # --> Last time step 
        out = self.fc(out)
        # print(out.shape)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Define input, hidden, and output sizes
input_size = 2  # Size of input vectors (x and y coordinates)
hidden_size = 20  # Size of hidden state (hyperparameter)
output_size = 2  # Size of output vectors (x and y coordinates)
num_layers = 2 # amount of layers (hyperparameter)
learning_rate = 0.01
num_epochs = 150

# Create an instance of the RNN model
rnn_model = SimpleRNN(input_size, hidden_size, output_size, num_layers, future_timesegments)

# ### Test output
# hidden = rnn_model.init_hidden(batch_size)
# output, _ = rnn_model(X, hidden)
# print("Output shape:", output.shape)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (past, future) in enumerate(train_loader):  
        # # origin shape: [N, 1, 28, 28]
        # # resized: [N, 28, 28]
        # images = images.reshape(-1, past_timesegments, input_size)
        # labels = labels
        # Forward pass
        
        hidden = rnn_model.init_hidden(batch_size)
        # if i == 21:
        #     hidden = torch.zeros(2, 2, 20)
        future_pred, _ = rnn_model(past,hidden)
        loss = criterion(future_pred, future)
        epoch_loss += loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('epoch: {epoch}, Loss: {epoch_loss}'.format(epoch=epoch+1,epoch_loss=epoch_loss))

# Test the model
batch = iter(train_loader)
X, y = next(batch)
print(X)
with torch.no_grad():
    hidden = rnn_model.init_hidden(batch_size)
    prediction, _ = rnn_model(X,hidden)
    print('Prediction: {prediction}, truth: {y}'.format(prediction=prediction,y=y))

    # for i, (past, future) in enumerate(train_loader):  
    #     # # origin shape: [N, 1, 28, 28]
    #     # # resized: [N, 28, 28]
    #     # images = images.reshape(-1, past_timesegments, input_size)
    #     # labels = labels
    #     # Forward pass
    #     if i == 10:
    #         hidden = rnn_model.init_hidden(batch_size)
    #         # if i == 21:
    #         #     hidden = torch.zeros(2, 2, 20)
    #         future_pred, _ = rnn_model(past,hidden)
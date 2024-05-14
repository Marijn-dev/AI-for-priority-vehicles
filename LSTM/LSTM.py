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
import torch.nn.functional as F


import torch
import torch.nn as nn


##### LOAD IN DATA ####
cwd = os.getcwd()
file1 = cwd + '/data/Coordinates_T30_run_1.csv'
past_timesegments = 5
future_timesegments = 3
batch_size = 15

X_train, y_train = create_data(file1,past_timesegments,future_timesegments)
train_val_ratio = 500 # ratio train validation
datalength = 599  # amount of data sets (CSV files) 

for i in range(1,datalength):
    file = cwd + '/data/Coordinates_T30_run_' + str(i+1)+'.csv'
    X_temp, y_temp = create_data(file1,past_timesegments,future_timesegments)

    # train set
    if i < train_val_ratio:
        X_train = np.concatenate((X_train,X_temp),axis=0)
        y_train = np.concatenate((y_train,y_temp),axis=0)
    elif i == train_val_ratio:
        X_val, y_val = X_temp, y_temp
    else:
        X_val = np.concatenate((X_val,X_temp),axis=0)
        y_val = np.concatenate((y_val,y_temp),axis=0)

# make dataloader from numpy arrays
## Trainloader
X_tensor, y_tensor = torch.from_numpy(X_train), torch.from_numpy(y_train)
X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)

## Validation loader
X_tensor, y_tensor = torch.from_numpy(X_val), torch.from_numpy(y_val)
X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)


# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments):
        super(SimpleRNN, self).__init__()
        self.future_timesegments = future_timesegments
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cell = nn.RNN(input_size, hidden_size,  num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 150)
        # self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(150,output_size)


    def forward(self, x, hidden):
        # Forward pass through the RNN layer
        out, hidden = self.rnn_cell(x, hidden)
        # Reshape the output to fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_size) many to one
        out = out[:, -future_timesegments:, :] # --> Last time step 
        out = F.relu(self.fc1(out))
        # out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc3(out)
        # print(out.shape)
        return out, hidden

    def init_hidden(self, x):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, x.size(0), self.hidden_size)

# Define input, hidden, and output sizes
input_size = 2  # Size of input vectors (x and y coordinates)
hidden_size = 350  # Size of hidden state (hyperparameter)
output_size = 2  # Size of output vectors (x and y coordinates)
num_layers = 1 # amount of layers (hyperparameter)
learning_rate = 0.0008
num_epochs = 15

# Create an instance of the RNN model
rnn_model = SimpleRNN(input_size, hidden_size, output_size, num_layers, future_timesegments)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss_val = 0
    epoch_loss_train = 0
    for i, (past, future) in enumerate(train_loader):  
       
        hidden = rnn_model.init_hidden(past)
        future_pred, _ = rnn_model(past,hidden)
        loss = criterion(future_pred, future)
        epoch_loss_train += loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    # validation losss
    # with torch.no_grad():
    for i, (past, future) in enumerate(val_loader):  
        hidden = rnn_model.init_hidden(past)
        # print(i)
        # if i == len(val_loader)-1:
        #     hidden = torch.zeros(2, 3, hidden_size)
        future_pred, _ = rnn_model(past,hidden)
        loss = criterion(future_pred, future)
        epoch_loss_val += loss.item()
    print('epoch: {epoch}: train_loss: {epoch_loss_train}, val_loss: {epoch_loss_val}'.format(epoch=epoch+1,epoch_loss_train=epoch_loss_train/len(train_loader),epoch_loss_val=epoch_loss_val/len(val_loader)))

# model_scripted = torch.jit.script(rnn_model) # Export to TorchScript
torch.save(rnn_model,'models/RNN_PAST8_FUTURE3_H350_L1.pt') # Save


# Test the model
batch = iter(train_loader)
X, y = next(batch)
with torch.no_grad():
    hidden = rnn_model.init_hidden(X)
    prediction, _ = rnn_model(X,hidden)
    print('Prediction: {prediction}, truth: {y}'.format(prediction=prediction,y=y))

#     for i, (past, future) in enumerate(train_loader):  
#         # # origin shape: [N, 1, 28, 28]
#         # # resized: [N, 28, 28]
#         # images = images.reshape(-1, past_timesegments, input_size)
#         # labels = labels
#         # Forward pass
#         if i == 10:
#             hidden = rnn_model.init_hidden(batch_size)
#             # if i == 21:
#             #     hidden = torch.zeros(2, 2, 20)
#             future_pred, _ = rnn_model(past,hidden)

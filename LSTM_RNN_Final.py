import torch.utils
import torch.utils.data
from LSTM.datapreprocessing import create_data

import csv
import os
import pandas as pd
print(pd.__version__)
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
future_timesegments = 4
batch_size = 12

X_train, y_train = create_data(file1,past_timesegments,future_timesegments)
train_length = 500 # amount of datasets for training
validation_length = 75  # amount of data sets for validation
test_length = 25    # amount of sets used for test set
datalength = 650

for i in range(1,datalength):
    file = cwd + '/data/Coordinates_T30_run_' + str(i+1)+'.csv'
    X_temp, y_temp = create_data(file1,past_timesegments,future_timesegments)

    # train set
    if i < train_length or i > train_length+validation_length:
        X_train = np.concatenate((X_train,X_temp),axis=0)
        y_train = np.concatenate((y_train,y_temp),axis=0)
    elif i == train_length:
        X_val, y_val = X_temp, y_temp
    elif i > train_length and i < (train_length+validation_length):
        X_val = np.concatenate((X_val,X_temp),axis=0)
        y_val = np.concatenate((y_val,y_temp),axis=0)
    elif i == (train_length+validation_length):
        X_test, y_test = X_temp, y_temp
    else:
        X_test = np.concatenate((X_test,X_temp),axis=0)
        y_test = np.concatenate((y_test,y_temp),axis=0)

# make dataloader from numpy arrays
## Trainloader
X_tensor, y_tensor = torch.from_numpy(X_train), torch.from_numpy(y_train)
X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True)

## Validation loader
X_tensor, y_tensor = torch.from_numpy(X_val), torch.from_numpy(y_val)
X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True)

# Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments):
        super(SimpleLSTM, self).__init__()
        self.future_timesegments = future_timesegments
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM_cell = nn.LSTM(input_size, hidden_size,  num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 150)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(150,2)
        # self.fc2 = nn.Linear(hidden_size, 50)
        self.fc3 = nn.Linear(500,100)
        self.fc4 = nn.Linear(100,output_size)


    def forward(self, x, hidden):
        # Forward pass through the RNN layer
        out, hidden = self.LSTM_cell(x, hidden)
        # Reshape the output to fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_size) many to one
        out = out[:, -future_timesegments:, :] # --> Last time step 
        # out = F.relu(self.fc1(out))
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        # out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        # out = self.fc3(out)
        # out = self.dropout(out)
        # out = self.fc4(out)
        # # print(out.shape)
        return out, hidden

    def init_hidden(self, x):
        # Initialize hidden state with zeros
        return (torch.zeros(self.num_layers, x.size(0), self.hidden_size),torch.zeros(self.num_layers, x.size(0), self.hidden_size))

# Define input, hidden, and output sizes
input_size = 2  # Size of input vectors (x and y coordinates)
hidden_size = 256  # Size of hidden state (hyperparameter)
output_size = 2  # Size of output vectors (x and y coordinates)
num_layers = 2 # amount of layers (hyperparameter)

# Create an instance of the LSTM model
LSTM_model = SimpleLSTM(input_size, hidden_size, output_size, num_layers, future_timesegments)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# LSTM_model.to(device)

Training = False
if Training == True:


    learning_rate = 0.005
    num_epochs = 3
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=learning_rate)  

    # Train the model
    n_total_steps = len(train_loader)
    val_losses = []
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss_val = 0
        epoch_loss_train = 0
        for i, (past, future) in enumerate(train_loader):  
            hidden = LSTM_model.init_hidden(past)
            future_pred, _ = LSTM_model(past,hidden)
            loss = criterion(future_pred, future)
            epoch_loss_train += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, (past, future) in enumerate(val_loader):  
            hidden = LSTM_model.init_hidden(past)

            future_pred, _ = LSTM_model(past,hidden)
            loss = criterion(future_pred, future)
            epoch_loss_val += loss.item()
        train_losses.append(epoch_loss_train/len(train_loader))
        val_losses.append(epoch_loss_val/len(val_loader))
        print('epoch: {epoch}: train_loss: {epoch_loss_train}, val_loss: {epoch_loss_val}'.format(epoch=epoch+1,epoch_loss_train=epoch_loss_train/len(train_loader),epoch_loss_val=epoch_loss_val/len(val_loader)))
    input = torch.tensor([[[  0.0000,   0.0000],
         [ -0.0506,  -3.6188],
         [ -0.0984,  -7.2159],
         [ -0.1446, -10.8376],
         [ -10.1925, -14.4658]]])
    hidden = LSTM_model.init_hidden(past)
    future_pred, _ = LSTM_model(past,hidden)
    loss = criterion(future_pred, future)
    print(input)
    print(future_pred)
    # model_scripted = torch.jit.script(rnn_model) # Export to TorchScript
    torch.save(LSTM_model,'LSTM_PAST5_FUTURE4_H500_L2_V2.pt') # Save

LSTM_model2 = torch.load('LSTM_PAST5_FUTURE4_H500_L2_V2.pt')
input = torch.tensor([[[  0.0000,   0.0000],
         [ -0.0506,  3.6188],
         [ -0.0984,  7.2159],
         [ -2.1446, 10.8376],
         [ -10.1925, 14.4658]]])
# input =  torch.tensor([[[ 0.0000e+00,  0.0000e+00],
#          [-5.5115e+00, -1.9052e-01],
#          [-1.2527e+01, -3.2767e-01],
#          [-1.8551e+01, -4.0936e-01],
#          [-2.0827e+01, -4.3579e-01]]])
print(input)
hidden = LSTM_model2.init_hidden(input)

future_pred, _ = LSTM_model2(input,hidden)
print(future_pred)

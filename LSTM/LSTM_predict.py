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

def prediction(input,global_path_to_model):
    # Define the RNN model
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments):
            super(SimpleRNN, self).__init__()
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
            out = out[:, -future_timesegments:, :] # --> Last time step 
            # out = F.relu(self.fc1(out))
            out = F.relu(self.fc1(out))
            # out = F.sigmoid(self.fc2(out))
            out = self.fc3(out)
            # print(out.shape)
            return out, hidden

        def init_hidden(self, x):
            # Initialize hidden state with zeros
            return (torch.zeros(self.num_layers, x.size(0), self.hidden_size),torch.zeros(self.num_layers, x.size(0), self.hidden_size))

    # Define input, hidden, and output sizes
    input_size = 2  # Size of input vectors (x and y coordinates)
    hidden_size = 500  # Size of hidden state (hyperparameter)
    output_size = 2  # Size of output vectors (x and y coordinates)
    num_layers = 2# amount of layers (hyperparameter)
    future_timesegments = 4
    model_name = 'AILSTM_PAST5_FUTURE4_H500_L2_RELU.pt'
    LSTM_model = SimpleRNN(input_size, hidden_size, output_size, num_layers, future_timesegments)
    LSTM_model = torch.load(global_path_to_model)
    hidden = LSTM_model.init_hidden(input)
    future_pred, _ = LSTM_model(input,hidden)
    return future_pred

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments):
        super(SimpleRNN, self).__init__()
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
        out = out[:, -4:, :] # --> Last time step 
        # out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        # out = F.sigmoid(self.fc2(out))
        out = self.fc3(out)
        # print(out.shape)
        return out, hidden

    def init_hidden(self, x):
        # Initialize hidden state with zeros
        return (torch.zeros(self.num_layers, x.size(0), self.hidden_size),torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    
# ###### TEST PERFORMANCE #####
# input = torch.tensor([[[ 0.0000e+00,  0.0000e+00],
#         [-1.6442e+00, -1.8951e-02],
#         [-5.6885e+00, -5.9998e-02],
#         [-9.8572e+00,  1.2701e-01],
#         [-1.4382e+01,  1.8763e+00]]])
# print('Output: ', prediction(input))

'''
### USE THIS TO EVAULATE MODEL PERFORMANCE ###
### LOAD IN MODEL ###


# Define input, hidden, and output sizes
input_size = 2  # Size of input vectors (x and y coordinates)
hidden_size = 500  # Size of hidden state (hyperparameter)
output_size = 2  # Size of output vectors (x and y coordinates)
num_layers = 2# amount of layers (hyperparameter)
future_timesegments = 4
model_name = 'LSTM_PAST5_FUTURE4_H500_L2_RELU.pt'
LSTM_model = SimpleRNN(input_size, hidden_size, output_size, num_layers, future_timesegments)
LSTM_model = torch.load('models/' + model_name)

##### LOAD IN DATA ####
cwd = os.getcwd()
file1 = cwd + '/data/Coordinates_T30_run_1.csv'
past_timesegments = 5
future_timesegments = 4
batch_size = 12

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

cost = np.zeros(len(val_loader))
cost_total = 0
criterion = nn.MSELoss()

for i, (past, future) in enumerate(val_loader): 
    
        
    hidden = LSTM_model.init_hidden(past)
    future_pred, _ = LSTM_model(past,hidden)
    if i == 10:
        print('past',past) 
        # print(future_pred)
        print('future',future)
    cost_total += criterion(future_pred, future)
    cost[i] = criterion(future_pred, future)

print('Mean cost: {cost}'.format(cost=np.mean(cost)))
print('Std cost: {cost}'.format(cost=np.std(cost)))
'''
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

if __name__ == '__main__':
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, future_timesegments):
            super(SimpleRNN, self).__init__()
            self.future_timesegments = future_timesegments
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn_cell = nn.RNN(input_size, hidden_size,  num_layers, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, 150)
            self.fc2 = nn.Linear(150, 50)
            self.fc3 = nn.Linear(50,output_size)


        def forward(self, x, hidden):
            out, hidden = self.rnn_cell(x, hidden)
            out = out[:, -future_timesegments:, :] # --> Last time step 
            out = F.relu(self.fc1(out))
            out = self.fc2(out)
            out = self.fc3(out)
            return out, hidden

        def init_hidden(self, x):
            return torch.zeros(self.num_layers, x.size(0), self.hidden_size)

    future_timesegments = 5 # use this to select which model to use
    model_name = 'RNN_PAST5_FUTURE' + str(future_timesegments) + '.pt'
    rnn_model = SimpleRNN(2, 250, 2, 3, future_timesegments) # create empty model to load pretrained model in
    rnn_model = torch.load('models/' + model_name)

    

    ### PURELY TO TEST VALUES AND TO SEE HOW SHAPE OF INPUTS AND OUTPUTS ###
    TEST_VALUES = True # set to false if you don't want to test values
    if TEST_VALUES:

        ##### LOAD IN DATA ####
        cwd = os.getcwd()
        file1 = cwd + '/data/Coordinates_T30_run_1.csv'
        past_timesegments = 5
        batch_size = 1

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
            
            # validation set
            elif i == train_val_ratio:
                X_val, y_val = X_temp, y_temp
            else:
                X_val = np.concatenate((X_val,X_temp),axis=0)
                y_val = np.concatenate((y_val,y_temp),axis=0)

        
        ### PUT DATA IN DATALOADERS ###
        X_tensor, y_tensor = torch.from_numpy(X_train), torch.from_numpy(y_train)
        X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)

        X_tensor, y_tensor = torch.from_numpy(X_val), torch.from_numpy(y_val)
        X_tensor, y_tensor = X_tensor.type(torch.FloatTensor), y_tensor.type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
        val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)
        print(len(val_loader))
        ### PUT SAMPLE THROUGH MODEL ###
        samples = [10,1010,2020,3030,4123,5666,6000] # select random datasamples
        for i, (past, future) in enumerate(val_loader):  
            if i in samples:
                hidden = rnn_model.init_hidden(past)
                future_pred, _ = rnn_model(past,hidden)
                print('Sample {i}: prediction:{future_pred}, truth:{future}'.format(i=i,future_pred=future_pred,future=future) )

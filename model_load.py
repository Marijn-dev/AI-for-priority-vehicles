import torch.utils
import torch.utils.data
from datapreprocessing import create_data
import matplotlib.pyplot as plt

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

def test_model(rnn_model,future_timesegments):

    # model_name = 'RNN_PAST5_FUTURE' + str(future_timesegments) + '.pt'
    # rnn_model = SimpleRNN(2, hidden_size, 2, num_layers, future_timesegments) # create empty model to load pretrained model in
    # rnn_model = torch.load('models/' + model_name)

    

    ### PURELY TO TEST VALUES AND TO SEE HOW SHAPE OF INPUTS AND OUTPUTS ###
    TEST_VALUES = True # set to false if you dongti't want to test values
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
        samples = [10,1010,2020,3030,4123,5666,6000,3200] # select random datasamples
        criterion = nn.MSELoss()
        cost = 0
        # Create a figure and a grid of subplots
        fig, axs = plt.subplots(2, 4)

        # Flatten the 2D array of subplots into a 1D array for easy iteration
        axs = axs.flatten()
        cost = np.zeros(len(val_loader))
        cost_total = 0
        for i, (past, future) in enumerate(val_loader): 
            
                
            hidden = rnn_model.init_hidden(past)
            future_pred, _ = rnn_model(past,hidden,future_timesegments)
            if i == 10:
                print('past',past) 
                # print(future_pred)
                print('future',future)
            cost_total += criterion(future_pred, future)
            cost[i] = criterion(future_pred, future)
            # for ax in enumerate(axs):
            #     if i in samples: #in samples:
            #         future_pred_np = future_pred.detach().numpy()
            #         future_pred_np_x, future_pred_np_y = future_pred_np[0,:,0], future_pred_np[0,:,1]
            #         future_np = future.detach().numpy()
            #         future_np_x, future_np_y = future[0,:,0], future[0,:,1]
            #         past_np = past.detach().numpy()
            #         past_np_x, past_np_y = past_np[0,:,0], past_np[0,:,1]
            #         # Plotting the trajectory
            #         # plt.figure(figsize=(8, 6))  # Adjust figure size if needed
            #         ax.plot(future_pred_np_x, future_pred_np_y, marker='o', linestyle='-',label='prediction')
            #         ax.plot(future_np_x, future_np_y, marker='o', linestyle='-',label='ground truth')
            #         ax.plot(past_np_x,past_np_y,marker='o', linestyle='-',label='past')
            #         # Adding labels and title
            #         ax.xlabel('X-axis')
            #         ax.ylabel('Y-axis')
            #         ax.ylim((-10,60))
            #         ax.xlim((-10,60))
            #         ax.grid()
            #         ax.legend()
            #         ax.title('Trajectory Plot, MSE: {mse}'.format(mse=criterion(future_pred, future)))
            # plt.tight_layout()

            # plt.show()

        print('Mean cost: {cost}'.format(cost=np.mean(cost)))
        print('Std cost: {cost}'.format(cost=np.std(cost)))
              
def prediction(model,data,future_timesegments):
    hidden = model.init_hidden(data)
    future_pred, _ = model(data,hidden,future_timesegments)
    return future_pred

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


        def forward(self, x, hidden,future_timesegments):
            out, hidden = self.rnn_cell(x, hidden)
            out = out[:, -future_timesegments:, :] # --> Last time step 
            out = F.relu(self.fc1(out))
            # out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
            return out, hidden

        def init_hidden(self, x):
            return torch.zeros(self.num_layers, x.size(0), self.hidden_size)
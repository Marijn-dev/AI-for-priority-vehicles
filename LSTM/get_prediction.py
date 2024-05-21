from model_load import SimpleRNN
import model_load
import torch
import numpy as np

hidden_size = 250
batch_size = 1
num_layers = 3
future_timesegments = 2
sample_input = torch.tensor([[[ 9.9623, 40.1296],
         [17.9857, 40.0803],
         [25.6637, 40.1170],
         [32.3959, 40.1765],
         [38.2145, 40.2339]]])

model_name = 'RNN_PAST5_FUTURE' + str(future_timesegments) + '.pt'
rnn_model = torch.load('AI-for-priority-vehicles/models/' + model_name)
pred = model_load.prediction(rnn_model,sample_input,future_timesegments)
print('pred:',pred.detach().numpy())


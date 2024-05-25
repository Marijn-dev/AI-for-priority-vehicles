import torch.utils
import torch.utils.data
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from LSTM_predict import SimpleRNN
import LSTM_predict

input = torch.tensor([[[ 0.0000e+00,  0.0000e+00],
        [-1.6442e+00, -1.8951e-02],
        [-5.6885e+00, -5.9998e-02],
        [-9.8572e+00,  1.2701e-01],
        [-1.4382e+01,  1.8763e+00]]])
print('Output: ', LSTM_predict.prediction(input))
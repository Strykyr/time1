

# import packages
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_input = 2; n_hidden = 64

class LSTM(nn.Module):

    def __init__(self,configs):
        super(LSTM, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # intialise weights of the attention mechanism
        self.weight = nn.Parameter(torch.zeros(1)).to(device)
        # intialise lstm structure
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(n_hidden, 2)


    def forward(self, x):
        _, (h_n, _)  = self.lstm(x)
        y_hat = self.linear(h_n[0,:,:])

        return y_hat[:, -self.pred_len:, :]



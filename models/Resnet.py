

# import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_input = 2; n_hidden = 64

class Resnet(nn.Module):

    def __init__(self,configs):
        super(Resnet, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # intialise weights of the attention mechanism
        self.weight = nn.Parameter(torch.zeros(1)).to(device)

        # intialise cnn structure
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_hidden, kernel_size=3, stride=1, padding=1), # ((5 + 1*2 - 3)/1 + 1) = 5
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden, eps=1e-5),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, stride=1, padding=1), # ((5 + 1*2 - 3)/1 + 1) = 5
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden, eps=1e-5),

            nn.Flatten(),
            nn.Linear(n_input * n_hidden, n_input)
        )

        # intialise lstm structure
        self.linear = nn.Linear(n_input, 2)


    def forward(self, x):

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(-1, 1, n_input)

        residuals = x + self.weight * cnn_output
        y_hat = self.linear(residuals)

        return y_hat[:, -self.pred_len:, :]



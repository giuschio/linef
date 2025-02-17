# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:43:20 2020

@author: Giulio Schiavi
"""

import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 35)
        self.fc11 = nn.Linear(35,10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        #also tried a sigmoid but it does not work as well as the relu
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc11(x))
        x = self.fc2(x)
        return x

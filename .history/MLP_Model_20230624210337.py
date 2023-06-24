import torch
import torch.nn as nn
import pandas as pd
import os

import numpy as np

import Data_Preprocess

import visualization as vis

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 11)
        )


    def forward(self, x):
        return F.softmax(self.net(x), dim=1)
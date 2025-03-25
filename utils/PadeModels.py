#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os, sys
import gc
import functools
import random
import warnings
from tqdm import tqdm
from IPython.display import clear_output

from functools import partial

import numba
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import mean_squared_error, mean_absolute_error as mae, r2_score, mean_absolute_percentage_error as mape
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler


import mplcyberpunk
plt.style.use('cyberpunk')

warnings.filterwarnings('ignore')


# In[8]:


seed_value = 15


# In[9]:


torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

class Model(nn.Module):

    def __init__(self, 
                 input_dim : int, 
                 hidden_size : int,
                 num_hidden_layers : int):

        assert input_dim>0, 'Input dim must be an integer > 0'
        assert hidden_size>0, 'Hidden size must be an integer > 0'

        super(Model, self).__init__()

        """
        Args:
            - input_dim - input dimensional value, = deg(Q)
            - hidden_size - size of hidden layers of MLP
            - num_hidden_layers - number of hidden layers
        """

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.PLayers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_size, bias = False, dtype = torch.float64)])
        self.QLayers = nn.ModuleList([nn.Linear(self.input_dim+1, self.hidden_size, bias = False, dtype = torch.float64)])

        self.HeadLayers = nn.ModuleList([nn.Linear(1, self.hidden_size, bias = False, dtype = torch.float64)])

        for i in range(self.num_hidden_layers):
            
            if i!=self.num_hidden_layers-1:
                self.PLayers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = False, dtype = torch.float64))
                self.QLayers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = False, dtype = torch.float64))
                self.HeadLayers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = False, dtype = torch.float64))

            else:
                self.PLayers.append(nn.Linear(self.hidden_size, 1, bias = False, dtype = torch.float64))
                self.QLayers.append(nn.Linear(self.hidden_size, 1, bias = False, dtype = torch.float64))
                self.HeadLayers.append(nn.Linear(self.hidden_size, 1, bias = False, dtype = torch.float64))

    def forward(self, x : torch.tensor):
        p = torch.stack([x**i for i in range(self.input_dim)], axis = 1)
        p = p.to(torch.float64)
        q = torch.stack([x**i for i in range(self.input_dim + 1)], axis = 1)
        q = q.to(torch.float64)

        for i, layer in enumerate(self.PLayers):
            p = self.PLayers[i](p)
            q = self.QLayers[i](q)

        y = p/q
        
        for layer in self.HeadLayers:
            y = layer(y)
            
        return y
    
    
    
class LaplacianModel(nn.Module):

    def __init__(self, 
                 input_dim : int,
                 physical_time_embedding_dim : int,
                 hidden_size : int,
                 num_hidden_layers : int,
                 hidden_size_laplacian : int,
                 num_hidden_layers_laplacian : int):

        assert input_dim>0, 'Input dim must be an integer > 0'
        assert hidden_size>0, 'Hidden size must be an integer > 0'

        super(LaplacianModel, self).__init__()

        """
        Args:
            - input_dim - input dimensional value, = deg(Q)
            - hidden_size - size of hidden layers of MLP
            - num_hidden_layers - number of hidden layers
        """

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_laplacian = hidden_size_laplacian
        self.num_hidden_layers_laplacian = num_hidden_layers_laplacian
        self.physical_time_embedding_dim = physical_time_embedding_dim

        self.PLayers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_size, bias = False, dtype = torch.float64)])
        self.QLayers = nn.ModuleList([nn.Linear(self.input_dim+1, self.hidden_size, bias = False, dtype = torch.float64)])

        self.HeadLayers = nn.ModuleList([nn.Linear(1, self.hidden_size, bias = False, dtype = torch.float64)])
        
        
        self.PhysicalEmbedding = nn.Linear(1, self.physical_time_embedding_dim, bias = False, dtype = torch.float64)
        self.YEmbedding = nn.Linear(1, self.physical_time_embedding_dim, bias = False, dtype = torch.float64)
        
        self.LaplacianLayers = nn.ModuleList([nn.Linear(self.physical_time_embedding_dim, self.hidden_size_laplacian, bias = False, dtype = torch.float64)])
        
        
        for i in range(self.num_hidden_layers_laplacian):
            self.LaplacianLayers.append(nn.Linear(self.hidden_size_laplacian, self.hidden_size_laplacian, bias = False, dtype = torch.float64))
        
        self.LaplacianLayers.append(nn.Linear(self.hidden_size_laplacian, 1, bias = False, dtype = torch.float64))

        for i in range(self.num_hidden_layers):
            
            if i!=self.num_hidden_layers-1:
                self.PLayers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = False, dtype = torch.float64))
                self.QLayers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = False, dtype = torch.float64))
                self.HeadLayers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = False, dtype = torch.float64))

            else:
                self.PLayers.append(nn.Linear(self.hidden_size, 1, bias = False, dtype = torch.float64))
                self.QLayers.append(nn.Linear(self.hidden_size, 1, bias = False, dtype = torch.float64))
                self.HeadLayers.append(nn.Linear(self.hidden_size, 1, bias = False, dtype = torch.float64))

    def forward(self, 
                x : torch.tensor, 
                t : torch.tensor):
        p = torch.stack([x**i for i in range(self.input_dim)], axis = 1)
        p = p.to(torch.float64)
        q = torch.stack([x**i for i in range(self.input_dim + 1)], axis = 1)
        q = q.to(torch.float64)
        
        t = t.reshape(-1, 1)

        for i, layer in enumerate(self.PLayers):
            p = self.PLayers[i](p)
            q = self.QLayers[i](q)

        y = p/q
        
        for layer in self.HeadLayers:
            y = layer(y)
            
        y_embed = self.YEmbedding(y)
        t_embed = self.PhysicalEmbedding(t)
        
        laplacian_inv_image = y_embed*t_embed
        
        for i in range(self.num_hidden_layers_laplacian):
            laplacian_inv_image = self.LaplacianLayers[i](laplacian_inv_image)
            laplacian_inv_image = F.relu(laplacian_inv_image)
            
        laplacian_inv_image = self.LaplacianLayers[-1](laplacian_inv_image)
            
        return y, laplacian_inv_image
            


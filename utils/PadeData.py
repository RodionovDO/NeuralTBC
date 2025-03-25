#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import mplcyberpunk
plt.style.use('cyberpunk')

warnings.filterwarnings('ignore')


# In[2]:


seed_value = 15


# In[3]:


torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

class PadeData(Dataset):

    def __init__(self, 
                 N : int,
                 x_l : float,
                 x_r : float,
                 mesh_len : int,
                 target_function : object):

        """
        Args:
            - N - denominator value, N = deg(Q)
            - x_l - initial point of mesh
            - x_r - final point of mesh
            - mesh_len - length of mesh
            - target_function - approximation function value
            - transform_data - indicator for data transformation
        """

        self.N = N
        self.x_l = x_l
        self.x_r = x_r
        self.mesh_len = mesh_len
        self.mesh = np.linspace(self.x_l, self.x_r, self.mesh_len)
        self.target_function = torch.tensor(target_function(self.mesh), dtype = torch.float64)
        self.transform_data = None
        

    def __len__(self):
        return self.mesh_len

    def __getitem__(self, idx):

        if self.transform_data:
            return self.P_data_scaled[idx], self.Q_data_scaled[idx], self.target_function[idx]
            
        else:
            return self.mesh[idx], self.target_function[idx]
        
        
class PadeLaplacianData(Dataset):

    def __init__(self, 
                 N : int,
                 x_l : float,
                 x_r : float,
                 t_min : float,
                 t_max : float,
                 mesh_len : int,
                 target_function : object,
                 target_function_original : object):

        """
        Args:
            - N - denominator value, N = deg(Q)
            - x_l - initial point of mesh
            - x_r - final point of mesh
            - mesh_len - length of mesh
            - target_function - approximation function value
            - transform_data - indicator for data transformation
        """

        self.N = N
        self.x_l = x_l
        self.x_r = x_r
        self.t_min = t_min
        self.t_max = t_max 
        self.mesh_len = mesh_len
        self.mesh = np.linspace(self.x_l, self.x_r, self.mesh_len)
        self.physical_mesh = np.linspace(self.t_min, self.t_max, self.mesh_len)
        self.target_function = torch.tensor(target_function(self.mesh), dtype = torch.float64)
        self.target_function_original = torch.tensor(target_function_original(self.physical_mesh), dtype = torch.float64)
        self.transform_data = None
        

    def __len__(self):
        return self.mesh_len

    def __getitem__(self, idx):

        if self.transform_data:
            return self.P_data_scaled[idx], self.Q_data_scaled[idx], self.target_function[idx]
            
        else:
            return self.mesh[idx], self.target_function[idx], self.physical_mesh[idx], self.target_function_original[idx]


# In[ ]:





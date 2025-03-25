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


class TBCPostprocessor(object):

    def __init__(self,
                 model : nn.Module):

        super(TBCPostprocessor, self).__init__()

        self.model = model
        self.p_coefs = None
        self.q_coefs = None

        self.p_roots = None
        self.q_roots = None

        self.collected_image = None

        self.s_mesh = np.linspace(0, 50, 1000)
        

    def _collect_kernel_image(self):

        model = self.model

        parameters_p_module = list(model.PLayers.parameters())
        parameters_q_module = list(model.QLayers.parameters())
        parameters_frac_module = list(model.HeadLayers.parameters())
        
        result_p_coefs = parameters_p_module[0]
        result_q_coefs = parameters_q_module[0]
        result_frac_coefs = parameters_frac_module[0]
        

        for i in range(1, len(parameters_p_module)):
            result_p_coefs = torch.matmul(parameters_p_module[i], result_p_coefs)
            result_q_coefs = torch.matmul(parameters_q_module[i], result_q_coefs)
            result_frac_coefs = torch.matmul(parameters_frac_module[i], result_frac_coefs)



        result_p_coefs = torch.matmul(result_frac_coefs, result_p_coefs)
        

        result_p_coefs = result_p_coefs.squeeze()
        result_q_coefs = result_q_coefs.squeeze()

        self.p_coefs = result_p_coefs.detach().numpy()[::-1]
        self.q_coefs = result_q_coefs.detach().numpy()[::-1]

        self.p_roots = np.roots(np.poly1d(self.p_coefs))
        self.q_roots = np.roots(np.poly1d(self.q_coefs))

        self.collected_image = 1

        return result_p_coefs.detach().numpy(), result_q_coefs.detach().numpy()

        
    def _plot_function(self):

        if not self.collected_image:
            self._collect_kernel_image()

        p = np.poly1d(self.p_coefs)
        q = np.poly1d(self.q_coefs)

        plt.plot(self.s_mesh, p(self.s_mesh)/q(self.s_mesh))
        plt.show()

        return
    

    def _plot_poles(self):

        r"""
        
        """
        if not self.collected_image:
            self._collect_kernel_image()

        plt.title(f'Distribution of $P(s)$ roots, real')
        plt.scatter(np.arange(len(self.p_roots)), self.p_roots.real)
        plt.show()

        plt.title(f'Distribution of $Q(s)$ roots, real')
        plt.scatter(np.arange(len(self.q_roots)), self.q_roots.real)
        plt.show()

        plt.title(f'Distribution of $P(s)$ roots, imag')
        plt.scatter(np.arange(len(self.p_roots)), self.p_roots.imag)
        plt.show()

        plt.title(f'Distribution of $Q(s)$ roots, imag')
        plt.scatter(np.arange(len(self.q_roots)), self.q_roots.imag)
        plt.show()

    def _frouassar_duplets(self):
        
        if not self.collected_image:
            self._collect_kernel_image()

        eps = 1e-4
        duples = np.array([])

        for p_root in self.p_roots:
            for q_root in self.q_roots:

                if (np.abs(p_root.real-q_root.real)<eps) and (np.abs(p_root.imag-q_root.imag)<eps):
                    duples = np.append(duples, [p_root, q_root])

        return duples


# In[ ]:





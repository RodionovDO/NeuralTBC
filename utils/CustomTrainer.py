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

from PadeModels import Model
from PadeData import PadeData
from TBCPostProcessor import TBCPostprocessor
from FractionDecomposer import FractionDecomposer


import mplcyberpunk
plt.style.use('cyberpunk')

warnings.filterwarnings('ignore')


# In[2]:
seed_value = 15
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

"""
def all_loss(model, target):
    
    model
    
"""


class Trainer(object):

    def __init__(self,
                 model : nn.Module,
                 train_data : Dataset,
                 val_data : Dataset,
                 num_epochs : int,
                 batch_size : int = 64,
                 lr_scheduler : np.array = None,
                 verbose : bool = None):

        """
        Args:
            -
        """

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_data, len(self.val_data), shuffle = True)
        self.plot_loader = DataLoader(self.train_data, len(self.train_data), shuffle = False)
        self.loss = nn.MSELoss()
        
        #self.all_loss = 

    def train(self):
        loss_history, mae_history, mape_history, r2_score_history, c_norm_train = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        mae_history_val, mape_history_val, c_norm_mean_val = np.array([]), np.array([]), np.array([])
        self.model = self.model.to(device)
        

        for epoch in tqdm(range(self.num_epochs)):
            if self.verbose==2:
                print(f'======================================================')
                print(f'==================Epoch {epoch}=======================')
                print(f'======================================================')

            self.model.train()

            loss_mean_value = 0
            mae_mean_value = 0
            mape_mean_value = 0
            r2_mean_value = 0
            c_norm_mean_value = 0

            for i, batch in enumerate(self.train_loader):
                global y_batch, y_predicted_
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_batch_ = y_batch.cpu().detach().numpy().squeeze()
    
                y_predicted = self.model(x_batch).squeeze()
                y_predicted_ = y_predicted.cpu().detach().numpy()
                loss_value = self.loss(y_batch.squeeze(), y_predicted)
                loss_value.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if y_predicted_.size==1:
                    y_predicted_ = np.array([y_predicted_])
                    y_batch_ = np.array([y_batch_])

                
                loss_mean_value += loss_value
                mae_mean_value += mae(y_batch_, y_predicted_)
                mape_mean_value += mape(y_batch_, y_predicted_)
                c_norm_mean_value += max(np.abs(torch.tensor(y_batch_) - y_predicted_)).detach().numpy()
                #r2_mean_value += r2_score(y_batch_, y_predicted_)
            


            loss_mean_value /= (i+1)
            mae_mean_value /= (i+1)
            mape_mean_value /= (i+1)
            c_norm_mean_value /= (i+1)
            #r2_mean_value /= i

            if self.verbose:
                """
                print(f'Epoch resume:')
                print(f'>>>>>>>>>>>>Loss: {loss_mean_value}')
                print(f'>>>>>>>>>>>>MAE: {mae_mean_value}')
                print(f'>>>>>>>>>>>>MAPE: {mape_mean_value}')
                print(f'>>>>>>>>>>>>R^2: {r2_mean_value}')
                """
            

            loss_history = np.append(loss_history, loss_mean_value.cpu().detach().numpy())
            mae_history = np.append(mae_history, mae_mean_value)
            mape_history = np.append(mape_history, mape_mean_value)
            c_norm_train = np.append(c_norm_train, c_norm_mean_value)
            #r2_score_history = np.append(r2_score_history, r2_mean_value)

            resume = np.array([np.arange(0, epoch+1), loss_history, mae_history, mape_history, c_norm_train]).T

            global resume_frame

            resume_frame = pd.DataFrame(resume, columns = ['Epoch', 'MSE (loss) train', 'MAE train', 'MAPE train', 'Train C[a;b]'])

            self.model.eval()
            with torch.no_grad():
                for _, batch in enumerate(self.val_loader):
                    X_batch_val, y_batch_val = batch
                    X_batch_val = X_batch_val.to(device)
                    y_predicted_val = self.model(X_batch_val).squeeze()
                    y_predicted_val = y_predicted_val.cpu().detach().numpy()
                    
                    if y_predicted_val.size == 1:
                        y_predicted_val = np.array([y_predicted_val])
                        y_batch_val = np.array([y_batch_val.squeeze().detach().numpy()])

                    mae_history_val = np.append(mae_history_val, mae(y_batch_val, y_predicted_val))
                    mape_history_val = np.append(mape_history_val, mape(y_batch_val, y_predicted_val))
                    c_norm_mean_val = np.append(c_norm_mean_val, max(np.abs(torch.tensor(y_batch_val) - y_predicted_val)).clone().detach().numpy())

                resume_frame['MAE validation'] = mae_history_val
                resume_frame['MAPE validation'] = mape_history_val
                resume_frame['Val C[a;b]'] = c_norm_mean_val

                for train_batch in self.plot_loader:
                    X_batch_train, y_batch_train = train_batch
                    X_batch_train = X_batch_train.to(device)
                    
                    y_predicted_plot = self.model(X_batch_train).squeeze()
                    y_predicted_plot = y_predicted_plot.cpu().detach().numpy()
                    
                    
                    
            
            if epoch%100==0 and self.verbose:
                """
                plt.title('Loss history')
                plt.plot(np.arange(len(loss_history)), loss_history)
                plt.show()
        
                plt.title('MAE history')
                plt.plot(np.arange(len(mae_history)), mae_history)
                plt.show()
        
                plt.title('MAPE history')
                plt.plot(np.arange(len(mape_history)), mape_history)
                plt.show()
        
                #plt.title('R^2 history')
                #plt.plot(np.arange(len(r2_score_history)), r2_score_history)
                #plt.show()
                """

                
                
                fig = px.line(resume_frame, x = 'Epoch', y = 'MSE (loss) train', title = 'Results : MSE loss')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['MAE train', 'MAE validation'], title = 'Results : MAE')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['MAPE train', 'MAPE validation'], title = 'Results : MAPE')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['Train C[a;b]', 'Val C[a;b]'], title = 'Results : C[a;b]')
                fig.show()
                

                plt.title('Function plot')
                plt.plot(self.train_data.mesh, y_predicted_plot)
                plt.show(block=False)
                plt.pause(10)
                clear_output(wait=True)
                

                
                


        return self.model, resume_frame
    

lambda_ = 0.1

def custom_loss(y_true, l_true, y, l):
    loss_value = ((y_true - y)**2).sum() + lambda_ * ((l_true - l)**2).sum()

    return loss_value
    

class TrainerLaplacian(object):

    def __init__(self,
                 model : nn.Module,
                 train_data : Dataset,
                 val_data : Dataset,
                 num_epochs : int,
                 batch_size : int = 64,
                 lambda_ : float = 0.1,
                 lr_scheduler : np.array = None,
                 verbose : bool = None):

        """
        Args:
            -
        """

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lambda_ = lambda_
        
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_data, len(self.val_data), shuffle = True)
        self.plot_loader = DataLoader(self.train_data, len(self.train_data), shuffle = False)
        self.loss = custom_loss
        
        #self.all_loss = 

    
    
    def train(self):
        loss_history, mae_history, mape_history, r2_score_history, c_norm_train = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        mae_history_val, mape_history_val, c_norm_mean_val = np.array([]), np.array([]), np.array([])
        
        mae_history_laplacian, mape_history_laplacian, r2_score_history_laplacian, c_norm_train_laplacian = np.array([]), np.array([]), np.array([]), np.array([])
        mae_history_val_laplacian, mape_history_val_laplacian, c_norm_mean_val_laplacian = np.array([]), np.array([]), np.array([])
        
        
        self.model = self.model.to(device)
        

        for epoch in tqdm(range(self.num_epochs)):
            if self.verbose==2:
                print(f'======================================================')
                print(f'==================Epoch {epoch}=======================')
                print(f'======================================================')

            self.model.train()

            loss_mean_value = 0
            mae_mean_value = 0
            mape_mean_value = 0
            r2_mean_value = 0
            c_norm_mean_value = 0
            
            
            mae_mean_value_laplacian = 0
            mape_mean_value_laplacian = 0
            r2_mean_value_laplacian = 0
            c_norm_mean_value_laplacian = 0

            for i, batch in enumerate(self.train_loader):
                global y_batch, y_predicted_
                x_batch, y_batch, t_batch, l_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                t_batch = t_batch.to(device)
                l_batch = l_batch.to(device)
                
                y_batch_ = y_batch.cpu().detach().numpy().squeeze()
                l_batch_ = l_batch.cpu().detach().numpy().squeeze()
                
                y_predicted, l_predicted = self.model(x_batch, t_batch)
                
                y_predicted = y_predicted.squeeze()
                y_predicted_ = y_predicted.cpu().detach().numpy()
                
                l_predicted = l_predicted.squeeze()
                l_predicted_ = l_predicted.cpu().detach().numpy()
                
                loss_value = self.loss(y_batch.squeeze(), l_batch.squeeze(), y_predicted, l_predicted)
                self.loss_value = loss_value
                loss_value.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if y_predicted_.size==1:
                    y_predicted_ = np.array([y_predicted_])
                    y_batch_ = np.array([y_batch_])
                    l_predicted_ = np.array([l_predicted_])
                    l_batch_ = np.array([l_batch])

                
                loss_mean_value += loss_value
                mae_mean_value += mae(y_batch_, y_predicted_)
                mape_mean_value += mape(y_batch_, y_predicted_)
                c_norm_mean_value += max(np.abs(torch.tensor(y_batch_) - y_predicted_)).detach().numpy()
                
                
                mae_mean_value_laplacian += mae(l_batch_, l_predicted_)
                mape_mean_value_laplacian += mape(l_batch_, l_predicted_)
                c_norm_mean_value_laplacian += max(np.abs(torch.tensor(l_batch_) - l_predicted_)).detach().numpy()
                #r2_mean_value += r2_score(y_batch_, y_predicted_)
                
            


            loss_mean_value /= (i+1)
            mae_mean_value /= (i+1)
            mape_mean_value /= (i+1)
            c_norm_mean_value /= (i+1)
            
            mae_mean_value_laplacian /= (i+1)
            mape_mean_value_laplacian /= (i+1)
            r2_mean_value_laplacian /= (i+1)
            c_norm_mean_value_laplacian /= (i+1)
            #r2_mean_value /= i

            if self.verbose:
                """
                print(f'Epoch resume:')
                print(f'>>>>>>>>>>>>Loss: {loss_mean_value}')
                print(f'>>>>>>>>>>>>MAE: {mae_mean_value}')
                print(f'>>>>>>>>>>>>MAPE: {mape_mean_value}')
                print(f'>>>>>>>>>>>>R^2: {r2_mean_value}')
                """
            

            loss_history = np.append(loss_history, loss_mean_value.cpu().detach().numpy())
            mae_history = np.append(mae_history, mae_mean_value)
            mape_history = np.append(mape_history, mape_mean_value)
            c_norm_train = np.append(c_norm_train, c_norm_mean_value)
            #r2_score_history = np.append(r2_score_history, r2_mean_value)
            
            mae_history_laplacian = np.append(mae_history_laplacian, mae_mean_value_laplacian)
            mape_history_laplacian = np.append(mape_history_laplacian, mape_mean_value_laplacian)
            c_norm_train_laplacian = np.append(c_norm_train_laplacian, c_norm_mean_value_laplacian)

            resume = np.array([np.arange(0, epoch+1), loss_history, mae_history, mape_history, c_norm_train, mae_history_laplacian, mape_history_laplacian, c_norm_train_laplacian]).T

            global resume_frame

            resume_frame = pd.DataFrame(resume, columns = ['Epoch', 'MSE (loss) train', 'MAE train', 'MAPE train', 'Train C[a;b]', 'MAE train Laplacian', 'MAPE train Laplacian', 'Train C[a;b] Laplacian'])
            
            self.model.eval()
            with torch.no_grad():
                for _, batch in enumerate(self.val_loader):
                    X_batch_val, y_batch_val, t_batch_val, l_batch_val = batch
                    X_batch_val = X_batch_val.to(device)
                    t_batch_val = t_batch_val.to(device)
                    
                    y_predicted_val, l_predicted_val = self.model(X_batch_val, t_batch_val)
                    y_predicted_val = y_predicted_val.cpu().detach().numpy().squeeze()
                    l_predicted_val = l_predicted_val.cpu().detach().numpy().squeeze()
                    
                    if y_predicted_val.size == 1:
                        y_predicted_val = np.array([y_predicted_val])
                        y_batch_val = np.array([y_batch_val.squeeze().detach().numpy()])
                        
                        l_predicted_val = np.array([l_predicted_val])
                        l_batch_val = np.array([l_batch_val.squeeze().detach().numpy()])

                    mae_history_val = np.append(mae_history_val, mae(y_batch_val, y_predicted_val))
                    mape_history_val = np.append(mape_history_val, mape(y_batch_val, y_predicted_val))
                    c_norm_mean_val = np.append(c_norm_mean_val, max(np.abs(torch.tensor(y_batch_val) - y_predicted_val)).clone().detach().numpy())
                    
                    mae_history_val_laplacian = np.append(mae_history_val_laplacian, mae(l_batch_val, l_predicted_val)) 
                    mape_history_val_laplacian = np.append(mape_history_val_laplacian, mape(l_batch_val, l_predicted_val))
                    c_norm_mean_val_laplacian = np.append(c_norm_mean_val_laplacian, max(np.abs(torch.tensor(l_batch_val) - l_predicted_val)).clone().detach().numpy())

                resume_frame['MAE validation'] = mae_history_val
                resume_frame['MAPE validation'] = mape_history_val
                resume_frame['Val C[a;b]'] = c_norm_mean_val
                
                resume_frame['MAE validation Laplacian'] = mae_history_val_laplacian
                resume_frame['MAPE validation Laplacian'] = mape_history_val_laplacian
                resume_frame['Val C[a;b] Laplacian'] = c_norm_mean_val_laplacian
                
                

                for train_batch in self.plot_loader:
                    X_batch_train, y_batch_train, t_batch_train, l_batch_train = train_batch
                    X_batch_train = X_batch_train.to(device)
                    t_batch_train = t_batch_train.to(device)
                    
                    y_predicted_plot, l_predicted_plot = self.model(X_batch_train, t_batch_train)
                    y_predicted_plot = y_predicted_plot.cpu().detach().numpy().squeeze()
                    l_predicted_plot = l_predicted_plot.cpu().detach().numpy().squeeze()
                    
                    
                    
            
            

                
            if epoch%100==0 and self.verbose:
                
                fig = px.line(resume_frame, x = 'Epoch', y = 'MSE (loss) train', title = 'Results : MSE loss')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['MAE train', 'MAE validation'], title = 'Results : MAE')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['MAPE train', 'MAPE validation'], title = 'Results : MAPE')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['Train C[a;b]', 'Val C[a;b]'], title = 'Results : C[a;b]')
                fig.show()
                
                
                fig = px.line(resume_frame, x = 'Epoch', y = ['MAE train Laplacian', 'MAE validation Laplacian'], title = 'Results : MAE')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['MAPE train Laplacian', 'MAPE validation Laplacian'], title = 'Results : MAPE')
                fig.show()

                fig = px.line(resume_frame, x = 'Epoch', y = ['Train C[a;b] Laplacian', 'Val C[a;b] Laplacian'], title = 'Results : C[a;b]')
                fig.show()
                

                plt.title('Function plot in Image space')
                plt.plot(self.train_data.mesh, y_predicted_plot)
                plt.show(block=False)
                
                plt.title('Function plot in original space')
                plt.plot(self.train_data.physical_mesh, l_predicted_plot)
                plt.show(block=False)
                plt.pause(10)
                
                plt.pause(10)
                clear_output(wait=True)
                
                

                
                


        return self.model, resume_frame

# In[ ]:





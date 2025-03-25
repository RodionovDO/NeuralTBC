#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
import gc
import random

from IPython.display import clear_output

import numpy as np
import pandas as pd
import torch
import scipy

from scipy.special import j1

import optuna
from optuna.trial import TrialState
from optuna.trial import TrialState
from optuna.samplers import TPESampler


from PadeModels import Model, LaplacianModel
from PadeData import PadeData, PadeLaplacianData
from CustomTrainer import Trainer, TrainerLaplacian
from TBCPostProcessor import TBCPostprocessor
from FractionDecomposer import FractionDecomposer


# In[11]:


seed_value = 42
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[17]:


torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

class ParameterOptimizer(object):
    
    def __init__(self,
                 target_function : object,
                 max_hidden_layers : int = 5,
                 max_hidden_size : int = 128,
                 max_num_epochs : int = 1001,
                 mesh_size : int = 5000,
                 train_s0 : float = 0.0,
                 train_s1 : float = 50,
                 val_s0 : float = 60,
                 val_s1 : float = 70,
                 physical_time : np.array = np.linspace(0.1, 1, 1000)):
        
        self.max_hidden_layers = max_hidden_layers
        self.max_hidden_size = max_hidden_size
        self.max_num_epochs = max_num_epochs
        
        
        self.mesh_size = mesh_size
        self.train_s0 = 0.0
        self.train_s1 = 50
        self.val_s0 = 60
        self.val_s1 = 70
        
        self.target_function = target_function
        self.physical_time = physical_time
        
        
        
    def objective(self,
                  trial):
        
        train_df_args = {'N': self.num_poles,
                         'x_l': self.train_s0,
                         'x_r' : self.train_s1,
                         'mesh_len' : self.mesh_size,
                         'target_function' : self.target_function}
        
        val_df_args = {'N': self.num_poles,
                         'x_l': self.val_s0,
                         'x_r' : self.val_s1,
                         'mesh_len' : self.mesh_size,
                         'target_function' : self.target_function}
        
        
        model_args = {"input_dim" : self.num_poles,
                      "hidden_size" : trial.suggest_int("hidden_size", self.num_poles, self.max_hidden_size, log=True),
                      "num_hidden_layers" : trial.suggest_int("num_hidden_layers", 3, self.max_hidden_layers, log=True)}
        
        num_epochs = trial.suggest_int("num_epochs", 101, self.max_num_epochs, log=True)
        
        train_df = PadeData(**train_df_args)
        val_df = PadeData(**val_df_args)
        model = Model(**model_args)
        trainer_ = Trainer(model, train_df, val_df, num_epochs)

        model, summary_df = trainer_.train()

        validation_mape = summary_df['MAPE validation'].to_numpy()[-1] #'MAPE validation'
        clear_output(wait = True)
        
        """
        TODO:
            - добавить вычисление обратного Лапласа как функцию для оптимизации
        """


        #if trial.should_prune():
        #    raise optuna.exceptions.TrialPruned()
        
        decomposer = FractionDecomposer()
        post_processor = TBCPostprocessor(model.cpu())
        
        p, q = post_processor._collect_kernel_image()
                 
        model_on_physical_time = decomposer.collect_exponential_stats(p, q, self.physical_time)
        original_metrics = max(abs(j1(self.physical_time)/self.physical_time - model_on_physical_time.real))
        

        return validation_mape
    
    def optimize(self, 
                 num_poles : int):
        self.num_poles = num_poles
        
        sampler = optuna.samplers.TPESampler(seed = seed_value)
        
        study = optuna.create_study(sampler = sampler, direction="minimize")
        study.optimize(self.objective, n_trials=1000, timeout=3600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")

        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


        trial_df = pd.DataFrame(columns = trial.params.keys())
        trial_df.loc[len(trial_df)] = trial.params.values()
        trial_df['Metric value'] = trial.value
        trial_df.to_csv(f'FixedMesh_results/Optimal_config_for_{num_poles}_poles.csv', index = False)
        
        
        
        
class ParameterOptimizerLaplacian(object):
    
    def __init__(self,
                 target_function : object,
                 target_function_original : object,
                 max_hidden_layers : int = 5,
                 max_hidden_size : int = 128,
                 max_num_epochs : int = 1001,
                 mesh_size : int = 1000,
                 train_s0 : float = 0.0,
                 train_s1 : float = 50,
                 val_s0 : float = 60,
                 val_s1 : float = 70,
                 train_t0 : float = 0.01,
                 train_t1 : float = 10.0,
                 val_t0 : float = 10.1,
                 val_t1 : float = 21.0,
                 physical_time : np.array = None):#= np.linspace(0.1, 1, 1000)):
        
        self.max_hidden_layers = max_hidden_layers
        self.max_hidden_size = max_hidden_size
        self.max_num_epochs = max_num_epochs
        
        
        self.mesh_size = mesh_size
        self.train_s0 = train_s0
        self.train_s1 = train_s1
        self.val_s0 = val_s0
        self.val_s1 = val_s1
        
        self.train_t0 = train_t0
        self.train_t1 = train_t1
        
        self.val_t0 = val_t0
        self.val_t1 = val_t1
        
        self.target_function = target_function
        self.target_function_original = target_function_original
        self.physical_time = np.linspace(self.train_t0, self.val_t1, self.mesh_size)
        
        
        
    def objective(self,
                  trial):
        
        train_df_args = {'N': self.num_poles,
                         'x_l': self.train_s0,
                         'x_r' : self.train_s1,
                         't_min' : self.train_t0,
                         't_max' : self.train_t1,
                         'mesh_len' : self.mesh_size,
                         'target_function' : self.target_function,
                         'target_function_original' : self.target_function_original}
        
        val_df_args = {'N': self.num_poles,
                         'x_l': self.val_s0,
                         'x_r' : self.val_s1,
                         't_min' : self.val_t0,
                         't_max' : self.val_t1,
                         'mesh_len' : self.mesh_size,
                         'target_function' : self.target_function,
                         'target_function_original' : self.target_function_original}
        
        
        model_args = {"input_dim" : self.num_poles,
                      "hidden_size" : trial.suggest_int("hidden_size", self.num_poles, self.max_hidden_size, log=True),
                      "num_hidden_layers" : trial.suggest_int("num_hidden_layers", 3, self.max_hidden_layers, log=True),
                     "num_hidden_layers_laplacian" : trial.suggest_int("num_hidden_layers_laplacian", 3, self.max_hidden_layers, log=True),
                     "hidden_size_laplacian" : trial.suggest_int("hidden_size_laplacian", self.num_poles, self.max_hidden_size, log=True),
                     "physical_time_embedding_dim" : trial.suggest_int("physical_time_embedding_dim", 16, 128, log=True)}
        
        num_epochs = trial.suggest_int("num_epochs", 101, self.max_num_epochs, log=True)
        
        train_df = PadeLaplacianData(**train_df_args)
        val_df = PadeLaplacianData(**val_df_args)
        model = LaplacianModel(**model_args)
        self.model = model
        trainer_ = TrainerLaplacian(model, train_df, val_df, num_epochs)

        model, summary_df = trainer_.train()

        validation_mape = summary_df['Val C[a;b] Laplacian'].to_numpy()[-1] #'MAPE validation'
        clear_output(wait = True)
        
        """
        TODO:
            - добавить вычисление обратного Лапласа как функцию для оптимизации
        """


        #if trial.should_prune():
        #    raise optuna.exceptions.TrialPruned()
        
        decomposer = FractionDecomposer()
        post_processor = TBCPostprocessor(model.cpu())
        
        p, q = post_processor._collect_kernel_image()
                 
        model_on_physical_time = decomposer.collect_exponential_stats(p, q, self.physical_time)
        original_metrics = max(abs(j1(self.physical_time)/self.physical_time - model_on_physical_time.real))
        

        return validation_mape
    
    def optimize(self, 
                 num_poles : int):
        self.num_poles = num_poles
        
        sampler = optuna.samplers.TPESampler(seed = seed_value)
        
        study = optuna.create_study(sampler = sampler, direction="minimize")
        study.optimize(self.objective, n_trials=1000, timeout=18000)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")

        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


        trial_df = pd.DataFrame(columns = trial.params.keys())
        trial_df.loc[len(trial_df)] = trial.params.values()
        trial_df['Metric value'] = trial.value
        trial_df.to_csv(f'Optimal_config_laplacian_for_{num_poles}_poles.csv', index = False)


# In[ ]:





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

from scipy.signal import residue

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


# ### Todo:
#     - Завернуть всё это в класс
#     - Добавить рассчет ядра свертки 
#     - Добавить сравнение с аналитикой






# In[2]:


seed_value = 15


# In[35]:


class FractionDecomposer(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def rational_decomposition(num_coeffs : np.array, 
                               den_coeffs : np.array):
        """
        Разложение рациональной дроби на сумму простейших дробей.

        :param num_coeffs: Коэффициенты числителя g(x)
        :param den_coeffs: Коэффициенты знаменателя f(x)
        :return: Коэффициенты A, B, C для простейших дробей, а также целую часть (если есть)
        """
        # Проверяем, если степень числителя >= степени знаменателя
        if len(num_coeffs) >= len(den_coeffs):
            # Выполняем деление полиномов
            quotient, remainder = np.polydiv(num_coeffs, den_coeffs)
            quotient = np.trim_zeros(quotient, 'f')  # Удаляем нулевые коэффициенты
            remainder = np.trim_zeros(remainder, 'f')
        else:
            quotient = np.array([])  # Целой части нет
            remainder = num_coeffs

        # Вычисляем разложение на простейшие дроби для остатка
        r, p, k = residue(remainder, den_coeffs)  # r - резидуумы, p - полюса, k - целая часть
        return r, p, k
    
    
    @staticmethod
    def evaluate_rational_function(r : np.array,
                                   p : np.array,
                                   k : np.array,
                                   x : np.array):
        """
        Вычисляет значение рациональной функции после разложения на простейшие дроби.

        :param r: Резидуумы
        :param p: Полюсы
        :param k: Целая часть
        :param x: Значения аргумента
        :return: Значения функции
        """
        result = np.zeros_like(x, dtype=complex)  # Инициализируем результат как комплексный массив
        for i in range(len(r)):
            result += r[i] / (x - p[i])  # Добавляем каждую простую дробь
        if len(k) > 0:
            poly_value = np.polyval(k, x)  # Вычисляем значение целой части
            result += poly_value
        return result
    
    
    @staticmethod
    def original_rational_function(num_coeffs : np.array,
                                   den_coeffs : np.array,
                                   x : np.array):
        """
        Вычисляет значение исходной рациональной функции.

        :param num_coeffs: Коэффициенты числителя
        :param den_coeffs: Коэффициенты знаменателя
        :param x: Значения аргумента
        :return: Значения функции
        """
        numerator = np.polyval(num_coeffs, x)
        denominator = np.polyval(den_coeffs, x)
        return numerator / denominator


    def check_convergence(self,
                          num_coeffs : np.array,
                          den_coeffs : np.array,
                          x_values : np.array = np.linspace(-100, 100, 100),
                          plot_report : bool = False):
        
        r, p, k = self.rational_decomposition(num_coeffs, den_coeffs)
        original_values = self.original_rational_function(num_coeffs, den_coeffs, x_values)
        decomposed_values = self.evaluate_rational_function(r, p, k, x_values)

        # Проверка точности
        error = np.abs(original_values - decomposed_values.real)
        max_error = np.max(error)

        if plot_report == True:

            # Вывод результатов
            print("Коэффициенты разложения:")
            for i in range(len(r)):
                print(f"({r[i]:.4f} / (x - {p[i]:.4f}))", end=" + " if i != len(r) - 1 else "\n")
            if len(k) > 0:
                print(f"Целая часть: {k}")

            print(f"\nМаксимальная ошибка: {max_error:.6e}")

            # Построение графиков
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, original_values.real, label="Исходная функция", color="blue")
            plt.plot(x_values, decomposed_values.real, label="Декомпозиция", linestyle="--", color="red")
            plt.title("Сравнение исходной функции и её декомпозиции")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid()
            plt.show()
            
        assert max_error < 1e-6, 'Fraction decomposition error'

        return max_error
    
    
    def collect_exponential_stats(self,
                                  num_coeffs : np.array, 
                                  den_coeffs : np.array,
                                  x : np.array = np.linspace(0, 100, 1000)):
        
        self.check_convergence(num_coeffs, den_coeffs, x)
        
        r, p, k = self.rational_decomposition(num_coeffs, den_coeffs)
        
        assert len(k) == 0, 'Decomposition has residual polynom'
        
        output = np.zeros_like(x, dtype=complex)
        stats = zip(r, p)
        for latitude, pole in stats:
            output += latitude*np.exp(pole*x)
            
        return output
            
        



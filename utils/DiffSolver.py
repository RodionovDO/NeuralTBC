#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import sys
import gc
import itertools
import tqdm
from IPython import display
from time import sleep, time

from numba import jit

import numpy as np
import scipy 
import torch
import sympy
from scipy.special import j1  
from scipy.fftpack import fft, ifft

import matplotlib.pyplot as plt


# In[5]:


#@jit(fastmath = True, parallel = True)
def convolution(u_slice, B):
    U = fft(u_slice, axis=0)
    convolved = ifft(U * B, axis=0).real
    return convolved


# In[6]:




# In[7]:


class HyperbolicalSolver(object):
    
    def __init__(self):
        pass
    
    
    @staticmethod
    #@jit(fastmath = True, parallel = True)
    def solve_iteration(self,
                        R : float,
                        ZL : float,
                        ZR : float,
                        T : float,
                        Nr : float,
                        Nt : float,
                        Nz : float,
                        Ntime : float,
                        c : float,
                        alpha : float,
                        beta : float,
                        B_function : object,
                        boudary_treshold : float = 0.1):
        
        """
        Args:
            -
        """
        # Шаги сетки
        dr = R / Nr
        dt = T / Ntime
        dtheta = 2 * np.pi / Nt
        dz = (ZR - ZL) / Nz

        # Создание сетки
        r = np.linspace(dr/2, R-dr/2, Nr)  # Центральные точки ячеек по r
        theta = np.linspace(0, 2*np.pi - dtheta, Nt)
        z = np.linspace(ZL + dz/2, ZR - dz/2, Nz)

        # Инициализация массивов для решения
        u = np.zeros((Ntime, Nr, Nt, Nz))
        u_prev = np.zeros((Nr, Nt, Nz))

        # Начальные условия (например, импульс в центре)
        for i in range(Nr):
            for j in range(Nt):
                for k in range(Nz):
                    if r[i] < 0.1 and abs(z[k] - (ZR+ZL)/2) < 0.1:
                        u[0, i, j, k] = np.exp(-100 * ((r[i])**2 + (z[k] - (ZR+ZL)/2)**2))
                        

        # Коэффициенты для метода конечных разностей
        alpha_r = (c * dt / dr)**2
        alpha_z = (c * dt / dz)**2

        # Основной цикл по времени
        for n in range(0, Ntime):
            u_new = np.zeros_like(u[n-1])
            for i in range(1, Nr-1):
                for j in range(Nt):
                    for k in range(1, Nz-1):
                        # Граничные условия для theta (периодические)
                        j_plus = (j + 1) % Nt
                        j_minus = (j - 1) % Nt

                        # Вычисление beta для текущего радиуса
                        beta_theta = (c * dt / (r[i] * dtheta))**2

                        # Метод конечных разностей для волнового уравнения
                        if i == 0:  # Центральная точка (особый случай)
                            u_new[i, j, k] = 2 * u[n-1, i, j, k] - u[n-2, i, j, k] + alpha_r * (u[n-1, i+1, j, k] - 2*u[n-1, i, j, k] + u[n-1, i, j, k]) + \
                                             beta_theta * (u[n-1, i, j_plus, k] - 2*u[n-1, i, j, k] + u[n-1, i, j_minus, k]) + \
                                             alpha_z * (u[n-1, i, j, k+1] - 2*u[n-1, i, j, k] + u[n-1, i, j, k-1])
                        else:
                            u_new[i, j, k] = 2 * u[n-1, i, j, k] - u[n-2, i, j, k] + alpha_r * (u[n-1, i+1, j, k] - 2*u[n-1, i, j, k] + u[n-1, i-1, j, k] + \
                                             (u[n-1, i+1, j, k] - u[n-1, i-1, j, k]) / (2*r[i])) + \
                                             beta_theta * (u[n-1, i, j_plus, k] - 2*u[n-1, i, j, k] + u[n-1, i, j_minus, k]) + \
                                             alpha_z * (u[n-1, i, j, k+1] - 2*u[n-1, i, j, k] + u[n-1, i, j, k-1])

            # Применение граничного условия третьего рода на боковой поверхности цилиндра
            u_new[-1, :, :] = (beta / (alpha + beta / dr)) * u_new[-2, :, :]

            # Применение граничного условия третьего рода на верхней и нижней границах
            B = np.array([B_function(t) for t in np.arange(0, T, dt)[:n]])  # Формируем массив B
            B = np.pad(B, (0, max(len(r) - len(B), 0)), mode='constant')  # Дополнение нулями до нужной длины

            for j in range(Nt):
                # При z = zL
                u_slice_left = u[n-1, :, j, 0]
                conv_left = convolution(u_slice_left, B[:len(u_slice_left)])
                u_new[:, j, 0] = u[n-1, :, j, 0] - dt * (c * (u[n-1, :, j, 1] - u[n-1, :, j, 0]) / dz + conv_left)

                # При z = zR
                u_slice_right = u[n-1, :, j, -1]
                conv_right = convolution(u_slice_right, B[:len(u_slice_right)])
                u_new[:, j, -1] = u[n-1, :, j, -1] + dt * (c * (u[n-1, :, j, -1] - u[n-1, :, j, -2]) / dz + conv_right)

            # Сохраняем решение для текущего временного шага
            u[n] = u_new.copy()
        
        return u
    

def CFL_criteria(alpha_r : float, 
                 beta_theta : float, 
                 alpha_z : float):
    
    C = alpha_r + beta_theta + alpha_z
    
    return C
    
class HyperbolicalSolverImp(object):
    
    def __init__(self):
        pass
    
    
    #@staticmethod
    #@jit(fastmath = True, parallel = True)
    def solve_iteration(self,
                        R : float,
                        ZL : float,
                        ZR : float,
                        T : float,
                        Nr : float,
                        Nt : float,
                        Nz : float,
                        Ntime : float,
                        c : float,
                        alpha : float,
                        beta : float,
                        B_function : object,
                        boudary_treshold : float = 0.1):
        
        """
        Args:
            -
        """
        # Шаги сетки
        dr = R / Nr
        dt = T / Ntime
        dtheta = 2 * np.pi / Nt
        dz = (ZR - ZL) / Nz

        # Создание сетки
        r = np.linspace(dr/2, R-dr/2, Nr)  # Центральные точки ячеек по r
        theta = np.linspace(0, 2*np.pi - dtheta, Nt)
        z = np.linspace(ZL + dz/2, ZR - dz/2, Nz)
        
        self.r = r
        self.theta = theta
        self.z = z
        self.time = np.linspace(0, T, Ntime)

        # Инициализация массивов для решения
        u = np.zeros((Ntime, Nr, Nt, Nz))
        u_prev = np.zeros((Nr, Nt, Nz))

        # Начальные условия (например, импульс в центре)
        for i in range(Nr):
            for j in range(Nt):
                for k in range(Nz):
                    if r[i] < 0.1 and abs(z[k] - (ZR+ZL)/2) < 0.1:
                        u[0, i, j, k] = np.exp(-100 * ((r[i])**2 + (z[k] - (ZR+ZL)/2)**2))
                        #u[0, i, j, -k] = -np.exp(-100 * ((r[i])**2 + (z[-k] - (ZR+ZL)/2)**2))
        
        
        """
        for i in range(Nr):
            for j in range(Nt):
                k = Nz//2
                if r[i] < 0.01:
                    u[0, i, j, k] = np.exp(-100 * ((r[i])**2 + (z[k] - (ZR+ZL)/2)**2))
                    u[0, i, j, k-1] = -np.exp(-100 * ((r[i])**2 + (z[k-1] - (ZR+ZL)/2)**2))
                    
        """
        
                        

        # Коэффициенты для метода конечных разностей
        alpha_r = (c * dt / dr)**2
        alpha_z = (c * dt / dz)**2

        # Основной цикл по времени
        for n in range(1, Ntime):
            u_new = np.zeros_like(u[n-1])
            for i in range(0, Nr-1):
                for j in range(Nt):
                    for k in range(0, Nz-1):
                        # Граничные условия для theta (периодические)
                        j_plus = (j + 1) % Nt
                        j_minus = (j - 1) % Nt
                        #j_plus = j + 1
                        #j_minus = j - 1
                        

                        # Вычисление beta для текущего радиуса
                        beta_theta = (c * dt / (r[i] * dtheta))**2
                        
                        CFL = CFL_criteria(alpha_r, beta_theta, alpha_z)
                        
                        #assert CFL <= 1, f'CFL criteria is not completed, CFL = {CFL}'

                        # Метод конечных разностей для волнового уравнения
                        if i == 0:  # Центральная точка (особый случай)
                            u_new[i, j, k] = 2 * u[n-1, i, j, k] - u[n-2, i, j, k] + alpha_r * (u[n-1, i+1, j, k] - 2*u[n-1, i, j, k] + u[n-1, i, j, k]) + \
                                             beta_theta * (u[n-1, i, j_plus, k] - 2*u[n-1, i, j, k] + u[n-1, i, j_minus, k]) + \
                                             alpha_z * (u[n-1, i, j, k+1] - 2*u[n-1, i, j, k] + u[n-1, i, j, k-1])
                        else:
                            u_new[i, j, k] = 2 * u[n-1, i, j, k] - u[n-2, i, j, k] + alpha_r * (u[n-1, i+1, j, k] - 2*u[n-1, i, j, k] + u[n-1, i-1, j, k] + \
                                             (u[n-1, i+1, j, k] - u[n-1, i-1, j, k]) / (2*r[i])) + \
                                             beta_theta * (u[n-1, i, j_plus, k] - 2*u[n-1, i, j, k] + u[n-1, i, j_minus, k]) + \
                                             alpha_z * (u[n-1, i, j, k+1] - 2*u[n-1, i, j, k] + u[n-1, i, j, k-1])

            # Применение граничного условия третьего рода на боковой поверхности цилиндра
            #u_new[-1, :, :] = (beta / (alpha + beta / dr)) * u_new[-2, :, :]
            
            u_new[-1, :, :] = (1 / (1 + (beta * dr)/alpha)) * u_new[-2, :, :]
            
            # Применение граничного условия третьего рода на верхней и нижней границах
            B = np.array([B_function(t) for t in np.arange(0, T, dt)[:n]])  # Формируем массив B
            B = np.pad(B, (0, max(len(r) - len(B), 0)), mode='constant')  # Дополнение нулями до нужной длины

            for j in range(Nt):
                # При z = zL
                u_slice_left = fft(u[n-1, :, j, 0]).real
                conv_left = convolution(u_slice_left, B[:len(u_slice_left)])
                conv_left = ifft(conv_left).real
                u_new[:, j, 0] = u[n-1, :, j, 0] - dt * (c * (u[n-1, :, j, 1] - u[n-1, :, j, 0]) / dz + conv_left)
                
                # При z = zR
                u_slice_right = fft(u[n-1, :, j, -1]).real
                conv_right = convolution(u_slice_right, B[:len(u_slice_right)])
                conv_right = ifft(conv_right).real
                u_new[:, j, -1] = u[n-1, :, j, -1] + dt * (c * (u[n-1, :, j, -1] - u[n-1, :, j, -2]) / dz + conv_right)
               
            # Сохраняем решение для текущего временного шага
            u[n] = u_new.copy()
        print(CFL)
        return u


# In[8]:


default_args = {'R' : 1.0,      # Радиус цилиндра
'ZL' : -1.0,     # Левая граница по z
'ZR' : 1.0,     # Правая граница по z
'T' : 2.0,      # Временной интервал
'Nr' : 100,      # Число узлов по радиальной координате
'Nt' : 100,      # Число узлов по угловой координате
'Nz' : 100,      # Число узлов по осевой координате
'Ntime' : 100,  # Число временных шагов
'c' : 1.0,      # Скорость распространения волны
'alpha' : 1.0,  # Коэффициент для граничного условия третьего рода на боковой поверхности
'beta' : 1.0}  # Коэффициент для граничного условия третьего рода на боковой поверхности


# In[ ]:





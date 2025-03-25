#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Пример трехмерного массива u
# u = np.random.rand(50, 10, 10)  # Замените на ваш массив

def create_animation(u : np.array,
                     x_ticks : np.array,
                     y_ticks : np.array,
                     output_filename="animation.gif"):
    # Проверяем размерность массива
    if u.ndim != 3:
        raise ValueError("Массив u должен быть трехмерным")

    # Создаем фигуру и оси для imshow
    fig, ax = plt.subplots()
    
    # Начальная картинка - первый слой по оси 0
    initial_frame = u[0, :, :]
    im = ax.imshow(initial_frame, cmap='viridis', aspect='auto')
    
    ax.set_xlabel(r'$z$')  # Радиальная координата
    ax.set_ylabel(r'$r$')  # Угловая координата
    #ax.set_xticks(x_ticks)
    #ax.set_yticks(y_ticks)

    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im)
    cbar.set_label('Значение')

    # Функция обновления для каждого кадра
    def update(frame):
        ax.set_title(f"Time stamp {frame}")
        im.set_data(u[frame, :, :])  # Обновляем данные изображения
        return [im]

    # Создаем анимацию
    anim = FuncAnimation(fig, update, frames=u.shape[0], interval=200, blit=True)

    # Сохраняем анимацию в формате .gif
    anim.save(output_filename, writer='pillow', fps=10)
    print(f"Анимация сохранена как {output_filename}")
    
    
    
    
def create_3d_animation(u, output_filename="3d_animation.gif"):
    # Проверяем размерность массива
    if u.ndim != 4:
        raise ValueError("Массив u должен быть четырехмерным")

    # Создаем фигуру и оси для трехмерной визуализации
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Генерируем сетку для трехмерного пространства
    x = np.arange(u.shape[1])
    y = np.arange(u.shape[2])
    X, Y = np.meshgrid(x, y)

    # Начальная картинка - первый слой по оси 0
    Z = u[0, :, :, int(u.shape[3] / 2)]  # Средний срез по четвертой оси
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Настройка пределов осей
    ax.set_xlim(0, u.shape[1] - 1)
    ax.set_ylim(0, u.shape[2] - 1)
    ax.set_zlim(np.min(u), np.max(u))
    
    ax.set_xlabel(r'$r$')  # Радиальная координата
    ax.set_ylabel(r'$\theta$')  # Угловая координата
    ax.set_zlabel(r'$z$')  # Вертикальная координата

    # Функция обновления для каждого кадра
    def update(frame):
        ax.clear()  # Очищаем предыдущий кадр
        ax.set_xlim(0, u.shape[1] - 1)
        ax.set_ylim(0, u.shape[2] - 1)
        ax.set_zlim(np.min(u), np.max(u))
        Z = u[frame, :, :, int(u.shape[3] / 2)]  # Берем средний срез по четвертой оси
        return ax.plot_surface(X, Y, Z, cmap='viridis')

    # Создаем анимацию
    anim = FuncAnimation(fig, update, frames=u.shape[0], interval=200, blit=False)

    # Сохраняем анимацию в формате .gif
    anim.save(output_filename, writer='pillow', fps=10)
    print(f"Анимация сохранена как {output_filename}")


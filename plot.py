import numpy as np
import matplotlib.pyplot as plt
from function import objective_function

# Define o intervalo para o plot
x_range = np.arange(-500, 505, 5)
y_range = np.arange(-500, 505, 5)
X, Y = np.meshgrid(x_range, y_range)

# Calcula os valores de Z usando a função objetivo
Z = objective_function(X, Y)

# Cria a figura 3D interativa
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12, 8))
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, linewidth=0.7)

ax.set_title('Visualização 3D da Função Objetivo w1+w4', fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
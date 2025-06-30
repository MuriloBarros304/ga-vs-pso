import numpy as np
import matplotlib.pyplot as plt
from function import ObjectiveFunction

objective_function = ObjectiveFunction()

# Define o intervalo para o plot
x_range = np.arange(-500, 501, 5)
y_range = np.arange(-500, 501, 5)
X, Y = np.meshgrid(x_range, y_range)

# Calcula os valores de Z usando a função objetivo
Z = objective_function(X, Y)

# Cria a figura 3D interativa
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12, 8))
# ax.plot_surface(X, Y, Z, cmap='autumn', edgecolor='none', alpha=0.9)
ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.5, linewidth=1.2)

ax.set_title('Visualização 3D da Função Objetivo w1+w4', fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(elev=50, azim=-65)
fig.tight_layout()

plt.show()
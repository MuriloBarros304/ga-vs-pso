import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from function import objective_function_w4
from pso import pso

NUM_PARTICLES = 50
MAX_ITERATIONS = 100
BOUNDS = (np.array([-500, -500]), np.array([500, 500]))
TOLERANCE = 1e-6
PATIENCE = 10
LEVEL_UPDATE_INTERVAL = 10 

print("Iniciando otimização com PSO...")
best_pos, best_cost, pos_history, fitness_history = pso(
    num_particles=NUM_PARTICLES,
    num_iterations=MAX_ITERATIONS,
    bounds=BOUNDS,
    tolerance=TOLERANCE,
    patience=PATIENCE
)
actual_iterations = len(pos_history)
print(f"Otimização concluída em {actual_iterations} iterações!")
print(f"Melhor posição encontrada: {best_pos}")
print(f"Custo (valor da função): {best_cost}")

x_range = np.arange(BOUNDS[0][0], BOUNDS[1][0] + 1, 10)
y_range = np.arange(BOUNDS[0][1], BOUNDS[1][1] + 1, 10)
X, Y = np.meshgrid(x_range, y_range)
Z_background = objective_function_w4(X, Y)

fig, ax = plt.subplots(figsize=(10, 8))

last_levels = None

def animate(i):
    global last_levels  # Permite modificar a variável fora da função
    ax.clear()

    current_positions = pos_history[i]
    current_fitnesses = fitness_history[i]
    
    if i % LEVEL_UPDATE_INTERVAL == 0 or last_levels is None:
        min_level = np.min(current_fitnesses)
        max_level = np.max(current_fitnesses)
        if max_level <= min_level:
            max_level = min_level + 1.0
        
        # Recalcula e armazena os novos níveis
        last_levels = np.linspace(min_level, max_level, 20)
    
    # Sempre usa os 'last_levels' para desenhar, garantindo estabilidade
    ax.contourf(X, Y, Z_background, levels=last_levels, cmap=plt.cm.viridis, extend='both')
    
    ax.scatter(current_positions[:, 0], current_positions[:, 1], marker='o', color='red', alpha=0.7, zorder=100, label='Partículas')

    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")
    ax.set_title(f'Iteração {i+1}/{actual_iterations} | Melhor Custo Atual: {np.min(current_fitnesses):.2f}', fontsize=16)
    ax.legend(loc='upper right')
    ax.set_xlim(BOUNDS[0][0], BOUNDS[1][0])
    ax.set_ylim(BOUNDS[0][1], BOUNDS[1][1])

anim = FuncAnimation(fig, animate, frames=actual_iterations, interval=150, blit=False)
anim.save('pso_animation.mp4', writer='ffmpeg', fps=5, dpi=120)
print("Animação 'pso_animation.mp4' salva com sucesso!")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from function import objective_function_w4
from pso import pso

NUM_PARTICLES = 40 # Número de partículas
MAX_ITERATIONS = 100 # Número máximo de iterações
BOUNDS = (np.array([-500, -500]), np.array([500, 500])) # Limites do espaço de busca
INERCIA_WEIGHT = 0.5 # Peso da inércia
COGNITIVE_COEFF = 1.5 # Coeficiente cognitivo
SOCIAL_COEFF = 1.5 # Coeficiente social
TOLERANCE = 1e-2 # Tolerância para considerar que não houve melhoria significativa
PATIENCE = 10 # Número de iterações sem melhoria antes de parar
LEVEL_UPDATE_INTERVAL = 10 # Intervalo para atualizar os níveis de contorno na animação

best_pos, best_cost, pos_history, fitness_history = pso(
    num_particles=NUM_PARTICLES,
    max_iterations=MAX_ITERATIONS,
    bounds=BOUNDS,
    inertia_weight=INERCIA_WEIGHT,
    cognitive_coeff=COGNITIVE_COEFF,
    social_coeff=SOCIAL_COEFF,
    tolerance=TOLERANCE,
    patience=PATIENCE
)
actual_iterations = len(pos_history)
print(f"Melhor posição encontrada: ({best_pos[0]:.4f}, {best_pos[1]:.4f})")
print(f"Z ótimo: {best_cost:.2f}")

x_range = np.arange(BOUNDS[0][0], BOUNDS[1][0] + 1, 10)
y_range = np.arange(BOUNDS[0][1], BOUNDS[1][1] + 1, 10)
X, Y = np.meshgrid(x_range, y_range)
Z_background = objective_function_w4(X, Y)

fig, ax = plt.subplots(figsize=(10, 8))

last_levels = None

def animate(i):
    global last_levels
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
    
    ax.contourf(X, Y, Z_background, levels=last_levels, cmap='autumn', alpha=0.7, zorder=5)
    
    ax.scatter(current_positions[:, 0], current_positions[:, 1], marker='o', color='blue', alpha=0.7, zorder=5, label='Partículas')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'Iteração {i+1}/{actual_iterations} | Melhor Custo Atual: {np.min(current_fitnesses):.2f}', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(BOUNDS[0][0], BOUNDS[1][0])
    ax.set_ylim(BOUNDS[0][1], BOUNDS[1][1])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

anim = FuncAnimation(fig, animate, frames=actual_iterations, interval=150, blit=False)
anim.save('animacoes/pso_animation.mp4', writer='ffmpeg', fps=2, dpi=100)
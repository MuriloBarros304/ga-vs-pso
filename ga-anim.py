import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from function import objective_function_w4
from ga import ga

NUM_INDIVIDUALS = 40       # Número de indivíduos na população
MAX_GENERATIONS = 100      # Número máximo de gerações
BOUNDS = (np.array([-500, -500]), np.array([500, 500])) # Limites
MUTATION_RATE = 3          # Taxa de mutação
CROSSOVER_RATE = 0.8       # Taxa de crossover
ELITISM_SIZE = 2           # Número de indivíduos de elite
TOURNAMENT_SIZE = 3        # Tamanho do torneio para seleção
TOLERANCE = 1e-2           # Tolerância para estagnação
PATIENCE = 5               # Paciência antes de parar
LEVEL_UPDATE_INTERVAL = 10 # Intervalo de atualização do contorno

# --- EXECUÇÃO DO GA ---
print("Iniciando otimização com Algoritmo Genético (GA)...")
best_ind, best_cost, population_history, fitness_history = ga(
    num_individuals=NUM_INDIVIDUALS,
    max_generations=MAX_GENERATIONS,
    bounds=BOUNDS,
    mutation_rate=MUTATION_RATE,
    crossover_rate=CROSSOVER_RATE,
    elitism_size=ELITISM_SIZE,
    tournament_size=TOURNAMENT_SIZE,
    tolerance=TOLERANCE,
    patience=PATIENCE
)
actual_generations = len(population_history)
print(f"Melhor indivíduo encontrado: ({best_ind[0]:.4f}, {best_ind[1]:.4f})")
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

    current_population = population_history[i]
    current_fitnesses = fitness_history[i]
    
    if i % LEVEL_UPDATE_INTERVAL == 0 or last_levels is None:
        min_level = np.min(current_fitnesses)
        max_level = np.max(current_fitnesses)
        if max_level <= min_level:
            max_level = min_level + 1.0
        
        last_levels = np.linspace(min_level, max_level, 40)
    
    ax.contourf(X, Y, Z_background, levels=last_levels, cmap='autumn', alpha=0.7, zorder=5)
    
    ax.scatter(current_population[:, 0], current_population[:, 1], 
               marker='o', color='green', alpha=0.7, zorder=10, label='Indivíduos')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'GA - Geração {i+1}/{actual_generations} | Melhor Custo: {np.min(current_fitnesses):.2f}', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(BOUNDS[0][0], BOUNDS[1][0])
    ax.set_ylim(BOUNDS[0][1], BOUNDS[1][1])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

anim = FuncAnimation(fig, animate, frames=actual_generations, interval=150, blit=False)
# Salva na mesma pasta e com os mesmos parâmetros do PSO
anim.save('animacoes/ga_animation.mp4', writer='ffmpeg', fps=2, dpi=100)

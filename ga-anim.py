import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from function import objective_function
from ga import ga

NUM_INDIVIDUALS = 100      # Número de indivíduos na população
MAX_GENERATIONS = 20       # Número máximo de gerações
BOUNDS = (np.array([-500, -500]), np.array([500, 500])) # Limites
MUTATION_RATE = 1.0        # Taxa de mutação (0 a 1)
MUTATION_STRENGTH = 1.0    # Força da mutação (0 a 1)
CROSSOVER_RATE = 0.9       # Taxa de crossover (0 a 1)
ELITISM_SIZE = 3           # Número de indivíduos de elite
TOURNAMENT_SIZE = 5        # Tamanho do torneio para seleção
TOLERANCE = 1e-4           # Tolerância para estagnação
PATIENCE = 5               # Paciência antes de parar
LEVEL_UPDATE_INTERVAL = 10 # Intervalo de atualização do contorno

print('------- GA --------')
best_ind, best_cost, population_history, fitness_history = ga(
    num_individuals=NUM_INDIVIDUALS,
    max_generations=MAX_GENERATIONS,
    bounds=BOUNDS,
    mutation_rate=MUTATION_RATE,
    mutation_strength=MUTATION_STRENGTH,
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
X, Y = np.meshgrid(x_range, y_range) # Cria uma grade de pontos
Z_background = objective_function(X, Y)

fig, ax = plt.subplots(figsize=(10, 8))
fitness_levels = np.linspace(np.min(objective_function(X, Y)), np.max(objective_function(X, Y)), 50)

def animate(i):
    ax.clear()

    current_population = population_history[i]
    current_fitnesses = fitness_history[i]
            
    ax.contourf(X, Y, Z_background, levels=fitness_levels, cmap='autumn', alpha=0.7, zorder=5)
    
    ax.scatter(current_population[:, 0], current_population[:, 1], 
               marker='o', color='green', alpha=0.7, zorder=10, label='Indivíduos')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'Geração {i}/{actual_generations-1} | Melhor Z: {np.min(current_fitnesses):.2f}', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(BOUNDS[0][0], BOUNDS[1][0])
    ax.set_ylim(BOUNDS[0][1], BOUNDS[1][1])
    ax.grid(True, linestyle='--', alpha=1)
    ax.set_aspect('equal', adjustable='box')

anim = FuncAnimation(fig, animate, frames=actual_generations, interval=150, blit=False)
anim.save('animacoes/ga_animation.mp4', writer='ffmpeg', fps=1, dpi=100)

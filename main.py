import numpy as np
from run_pso import run_pso_and_animate
from run_ga import run_ga_and_animate

PSO_CONFIG = {
    'num_particles': 30,
    'max_iterations': 50,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'cognitive_coeff': 1.2,
    'social_coeff': 1.0,
    'inertia_weight': 0.5,
    'tolerance': 1e-4,
    'patience': 5
}

GA_CONFIG = {
    'num_individuals': 50,
    'max_generations': 50,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'mutation_rate': 0.1,
    'mutation_strength': 5.0,
    'crossover_rate': 0.9,
    'elitism_size': 2,
    'tournament_size': 3,
    'tolerance': 1e-4,
    'patience': 5
}

if __name__ == '__main__':
    
    # Executa o experimento com PSO
    run_pso_and_animate(params=PSO_CONFIG)
    
    # Executa o experimento com GA
    run_ga_and_animate(params=GA_CONFIG)
    
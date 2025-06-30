import numpy as np
from run_pso import run_pso_and_animate
from run_ga import run_ga_and_animate
from function import ObjectiveFunction

function = ObjectiveFunction()

PSO_CONFIG = {
    'obj_func': function,
    'num_particles': 30,
    'max_iterations': 50,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'cognitive_coeff': 1.2,
    'social_coeff': 1.0,
    'inertia_weight': 0.5,
    'tolerance': 1e-3,
    'patience': 5
}

GA_CONFIG = {
    'obj_func': function,
    'num_individuals': 50,
    'max_generations': 50,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'mutation_rate': 0.25,
    'mutation_strength': 7.5,
    'crossover_rate': 0.7,
    'elitism_size': 2,
    'tournament_size': 3,
    'tolerance': 1e-3,
    'patience': 5
}

if __name__ == '__main__':
    function.reset()
    run_pso_and_animate(params=PSO_CONFIG)
    
    function.reset()
    run_ga_and_animate(params=GA_CONFIG)

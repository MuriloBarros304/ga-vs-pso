import numpy as np
from run_pso import run_pso_and_animate
from run_ga import run_ga_and_animate
from function import ObjectiveFunction

function = ObjectiveFunction()

PSO_CONFIG = {
    'obj_func': function,
    'num_particles': 40,
    'max_iterations': 50,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'cognitive_coeff': 0.5,
    'social_coeff': 0.4,
    'min_w': 0.01,
    'max_w': 0.75,
    'tolerance': 1e-3,
    'patience': 10
}

GA_CONFIG = {
    'obj_func': function,
    'num_individuals': 40,
    'max_generations': 65,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'mutation_rate': 0.07,
    'mutation_strength': 20,
    'crossover_rate': 0.7,
    'elitism_size': 10,
    'tolerance': 1e-3,
    'patience': 5
}

if __name__ == '__main__':
    function.reset()
    run_pso_and_animate(params=PSO_CONFIG)
    
    function.reset()
    run_ga_and_animate(params=GA_CONFIG)

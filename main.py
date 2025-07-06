import numpy as np
from run_pso import run_pso_and_animate
from run_ga import run_ga_and_animate
from function import ObjectiveFunction

function = ObjectiveFunction()

PSO_CONFIG = {
    'obj_func': function,
    'num_particles': 40,
    'max_iterations': 100,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'cognitive_coeff': 0.5,
    'social_coeff': 0.4,
    'min_w': 0.01,
    'max_w': 0.75,
    'tolerance': 1e-5,
    'patience': 10
}

GA_CONFIG = {
    'obj_func': function,
    'num_individuals': 40,
    'max_generations': 100,
    'bounds': (np.array([-500, -500]), np.array([500, 500])),
    'mutation_rate': 0.07,
    'mutation_strength': 20,
    'crossover_rate': 0.7,
    'elitism_size': 10,
    'tolerance': 1e-5,
    'patience': 10
}

# PSO
""" valores_de_n_particulas = [10, 20, 30, 50, 80]
valores_do_coeficiente_cognitivo = [0.6, 0.9, 1.4, 1.8, 2.2]
valores_do_coeficiente_social = [0.5, 0.8, 0.9, 1.5, 2]
valores_de_max_w = [0.4, 0.5, 0.6, 0.7, 0.9]
valores_de_min_w = [0.01, 0.02, 0.05, 0.09, 0.1]

for num_particles in valores_de_n_particulas:
    PSO_CONFIG['num_particles'] = num_particles
    for cognitive_coeff in valores_do_coeficiente_cognitivo:
        PSO_CONFIG['cognitive_coeff'] = cognitive_coeff
        for social_coeff in valores_do_coeficiente_social:
            PSO_CONFIG['social_coeff'] = social_coeff
            for i in range(len(valores_de_max_w)):
                PSO_CONFIG['max_w'] = valores_de_max_w[i]
                PSO_CONFIG['min_w'] = valores_de_min_w[i]
                run_pso_and_animate(params=PSO_CONFIG) """

# GA
""" valores_de_num_individuos = [10, 20, 30, 50, 80]
valores_da_taxa_de_crossover = [0.6, 0.7, 0.75, 0.8, 0.85]
valores_da_taxa_de_mutacao = [0.02, 0.03, 0.07, 0.08, 0.1]
valores_do_tamanho_da_elite = [1, 2, 4, 7, 10]

for num_individuals in valores_de_num_individuos:
    GA_CONFIG['num_individuals'] = num_individuals
    for elitism_size in valores_do_tamanho_da_elite:
        GA_CONFIG['elitism_size'] = elitism_size
        for crossover_rate in valores_da_taxa_de_crossover:
            GA_CONFIG['crossover_rate'] = crossover_rate
            for mutation_rate in valores_da_taxa_de_mutacao:
                GA_CONFIG['mutation_rate'] = mutation_rate
                run_ga_and_animate(params=GA_CONFIG) """

if __name__ == '__main__':
    function.reset()
    run_pso_and_animate(params=PSO_CONFIG)
    
    function.reset()
    run_ga_and_animate(params=GA_CONFIG)

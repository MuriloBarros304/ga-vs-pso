# run_ga.py
import numpy as np
import os
from function import objective_function
from ga import ga
from animator import create_animation

def run_ga_and_animate():
    """
    Define os parâmetros, executa o GA, printa os resultados e gera a animação.
    """
    # --- PARÂMETROS DO ALGORITMO GENÉTICO ---
    params = {
        'num_individuals': 80,
        'max_generations': 100,
        'bounds': (np.array([-500, -500]), np.array([500, 500])),
        'mutation_rate': 0.2,
        'mutation_strength': 10.0,
        'crossover_rate': 0.8,
        'elitism_size': 2,
        'tournament_size': 3,
        'tolerance': 1e-6,
        'patience': 10
    }
    
    # --- EXECUÇÃO DO GA ---
    print('------------ GA -------------')
    best_ind, best_cost, population_history, fitness_history = ga(**params)
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    print(f"Otimização concluída em {len(population_history) - 1} gerações.")
    print(f"Melhor indivíduo encontrado: ({best_ind[0]:.4f}, {best_ind[1]:.4f})")
    print(f"Valor da função no ponto ótimo (Z): {best_cost:.4f}")

    # --- GERAÇÃO DA ANIMAÇÃO ---
    create_animation(
        population_history=population_history,
        fitness_history=fitness_history,
        objective_function=objective_function,
        bounds=params['bounds'],
        filename="animacoes/ga_animation.mp4",
        title="Otimização por Algoritmo Genético (GA)",
        particle_color='green',
        particle_label='Indivíduos'
    )
    print("------------------------------------------\n")
    
if __name__ == '__main__':
    run_ga_and_animate()
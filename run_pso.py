import numpy as np
import os
from function import objective_function
from pso import pso
from animator import create_animation

def run_pso_and_animate():
    """
    Define os parâmetros, executa o PSO, printa os resultados e gera a animação.
    """
    params = {
        'num_particles': 40,
        'max_iterations': 100,
        'bounds': (np.array([-500, -500]), np.array([500, 500])),
        'cognitive_coeff': 1.5,
        'social_coeff': 1.5,
        'tolerance': 1e-6,
        'patience': 10
    }

    # --- EXECUÇÃO DO PSO ---
    print('------------ PSO -------------')
    best_pos, best_cost, pos_history, fitness_history = pso(**params)
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    print(f"Otimização concluída em {len(pos_history) - 1} iterações.")
    print(f"Melhor posição encontrada: ({best_pos[0]:.4f}, {best_pos[1]:.4f})")
    print(f"Valor da função no ponto ótimo (Z): {best_cost:.4f}")
    
    # --- GERAÇÃO DA ANIMAÇÃO ---
    os.makedirs('animacoes', exist_ok=True)
    create_animation(
        population_history=pos_history,
        fitness_history=fitness_history,
        objective_function=objective_function,
        bounds=params['bounds'],
        filename="animacoes/pso_animation.mp4",
        title="Otimização por Enxame de Partículas (PSO)",
        particle_color='blue',
        particle_label='Partículas'
    )
    print("-------------------------------------------\n")

if __name__ == '__main__':
    run_pso_and_animate()
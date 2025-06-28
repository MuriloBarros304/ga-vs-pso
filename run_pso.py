from function import ObjectiveFunction
from pso import pso
from animator import create_animation

def run_pso_and_animate(params: dict):
    """
    Define os parâmetros, executa o PSO, printa os resultados e gera a animação.
    Args:
        params (dict): Dicionário contendo os parâmetros do PSO.
    """

    # --- EXECUÇÃO DO PSO ---
    print('------------ PSO -------------')
    best_pos, best_cost, pos_history, fitness_history = pso(**params)
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    print(f"Melhor posição encontrada: ({best_pos[0]:.4f}, {best_pos[1]:.4f})")
    print(f"Valor da função no ponto ótimo (Z): {best_cost:.4f}")
    
    # --- GERAÇÃO DA ANIMAÇÃO ---
    create_animation(
        population_history=pos_history,
        fitness_history=fitness_history,
        objective_function=params['obj_func'],
        bounds=params['bounds'],
        filename="animacoes/pso_animation.mp4",
        title="PSO",
        particle_color='blue',
        particle_label='Partículas'
    )

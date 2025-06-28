from function import ObjectiveFunction
from ga import ga
from animator import create_animation

def run_ga_and_animate(params: dict):
    """
    Define os parâmetros, executa o GA, printa os resultados e gera a animação.
    Args:
        params (dict): Dicionário contendo os parâmetros do GA.
    """
    
    # --- EXECUÇÃO DO GA ---
    print('------------ GA -------------')
    best_ind, best_cost, population_history, fitness_history = ga(**params)
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    print(f"Melhor indivíduo encontrado: ({best_ind[0]:.4f}, {best_ind[1]:.4f})")
    print(f"Valor da função no ponto ótimo (Z): {best_cost:.4f}")

    # --- GERAÇÃO DA ANIMAÇÃO ---
    create_animation(
        population_history=population_history,
        fitness_history=fitness_history,
        objective_function=params['obj_func'],
        bounds=params['bounds'],
        filename="animacoes/ga_animation.mp4",
        title="GA",
        particle_color='green',
        particle_label='Indivíduos'
    )

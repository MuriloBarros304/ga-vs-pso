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
    best_ind, best_cost, population_history, fitness_history, cont = ga(**params)
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    total_evaluations = params['obj_func'].evaluations
    total_multiplications = params['obj_func'].multiplications + cont['multiplications']
    total_divisions = params['obj_func'].divisions + cont['divisions']
    print(f"Ponto ótimo: ({best_ind[0]:.8f}, {best_ind[1]:.8f})")
    print(f"Z ótimo: {best_cost:.8f}")
    print(f"Avaliações da função: {total_evaluations}")
    print(f"Multiplicações: {total_multiplications}")
    print(f"Divisões: {total_divisions}")

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

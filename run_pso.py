from pso import pso
from animator import create_animation
from analysis import find_discovery
import os

def run_pso_and_animate(params: dict):
    """
    Define os parâmetros, executa o PSO, printa os resultados e gera a animação.
    Args:
        params (dict): Dicionário contendo os parâmetros do PSO.
    """

    # Identifica qual função estamos rodando
    func_name = params['obj_func'].target_func
    print(f'------------ PSO ({func_name}) -------------')

    # --- EXECUÇÃO DO PSO ---
    best_pos, best_cost, pos_history, fitness_history, cont = pso(**params)

     # --- ANÁLISE PÓS-EXECUÇÃO ---
    discovery_gen = find_discovery(fitness_history, best_cost, threshold_percent=0.1)
    
    evaluations = params['obj_func'].evaluations
    discovery_nfe = (discovery_gen + 1) * params['num_particles']

    # --- EXIBIÇÃO DOS RESULTADOS ---
    total_multiplications = params['obj_func'].multiplications + cont['multiplications']
    total_divisions = params['obj_func'].divisions + cont['divisions']
    
    print(f"Função: {func_name}")
    print(f"Ponto ótimo: ({best_pos[0]:.8f}, {best_pos[1]:.8f})")
    print(f"Z ótimo: {best_cost:.8f}")
    print(f"Avaliações até encontrar o mínimo global: {discovery_nfe} (Iteração {discovery_gen})")
    print(f"Total de avaliações da função: {evaluations}")
    print(f"Multiplicações: {total_multiplications}")
    print(f"Divisões: {total_divisions}\n")
    
    # --- GERAÇÃO DA ANIMAÇÃO DINÂMICA ---
    if not os.path.exists("animacoes"):
        os.makedirs("animacoes")

    filename = f"animacoes/pso_{func_name}.mp4"
    plot_title = f"PSO - {func_name}"

    create_animation(
        population_history=pos_history,
        fitness_history=fitness_history,
        objective_function=params['obj_func'],
        bounds=params['bounds'],
        filename=filename,
        title=plot_title,
        particle_color='blue',
        particle_label='Partículas'
    )
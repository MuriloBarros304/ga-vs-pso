from ga import ga
from animator import create_animation
from analysis import find_discovery
import os

def run_ga_and_animate(params: dict):
    """
    Define os parâmetros, executa o GA, printa os resultados e gera a animação.
    Args:
        params (dict): Dicionário contendo os parâmetros do GA.
    """
    
    # Identifica qual função estamos rodando para personalizar logs e arquivos
    func_name = params['obj_func'].target_func
    print(f'------------ GA ({func_name}) -------------')
    
    # --- EXECUÇÃO DO GA ---
    best_ind, best_cost, population_history, fitness_history, cont = ga(**params)

     # --- ANÁLISE PÓS-EXECUÇÃO ---
    discovery_gen = find_discovery(fitness_history, best_cost, threshold_percent=0.1)
    
    evaluations = params['obj_func'].evaluations
    discovery_nfe = (discovery_gen + 1) * params['num_individuals']
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    total_multiplications = params['obj_func'].multiplications + cont['multiplications']
    total_divisions = params['obj_func'].divisions + cont['divisions']
    
    print(f"Função: {func_name}")
    print(f"Ponto ótimo: ({best_ind[0]:.8f}, {best_ind[1]:.8f})")
    print(f"Z ótimo: {best_cost:.8f}")
    print(f"Avaliações até encontrar o mínimo global: {discovery_nfe} (Geração {discovery_gen})")
    print(f"Total de avaliações da função: {evaluations}")
    print(f"Multiplicações: {total_multiplications}")
    print(f"Divisões: {total_divisions}\n")

    # --- GERAÇÃO DA ANIMAÇÃO DINÂMICA ---
    # Cria diretório se não existir
    if not os.path.exists("animacoes"):
        os.makedirs("animacoes")

    filename = f"animacoes/ga_{func_name}.mp4"
    plot_title = f"GA - {func_name}"

    create_animation(
        population_history=population_history,
        fitness_history=fitness_history,
        objective_function=params['obj_func'],
        bounds=params['bounds'],
        filename=filename,
        title=plot_title,
        particle_color='green',
        particle_label='Indivíduos'
    )
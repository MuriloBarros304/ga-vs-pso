from ga import ga
from animator import create_animation
from analysis import find_discovery

def run_ga_and_animate(params: dict):
    """
    Define os parâmetros, executa o GA, printa os resultados e gera a animação.
    Args:
        params (dict): Dicionário contendo os parâmetros do GA.
    """
    
    # --- EXECUÇÃO DO GA ---
    print('------------ GA -------------')
    best_ind, best_cost, population_history, fitness_history, cont = ga(**params)

     # --- ANÁLISE PÓS-EXECUÇÃO ---
    # Encontra a geração em que a solução final foi "descoberta" (chegou a 0.1% do valor final)
    discovery_gen = find_discovery(fitness_history, best_cost, threshold_percent=0.1)
    
    # Calcula o NFE no momento da descoberta
    # O histórico tem N+1 estados, então o NFE da geração 'i' é (i * pop_size) + pop_size inicial
    # Como o nosso contador já faz isso, podemos simplesmente calcular de forma proporcional
    evaluations = params['obj_func'].evaluations
    discovery_nfe = -1
    if discovery_gen != -1:
        # Estimativa do NFE no momento da descoberta
        discovery_nfe = (discovery_gen + 1) * params['num_individuals']
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    total_multiplications = params['obj_func'].multiplications + cont['multiplications']
    total_divisions = params['obj_func'].divisions + cont['divisions']
    """ print(f"Ponto ótimo: ({best_ind[0]:.8f}, {best_ind[1]:.8f})")
    print(f"Z ótimo: {best_cost:.8f}")
    print(f"Avaliações até encontrar o mínimo global: {discovery_nfe} (Geração {discovery_gen})")
    print(f"Total de avaliações da função: {evaluations}")
    print(f"Multiplicações: {total_multiplications}")
    print(f"Divisões: {total_divisions}") """

    # --- GERAÇÃO DA ANIMAÇÃO ---
    """ create_animation(
        population_history=population_history,
        fitness_history=fitness_history,
        objective_function=params['obj_func'],
        bounds=params['bounds'],
        filename="animacoes/ga_animation.mp4",
        title="GA",
        particle_color='green',
        particle_label='Indivíduos'
    ) """

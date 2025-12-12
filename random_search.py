import numpy as np
import os
from function import ObjectiveFunction
from pso import pso
from ga import ga

# ==============================================================================
# CONFIGURAÇÕES DA BUSCA
# ==============================================================================
SEARCH_ITERATIONS = 20
BOUNDS = (np.array([-500, -500]), np.array([500, 500]))

# Instâncias das funções
obj_func_rastrigin = ObjectiveFunction('rastrigin')
obj_func_w1w4 = ObjectiveFunction('w1_w4')

def format_params_for_display(params):
    """
    Formata o dicionário de parâmetros para parecer código Python limpo.
    Remove objetos como 'obj_func' e 'bounds' para facilitar o copy-paste.
    """
    clean_params = {k: v for k, v in params.items() if k not in ['obj_func', 'bounds']}
    
    formatted_str = "{\n"
    for key, value in clean_params.items():
        # Formata floats para não ficarem gigantes, se necessário
        if isinstance(value, float):
            formatted_str += f"    '{key}': {value:.5f},\n"
        else:
            formatted_str += f"    '{key}': {value},\n"
    formatted_str += "}"
    return formatted_str

def save_results_to_txt(filename, func_name, pso_params, ga_params, pso_score, ga_score):
    """
    Salva os resultados formatados em um arquivo txt.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== MELHORES HIPERPARÂMETROS PARA: {func_name.upper()} ===\n\n")
        
        f.write(f"--- PSO (Melhor Z: {pso_score:.8f}) ---\n")
        f.write(format_params_for_display(pso_params))
        f.write("\n\n")
        
        f.write(f"--- GA (Melhor Z: {ga_score:.8f}) ---\n")
        f.write(format_params_for_display(ga_params))
        f.write("\n")
    
    print(f"Arquivo salvo: {filename}")

def tune_pso(obj_func, bounds, iterations=20):
    func_name = obj_func.target_func
    print(f"\n>>> [TUNING] Iniciando Random Search PSO para '{func_name}' ({iterations} iterações)...")
    
    best_config = None
    best_global_fitness = np.inf
    
    for i in range(iterations):
        # 1. Amostragem Aleatória
        current_max_w = np.random.uniform(0.5, 0.95)
        
        config = {
            'obj_func': obj_func,
            'num_particles': np.random.randint(20, 80),
            'max_iterations': 200, 
            'bounds': bounds,
            'cognitive_coeff': np.random.uniform(0.5, 2.5),
            'social_coeff': np.random.uniform(0.5, 2.5),
            'max_w': current_max_w,
            'min_w': np.random.uniform(0.1, current_max_w - 0.05),
            'tolerance': 1e-5,
            'patience': 25
        }
        
        # 2. Execução Rápida
        obj_func.reset()
        _, cost, _, _, _ = pso(**config)
        
        # 3. Comparação
        if cost < best_global_fitness:
            best_global_fitness = cost
            best_config = config
            
    return best_config, best_global_fitness

def tune_ga(obj_func, bounds, iterations=20):
    func_name = obj_func.target_func
    print(f"\n>>> [TUNING] Iniciando Random Search GA para '{func_name}' ({iterations} iterações)...")
    
    best_config = None
    best_global_fitness = np.inf
    
    for i in range(iterations):
        # 1. Amostragem Aleatória
        pop_size = np.random.randint(30, 100)
        elitism = np.random.randint(1, max(2, int(pop_size * 0.2)))
        
        config = {
            'obj_func': obj_func,
            'num_individuals': pop_size,
            'max_generations': 200, 
            'bounds': bounds,
            'mutation_rate': np.random.uniform(0.01, 0.4),
            'mutation_strength': np.random.uniform(1.0, 40.0), 
            'crossover_rate': np.random.uniform(0.6, 0.95),
            'elitism_size': elitism,
            'tolerance': 1e-5,
            'patience': 25
        }
        
        # 2. Execução Rápida
        obj_func.reset()
        _, cost, _, _, _ = ga(**config)
        
        # 3. Comparação
        if cost < best_global_fitness:
            best_global_fitness = cost
            best_config = config

    return best_config, best_global_fitness

if __name__ == "__main__":
    # --- EXECUÇÃO PARA RASTRIGIN ---
    best_pso_rastrigin, score_pso_rast = tune_pso(obj_func_rastrigin, BOUNDS, iterations=SEARCH_ITERATIONS)
    best_ga_rastrigin, score_ga_rast = tune_ga(obj_func_rastrigin, BOUNDS, iterations=SEARCH_ITERATIONS)
    
    # Salvar Rastrigin
    save_results_to_txt(
        "best_params_rastrigin.txt", 
        "rastrigin", 
        best_pso_rastrigin, 
        best_ga_rastrigin,
        score_pso_rast,
        score_ga_rast
    )

    # --- EXECUÇÃO PARA W1+W4 ---
    best_pso_w1w4, score_pso_w1 = tune_pso(obj_func_w1w4, BOUNDS, iterations=SEARCH_ITERATIONS)
    best_ga_w1w4, score_ga_w1 = tune_ga(obj_func_w1w4, BOUNDS, iterations=SEARCH_ITERATIONS)

    # Salvar W1+W4
    save_results_to_txt(
        "best_params_w1_w4.txt", 
        "w1_w4", 
        best_pso_w1w4, 
        best_ga_w1w4,
        score_pso_w1,
        score_ga_w1
    )

    print("\n" + "="*50)
    print(">>> TUNING CONCLUÍDO <<<")
    print("Os parâmetros foram salvos em 'best_params_rastrigin.txt' e 'best_params_w1_w4.txt'.")
    print("Copie o conteúdo destes arquivos para sua main.py.")
    print("="*50)
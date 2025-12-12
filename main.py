import numpy as np
from function import ObjectiveFunction
from run_pso import run_pso_and_animate
from run_ga import run_ga_and_animate

# ==============================================================================
# CONFIGURAÇÃO GLOBAL
# ==============================================================================
# Limites fixos [-500, 500] (A classe Function normaliza Rastrigin internamente)
BOUNDS = (np.array([-500, -500]), np.array([500, 500]))

default_pso_params = {
    'num_particles': 30,
    'max_iterations': 200,
    'bounds': BOUNDS,
}

default_ga_params = {
    'num_individuals': 50,
    'max_generations': 200,
    'bounds': BOUNDS,
}

def run_func(target_func_name, pso_params=None, ga_params=None):
    """
    Executa o fluxo completo (PSO + GA) para uma função alvo específica,
    gerando as animações correspondentes.
    Args:
        target_func_name (str): Nome da função alvo ('schwefel_rosenbrock' ou 'rastrigin').
        pso_params (dict, optional): Parâmetros específicos para o PSO. Usa default se None.
        ga_params (dict, optional): Parâmetros específicos para o GA. Usa default se None.
    Returns:
        None
    """
    print(f"\n{'='*60}")
    print(f"CENÁRIO: {target_func_name.upper()}")
    print(f"{'='*60}")

    # Instancia a função objetivo
    obj_func = ObjectiveFunction(target_func=target_func_name)

    # ------------------- PSO -------------------
    # Usa params passados ou o default
    current_pso_params = default_pso_params.copy() if pso_params is None else pso_params.copy()
    
    # [CORREÇÃO] Injeta a função objetivo no dicionário antes de enviar
    current_pso_params['obj_func'] = obj_func
    
    print(f"\n... Gerando animação PSO para {target_func_name} ...")
    obj_func.reset() # Reset obrigatório antes da rodada final
    run_pso_and_animate(current_pso_params)

    # ------------------- GA --------------------
    # Usa params passados ou o default
    current_ga_params = default_ga_params.copy() if ga_params is None else ga_params.copy()
    
    # [CORREÇÃO] Injeta a função objetivo no dicionário antes de enviar
    current_ga_params['obj_func'] = obj_func
    
    print(f"\n... Gerando animação GA para {target_func_name} ...")
    obj_func.reset() # Reset obrigatório antes da rodada final
    run_ga_and_animate(current_ga_params)

if __name__ == '__main__':
    # ==========================================================================
    # 1. Executa o fluxo para Schwefel-Rosenbrock
    # ==========================================================================
    print("\n>>> INICIANDO PROCESSO DE OTIMIZAÇÃO E GERAÇÃO DE ANIMAÇÕES <<<")
    schwefel_rosenbrock_pso_params = {
        'num_particles': 45,
        'max_iterations': 200,
        'cognitive_coeff': 0.6,
        'social_coeff': 1.6,
        'max_w': 0.6,
        'min_w': 0.1,
        'tolerance': 0.00001,
        'patience': 25,
        'bounds': BOUNDS,
    }
    schwefel_rosenbrock_ga_params = {
        'num_individuals': 94,
        'max_generations': 200,
        'mutation_rate': 0.06,
        'mutation_strength': 26,
        'crossover_rate': 0.7,
        'elitism_size': 6,
        'tolerance': 0.00001,
        'patience': 25,
        'bounds': BOUNDS,
    }
    run_func('schwefel_rosenbrock', pso_params=schwefel_rosenbrock_pso_params, ga_params=schwefel_rosenbrock_ga_params)

    # ==========================================================================
    # 2. Executa o fluxo para Rastrigin
    # ==========================================================================
    rastrigin_pso_params = {
        'num_particles': 49,
        'max_iterations': 200,
        'cognitive_coeff': 2.2,
        'social_coeff': 0.7,
        'max_w': 0.55,
        'min_w': 0.15,
        'tolerance': 1e-5,
        'patience': 25,
        'bounds': BOUNDS,
    }
    rastrigin_ga_params = {
        'num_individuals': 63,
        'max_generations': 200,
        'mutation_rate': 0.04,
        'mutation_strength': 34,
        'crossover_rate': 0.9,
        'elitism_size': 7,
        'tolerance': 1e-5,
        'patience': 25,
        'bounds': BOUNDS,
    }
    run_func('rastrigin', pso_params=rastrigin_pso_params, ga_params=rastrigin_ga_params)
    
    print("\n>>> PROCESSO CONCLUÍDO. <<<")
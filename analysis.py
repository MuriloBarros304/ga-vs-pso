import numpy as np

def find_discovery(fitness_history, final_best_z, threshold_percent=0.1):
    """
    Analisa o histórico de fitness para encontrar a primeira geração que se
    aproximou do resultado final.

    Args:
        fitness_history (list): Lista de arrays com o fitness da população a cada geração.
        final_best_cost (float): O melhor valor de fitness encontrado ao final da execução.
        threshold_percent (float): A porcentagem de proximidade para considerar "descoberto". (ex: 1.0 para 1%, 0.1 para 0.1%)

    Returns:
        int: O índice da geração da descoberta, ou -1 se não for encontrado.
    """
    # Define o que é "próximo o suficiente" do valor final
    threshold_val = abs(final_best_z * (threshold_percent / 100.0)) + 1e-9
    target_fitness = final_best_z + threshold_val

    # Itera sobre o histórico de fitness de cada geração
    for i, generation_fitnesses in enumerate(fitness_history):
        # Se o melhor indivíduo da geração 'i' atingiu o nosso alvo...
        if np.min(generation_fitnesses) <= target_fitness:
            # ...retorna o número da geração e para a busca.
            return i
    
    # Se o loop terminar, significa que o alvo nunca foi atingido
    return -1
from function import objective_function
import numpy as np

def ga(num_individuals: int, max_generations: int, bounds: tuple,
       crossover_rate: float=0.7, mutation_rate: float=0.1, mutation_strength: float=1.0,
       elitism_size: int=1, tournament_size: int=3, tolerance: float=1e-6, patience: int=10) -> tuple:
    """
    Algoritmo Genético para otimização de uma função objetivo.
    Args:
        num_individuals (int): Número de indivíduos na população.
        max_generations (int): Número máximo de gerações.
        bounds (tuple): Limites inferior e superior para os indivíduos.
        crossover_rate (float): Taxa de crossover.
        mutation_rate (float): Taxa de mutação.
        mutation_strength (float): Força da mutação.
        elitism_size (int): Número de indivíduos a serem mantidos na próxima geração (elitismo).
        tournament_size (int): Tamanho do torneio para seleção.
        tolerance (float): Tolerância para considerar convergência.
        patience (int): Número de gerações sem melhoria antes de parar.
    Returns:
        tuple: Melhor indivíduo encontrado, seu valor de fitness, histórico da população e histórico de fitness.
    """
    
    # --- INICIALIZAÇÃO ---
    population = np.random.uniform(bounds[0], bounds[1], (num_individuals, 2))
    fitness = objective_function(population[:, 0], population[:, 1])
    
    # --- HISTÓRICO ---
    population_history = []
    fitness_history = []
    
    best_overall_fitness = np.inf
    best_overall_individual = None

    # Variáveis para rastrear a estagnação
    stagnation_counter = 0
    last_overall_best_fitness = np.inf

    # --- CICLO EVOLUTIVO ---
    for generation in range(max_generations):
        population_history.append(population.copy())
        fitness_history.append(fitness.copy())

        # --- ELITISMO ---
        elite_indices = np.argsort(fitness)[:elitism_size] # Seleciona os 'elitism_size' melhores indivíduos
        new_population = [population[i].copy() for i in elite_indices]

        # --- SELEÇÃO POR TORNEIO ---
        mating_pool = [] # Lista para armazenar os pais selecionados no torneio
        for _ in range(num_individuals - elitism_size): # Preencher o restante da população
            competitor_indices = np.random.choice(num_individuals, tournament_size, replace=False)
            winner_index = competitor_indices[np.argmin(fitness[competitor_indices])]
            mating_pool.append(population[winner_index])

        # --- CROSSOVER ---
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            if i + 1 < len(mating_pool):
                parent2 = mating_pool[i+1]
                if np.random.rand() < crossover_rate:
                    alpha = 0.5
                    children = []
                    for _ in range(2):
                        child = np.zeros_like(parent1)
                        for j in range(len(parent1)):
                            d = np.abs(parent1[j] - parent2[j])
                            min_val = min(parent1[j], parent2[j]) - alpha * d
                            max_val = max(parent1[j], parent2[j]) + alpha * d
                            child[j] = np.random.uniform(min_val, max_val)
                        children.append(child)
                    new_population.extend(children)
                else:
                    new_population.extend([parent1, parent2])
            else:
                new_population.append(parent1)
        
        population = np.array(new_population)

        # --- MUTAÇÃO ---
        if elitism_size < len(population):
            mutation_candidates = population[elitism_size:]
            mask = np.random.rand(*mutation_candidates.shape) < mutation_rate
            mutation_candidates[mask] += np.random.normal(0, mutation_strength, size=mutation_candidates[mask].shape)
        
        population = np.clip(population, bounds[0], bounds[1])
        fitness = objective_function(population[:, 0], population[:, 1])

        # --- ATUALIZAÇÃO DO MELHOR GLOBAL ---
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_overall_fitness:
            best_overall_fitness = fitness[current_best_index]
            best_overall_individual = population[current_best_index].copy()
        
        # MODIFICAÇÃO: LÓGICA DE PARADA POR TOLERÂNCIA E PACIÊNCIA
        improvement = last_overall_best_fitness - best_overall_fitness
        if improvement > tolerance:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if stagnation_counter >= patience:
            print(f"Convergência atingida na geração {generation + 1} devido à estagnação.")
            # Adiciona o último estado antes de parar, para a animação ficar completa
            population_history.append(population.copy())
            fitness_history.append(fitness.copy())
            break
            
        last_overall_best_fitness = best_overall_fitness
    
    # Garante que o último estado seja salvo se o loop terminar por max_generations
    if len(population_history) == max_generations:
        print(f"Número máximo de gerações {max_generations} atingido.")
        population_history.append(population.copy())
        fitness_history.append(fitness.copy())

    return best_overall_individual, best_overall_fitness, population_history, fitness_history
from function import objective_function_w4
import numpy as np

def ga(num_individuals, max_generations, bounds, mutation_rate=0.1, crossover_rate=0.7, 
       tournament_size=3, elitism_size=1, tolerance=1e-6, patience=10, mutation_strength=1.0):
    
    # --- INICIALIZAÇÃO ---
    population = np.random.uniform(bounds[0], bounds[1], (num_individuals, 2))
    fitness = objective_function_w4(population[:, 0], population[:, 1])
    
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
        elite_indices = np.argsort(fitness)[:elitism_size]
        new_population = [population[i].copy() for i in elite_indices]

        # --- SELEÇÃO POR TORNEIO ---
        mating_pool = []
        for _ in range(num_individuals - elitism_size):
            competitor_indices = np.random.choice(num_individuals, tournament_size, replace=False)
            winner_index = competitor_indices[np.argmin(fitness[competitor_indices])]
            mating_pool.append(population[winner_index])

        # --- CROSSOVER (BLX-alpha é uma ótima opção aqui) ---
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
        fitness = objective_function_w4(population[:, 0], population[:, 1])

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
            print(f"\nConvergência atingida na geração {generation + 1} devido à estagnação.")
            # Adiciona o último estado antes de parar, para a animação ficar completa
            population_history.append(population.copy())
            fitness_history.append(fitness.copy())
            break
            
        last_overall_best_fitness = best_overall_fitness
    
    # Garante que o último estado seja salvo se o loop terminar por max_generations
    if len(population_history) == max_generations:
        population_history.append(population.copy())
        fitness_history.append(fitness.copy())

    return best_overall_individual, best_overall_fitness, population_history, fitness_history
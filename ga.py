from function import objective_function
import numpy as np

def ga(num_individuals: int, max_generations: int, bounds: tuple, crossover_rate: float=0.9,
        mutation_rate: float=0.5, mutation_strength: float=1.0, elitism_size: int=1,
        tournament_size: int=3, tolerance: float=1e-6, patience: int=10) -> tuple:
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
    
    # Inicializa o melhor global
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
        new_population = [population[i].copy() for i in elite_indices] # A nova população começa com os indivíduos de elite

        # --- SELEÇÃO POR TORNEIO ---
        mating_pool = [] # Lista para armazenar os pais selecionados no torneio
        for _ in range(num_individuals - elitism_size): # Preencher o restante da população
            competitor_indices = np.random.choice(num_individuals, tournament_size, replace=False)
            winner_index = competitor_indices[np.argmin(fitness[competitor_indices])]
            mating_pool.append(population[winner_index])

        # --- BLEND CROSSOVER (BLX-⍺) ---
        i = 0
        alpha = 0.5 # Fator de mistura
        while len(new_population) < num_individuals: # Enquanto a nova população não estiver completa
            parent1 = mating_pool[i]
            if i + 1 >= len(mating_pool): # Condição de segurança para o último indivíduo em um pool de tamanho ímpar
                new_population.append(parent1.copy())
                break # Encerra o loop while
                
            parent2 = mating_pool[i+1]

            if np.random.rand() < crossover_rate:
                child1 = np.zeros_like(parent1) # Arrays que receberão os genes dos filhos
                child2 = np.zeros_like(parent2)
                
                for j in range(len(parent1)): # Itera sobre cada gene (x, y)
                    d = np.abs(parent1[j] - parent2[j])
                    min_val = min(parent1[j], parent2[j]) - alpha * d
                    max_val = max(parent1[j], parent2[j]) + alpha * d
                    
                    new_gene1 = np.random.uniform(min_val, max_val)
                    new_gene2 = np.random.uniform(min_val, max_val)
                    
                    child1[j] = np.clip(new_gene1, bounds[0][j], bounds[1][j])
                    child2[j] = np.clip(new_gene2, bounds[0][j], bounds[1][j])

                new_population.append(child1) # Adiciona o primeiro filho à nova população
                if len(new_population) < num_individuals: # Verifica se ainda há espaço para o segundo filho
                    new_population.append(child2) # Adiciona o segundo filho à nova população
            else:
                new_population.append(parent1.copy()) # Se não houver crossover, os pais sobrevivem
                if len(new_population) < num_individuals:
                    new_population.append(parent2.copy())
            
            i += 2 # Avança para o próximo par de pais
        
        population = np.array(new_population)

        # --- MUTAÇÃO ---
        if elitism_size < len(population):
            mutation_candidates = population[elitism_size:] # Todos os indivíduos exceto os de elite
            mask = np.random.rand(*mutation_candidates.shape) < mutation_rate
            mutation_candidates[mask] += np.random.normal(0, mutation_strength, size=mutation_candidates[mask].shape)
        
        population = np.clip(population, bounds[0], bounds[1])
        fitness = objective_function(population[:, 0], population[:, 1])

        # --- ATUALIZAÇÃO DO MELHOR GLOBAL ---
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_overall_fitness:
            best_overall_fitness = fitness[current_best_index]
            best_overall_individual = population[current_best_index].copy()
        
        # --- PARADA POR TOLERÂNCIA ---
        improvement = last_overall_best_fitness - best_overall_fitness
        if improvement > tolerance:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if stagnation_counter >= patience:
            print(f"Convergência atingida na geração {generation} devido à estagnação.")
            # Adiciona o último estado antes de parar, para a animação ficar completa
            population_history.append(population.copy())
            fitness_history.append(fitness.copy())
            break
            
        last_overall_best_fitness = best_overall_fitness
    
    # Garante que o último estado seja salvo se o loop terminar por max_generations
    if len(population_history) == max_generations:
        print(f"Número máximo de gerações ({max_generations}) atingido.")
        population_history.append(population.copy())
        fitness_history.append(fitness.copy())

    return best_overall_individual, best_overall_fitness, population_history, fitness_history
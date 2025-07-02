from function import ObjectiveFunction
import numpy as np

def ga(obj_func: ObjectiveFunction, num_individuals: int, max_generations: int,
        bounds: tuple, crossover_rate: float=0.9, mutation_rate: float=0.5,
        mutation_strength: float=1.0, elitism_size: int=1, tolerance: float=1e-6,
        patience: int=10) -> tuple:
    """
    Algoritmo Genético para otimização de uma função objetivo.
    Args:
        obj_func (ObjectiveFunction): Instância da função objetivo a ser minimizada.
        num_individuals (int): Número de indivíduos na população.
        max_generations (int): Número máximo de gerações.
        bounds (tuple): Limites inferior e superior para os indivíduos.
        crossover_rate (float): Taxa de crossover.
        mutation_rate (float): Taxa de mutação.
        mutation_strength (float): Força da mutação.
        elitism_size (int): Número de indivíduos a serem mantidos na próxima geração (elitismo).
        tolerance (float): Tolerância para considerar convergência.
        patience (int): Número de gerações sem melhoria antes de parar.
    Returns:
        tuple: Melhor indivíduo encontrado, seu valor de fitness, histórico da população e histórico de fitness.
    """
    
    # --- INICIALIZAÇÃO ---
    population = np.random.uniform(bounds[0], bounds[1], (num_individuals, 2))
    fitness = obj_func(population[:, 0], population[:, 1])
    counter = {'multiplications': 0, 'divisions': 0} # Contador de operações
    
    # --- HISTÓRICO ---
    population_history = [population.copy()] # Armazena o histórico da população
    fitness_history = [fitness.copy()] # Armazena o histórico de fitness
    
    # Inicializa o melhor global
    best_overall_fitness = np.inf
    best_overall_individual = None

    # Variáveis para rastrear a estagnação
    stagnation_counter = 0
    last_overall_best_fitness = np.inf

    # --- CICLO EVOLUTIVO ---
    stagnation_reached = False
    for generation in range(max_generations):

        # --- ELITISMO ---
        elite_indices = np.argsort(fitness)[:elitism_size] # Seleciona os 'elitism_size' melhores indivíduos
        new_population = [population[i].copy() for i in elite_indices] # A nova população começa com os indivíduos de elite

        # --- SELEÇÃO POR ROLETA ---
        # Aptidão é a diferença entre o pior fitness e o fitness de cada indivíduo
        # A aptidões são normalizadas para probabilidades de seleção, indivíduos mais aptos têm maior chance de serem selecionados

        """ non_elite_mask = np.ones(num_individuals, dtype=bool)
        non_elite_mask[elite_indices] = False
        non_elite_population = population[non_elite_mask]
        non_elite_fitness = fitness[non_elite_mask]
        non_elite_num_ind = len(non_elite_population) """
        
        num_parents_to_select = num_individuals - elitism_size # Número de pais a serem selecionados
        
        ranked_indices = np.argsort(fitness) # Índices dos indivíduos ordenados por fitness
        # ranked_indices = np.argsort(non_elite_fitness) # Sem elite
        rank_aptitude = np.arange(num_individuals, 0, -1) # Lista de aptidões, de num_individuals a 1
        # rank_aptitude = np.arange(non_elite_num_ind, 0, -1) # Sem elite

        total_aptitude = np.sum(rank_aptitude) # Soma das aptidões
    
        selection_probabilities = rank_aptitude / total_aptitude
        counter['divisions'] += num_individuals # Contabiliza as divisões
        # counter['divisions'] += non_elite_num_ind # Sem elite

        final_probabilities = np.zeros(num_individuals) # Inicializa as probabilidades finais
        # final_probabilities = np.zeros(non_elite_num_ind) #Sem elite
        final_probabilities[ranked_indices] = selection_probabilities # Atribui as probabil
        
        parent_indices = np.random.choice( # Sorteia com base nas probabilidades de seleção
            a=num_individuals,
            # a=non_elite_num_ind, # Sem elite
            size=num_parents_to_select,
            replace=True,
            p=final_probabilities
        )

        mating_pool = population[parent_indices] # Cria o pool de pais selecionados
        # mating_pool = non_elite_population[parent_indices] # Sem elite

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
                    counter['multiplications'] += 2 # Contabiliza as multiplicações
                    
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
            # --- DEBUG ---
            # print(f"P1: ({parent1[0]:.2f}, {parent1[1]:.2f}) ; P2: ({parent2[0]:.2f}, {parent2[1]:.2f})   ->   C1: ({new_population[-2][0]:.2f}, {new_population[-2][1]:.2f}) , ({new_population[-1][0] if len(new_population) % 2 == 0 else 'N/A':.2f}, {new_population[-1][1] if len(new_population) % 2 == 0 else 'N/A':.2f})")
            # --- FIM DEBUG ---
            
            i += 2 # Avança para o próximo par de pais
        
        population = np.array(new_population)

        # --- MUTAÇÃO ---
        if elitism_size < len(population):
            mutation_candidates = population[elitism_size:] # Todos os indivíduos exceto os de elite
            mask = np.random.rand(*mutation_candidates.shape) < mutation_rate
            mutation_candidates[mask] += np.random.normal(0, mutation_strength, size=mutation_candidates[mask].shape)
            num_mutations = np.sum(mask)
            counter['multiplications'] += num_mutations

        cliped_population = np.clip(population, bounds[0], bounds[1]) # Garante que os indivíduos estejam dentro dos limites
        
        population = cliped_population

        fitness = obj_func(population[:, 0], population[:, 1])

        # --- ATUALIZAÇÃO DO MELHOR GLOBAL ---
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_overall_fitness:
            best_overall_fitness = fitness[current_best_index]
            best_overall_individual = population[current_best_index].copy()
        
        population_history.append(population.copy()) # Armazena o estado atual da população
        fitness_history.append(fitness.copy())
        
        # --- PARADA POR TOLERÂNCIA ---
        improvement = last_overall_best_fitness - best_overall_fitness
        if improvement > tolerance: # Se houve melhoria significativa
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if stagnation_counter >= patience: # Se acabou a paciência
            stagnation_reached = True
            last_overall_best_fitness = best_overall_fitness # Atualiza o melhor fitness para a próxima iteração
            break
            
        last_overall_best_fitness = best_overall_fitness # Para ser usado na próxima iteração
    
    if stagnation_reached:
        print(f"Convergência atingida na geração {generation + 1} devido à estagnação.")
    else:
        print(f"Número máximo de gerações ({max_generations}) atingido")
        

    return best_overall_individual, best_overall_fitness, population_history, fitness_history, counter
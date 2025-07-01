from function import ObjectiveFunction
import numpy as np

def pso(obj_func: ObjectiveFunction, num_particles: int, max_iterations: int, bounds: tuple, 
        cognitive_coeff: float=1.5, social_coeff: float=1.5, min_w: float=0.2, max_w: float=0.9,
        tolerance: float=1e-6, patience: int=10):
    """Algoritmo de Otimização por Enxame de Partículas (PSO).
    Args:
        obj_func (ObjectiveFunction): Instância da função objetivo a ser minimizada.
        num_particles (int): Número de partículas no enxame.
        max_iterations (int): Número máximo de iterações.
        bounds (tuple): Limites inferior e superior para as partículas.
        cognitive_coeff (float): Coeficiente cognitivo (peso da experiência pessoal).
        social_coeff (float): Coeficiente social (peso da experiência do grupo).
        min_w (float): Peso mínimo da inércia. (maior que 0)
        max_w (float): Peso máximo da inércia (max 1)
        tolerance (float): Tolerância para considerar que não houve melhoria significativa.
        patience (int): Número de iterações sem melhoria antes de parar.
    Returns:
        tuple: Melhor posição encontrada, seu valor de fitness, histórico de posições e histórico de fitness.
    """

    # --- INICIALIZAÇÃO ---
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
    velocities = np.zeros_like(particles)
    fitness = obj_func(particles[:, 0], particles[:, 1])
    counter = {'multiplications': 0, 'divisions': 0} # Contador de operações
    personal_best_positions = particles.copy()
    personal_best_fitness = fitness.copy()
    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index].copy()
    pos_history = [particles.copy()] # Histórico de posições
    fitness_history = [fitness.copy()] # Histórico de fitness
    stagnation_counter = 0
    last_global_best_fitness = np.inf

    # --- ITERAÇÕES ---
    stagnation_reached = False
    for iteration in range(max_iterations): # Iterações do PSO
        r1 = np.random.rand(num_particles, 2) # Fator aleatório para componente cognitivo
        r2 = np.random.rand(num_particles, 2) # Fator aleatório para componente social

        # --- COMPONENTE COGNITIVO ---
        cognitive_component = cognitive_coeff * r1 * (personal_best_positions - particles)
        counter['multiplications'] += 2 * particles.size # Contabiliza as multiplicações

        # --- COMPONENTE SOCIAL ---
        social_component = social_coeff * r2 * (global_best_position - particles)
        counter['multiplications'] += 2 * particles.size # Contabiliza as multiplicações

        # --- PESO DA INÉRCIA DECRESCENTE ---
        inertia_weight = max_w - (max_w - min_w) * (iteration / max_iterations) # Peso da inércia decrescente
        counter['divisions'] += num_particles # Contabiliza as divisões
        inertia_weight = max(min(inertia_weight, max_w), min_w) # Garante que o peso da inércia esteja dentro dos limites
        
        # --- ATUALIZAÇÃO DAS VELOCIDADES ---
        velocities = (inertia_weight * velocities) + cognitive_component + social_component # Atualiza as velocidades
        counter['multiplications'] += particles.size # Contabiliza as multiplicações
        
        # --- ATUALIZAÇÃO DAS POSIÇÕES ---
        particles += velocities # Atualiza as posições das partículas adicionando as velocidades
        particles = np.clip(particles, bounds[0], bounds[1]) # Garante que as partículas permaneçam dentro dos limites
        
        # --- AVALIAÇÃO DA FUNÇÃO OBJETIVO ---
        fitness = obj_func(particles[:, 0], particles[:, 1]) # Avalia a função objetivo para as novas posições
        
        # --- ATUALIZAÇÃO DE MELHORES ---
        # Encontra o melhor da iteração atual
        current_iter_best_index = np.argmin(fitness)
        current_iter_best_fitness = fitness[current_iter_best_index]
        
        # Compara o melhor da iteração atual com o melhor global
        if current_iter_best_fitness < obj_func(global_best_position[0], global_best_position[1]):
            global_best_position = particles[current_iter_best_index].copy()
        
        # As partículas atualizam seu pbest com base na nova posição
        update_mask = fitness < personal_best_fitness # Apenas as partículas que melhoraram
        personal_best_positions[update_mask] = particles[update_mask] # Atualiza as melhores posições pessoais
        personal_best_fitness[update_mask] = fitness[update_mask] # Atualiza os melhores fitness pessoais 
        
        current_global_best_fitness = obj_func(global_best_position[0], global_best_position[1])
        improvement = last_global_best_fitness - current_global_best_fitness
        pos_history.append(particles.copy())
        fitness_history.append(fitness.copy())

        # --- DEBUG ---
        # print(f"Iteração {iteration + 1}: Melhor posição: ({global_best_position[0]:.4f}, {global_best_position[1]:.4f}), Z ótimo: {current_global_best_fitness:.2f}, Melhoria: {improvement:.6f}")
        # --- FIM DEBUG ---

        # --- VERIFICAÇÃO DE CONVERGÊNCIA ---
        if improvement > tolerance: # Se houve melhoria significativa
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= patience:
            stagnation_reached = True
            break
    
    if stagnation_reached:
        print(f"Convergência atingida na iteração {iteration + 1} devido à estagnação.")
    else:
        print(f"Número máximo de iterações ({max_iterations}) atingido pelo PSO.")

        last_global_best_fitness = current_global_best_fitness # Para ser usado na próxima iteração

    final_cost = obj_func(global_best_position[0], global_best_position[1])
    return global_best_position, final_cost, pos_history, fitness_history, counter
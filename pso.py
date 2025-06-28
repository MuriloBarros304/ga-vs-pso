from function import ObjectiveFunction
import numpy as np

def pso(obj_func: ObjectiveFunction, num_particles: int, max_iterations: int, bounds: tuple, cognitive_coeff: float=1.5,
        social_coeff: float=1.5, inertia_weight: float=0.5, tolerance: float=1e-6, patience: int=10):
    """Algoritmo de Otimização por Enxame de Partículas (PSO).
    Args:
        obj_func (ObjectiveFunction): Instância da função objetivo a ser minimizada.
        num_particles (int): Número de partículas no enxame.
        max_iterations (int): Número máximo de iterações.
        bounds (tuple): Limites inferior e superior para as partículas.
        cognitive_coeff (float): Coeficiente cognitivo (peso da experiência pessoal).
        social_coeff (float): Coeficiente social (peso da experiência do grupo).
        inertia_weight (float): Peso da inércia (influência da velocidade anterior).
        tolerance (float): Tolerância para considerar que não houve melhoria significativa.
        patience (int): Número de iterações sem melhoria antes de parar.
    Returns:
        tuple: Melhor posição encontrada, seu valor de fitness, histórico de posições e histórico de fitness.
    """

    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
    velocities = np.zeros_like(particles)
    fitness = obj_func(particles[:, 0], particles[:, 1])
    
    personal_best_positions = particles.copy()
    personal_best_fitness = fitness.copy()
    
    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index].copy()
    
    pos_history = [particles.copy()] # Histórico de posições
    fitness_history = [fitness.copy()] # Histórico de fitness
    
    stagnation_counter = 0
    last_global_best_fitness = np.inf

    for iteration in range(max_iterations): # Iterações do PSO
        r1 = np.random.rand(num_particles, 2) # Fator aleatório para componente cognitivo
        r2 = np.random.rand(num_particles, 2) # Fator aleatório para componente social

        cognitive_component = cognitive_coeff * r1 * (personal_best_positions - particles)
        social_component = social_coeff * r2 * (global_best_position - particles)
        
        velocities = (inertia_weight * velocities) + cognitive_component + social_component # Atualiza as velocidades
        
        particles += velocities # Atualiza as posições das partículas adicionando as velocidades
        particles = np.clip(particles, bounds[0], bounds[1]) # Garante que as partículas permaneçam dentro dos limites
        
        fitness = obj_func(particles[:, 0], particles[:, 1]) # Avalia a função objetivo para as novas posições
        
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
        # --- DEBUG ---
        # print(f"Iteração {iteration + 1}: Melhor posição: ({global_best_position[0]:.4f}, {global_best_position[1]:.4f}), Z ótimo: {current_global_best_fitness:.2f}, Melhoria: {improvement:.6f}")
        # --- FIM DEBUG ---
        if improvement > tolerance: # Se houve melhoria significativa
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= patience:
            print(f"Convergência atingida na iteração {iteration} devido à estagnação.")
            last_global_best_fitness = current_global_best_fitness
            break

        if len(pos_history) == max_iterations:
            print(f"Número máximo de iterações ({max_iterations}) atingido.")
            
        last_global_best_fitness = current_global_best_fitness
        pos_history.append(particles.copy())
        fitness_history.append(fitness.copy())

    final_cost = obj_func(global_best_position[0], global_best_position[1])
    return global_best_position, final_cost, pos_history, fitness_history
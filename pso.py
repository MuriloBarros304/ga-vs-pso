from function import objective_function_w4
import numpy as np

W_MAX = 0.9  # Peso máximo da inércia
W_MIN = 0.1  # Peso mínimo da inércia

def pso(num_particles, max_iterations, bounds, 
        cognitive_coeff=1.5, social_coeff=1.5, 
        inertia_weight=0.5, tolerance=1e-6, patience=10):

    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
    velocities = np.zeros_like(particles)
    fitness = objective_function_w4(particles[:, 0], particles[:, 1])
    
    # Controle de diversidade inicial
    worst_particle_index = np.argmax(fitness)
    swarm_mean_position = np.mean(particles, axis=0)
    particles[worst_particle_index] = swarm_mean_position
    fitness[worst_particle_index] = objective_function_w4(particles[worst_particle_index, 0], particles[worst_particle_index, 1])
    
    personal_best_positions = particles.copy()
    personal_best_fitness = fitness.copy()
    
    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index].copy()
    
    pos_history = []
    fitness_history = []
    
    stagnation_counter = 0
    last_global_best_fitness = np.inf

    for iteration in range(max_iterations):
        pos_history.append(particles.copy())
        fitness_history.append(fitness.copy())

        r1 = np.random.rand(num_particles, 2)
        r2 = np.random.rand(num_particles, 2)

        cognitive_component = cognitive_coeff * r1 * (personal_best_positions - particles)
        social_component = social_coeff * r2 * (global_best_position - particles)
        
        velocities = inertia_weight * velocities + cognitive_component + social_component
        
        particles += velocities
        particles = np.clip(particles, bounds[0], bounds[1])
        
        fitness = objective_function_w4(particles[:, 0], particles[:, 1])
        
        # Encontra o melhor da geração atual
        current_gen_best_index = np.argmin(fitness)
        current_gen_best_fitness = fitness[current_gen_best_index]
        
        # Compara o melhor da geração atual com o melhor GLOBAL histórico
        if current_gen_best_fitness < objective_function_w4(global_best_position[0], global_best_position[1]):
            global_best_position = particles[current_gen_best_index].copy()
        
        # As partículas atualizam seu pbest com base na nova posição
        update_mask = fitness < personal_best_fitness
        personal_best_positions[update_mask] = particles[update_mask]
        personal_best_fitness[update_mask] = fitness[update_mask]
        
        current_global_best_fitness = objective_function_w4(global_best_position[0], global_best_position[1])
        improvement = last_global_best_fitness - current_global_best_fitness
        if improvement > tolerance:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= patience:
            print(f"\nConvergência atingida na iteração {iteration + 1} devido à estagnação.")
            break
            
        last_global_best_fitness = current_global_best_fitness

    pos_history.append(particles.copy())
    fitness_history.append(fitness.copy())

    final_cost = objective_function_w4(global_best_position[0], global_best_position[1])
    return global_best_position, final_cost, pos_history, fitness_history
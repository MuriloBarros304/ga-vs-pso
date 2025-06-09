from function import objective_function_w4
import numpy as np

def pso(num_particles, num_iterations, bounds, 
        inertia_weight=0.5, cognitive_coeff=1.5, social_coeff=1.5, 
        tolerance=1e-6, patience=10):

    pos_history = []
    fitness_history = [] 
    
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
    velocities = np.random.uniform(-1, 1, (num_particles, 2))
    fitness = objective_function_w4(particles[:, 0], particles[:, 1])
    
    personal_best_positions = particles.copy()
    personal_best_fitness = fitness.copy()
    global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
    
    stagnation_counter = 0
    last_global_best_fitness = np.inf 

    for iteration in range(num_iterations):
        pos_history.append(particles.copy())
        fitness_history.append(fitness.copy())

        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                             social_coeff * r2 * (global_best_position - particles[i]))
            
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])
            fitness[i] = objective_function_w4(particles[i, 0], particles[i, 1])
            
            if fitness[i] < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness[i]
                personal_best_positions[i] = particles[i].copy()
        
        current_best_fitness = np.min(personal_best_fitness)
        global_best_fitness = objective_function_w4(global_best_position[0], global_best_position[1])

        if current_best_fitness < global_best_fitness:
            global_best_position = personal_best_positions[np.argmin(personal_best_fitness)].copy()
            global_best_fitness = current_best_fitness
            
        improvement = last_global_best_fitness - global_best_fitness
        if improvement > tolerance:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= patience:
            print(f"\nConvergência atingida na iteração {iteration + 1} devido à estagnação.")
            # Adiciona o último estado antes de parar
            pos_history.append(particles.copy())
            fitness_history.append(fitness.copy())
            break
            
        last_global_best_fitness = global_best_fitness

    final_cost = objective_function_w4(global_best_position[0], global_best_position[1])

    return global_best_position, final_cost, pos_history, fitness_history

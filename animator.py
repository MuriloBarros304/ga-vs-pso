# animator.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation(population_history, fitness_history, objective_function, bounds, filename="animation.mp4", title="Animação de Otimização", particle_color='blue', particle_label='Partículas'
):
    """
    Cria e salva uma animação do processo de otimização.

    Args:
        population_history (list): Lista de arrays 2D com as posições da população a cada iteração.
        fitness_history (list): Lista com o fitness da população a cada iteração.
        objective_function (callable): A função objetivo para plotar o fundo.
        bounds (tuple): Tupla com os limites ( (mins), (maxs) ).
        filename (str): Nome do arquivo de vídeo a ser salvo.
        title (str): Título base para a animação.
        particle_color (str): Cor para as partículas/indivíduos.
        particle_label (str): Legenda para as partículas/indivíduos.
    """
    actual_iterations = len(population_history)

    # Preparação do fundo do gráfico (contour plot)
    x_range = np.arange(bounds[0][0], bounds[1][0] + 1, 10)
    y_range = np.arange(bounds[0][1], bounds[1][1] + 1, 10)
    X, Y = np.meshgrid(x_range, y_range)
    Z_background = objective_function(X, Y)
    
    # Níveis de contorno estáticos para uma visualização consistente
    fitness_levels = np.linspace(np.min(Z_background), np.max(Z_background), 50)

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Função que desenha cada quadro da animação
    def animate(i):
        ax.clear()
        
        current_population = population_history[i]
        current_fitness = np.min(fitness_history[i])
        
        # Desenha o fundo
        ax.contourf(X, Y, Z_background, levels=fitness_levels, cmap='autumn', alpha=0.7, zorder=5)
        
        # Desenha as partículas/indivíduos
        ax.scatter(current_population[:, 0], current_population[:, 1], 
                   marker='o', color=particle_color, alpha=0.7, zorder=10, label=particle_label)

        # Formatação dos eixos e títulos
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f'{title} - Iteração {i+1}/{actual_iterations} | Melhor Z: {current_fitness:.2f}', fontsize=16)
        ax.legend(loc='upper right', fontsize=12)
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        ax.grid(True, linestyle='--', alpha=1)
        ax.set_aspect('equal', adjustable='box')

    # Cria e salva a animação
    anim = FuncAnimation(fig, animate, frames=actual_iterations, interval=150, blit=False)
    try:
        anim.save(filename, writer='ffmpeg', fps=1, dpi=120)
    except Exception as e:
        print(f"Erro ao salvar a animação: {e}")
        print("Verifique se o FFMpeg está instalado e acessível no PATH do sistema.")
    
    plt.close(fig) # Fecha a figura para liberar memória
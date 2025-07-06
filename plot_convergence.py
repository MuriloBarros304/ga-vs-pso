import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def parse_full_log_file(filepath: str) -> dict:
    """
    Lê um arquivo de log e agrupa a LISTA COMPLETA de fitness de cada 
    execução por número de iteração.

    Args:
        filepath (str): O caminho para o arquivo .txt.

    Returns:
        dict: Dicionário onde chaves são iterações e valores são listas de listas.
              Ex: {1: [[run1_fitnesses], [run2_fitnesses]], 2: [...]}
    """
    # Usamos defaultdict(list) para agrupar as execuções por iteração
    data_by_iteration = defaultdict(list)
    print(f"Lendo e processando o arquivo de log completo: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            full_log_content = f.read()

        # Separa o log em blocos de execução, usando o separador "---"
        # Filtra blocos vazios que podem surgir de linhas em branco
        execution_blocks = [block for block in re.split(r'-{10,}', full_log_content) if block.strip()]

        for block in execution_blocks:
            # Padrão para encontrar linhas no formato: "1, [ ... ]"
            # O re.DOTALL permite que o '.' capture também as quebras de linha
            matches = re.finditer(r'^\s*(\d+),\s*\[(.*?)\]', block, re.DOTALL | re.MULTILINE)
            for match in matches:
                iteration = int(match.group(1))
                list_str = match.group(2).replace('\n', ' ')
                
                try:
                    fitness_values = [float(num) for num in list_str.split() if num]
                    if fitness_values:
                        # Adiciona a lista completa de fitness para esta iteração/execução
                        data_by_iteration[iteration].append(fitness_values)
                except ValueError:
                    print(f"Aviso: Não foi possível processar a linha da iteração {iteration}.")

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{filepath}'.")
        return None
        
    print("Processamento do arquivo concluído.")
    return data_by_iteration


def generate_convergence_plot(data: dict, output_filename: str, algorithm_name: str):
    """
    Calcula e plota as métricas de convergência, incluindo a média e o desvio
    padrão de toda a população.
    """
    if not data:
        print(f"Nenhum dado para plotar para o {algorithm_name}.")
        return

    print(f"Para {algorithm_name}: Calculando métricas de convergência...")
    
    iterations = sorted(data.keys())
    
    # --- NOVOS CÁLCULOS ---
    # Média de toda a população a cada iteração
    avg_population_fitness = []
    # Desvio padrão de toda a população a cada iteração
    std_population_fitness = []
    # Melhor valor médio encontrado (mesma lógica de antes)
    avg_best_so_far = []
    
    # Para calcular o "melhor até agora", primeiro estruturamos os dados por execução
    num_runs = max(len(v) for v in data.values()) if data else 0
    max_iter = max(data.keys())
    
    # Matriz para os melhores de cada run/iteração
    best_per_run_matrix = np.full((num_runs, max_iter), np.nan)
    for i_iter, runs_for_iter in data.items():
        for i_run, run_fitness_list in enumerate(runs_for_iter):
            best_per_run_matrix[i_run, i_iter-1] = min(run_fitness_list)

    # Preenche os NaNs e calcula a curva de convergência média
    for i_run in range(num_runs):
        last_val = np.nan
        for i_iter in range(max_iter):
            if not np.isnan(best_per_run_matrix[i_run, i_iter]):
                last_val = best_per_run_matrix[i_run, i_iter]
            else:
                best_per_run_matrix[i_run, i_iter] = last_val

    best_so_far_matrix = np.minimum.accumulate(best_per_run_matrix, axis=1)
    avg_best_so_far = np.nanmean(best_so_far_matrix, axis=0)

    # Agora, calcula as estatísticas da população inteira
    for i in iterations:
        # Junta todas as listas de fitness de uma iteração em uma única lista gigante
        all_fitness_at_iter = [item for sublist in data[i] for item in sublist]
        avg_population_fitness.append(np.mean(all_fitness_at_iter))
        std_population_fitness.append(np.std(all_fitness_at_iter))

    print("Gerando o gráfico...")
    # --- PLOTAGEM ATUALIZADA ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Converte para arrays numpy
    avg_population_fitness = np.array(avg_population_fitness)
    std_population_fitness = np.array(std_population_fitness)

    # Plota a MÉDIA DE FITNESS DE TODA A POPULAÇÃO (linha tracejada)
    ax.plot(iterations, avg_population_fitness, color='deepskyblue', linestyle='--', label=f'Média da População ({algorithm_name})')
    
    # Plota o DESVIO PADRÃO em torno dessa média
    ax.fill_between(
        iterations, 
        avg_population_fitness - std_population_fitness, 
        avg_population_fitness + std_population_fitness, 
        color='deepskyblue', 
        alpha=0.2, 
        label='Desvio Padrão da População (±σ)'
    )
    
    # Plota a CURVA DE CONVERGÊNCIA REAL (melhor valor encontrado)
    ax.plot(iterations, avg_best_so_far, color='red', linewidth=2.5, label='Média do Melhor')
    
    # --- FORMATAÇÃO E TÍTULOS ---
    ax.set_title(f'Gráfico de Convergência - {algorithm_name}', fontsize=16, weight='bold')
    ax.set_xlabel('Número de Iterações/Gerações', fontsize=12)
    ax.set_ylabel('Valor da Função Objetivo (Z)', fontsize=12)
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo com sucesso como '{output_filename}'")

if __name__ == '__main__':
    # =================================================================
    log_filepath = 'pso.txt'
    # =================================================================

    data = parse_full_log_file(log_filepath)
    if data:
        generate_convergence_plot(data, 'imgs/grafico_convergencia_pso.png', "PSO")
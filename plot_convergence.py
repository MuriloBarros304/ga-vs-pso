import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def parse_log_file_multiline(filepath: str) -> dict:
    """
    Lê um arquivo de log onde os dados de uma iteração podem se estender por
    múltiplas linhas e agrupa o MELHOR fitness de cada execução por iteração.

    Args:
        filepath (str): O caminho para o arquivo .txt.

    Returns:
        dict: Dicionário onde chaves são iterações e valores são listas
              dos melhores fitness (mínimos) de cada execução para aquela iteração.
    """
    data_by_iteration = defaultdict(list)
    
    print(f"Lendo e processando o arquivo: {filepath}")
    try:
        with open(filepath, 'r') as f:
            in_record = False  # Flag para saber se estamos dentro de um bloco [...]
            current_iteration = None
            current_values_str = ""

            for line in f:
                # Se encontrarmos o início de um novo registro (ex: "11, [...")
                if re.match(r'^\s*\d+,\s*\[', line):
                    # Pega o número da iteração
                    iteration_str, data_str = line.split(',', 1)
                    current_iteration = int(iteration_str.strip())
                    
                    # Limpa e armazena o início da lista de números
                    current_values_str = data_str.strip().lstrip('[')
                    in_record = True
                
                # Se já estivermos dentro de um registro, continua lendo os números
                elif in_record:
                    current_values_str += " " + line.strip()

                # Se a linha atual contiver o final do registro "]"
                if ']' in line and in_record:
                    # Limpa o colchete final
                    current_values_str = current_values_str.rsplit(']', 1)[0]
                    
                    try:
                        # Converte a string de números em uma lista de floats
                        fitness_values = [float(num) for num in current_values_str.split() if num]
                        if fitness_values:
                            # Adiciona o melhor (mínimo) valor ao nosso dicionário
                            data_by_iteration[current_iteration].append(min(fitness_values))
                    except ValueError:
                        print(f"Aviso: Não foi possível processar os dados para a iteração {current_iteration}.")
                    
                    # Reseta o estado para a próxima iteração
                    in_record = False
                    current_values_str = ""

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{filepath}'.")
        return None
        
    print("Processamento do arquivo concluído.")
    return data_by_iteration


def generate_convergence_plot(data: dict, output_filename: str, algorithm_name: str):
    """
    Calcula e plota 3 métricas de convergência:
    1. A média do melhor fitness de cada geração.
    2. O desvio padrão em torno dessa média.
    3. A curva do melhor valor global encontrado ao longo do tempo.
    """
    if not data or not any(data.values()):
        print(f"Nenhum dado válido encontrado para plotar para o {algorithm_name}.")
        return

    print(f"Para {algorithm_name}: Calculando as métricas de convergência...")
    
    # --- LÓGICA DE CÁLCULO CORRIGIDA E COMPLETA ---

    # 1. Encontra o número máximo de iterações e o número de execuções
    max_iter = max(data.keys())
    # O número de execuções pode variar se algumas pararam antes. Pegamos o máximo.
    num_runs = max(len(v) for v in data.values()) if data else 0
    iterations = np.arange(1, max_iter + 1)
    
    print(f"  - Encontradas {num_runs} execuções com até {max_iter} iterações.")

    # 2. Cria uma matriz (runs x iterações) para facilitar os cálculos.
    #    Preenche com NaN se uma execução terminou mais cedo.
    all_runs_matrix = np.full((num_runs, max_iter), np.nan)
    for i_iter, fitness_list in data.items():
        # Garante que a lista de fitness tenha o mesmo tamanho que o num_runs
        # preenchendo com NaN se necessário
        padded_list = fitness_list + [np.nan] * (num_runs - len(fitness_list))
        all_runs_matrix[:, i_iter-1] = padded_list

    # 3. Métrica 1: Média do Melhor Fitness da Geração
    #    Calcula a média de cada coluna da matriz.
    mean_of_generation_bests = np.nanmean(all_runs_matrix, axis=0)

    # 4. Métrica 2: Desvio Padrão do Melhor Fitness da Geração
    #    Calcula o desvio padrão de cada coluna.
    std_of_generation_bests = np.nanstd(all_runs_matrix, axis=0)

    # 5. Métrica 3: Curva do Melhor Valor Encontrado
    #    Primeiro, calcula a curva de melhor valor para CADA execução (linha).
    best_so_far_matrix = np.minimum.accumulate(all_runs_matrix, axis=1, dtype=float)
    #    Depois, calcula a média dessas curvas.
    avg_of_best_so_far = np.nanmean(best_so_far_matrix, axis=0)


    print("Gerando o gráfico...")
    # --- PLOTAGEM COM AS TRÊS MÉTRICAS ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plota a MÉDIA DO MELHOR DE CADA GERAÇÃO (linha tracejada)
    ax.plot(iterations, mean_of_generation_bests, color='deepskyblue', linestyle='--', label=f'Média do Melhor Fitness da Geração ({algorithm_name})')
    
    # Plota o DESVIO PADRÃO em torno da média da geração
    ax.fill_between(
        iterations, 
        mean_of_generation_bests - std_of_generation_bests, 
        mean_of_generation_bests + std_of_generation_bests, 
        color='deepskyblue', 
        alpha=0.2, 
        label='Desvio Padrão (±1σ)'
    )
    
    # Plota a CURVA DE CONVERGÊNCIA REAL (média dos melhores valores acumulados)
    ax.plot(iterations, avg_of_best_so_far, color='red', linewidth=2.5, label='Melhor Valor Médio Encontrado')
    
    # --- FORMATAÇÃO E TÍTULOS ---
    ax.set_title(f'Gráfico de Convergência Média - {algorithm_name}', fontsize=16, weight='bold')
    ax.set_xlabel('Número de Iterações/Gerações', fontsize=12)
    ax.set_ylabel('Valor da Função Objetivo (Z Ótimo)', fontsize=12)
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    
        
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo com sucesso como '{output_filename}'")


if __name__ == '__main__':
    # =================================================================
    log_filepath = 'resultados_ga.txt' 
    # =================================================================

    # --- Processa e Plota para o GA ---
    data = parse_log_file_multiline(log_filepath)
    if data:
        generate_convergence_plot(data, 'imgs/grafico_convergencia_ga.png', "GA")
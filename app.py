import streamlit as st
import chat
import os
import base64

st.session_state.setdefault("chat_history", [])

st.set_page_config(
    page_title="Página do Projeto GA e PSO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Projeto de Otimização com GA e PSO")
tab_desc, tab_chat = st.tabs(["Descrição do Projeto", "Chatbot"])

with tab_desc:
    st.header("Descrição do Projeto")
    st.markdown(
        """
        Este projeto implementa algoritmos de otimização baseados em Enxame de Partículas (PSO) e Algoritmos Genéticos (GA) para resolver funções matemáticas complexas, como a função Schwefel-Rosenbrock e a função Rastrigin. O objetivo é comparar o desempenho desses algoritmos na busca por soluções ótimas, bem como gerar animações que ilustram o processo de otimização ao longo das iterações.

        ### Funcionalidades Principais:
        - Implementação dos algoritmos PSO e GA com parâmetros configuráveis.
        - Geração de animações que mostram a evolução das soluções ao longo do tempo.
        - Comparação de desempenho entre os dois algoritmos em diferentes funções objetivo.

        ### Tecnologias Utilizadas:
        - Python para implementação dos algoritmos.
        - Bibliotecas de visualização para criação das animações.
        - Streamlit para a interface interativa do chatbot.

        ### Como Usar o Chatbot:
        - Na aba "Chatbot", você pode fazer perguntas relacionadas ao projeto, aos algoritmos implementados, aos resultados obtidos e às animações geradas. O chatbot utiliza um modelo de linguagem avançado para fornecer respostas detalhadas e relevantes, além de exibir gráficos relacionados às suas perguntas.
        ----------------------------------------------------------------------
        """
    )
    st.markdown(
        r"""
            ## Algoritmo Genético (GA)

            A metodologia de cálculo iterativo do Algoritmo Genético (GA) implementada neste projeto segue o princípio da evolução biológica para refinar progressivamente uma população de soluções candidatas. Ao contrário de métodos determinísticos, o GA utiliza operações estocásticas (aleatórias) guiadas pela aptidão (fitness) dos indivíduos.

            Abaixo, detalha-se o ciclo evolutivo executado a cada geração:

            ### 1. Inicialização e Avaliação
            O processo começa com a criação de uma população inicial aleatória dentro dos limites ($bounds$) definidos. Imediatamente, a Função Objetivo é chamada para calcular o fitness de cada indivíduo. Este valor representa a qualidade da solução (quanto menor o valor, melhor a solução para problemas de minimização).

            ### 2. Elitismo (Preservação dos Melhores)
            Para garantir que a qualidade da solução nunca regrida, aplica-se o conceito de elitismo.
            * Os indivíduos são ordenados com base em seu fitness.
            * Os $N$ melhores indivíduos (definido por `elitism_size`) são copiados integralmente e sem alterações para a próxima geração. Isso assegura a convergência monotônica do algoritmo.

            ### 3. Seleção de Pais (Método da Roleta Baseado em Rank)
            Os indivíduos que formarão a próxima geração (além da elite) são selecionados probabilisticamente:
            * **Rankeamento:** Em vez de usar o valor bruto do fitness (que pode causar convergência prematura se houver "super indivíduos"), os indivíduos são classificados por ordem de mérito.
            * **Probabilidade:** A probabilidade de seleção é proporcional ao rank. O melhor indivíduo tem a maior chance de ser pai, mas os piores ainda têm uma chance pequena, mantendo a diversidade genética.
            * **Sorteio:** Utiliza-se a função `np.random.choice` para selecionar os índices dos pais com base nessas probabilidades calculadas.

            ### 4. Crossover (Recombinação Genética BLX-α)
            Os pais selecionados são agrupados em pares para a reprodução. O método utilizado é o Blend Crossover (BLX-α), ideal para espaços de busca contínuos:
            * Se o sorteio de crossover for bem-sucedido (baseado em `crossover_rate`), dois filhos são gerados.
            * O gene do filho não é apenas uma cópia do pai ou da mãe, mas um valor sorteado dentro de um intervalo expandido entre os genes dos pais.
            * Fórmula: $$Intervalo = [min(P_1, P_2) - \alpha \cdot d, \quad max(P_1, P_2) + \alpha \cdot d]$$, onde $d$ é a distância entre os pais e $\alpha$ é o fator de expansão (0.5).
            * Isso permite que o algoritmo explore novas regiões próximas aos pais, inovando além do que já existia na população.

            ### 5. Mutação (Perturbação Gaussiana)
            Após o crossover, os novos indivíduos (exceto a elite) passam pelo operador de mutação para introduzir variabilidade e evitar mínimos locais.
            * Uma máscara booleana define quais genes sofrerão mutação com base na `mutation_rate`.
            * Aos genes selecionados, adiciona-se um ruído aleatório proveniente de uma distribuição Normal (Gaussiana) com média 0 e desvio padrão definido por `mutation_strength`.
            * O resultado é "clipado" para garantir que o indivíduo permaneça dentro dos limites do espaço de busca.

            ### 6. Critérios de Parada
            O ciclo se repete até que uma das condições seja atendida:
            * Máximo de Gerações: O limite de iterações (`max_generations`) é atingido.
            * Estagnação (Paciência): Se o melhor fitness global não apresentar uma melhoria superior à `tolerance` (ex: $1e^{-6}$) por um número consecutivo de gerações (`patience`), o algoritmo assume convergência e encerra a execução antecipadamente para economizar recursos.
            -----------------------------------------------------------------------
        """
    )
    st.markdown(
        r"""
            ## Enxame de Partículas (PSO)

            A metodologia do PSO implementada neste projeto baseia-se na simulação do comportamento social de bandos de pássaros ou cardumes de peixes. Ao contrário do GA, que altera a estrutura genética dos indivíduos, o PSO altera a velocidade e posição das partículas no espaço de busca, guiado por uma combinação de inércia, memória pessoal e influência social.

            Abaixo, detalha-se o ciclo iterativo executado pelo algoritmo:

            ### 1. Inicialização
            O enxame é inicializado com partículas distribuídas aleatoriamente (distribuição uniforme) dentro dos limites ($bounds$).
            * Posição ($x$): Coordenada atual da partícula.
            * Velocidade ($v$): Inicializada como zero.
            * Memória ($pbest$ e $gbest$): Inicialmente, a melhor posição pessoal ($pbest$) é a posição atual, e a melhor global ($gbest$) é a melhor entre todas da população inicial.

            ### 2. Cálculo da Inércia Dinâmica
            O algoritmo utiliza um peso de inércia decrescente ($w$) para equilibrar a exploração (busca global) e a explotação (busca local/refinamento).
            * No início, $w$ é alto (`max_w`), permitindo que as partículas se movam rapidamente e cubram grandes áreas.
            * Com o passar das iterações, $w$ decai linearmente até `min_w`. Isso faz com que as partículas "freiem" progressivamente para refinar a busca ao redor do ótimo.
            * **Fórmula:** $w(t) = w_{max} - (w_{max} - w_{min}) \cdot \frac{t}{T_{max}}$.

            ### 3. Atualização da Velocidade
            A nova velocidade de cada partícula é calculada somando três vetores vetoriais:
            1.  Inércia: A tendência da partícula de manter seu movimento anterior ($w \cdot v$).
            2.  Componente Cognitivo (Memória Pessoal): A atração da partícula de volta para sua melhor posição histórica ($pbest$). Calculado como $$c_1 \cdot r_1 \cdot (pbest - x)$$.
            3.  Componente Social (Influência do Grupo): A atração da partícula em direção à melhor posição encontrada por todo o enxame ($gbest$). Calculado como $c_2 \cdot r_2 \cdot (gbest - x)$.

            Onde $r_1$ e $r_2$ são vetores aleatórios $[0, 1]$ que garantem a estocasticidade do movimento.

            ### 4. Atualização da Posição e Limites
            Com a nova velocidade calculada, a posição é atualizada fisicamente:
            * $x_{novo} = x_{atual} + v_{novo}$
            * Restrição de Fronteira: Aplica-se uma função de "clip" (`np.clip`). Se uma partícula tentar sair do espaço de busca definido (ex: > 500), ela é forçada a permanecer no limite da borda, garantindo a viabilidade da solução.

            ### 5. Avaliação e Atualização de Memórias
            Após o movimento, a Função Objetivo é calculada para as novas posições:
            * Atualização do PBest: Se o novo fitness da partícula for melhor que seu histórico pessoal, o $pbest$ é atualizado.
            * Atualização do GBest: Se a melhor partícula desta iteração for superior ao líder global histórico, o $gbest$ é atualizado. O algoritmo utiliza uma topologia Global (todas as partículas enxergam o mesmo líder).

            ### 6. Critérios de Parada
            O loop de movimento continua até:
            * O número máximo de iterações (`max_iterations`) ser atingido.
            * Ocorrer estagnação: Se a melhoria do líder global for insignificante (menor que `tolerance`) por um número determinado de iterações consecutivas (`patience`), o enxame é considerado convergido.
        """
    )

with tab_chat:
    st.header("Chatbot do Projeto")
    st.markdown("Faça perguntas sobre o projeto, as funções e os resultados")


    for interaction in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(interaction["question"])
            
        with st.chat_message("model"):
            st.markdown(interaction["answer"])
            
            if interaction["images"]:
                for img_path in interaction["images"]:
                    if os.path.exists(img_path):
                        # Detecta se é vídeo ou imagem para exibir corretamente
                        _, ext = os.path.splitext(img_path)
                        if ext.lower() == '.mp4':
                            st.video(img_path)
                        elif ext.lower() == '.gif':
                            # Lê o GIF e converte para base64 para garantir a animação
                            try:
                                with open(img_path, "rb") as f:
                                    contents = f.read()
                                    data_url = base64.b64encode(contents).decode("utf-8")
                                
                                # Renderiza via Markdown/HTML
                                st.markdown(
                                    f'<img src="data:image/gif;base64,{data_url}" width="800" alt="gif animado">',
                                    unsafe_allow_html=True,
                                )
                            except Exception as e:
                                st.error(f"Erro ao carregar GIF: {e}")
                        
                        # Imagens estáticas normais (png, jpg)
                        else:
                            st.image(img_path, caption=f"Imagem: {img_path}", width=600)
                    else:
                        st.warning(f"Arquivo {img_path} não encontrado no servidor.")

    with st.form(key="chat_form", clear_on_submit=True):
        user_prompt = st.text_input("Faça sua pergunta:", placeholder="Ex: Qual algoritmo teve o melhor desempenho?", key="chat_input")
        submit_button = st.form_submit_button("Enviar")

    if submit_button and user_prompt:
        with st.spinner("Analisando sua pergunta e buscando conteúdo..."):
            try:
                api_text, api_images = chat.run_ai(user_prompt)
                
                st.session_state.chat_history.append({
                    "question": user_prompt,
                    "answer": api_text,
                    "images": api_images
                })
                
            except Exception as e:
                error_message = f"Ocorreu um erro ao processar sua pergunta: {e}"
                st.session_state.chat_history.append({
                    "question": user_prompt,
                    "answer": error_message,
                    "images": []
                })
        
        st.rerun()
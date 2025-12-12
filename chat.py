import google.genai as genai  # type: ignore
import streamlit as st
import os
import re

IMAGE_MAP = {
}

def load_context():
    """Carrega o arquivo de texto de contextualização."""
    try:
        with open("context.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Arquivo 'contexto_projeto.txt' não encontrado.")
        return "" # Retorna um contexto vazio se o arquivo não for encontrado

def find_images_in_response(response_text):
    """
    Verifica o texto de resposta por palavras-chave e retorna
    os caminhos das imagens correspondentes.
    Suporta mapeamento 1-para-1 ou 1-para-muitos (listas).
    """
    images_to_show = set()
    text_lower = response_text.lower()
    
    for keyword, path_or_list in IMAGE_MAP.items():
        # Verifica se a palavra-chave está no texto
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower, re.IGNORECASE):
            
            # Se o valor for uma lista (caso do "mapeamento"), adiciona todos os itens
            if isinstance(path_or_list, list):
                for path in path_or_list:
                    if os.path.exists(path):
                        images_to_show.add(path)
            # Se for uma string única, adiciona direto
            else:
                if os.path.exists(path_or_list):
                    images_to_show.add(path_or_list)
            
    # Retorna como lista para o Streamlit renderizar
    return list(images_to_show)

def run_ai(user_prompt):
    """
    Executa a consulta à API do Gemini, combinando o contexto e o prompt do usuário.
    Retorna o texto da resposta e uma lista de imagens para exibir.
    """
    client = genai.Client()  # type: ignore
    context_text = load_context()
    if not context_text:
        return "Erro: Não foi possível carregar o contexto do projeto.", []

    full_prompt = f"""
    {context_text}
    
    ---
    
    ENTRADA DO USUÁRIO:
    {user_prompt}
    
    ---
    
    Com base estritamente no contexto fornecido, responda à pergunta do usuário.
    Se a resposta exigir a visualização de um gráfico, mencione palavras-chave
    como 'comparação', 'dispersão', 'importância', ou 'predição diária'
    em sua resposta para que a imagem correta possa ser exibida.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=full_prompt
        )
        response_text = response.text
    except Exception as e:
        return f"Erro ao gerar resposta da IA: {e}", []

    images = find_images_in_response(response_text)
    
    return response_text, images
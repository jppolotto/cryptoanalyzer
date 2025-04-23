import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime, timedelta
import time

def render_crypto_data_page():
    """Renderiza a página de análise de dados do CryptoDataDownload."""
    st.title("📊 Análise de Dados CryptoDataDownload")
    
    # Layout em duas colunas para a parte superior
    col1, col2 = st.columns([3, 1])
    
    # Coluna principal para seleção de dados
    with col1:
        st.markdown("""
        Esta página permite analisar e visualizar os dados baixados do CryptoDataDownload.
        Selecione uma categoria de dados e um arquivo específico para visualizar os resultados.
        """)
    
    # Coluna para configuração e atualização
    with col2:
        st.markdown("### Configuração")
        data_dir = st.text_input("Diretório dos Dados", value="data")
        if st.button("Atualizar Lista de Arquivos"):
            st.success("Lista de arquivos atualizada!")
    
    # Obter lista de arquivos disponíveis
    try:
        files = get_available_files(data_dir)
    except Exception as e:
        st.error(f"Erro ao acessar diretório de dados: {str(e)}")
        return
    
    # Verificar se temos arquivos para exibir
    if not any(files.values()):
        st.warning("Nenhum arquivo de dados disponível. Baixe os dados primeiro.")
        
        # Exibir instruções de uso
        st.subheader("Como usar")
        st.markdown("""
        Para analisar dados do CryptoDataDownload, siga estes passos:
        
        1. Baixe os dados do CryptoDataDownload
        2. Coloque os arquivos no diretório especificado
        3. Clique no botão "Atualizar Lista de Arquivos"
        4. Selecione uma categoria e um arquivo para visualizar os dados
        """)
        return
    
    # Exibir categorias e arquivos em abas
    st.subheader("Explorar Dados")
    
    tabs = st.tabs([
        "Market Breadth", 
        "OHLC", 
        "Gov. Data", 
        "Risk", 
        "Summary", 
        "Technicals",
        "Outros"
    ])
    
    # Mapeamento de categorias para nomes de exibição
    category_map = {
        "Market Breadth": "breadth",
        "OHLC": "ohlc",
        "Gov. Data": "gov",
        "Risk": "risk",
        "Summary": "summary",
        "Technicals": "technicals",
        "Outros": "other"
    }
    
    # Para cada aba (categoria)
    for i, tab in enumerate(tabs):
        with tab:
            category_name = list(category_map.keys())[i]
            category_key = category_map[category_name]
            
            if not files[category_key]:
                st.info(f"Nenhum arquivo disponível na categoria {category_name}.")
                continue
            
            # Ordenar arquivos por data de modificação (mais recentes primeiro)
            sorted_files = sorted(files[category_key], key=lambda x: x['modified'], reverse=True)
            
            # Opções para o selectbox
            file_options = [f"{f['name']} ({f['modified']})" for f in sorted_files]
            selected_file_idx = st.selectbox(
                f"Selecione um arquivo ({len(sorted_files)} disponíveis):", 
                range(len(file_options)),
                format_func=lambda x: file_options[x],
                key=f"file_{category_key}"
            )
            
            selected_file = sorted_files[selected_file_idx]
            
            # Mostrar informações do arquivo
            st.markdown(f"**Arquivo:** {selected_file['name']}")
            st.markdown(f"**Caminho:** {selected_file['path']}")
            st.markdown(f"**Tamanho:** {format_file_size(selected_file['size'])}")
            st.markdown(f"**Última modificação:** {selected_file['modified']}")
            
            # Carregar e exibir dados
            try:
                df = load_data_file(selected_file['path'])
                
                if df is None:
                    st.error("Erro ao carregar o arquivo.")
                    continue
                
                if df.empty:
                    st.warning("O arquivo está vazio ou não contém dados válidos.")
                    continue
                    
                # Exibir primeiras linhas dos dados
                with st.expander("Visualizar dados brutos", expanded=False):
                    st.dataframe(df)
                    
                    # Opção para download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=selected_file['name'],
                        mime='text/csv',
                    )
                
                # Visualização de dados específica para cada categoria
                visualize_data(df, category_key, selected_file['name'])
                
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")

def get_available_files(data_dir):
    """Retorna um dicionário com os arquivos disponíveis por categoria."""
    files = {
        "breadth": [],
        "ohlc": [],
        "gov": [],
        "risk": [],
        "summary": [],
        "technicals": [],
        "other": []
    }
    
    if not os.path.exists(data_dir):
        return files
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            file_info = {
                'name': filename,
                'path': file_path,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Categorizar o arquivo
            category = categorize_file(filename)
            if category in files:
                files[category].append(file_info)
            else:
                files['other'].append(file_info)
    
    return files

def categorize_file(filename):
    """Categoriza um arquivo baseado em seu nome."""
    filename_lower = filename.lower()
    
    if 'breadth' in filename_lower:
        return 'breadth'
    elif 'ohlc' in filename_lower:
        return 'ohlc'
    elif 'gov' in filename_lower:
        return 'gov'
    elif 'risk' in filename_lower:
        return 'risk'
    elif 'summary' in filename_lower:
        return 'summary'
    elif 'technicals' in filename_lower:
        return 'technicals'
    else:
        return 'other'

def load_data_file(file_path):
    """Carrega um arquivo de dados."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def format_file_size(size_bytes):
    """Formata o tamanho do arquivo em unidades legíveis."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def visualize_data(df, category, filename):
    """
    Cria visualizações específicas para cada categoria de dados.
    
    Args:
        df: DataFrame com os dados
        category: Categoria dos dados
        filename: Nome do arquivo
    """
    st.subheader("Visualização de Dados")
    
    # Converter colunas de data/timestamp, se existirem
    date_columns = ['timestamp', 'date', 'time']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                st.warning(f"Não foi possível converter a coluna {col} para datetime: {e}")
    
    # Detectar colunas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    # Criar visualizações específicas por categoria
    if category == "breadth":
        visualize_market_breadth(df, numeric_cols, date_cols, filename)
    elif category == "ohlc":
        visualize_ohlc(df, numeric_cols, date_cols, filename)
    elif category == "gov":
        visualize_gov_data(df, numeric_cols, date_cols, filename)
    elif category == "risk":
        visualize_risk_data(df, numeric_cols, date_cols, filename)
    elif category == "summary":
        visualize_summary_data(df, numeric_cols, date_cols, filename)
    elif category == "technicals":
        visualize_technical_data(df, numeric_cols, date_cols, filename)
    else:
        visualize_generic_data(df, numeric_cols, date_cols, filename)
    
    # Mostrar estatísticas das colunas
    with st.expander("Estatísticas das Colunas", expanded=False):
        if numeric_cols:
            stats = df[numeric_cols].describe().transpose()
            st.dataframe(stats)
        else:
            st.info("Não há colunas numéricas para calcular estatísticas.")
        
        # Mostrar informações sobre dados ausentes
        st.subheader("Dados Ausentes")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Valores Ausentes': missing_data,
            'Porcentagem (%)': missing_pct
        })
        missing_df = missing_df[missing_df['Valores Ausentes'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.info("Não há dados ausentes no arquivo.")

def visualize_market_breadth(df, numeric_cols, date_cols, filename):
    """Visualização para dados de Market Breadth."""
    if date_cols:
        x_col = date_cols[0]
        y_cols = [col for col in numeric_cols if col not in date_cols]
        
        if y_cols:
            selected_cols = st.multiselect("Selecione colunas para visualizar:", y_cols, default=y_cols[:3])
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], 
                        y=df[col],
                        mode='lines+markers',
                        name=col
                    ))
                
                fig.update_layout(
                    title=f"Métricas de Amplitude do Mercado - {filename}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna para visualizar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas para visualização.")
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

def visualize_ohlc(df, numeric_cols, date_cols, filename):
    """Visualização para dados OHLC."""
    if date_cols:
        x_col = date_cols[0]
        
        # Verificar se temos colunas OHLC
        ohlc_cols = {
            'open': [col for col in df.columns if 'open' in col.lower()],
            'high': [col for col in df.columns if 'high' in col.lower()],
            'low': [col for col in df.columns if 'low' in col.lower()],
            'close': [col for col in df.columns if 'close' in col.lower()]
        }
        
        if all(ohlc_cols.values()):
            # Criar gráfico de velas
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df[x_col],
                open=df[ohlc_cols['open'][0]],
                high=df[ohlc_cols['high'][0]],
                low=df[ohlc_cols['low'][0]],
                close=df[ohlc_cols['close'][0]],
                name="OHLC"
            ))
            
            fig.update_layout(
                title=f"Gráfico OHLC - {filename}",
                xaxis_title=x_col,
                yaxis_title="Preço",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Se não tiver OHLC, mostrar gráfico de linhas
            visualize_generic_data(df, numeric_cols, date_cols, filename)
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

def visualize_gov_data(df, numeric_cols, date_cols, filename):
    """Visualização para dados governamentais."""
    if date_cols:
        x_col = date_cols[0]
        y_cols = [col for col in numeric_cols if col not in date_cols]
        
        if y_cols:
            selected_cols = st.multiselect("Selecione colunas para visualizar:", y_cols, default=y_cols[:3])
            
            if selected_cols:
                chart_type = st.radio(
                    "Tipo de gráfico:",
                    ["Linhas", "Barras", "Área"],
                    horizontal=True
                )
                
                fig = go.Figure()
                
                for col in selected_cols:
                    if chart_type == "Linhas":
                        fig.add_trace(go.Scatter(
                            x=df[x_col], 
                            y=df[col],
                            mode='lines+markers',
                            name=col
                        ))
                    elif chart_type == "Barras":
                        fig.add_trace(go.Bar(
                            x=df[x_col], 
                            y=df[col],
                            name=col
                        ))
                    elif chart_type == "Área":
                        fig.add_trace(go.Scatter(
                            x=df[x_col], 
                            y=df[col],
                            mode='lines',
                            fill='tozeroy',
                            name=col
                        ))
                
                fig.update_layout(
                    title=f"Dados Governamentais - {filename}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna para visualizar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas para visualização.")
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

def visualize_risk_data(df, numeric_cols, date_cols, filename):
    """Visualização para dados de risco."""
    if date_cols:
        x_col = date_cols[0]
        y_cols = [col for col in numeric_cols if col not in date_cols]
        
        if y_cols:
            selected_cols = st.multiselect("Selecione colunas para visualizar:", y_cols, default=y_cols[:3])
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], 
                        y=df[col],
                        mode='lines+markers',
                        name=col
                    ))
                
                fig.update_layout(
                    title=f"Métricas de Risco - {filename}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna para visualizar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas para visualização.")
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

def visualize_summary_data(df, numeric_cols, date_cols, filename):
    """Visualização para dados resumidos."""
    if date_cols:
        x_col = date_cols[0]
        y_cols = [col for col in numeric_cols if col not in date_cols]
        
        if y_cols:
            selected_cols = st.multiselect("Selecione colunas para visualizar:", y_cols, default=y_cols[:3])
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], 
                        y=df[col],
                        mode='lines+markers',
                        name=col
                    ))
                
                fig.update_layout(
                    title=f"Dados Resumidos - {filename}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna para visualizar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas para visualização.")
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

def visualize_technical_data(df, numeric_cols, date_cols, filename):
    """Visualização para dados técnicos."""
    if date_cols:
        x_col = date_cols[0]
        y_cols = [col for col in numeric_cols if col not in date_cols]
        
        if y_cols:
            selected_cols = st.multiselect("Selecione colunas para visualizar:", y_cols, default=y_cols[:3])
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], 
                        y=df[col],
                        mode='lines+markers',
                        name=col
                    ))
                
                fig.update_layout(
                    title=f"Indicadores Técnicos - {filename}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna para visualizar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas para visualização.")
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

def visualize_generic_data(df, numeric_cols, date_cols, filename):
    """Visualização genérica para qualquer tipo de dados."""
    if date_cols:
        x_col = date_cols[0]
        y_cols = [col for col in numeric_cols if col not in date_cols]
        
        if y_cols:
            selected_cols = st.multiselect("Selecione colunas para visualizar:", y_cols, default=y_cols[:3])
            
            if selected_cols:
                chart_type = st.radio(
                    "Tipo de gráfico:",
                    ["Linhas", "Barras", "Área"],
                    horizontal=True
                )
                
                fig = go.Figure()
                
                for col in selected_cols:
                    if chart_type == "Linhas":
                        fig.add_trace(go.Scatter(
                            x=df[x_col], 
                            y=df[col],
                            mode='lines+markers',
                            name=col
                        ))
                    elif chart_type == "Barras":
                        fig.add_trace(go.Bar(
                            x=df[x_col], 
                            y=df[col],
                            name=col
                        ))
                    elif chart_type == "Área":
                        fig.add_trace(go.Scatter(
                            x=df[x_col], 
                            y=df[col],
                            mode='lines',
                            fill='tozeroy',
                            name=col
                        ))
                
                fig.update_layout(
                    title=f"Dados - {filename}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna para visualizar.")
        else:
            st.warning("Não foi possível identificar colunas numéricas para visualização.")
    else:
        st.warning("Não foi possível identificar uma coluna de data/hora para visualização temporal.")

if __name__ == "__main__":
    render_crypto_data_page() 
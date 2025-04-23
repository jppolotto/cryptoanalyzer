"""
Módulo para análise de dados criptográficos do CryptoDataDownload.
Versão aprimorada com análises mais detalhadas e explicações dos resultados.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import json
import io
import re
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from scipy.signal import argrelextrema
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator


class CryptoDataAnalyzer:
    def __init__(self, fetcher):
        """
        Inicializa o analisador de dados.
        
        Args:
            fetcher: Instância de CryptoAPIFetcher
        """
        self.fetcher = fetcher
        # Paletas de cores para gráficos
        self.color_palette = {
            'bullish': 'green',
            'bearish': 'red',
            'neutral': 'gray',
            'primary': 'blue',
            'secondary': 'orange',
            'tertiary': 'purple',
            'background': 'rgba(242, 242, 242, 0.8)',
            'grid': 'rgba(189, 189, 189, 0.5)'
        }
    
    def _process_api_data(self, data):
        """
        Processa dados da API em um DataFrame.
        
        Args:
            data: Dados retornados pela API
        
        Returns:
            DataFrame processado
        """
        df = pd.DataFrame() # Initialize df
        try:
            if isinstance(data, pd.DataFrame):
                # Check if DataFrame has only a 'result' column - this is our specific case
                if data.shape[1] == 1 and 'result' in data.columns:
                    parsed_records = []
                    for _, row in data.iterrows():
                        result_str = row['result']
                        if isinstance(result_str, str):
                            if result_str.startswith('"') and result_str.endswith('"'):
                                result_str = result_str[1:-1]
                            result_str = result_str.replace("'", '"')
                            try:
                                record = json.loads(result_str)
                                parsed_records.append(record)
                            except json.JSONDecodeError as e:
                                try:
                                    pattern = r'{\s*([^{}]+)\s*}'
                                    if re.match(pattern, result_str):
                                        record = {}
                                        pairs = re.findall(r'([^:,]+):\s*([^,]+)', result_str)
                                        for key, value in pairs:
                                            key = key.strip().strip('"')
                                            value = value.strip().strip('"')
                                            try:
                                                if value.lower() == 'true': value = True
                                                elif value.lower() == 'false': value = False
                                                elif value.lower() == 'null' or value.lower() == 'none': value = None
                                                elif '.' in value and value.replace('.', '').isdigit(): value = float(value)
                                                elif value.isdigit(): value = int(value)
                                            except: pass
                                            record[key] = value
                                        if record: parsed_records.append(record)
                                except: 
                                    st.warning(f"Não foi possível processar o registro: {result_str[:100]}... Erro: {str(e)}")
                                    continue
                    if parsed_records:
                        df = pd.DataFrame(parsed_records)
                    else:
                        # If parsing failed for all rows, return the original (or empty)
                        df = data
                else:
                     df = data # Use the original DataFrame if it doesn't match the 'result' case

            elif isinstance(data, str):
                try:
                    # Try direct JSON
                    json_data = json.loads(data)
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict):
                        if 'result' in json_data:
                            df = pd.DataFrame(json_data['result'])
                        else:
                            df = pd.DataFrame([json_data])
                except json.JSONDecodeError:
                    # Try line-separated JSON
                    lines = [line.strip() for line in data.split('\\n') if line.strip()]
                    if lines:
                        try:
                            json_list = [json.loads(line) for line in lines]
                            df = pd.DataFrame(json_list)
                        except json.JSONDecodeError:
                             # Try CSV
                            try: df = pd.read_csv(io.StringIO(data))
                            except: pass # Keep df empty if all parsing fails
                except Exception as e:
                     st.error(f"Erro ao processar string: {str(e)}")
            elif isinstance(data, dict):
                if 'result' in data:
                    result = data['result']
                    if isinstance(result, list): df = pd.DataFrame(result)
                    elif isinstance(result, dict): df = pd.DataFrame([result])
                    else: df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)

            # Ensure df is a DataFrame before proceeding
            if not isinstance(df, pd.DataFrame):
                 df = pd.DataFrame() # Return empty DataFrame if conversion failed
            
            # Normalize columns
            if not df.empty:
                df = self._normalize_column_names(df)
                # Remove duplicate columns, keeping the first occurrence
                df = df.loc[:, ~df.columns.duplicated()]
            
            return df
            
        except Exception as e:
            st.error(f"Erro fatal ao processar dados: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return pd.DataFrame() # Return empty DataFrame on fatal error

    def _normalize_column_names(self, df):
        """
        Normaliza os nomes das colunas do DataFrame para um formato consistente.
        
        Args:
            df: DataFrame a ser normalizado
        
        Returns:
            DataFrame com colunas normalizadas
        """
        if not isinstance(df, pd.DataFrame):
            return df
            
        rename_map = {
            # Colunas de timestamp (Geral)
            'date': 'timestamp',
            'time': 'timestamp',
            'Date': 'timestamp',
            'Time': 'timestamp',
            'Report_Date_as_MM_DD_YYYY': 'timestamp', # COT
            'worstDate': 'timestamp', # VaR
            'Unix': 'unix_timestamp',
            
            # Colunas de preço/volume
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            
            # Colunas de highs/lows
            'new_52week_highs': 'new_highs',
            'new_52week_lows': 'new_lows',
            
            # Colunas de médias móveis
            'percent_above_ma50': 'above_ma50',
            'percent_above_ma100': 'above_ma100',
            'percent_above_ma200': 'above_ma200',
            
            # Colunas de funding
            'funding_rate': 'funding',
            'fundingRate': 'funding',
            'last_funding_rate': 'funding',
            
            # Colunas de opções
            'strike_price': 'strike',
            'calls': 'calls_oi',
            'puts': 'puts_oi',
            'total_calls_oi': 'total_calls_oi',
            'total_puts_oi': 'total_puts_oi',
            
            # Colunas COT
            'long_positions': 'longs', # Nome genérico
            'short_positions': 'shorts', # Nome genérico
            'NonComm_Positions_Long_All': 'longs', # Não Comercial (se existir)
            'NonComm_Positions_Short_All': 'shorts', # Não Comercial (se existir)
            'Lev_Money_Positions_Long_All': 'longs', # << Fundos Alavancados (Prioridade)
            'Lev_Money_Positions_Short_All': 'shorts', # << Fundos Alavancados (Prioridade)
            'Open_Interest_All': 'open_interest',

            # Colunas VaR
            '99% VaR': 'var_99',
            '95% VaR': 'var_95',
            '90% VaR': 'var_90',
            'worstLoss': 'worst_loss',

            # Colunas de Symbol/Nome
            'Symbol': 'symbol',
            'symbol': 'symbol',
            'Market_and_Exchange_Names': 'market_name' # COT
        }
        
        # Aplicar renomeações apenas para colunas que existem
        # Priorizar Lev_Money se ambos Lev_Money e NonComm existirem
        cols_to_rename = {}
        existing_cols = df.columns
        for old, new in rename_map.items():
            if old in existing_cols:
                # Se o novo nome já foi mapeado por uma coluna de maior prioridade, não sobrescrever
                # (Ex: Se Lev_Money já mapeou para 'longs', não deixar NonComm sobrescrever)
                if new not in cols_to_rename.values():
                    # Especificamente para longs/shorts, dar prioridade a Lev_Money
                    if new in ['longs', 'shorts']:
                         if ('Lev_Money_Positions' in old) or \
                            all(f'Lev_Money_Positions{p}_All' not in existing_cols for p in ['_Long', '_Short']):
                             cols_to_rename[old] = new
                    else:
                         cols_to_rename[old] = new
        
        if cols_to_rename:
            df = df.rename(columns=cols_to_rename)
        
        # Converter colunas de timestamp para datetime se necessário
        if 'timestamp' in df.columns:
             try:
                 df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
             except Exception:
                 try:
                     df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                 except Exception:
                      st.warning("Não foi possível converter a coluna 'timestamp' para datetime.")
                      pass
        elif 'unix_timestamp' in df.columns:
             try:
                 df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='ms', errors='coerce')
             except Exception:
                  st.warning("Não foi possível converter a coluna 'unix_timestamp' para datetime.")
                  pass
        
        return df
    
    def analyze_breadth_data(self):
        """
        Analisa dados de amplitude do mercado com métricas melhoradas e explicações detalhadas.
        """
        st.markdown("### 📊 Análise de Amplitude do Mercado")
        
        try:
            # Carregar dados
            highs_lows = self.fetcher.load_data_file("breadth/52wk_highs_lows.csv")
            ma_tracking = self.fetcher.load_data_file("breadth/moving_average_tracking.csv")
            
            st.info(f"Arquivos consultados: breadth/52wk_highs_lows.csv, breadth/moving_average_tracking.csv")
            
            # Processar dados de highs/lows
            if highs_lows is not None:
                highs_lows = self._process_api_data(highs_lows)
                if not highs_lows.empty:
                    st.success(f"Dados de highs/lows processados com sucesso. Colunas: {list(highs_lows.columns)}")
                    
                    # Verificar se as colunas necessárias existem
                    needed_cols = ['timestamp', 'new_highs', 'new_lows']
                    if all(col in highs_lows.columns for col in needed_cols):
                        # Converter colunas numéricas e timestamp
                        for col in ['new_highs', 'new_lows', 'total']:
                            if col in highs_lows.columns:
                                highs_lows[col] = pd.to_numeric(highs_lows[col], errors='coerce')
                        if highs_lows['timestamp'].dtype == 'object':
                            highs_lows['timestamp'] = pd.to_datetime(highs_lows['timestamp'], errors='coerce')
                        highs_lows = highs_lows.dropna(subset=needed_cols)

                        if len(highs_lows) >= 2:
                            # Calcular métricas adicionais
                            highs_lows['net_breadth'] = highs_lows['new_highs'] - highs_lows['new_lows']
                            highs_lows['breadth_ratio'] = highs_lows['new_highs'] / highs_lows['new_lows'].replace(0, 1)
                            highs_lows['strength_index'] = (highs_lows['new_highs'] - highs_lows['new_lows']) / \
                                                          (highs_lows['new_highs'] + highs_lows['new_lows']).replace(0, 1)
                            highs_lows['strength_ma20'] = highs_lows['strength_index'].rolling(window=20).mean()
                            highs_lows['strength_ma50'] = highs_lows['strength_index'].rolling(window=50).mean()
                            
                            # Identificar fases de mercado com base no Índice de Força
                            highs_lows['market_phase'] = pd.cut(
                                highs_lows['strength_index'],
                                bins=[-1.1, -0.5, -0.2, 0.2, 0.5, 1.1],
                                labels=['Extremo Pessimismo', 'Pessimismo', 'Neutro', 'Otimismo', 'Extremo Otimismo']
                            )
                            
                            # Calcular Z-score do Índice de Força (20 dias)
                            if len(highs_lows) >= 60:  # Precisa de pelo menos 60 pontos para Z-score significativo
                                lookback = 60
                                highs_lows['strength_zscore'] = (
                                    highs_lows['strength_index'] - 
                                    highs_lows['strength_index'].rolling(lookback).mean()
                                ) / highs_lows['strength_index'].rolling(lookback).std()
                            
                            # Métricas principais
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                current_highs = highs_lows['new_highs'].iloc[-1]
                                prev_highs = highs_lows['new_highs'].iloc[-2]
                                delta_highs = current_highs - prev_highs
                                st.metric("Novas Máximas (Hoje)", 
                                        f"{int(current_highs)}", 
                                        f"{int(delta_highs)}")
                            with col2:
                                current_lows = highs_lows['new_lows'].iloc[-1]
                                prev_lows = highs_lows['new_lows'].iloc[-2]
                                delta_lows = current_lows - prev_lows
                                st.metric("Novas Mínimas (Hoje)", 
                                        f"{int(current_lows)}", 
                                        f"{int(delta_lows)}")
                            with col3:
                                current_ratio = highs_lows['breadth_ratio'].iloc[-1]
                                prev_ratio = highs_lows['breadth_ratio'].iloc[-2]
                                delta_ratio = current_ratio - prev_ratio
                                st.metric("Ratio Máximas/Mínimas", 
                                        f"{current_ratio:.2f}", 
                                        f"{delta_ratio:.2f}")
                            with col4:
                                current_phase = highs_lows['market_phase'].iloc[-1]
                                total = highs_lows['total'].iloc[-1] if 'total' in highs_lows.columns and not pd.isna(highs_lows['total'].iloc[-1]) else 0
                                st.metric("Fase do Mercado", str(current_phase), f"Total: {int(total)} pares")
                        
                            # Visualizações principais - 2 colunas com 2 gráficos cada
                            with st.expander("📈 Visualizações Detalhadas de Amplitude do Mercado", expanded=True):
                                tab1, tab2, tab3, tab4 = st.tabs([
                                    "Máximas vs Mínimas", 
                                    "Índice de Força", 
                                    "Distribuição de Máximas/Mínimas",
                                    "Mapa de Calor do Índice de Força"
                                ])
                                
                                with tab1:
                                    st.markdown("#### 📈 Máximas e Mínimas de 52 Semanas")
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=highs_lows['timestamp'],
                                        y=highs_lows['new_highs'],
                                        name='Novas Máximas',
                                        fill='tozeroy',
                                        line=dict(color=self.color_palette['bullish'])
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=highs_lows['timestamp'],
                                        y=-highs_lows['new_lows'],
                                        name='Novas Mínimas',
                                        fill='tozeroy',
                                        line=dict(color=self.color_palette['bearish'])
                                    ))
                                    fig.update_layout(
                                        title="Novas Máximas vs Mínimas (52 semanas)",
                                        hovermode='x unified',
                                        showlegend=True,
                                        yaxis_title="Número de Criptos",
                                        xaxis_title="Data"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Gráfico adicional: Net Breadth (Novas Máximas - Novas Mínimas)
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        x=highs_lows['timestamp'],
                                        y=highs_lows['net_breadth'],
                                        name='Net Breadth',
                                        marker=dict(
                                            color=np.where(highs_lows['net_breadth'] >= 0, 
                                                          self.color_palette['bullish'], 
                                                          self.color_palette['bearish'])
                                        )
                                    ))
                                    fig.update_layout(
                                        title="Net Breadth (Novas Máximas - Novas Mínimas)",
                                        hovermode='x unified',
                                        showlegend=False,
                                        yaxis_title="Net Breadth",
                                        xaxis_title="Data"
                                    )
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Explicação
                                    st.markdown("""
                                    **Interpretação do Gráfico de Máximas vs Mínimas:**
                                    
                                    1. **Dominância de Máximas (Verde Acima):** 
                                       - Indica um mercado forte e em ascensão
                                       - Sinal característico de tendências de alta sustentáveis
                                       - Sugere ampla participação e força do mercado
                                    
                                    2. **Dominância de Mínimas (Vermelho Abaixo):** 
                                       - Indica um mercado em queda
                                       - Sinal de tendência de baixa predominante
                                       - Sugere fraqueza generalizada no mercado
                                    
                                    3. **Net Breadth (Gráfico de Barras):**
                                       - Barras verdes: Mais criptos fazendo novas máximas que mínimas
                                       - Barras vermelhas: Mais criptos fazendo novas mínimas que máximas
                                       - Oscilações de positivo para negativo podem sinalizar mudanças de tendência
                                    
                                    4. **Padrões a Observar:**
                                       - Divergências: Preço do bitcoin subindo mas Net Breadth caindo sugere rally instável
                                       - Extremos: Valores extremamente positivos ou negativos podem indicar condições de sobrecompra/sobrevenda
                                    """)
                                
                                with tab2:
                                    st.markdown("#### 💪 Índice de Força do Mercado")
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=highs_lows['timestamp'],
                                        y=highs_lows['strength_index'],
                                        name='Índice de Força',
                                        line=dict(color=self.color_palette['primary'], width=1.5)
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=highs_lows['timestamp'],
                                        y=highs_lows['strength_ma20'],
                                        name='Média Móvel 20 períodos',
                                        line=dict(color=self.color_palette['secondary'], width=2)
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=highs_lows['timestamp'],
                                        y=highs_lows['strength_ma50'],
                                        name='Média Móvel 50 períodos',
                                        line=dict(color=self.color_palette['tertiary'], width=2, dash='dash')
                                    ))
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                    fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5)
                                    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5)
                                    fig.add_hline(y=0.2, line_dash="dot", line_color="lightgreen", opacity=0.3)
                                    fig.add_hline(y=-0.2, line_dash="dot", line_color="lightcoral", opacity=0.3)
                                    
                                    fig.update_layout(
                                        title="Índice de Força do Mercado",
                                        hovermode='x unified',
                                        showlegend=True,
                                        yaxis_title="Força do Mercado",
                                        xaxis_title="Data",
                                        yaxis=dict(range=[-1, 1])
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Mostrar Z-score se disponível
                                    if 'strength_zscore' in highs_lows.columns:
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=highs_lows['timestamp'],
                                            y=highs_lows['strength_zscore'],
                                            name='Z-Score do Índice de Força',
                                            line=dict(color='purple', width=1.5)
                                        ))
                                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                        fig.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5)
                                        fig.add_hline(y=-2, line_dash="dot", line_color="red", opacity=0.5)
                                        fig.update_layout(
                                            title="Z-Score do Índice de Força (Normalizado)",
                                            hovermode='x unified',
                                            showlegend=False,
                                            yaxis_title="Z-Score",
                                            xaxis_title="Data"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    if len(highs_lows) > 20: # Suficiente para MA20
                                        current_strength = highs_lows['strength_index'].iloc[-1]
                                        strength_ma20 = highs_lows['strength_ma20'].iloc[-1]
                                        strength_ma50 = highs_lows['strength_ma50'].iloc[-1] if len(highs_lows) > 50 else None
                                        
                                        # Análise melhorada das tendências
                                        trend_direction = ""
                                        if strength_ma20 > 0 and current_strength > strength_ma20:
                                            trend_direction = "Alta Acelerando"
                                        elif strength_ma20 > 0 and current_strength < strength_ma20:
                                            trend_direction = "Alta Desacelerando"
                                        elif strength_ma20 < 0 and current_strength < strength_ma20:
                                            trend_direction = "Baixa Acelerando"
                                        elif strength_ma20 < 0 and current_strength > strength_ma20:
                                            trend_direction = "Baixa Desacelerando"
                                        elif abs(strength_ma20) < 0.1:
                                            trend_direction = "Consolidação/Neutro"
                                            
                                        # Sinal de cruzamento de médias se tivermos MA50
                                        cross_signal = ""
                                        if strength_ma50 is not None:
                                            if strength_ma20 > strength_ma50 and highs_lows['strength_ma20'].iloc[-2] <= highs_lows['strength_ma50'].iloc[-2]:
                                                cross_signal = "💹 SINAL DE ALTA: Cruzamento MA20 acima da MA50"
                                            elif strength_ma20 < strength_ma50 and highs_lows['strength_ma20'].iloc[-2] >= highs_lows['strength_ma50'].iloc[-2]:
                                                cross_signal = "🔻 SINAL DE BAIXA: Cruzamento MA20 abaixo da MA50"
                                        
                                        # Status do mercado baseado no valor atual do índice
                                        if current_strength > 0.5:
                                            market_status = "Mercado Extremamente Forte"
                                        elif current_strength > 0.2:
                                            market_status = "Mercado Forte"
                                        elif current_strength > -0.2:
                                            market_status = "Mercado Neutro"
                                        elif current_strength > -0.5:
                                            market_status = "Mercado Fraco"
                                        else:
                                            market_status = "Mercado Extremamente Fraco"
                                        
                                        st.info(f"""
                                        **Análise Avançada do Índice de Força:**
                                        
                                        - **Valor Atual:** {current_strength:.3f}
                                        - **Média 20 períodos:** {strength_ma20:.3f}
                                        {f"- **Média 50 períodos:** {strength_ma50:.3f}" if strength_ma50 is not None else ""}
                                        - **Status do Mercado:** {market_status}
                                        - **Tendência:** {trend_direction}
                                        {f"- **{cross_signal}**" if cross_signal else ""}
                                        """)
                                        
                                        # Explicação
                                        st.markdown("""
                                        **Interpretação do Índice de Força:**
                                        
                                        Este índice é calculado como `(Novas Máximas - Novas Mínimas) / (Novas Máximas + Novas Mínimas)` e varia de -1 a +1:
                                        
                                        - **Acima de +0.5:** Mercado extremamente forte (fase de euforia/sobrecompra)
                                        - **Entre +0.2 e +0.5:** Mercado em tendência de alta forte
                                        - **Entre -0.2 e +0.2:** Mercado em consolidação/neutro
                                        - **Entre -0.5 e -0.2:** Mercado em tendência de baixa forte
                                        - **Abaixo de -0.5:** Mercado extremamente fraco (fase de pânico/sobrevenda)
                                        
                                        **Sinais estratégicos:**
                                        1. **Cruzamentos de médias móveis:** A MA20 cruzando a MA50 oferece sinais de mudança de tendência
                                        2. **Divergências:** Se o preço do Bitcoin formar novos topos, mas o índice de força formar topos mais baixos, isso pode sinalizar fraqueza da tendência
                                        3. **Retorno à média:** Valores extremos (+0.8 ou -0.8) geralmente retornam à média, oferecendo oportunidades de reversão
                                        """)
                                
                                with tab3:
                                    st.markdown("#### 📊 Distribuição de Máximas e Mínimas")
                                    # Criar histograma/distribuição dos dados de máximas/mínimas
                                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribuição de Novas Máximas", "Distribuição de Novas Mínimas"))
                                    
                                    # Histograma de máximas
                                    fig.add_trace(
                                        go.Histogram(
                                            x=highs_lows['new_highs'],
                                            nbinsx=20,
                                            marker_color=self.color_palette['bullish'],
                                            opacity=0.7,
                                            name="Novas Máximas"
                                        ),
                                        row=1, col=1
                                    )
                                    
                                    # Histograma de mínimas
                                    fig.add_trace(
                                        go.Histogram(
                                            x=highs_lows['new_lows'],
                                            nbinsx=20,
                                            marker_color=self.color_palette['bearish'],
                                            opacity=0.7,
                                            name="Novas Mínimas"
                                        ),
                                        row=1, col=2
                                    )
                                    
                                    fig.update_layout(
                                        title="Distribuição Histórica de Máximas e Mínimas",
                                        showlegend=True,
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Estatísticas descritivas
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("Estatísticas - Novas Máximas")
                                        stats_highs = highs_lows['new_highs'].describe()
                                        stats_df_highs = pd.DataFrame({
                                            'Métrica': ['Média', 'Desvio Padrão', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo'],
                                            'Valor': [
                                                f"{stats_highs['mean']:.2f}",
                                                f"{stats_highs['std']:.2f}",
                                                f"{stats_highs['min']:.0f}",
                                                f"{stats_highs['25%']:.0f}",
                                                f"{stats_highs['50%']:.0f}",
                                                f"{stats_highs['75%']:.0f}",
                                                f"{stats_highs['max']:.0f}"
                                            ]
                                        })
                                        st.table(stats_df_highs)
                                    
                                    with col2:
                                        st.subheader("Estatísticas - Novas Mínimas")
                                        stats_lows = highs_lows['new_lows'].describe()
                                        stats_df_lows = pd.DataFrame({
                                            'Métrica': ['Média', 'Desvio Padrão', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo'],
                                            'Valor': [
                                                f"{stats_lows['mean']:.2f}",
                                                f"{stats_lows['std']:.2f}",
                                                f"{stats_lows['min']:.0f}",
                                                f"{stats_lows['25%']:.0f}",
                                                f"{stats_lows['50%']:.0f}",
                                                f"{stats_lows['75%']:.0f}",
                                                f"{stats_lows['max']:.0f}"
                                            ]
                                        })
                                        st.table(stats_df_lows)
                                    
                                    # Correlação entre máximas e mínimas
                                    corr = np.corrcoef(highs_lows['new_highs'], highs_lows['new_lows'])[0, 1]
                                    
                                    st.markdown(f"""
                                    **Análise da Distribuição:**
                                    
                                    - **Correlação entre Máximas e Mínimas:** {corr:.3f}
                                    - Uma correlação negativa indica comportamento anticíclico esperado (quando máximas aumentam, mínimas diminuem)
                                    - Uma correlação próxima de zero indica independência entre máximas e mínimas
                                    - Uma correlação positiva pode indicar alta volatilidade geral do mercado
                                    
                                    A distribuição mostra a frequência histórica de diferentes níveis de máximas e mínimas, ajudando a identificar o que constitui valores "normais" versus valores "extremos" no contexto atual do mercado.
                                    """)
                                
                                with tab4:
                                    st.markdown("#### 🔥 Mapa de Calor do Índice de Força")
                                    
                                    # Criar mapa de calor do Índice de Força por mês/ano
                                    if len(highs_lows) > 20:  # Precisa de dados suficientes
                                        # Extrair mês e ano para o heatmap
                                        highs_lows['year'] = highs_lows['timestamp'].dt.year
                                        highs_lows['month'] = highs_lows['timestamp'].dt.month
                                        
                                        # Calcular média do índice de força por mês/ano
                                        heatmap_data = highs_lows.groupby(['year', 'month'])['strength_index'].mean().reset_index()
                                        
                                        # Criar pivot table para o heatmap
                                        heatmap_pivot = heatmap_data.pivot(index='month', columns='year', values='strength_index')
                                        
                                        # Renomear os meses para nomes
                                        month_names = {
                                            1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                                            7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
                                        }
                                        heatmap_pivot.index = heatmap_pivot.index.map(lambda x: month_names.get(x, x))
                                        
                                        # Criar o heatmap
                                        fig = px.imshow(
                                            heatmap_pivot,
                                            labels=dict(x="Ano", y="Mês", color="Índice de Força"),
                                            x=heatmap_pivot.columns,
                                            y=heatmap_pivot.index,
                                            color_continuous_scale='RdBu_r',  # Escala de cores vermelho-branco-azul
                                            zmin=-1, zmax=1,  # Limitar a escala de -1 a 1
                                            title="Mapa de Calor do Índice de Força por Mês/Ano"
                                        )
                                        
                                        fig.update_layout(height=500)
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Resumo de sazonalidade
                                        # Calcular média por mês (todos os anos)
                                        monthly_avg = highs_lows.groupby(highs_lows['timestamp'].dt.month)['strength_index'].mean()
                                        best_month = monthly_avg.idxmax()
                                        worst_month = monthly_avg.idxmin()
                                        
                                        st.markdown(f"""
                                        **Análise de Sazonalidade do Índice de Força:**
                                        
                                        - **Mês historicamente mais forte:** {month_names[best_month]} (média: {monthly_avg[best_month]:.3f})
                                        - **Mês historicamente mais fraco:** {month_names[worst_month]} (média: {monthly_avg[worst_month]:.3f})
                                        
                                        O mapa de calor revela padrões de sazonalidade e tendências de longo prazo no mercado de criptomoedas:
                                        
                                        - **Cores azuis (positivas):** Períodos de força de mercado, quando máximas dominaram mínimas
                                        - **Cores vermelhas (negativas):** Períodos de fraqueza de mercado, quando mínimas dominaram máximas
                                        - **Padrões horizontais:** Indicam sazonalidade consistente (certos meses tendem a ser mais fortes/fracos)
                                        - **Padrões verticais:** Indicam anos completos de bull market (azul) ou bear market (vermelho)
                                        
                                        Esta visualização é útil para identificar se existe sazonalidade no mercado de criptomoedas e 
                                        para colocar o momento atual em perspectiva histórica.
                                        """)
                                    else:
                                        st.warning("Dados insuficientes para criar o mapa de calor. São necessários pelo menos 20 pontos de dados.")
                        else:
                            st.error("Dados de máximas/mínimas não contêm as colunas necessárias ou válidas")
                            st.write("Colunas necessárias:", needed_cols)
                            st.write("Colunas disponíveis:", list(highs_lows.columns))
                else:
                    st.error("Não foi possível processar os dados de highs/lows")
            
            # Processar dados de médias móveis
            if ma_tracking is not None:
                ma_tracking = self._process_api_data(ma_tracking)
                if not ma_tracking.empty:
                    st.success(f"Dados de médias móveis processados com sucesso. Colunas: {list(ma_tracking.columns)}")
                    
                    ma_cols = [col for col in ma_tracking.columns if 'above_ma' in col.lower()]
                    if 'timestamp' in ma_tracking.columns and ma_cols:
                        # Converter colunas numéricas e timestamp
                        for col in ma_cols:
                            ma_tracking[col] = pd.to_numeric(ma_tracking[col], errors='coerce')
                        if ma_tracking['timestamp'].dtype == 'object':
                            ma_tracking['timestamp'] = pd.to_datetime(ma_tracking['timestamp'], errors='coerce')
                        ma_tracking = ma_tracking.dropna(subset=[*ma_cols, 'timestamp'])

                        if len(ma_tracking) >= 2:
                            with st.expander("📉 Análise de Tendência por Médias Móveis", expanded=True):
                                st.markdown("#### 📉 Rastreamento de Médias Móveis")
                                
                                # Adicionar colunas derivadas para análise mais profunda
                                if 'above_ma50' in ma_cols and 'above_ma200' in ma_cols:
                                    ma_tracking['golden_cross_ratio'] = ma_tracking['above_ma50'] / ma_tracking['above_ma200']
                                
                                # Métricas principais e tendências
                                cols = st.columns(len(ma_cols))
                                for i, col in enumerate(ma_cols):
                                    with cols[i]:
                                        current_value = float(ma_tracking[col].iloc[-1])
                                        previous_value = float(ma_tracking[col].iloc[-2])
                                        delta = current_value - previous_value
                                        ma_period = col.split('_')[-1].upper()
                                        
                                        # Adicionar interpretação de tendência com base no valor
                                        if current_value > 80:
                                            trend = "Muito Forte"
                                        elif current_value > 60:
                                            trend = "Forte"
                                        elif current_value > 40:
                                            trend = "Neutra"
                                        elif current_value > 20:
                                            trend = "Fraca"
                                        else:
                                            trend = "Muito Fraca"
                                            
                                        st.metric(
                                            f"Acima da MA{ma_period}",
                                            f"{current_value:.1f}%",
                                            f"{delta:.1f}% ({trend})"
                                        )
                                
                                # Gráfico principal com zonas de sobrecompra/sobrevenda destacadas
                                fig = go.Figure()
                                colors = ['rgb(26, 118, 255)', 'rgb(178, 107, 255)', 'rgb(55, 83, 109)']
                                
                                # Adicionar zonas de sobrecompra/sobrevenda
                                fig.add_shape(
                                    type="rect",
                                    x0=ma_tracking['timestamp'].min(),
                                    x1=ma_tracking['timestamp'].max(),
                                    y0=80,
                                    y1=100,
                                    fillcolor="rgba(0, 255, 0, 0.1)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                                fig.add_shape(
                                    type="rect",
                                    x0=ma_tracking['timestamp'].min(),
                                    x1=ma_tracking['timestamp'].max(),
                                    y0=0,
                                    y1=20,
                                    fillcolor="rgba(255, 0, 0, 0.1)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                                
                                for i, col in enumerate(ma_cols):
                                    ma_period = col.split('_')[-1].upper()
                                    fig.add_trace(go.Scatter(
                                        x=ma_tracking['timestamp'],
                                        y=ma_tracking[col],
                                        name=f'MA {ma_period}',
                                        line=dict(color=colors[i % len(colors)], width=2),
                                        hovertemplate="%{y:.1f}%<extra></extra>"
                                    ))
                                
                                fig.add_hline(y=80, line_dash="dot", line_color="green", opacity=0.7)
                                fig.add_hline(y=20, line_dash="dot", line_color="red", opacity=0.7)
                                fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.7)
                                
                                fig.update_layout(
                                    title="Porcentagem de Criptos Acima das Médias Móveis",
                                    hovermode='x unified',
                                    showlegend=True,
                                    yaxis_title="% de Criptos",
                                    xaxis_title="Data",
                                    yaxis=dict(range=[0, 100])
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Adicionar gráfico derivado: Golden Cross Ratio
                                if 'golden_cross_ratio' in ma_tracking.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=ma_tracking['timestamp'],
                                        y=ma_tracking['golden_cross_ratio'],
                                        name='Golden Cross Ratio',
                                        line=dict(color='gold', width=2),
                                        hovertemplate="%{y:.2f}<extra></extra>"
                                    ))
                                    
                                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
                                    fig.add_hline(y=1.2, line_dash="dot", line_color="green", opacity=0.5)
                                    fig.add_hline(y=0.8, line_dash="dot", line_color="red", opacity=0.5)
                                    
                                    fig.update_layout(
                                        title="Golden Cross Ratio (MA50/MA200)",
                                        hovermode='x unified',
                                        showlegend=True,
                                        yaxis_title="Ratio",
                                        xaxis_title="Data"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Análise de tendência avançada
                                current_values = {col: float(ma_tracking[col].iloc[-1]) for col in ma_cols}
                                ma50 = current_values.get('above_ma50', 0.0)
                                ma200 = current_values.get('above_ma200', 0.0)
                                
                                # Determinar fase de mercado por múltiplos indicadores
                                if ma50 > 80 and ma200 > 80:
                                    market_phase = "🚀 Bull Market Forte (Possível Euforia)"
                                    risk_level = "Alto - Considere proteger lucros"
                                elif ma50 > ma200 and ma50 > 60:
                                    market_phase = "🟢 Bull Market"
                                    risk_level = "Moderado - Tendência de alta em andamento"
                                elif ma50 > ma200:
                                    market_phase = "🟡 Início de Alta/Recuperação"
                                    risk_level = "Médio-Baixo - Tendência de alta iniciando"
                                elif ma50 < 20 and ma200 < 20:
                                    market_phase = "⚠️ Bear Market Forte (Possível Capitulação)"
                                    risk_level = "Extremo - Potencial de reversão, mas cautela necessária"
                                elif ma50 < ma200 and ma50 < 40:
                                    market_phase = "🔴 Bear Market"
                                    risk_level = "Alto - Tendência de baixa em andamento"
                                elif ma50 < ma200:
                                    market_phase = "🟠 Início de Baixa/Correção"
                                    risk_level = "Médio-Alto - Tendência de baixa iniciando"
                                else:
                                    market_phase = "⚖️ Mercado Neutro/Indeciso"
                                    risk_level = "Médio - Sem tendência clara"
                                
                                # Calcular força da tendência
                                trend = "🟢 Tendência de Alta" if ma50 > ma200 else "🔴 Tendência de Baixa"
                                trend_strength = "Forte" if abs(ma50 - ma200) > 20 else "Moderada" if abs(ma50 - ma200) > 10 else "Fraca"
                                
                                # Identificar condições de divergência
                                divergence = ""
                                if len(ma_tracking) > 10:
                                    # Verificar divergências nos últimos 10 períodos
                                    recent_ma50 = ma_tracking['above_ma50'].iloc[-10:].values
                                    if ma50 > ma200 and np.all(np.diff(recent_ma50[-5:]) < 0):
                                        divergence = "⚠️ Alerta: MA50 em tendência de alta mas com momentum de baixa (possível topo)"
                                    elif ma50 < ma200 and np.all(np.diff(recent_ma50[-5:]) > 0):
                                        divergence = "💡 Oportunidade: MA50 em tendência de baixa mas com momentum de alta (possível fundo)"
                                
                                st.info(f"""
                                **Análise Avançada de Tendência:**
                                
                                - **Fase do Mercado:** {market_phase}
                                - **Tendência Dominante:** {trend} ({trend_strength})
                                - **Diferença MA50-MA200:** {ma50 - ma200:.1f}%
                                - **Nível de Risco Atual:** {risk_level}
                                {f"- **{divergence}**" if divergence else ""}
                                """)
                                
                                # Explicação dos indicadores
                                st.markdown("""
                                **Interpretação dos Indicadores de Médias Móveis:**
                                
                                1. **Porcentagem acima da MA:**
                                   - **> 80%**: Mercado fortemente positivo, possível sobrecompra
                                   - **50-80%**: Mercado em tendência de alta saudável
                                   - **20-50%**: Mercado neutro ou em transição
                                   - **< 20%**: Mercado fortemente negativo, possível sobrevenda
                                
                                2. **Golden Cross Ratio (MA50/MA200)**:
                                   - **> 1.0**: Mais ativos acima da MA50 que da MA200, sinal positivo
                                   - **< 1.0**: Menos ativos acima da MA50 que da MA200, sinal negativo
                                   - **Cruzamento acima de 1.0**: Sinal de início de tendência de alta
                                   - **Cruzamento abaixo de 1.0**: Sinal de início de tendência de baixa
                                
                                3. **Fases do Mercado:**
                                   - Bull Market: MA50 > MA200 e altas % acima de MAs
                                   - Bear Market: MA50 < MA200 e baixas % acima de MAs
                                   - Transição: Divergências entre MAs ou momentum contrário à tendência
                                
                                4. **Divergências:**
                                   - Divergência de topo: Preço subindo mas % acima de MAs caindo
                                   - Divergência de fundo: Preço caindo mas % acima de MAs subindo
                                """)
                        else:
                            st.error("Dados de médias móveis não contêm as colunas necessárias ou válidas, ou não há dados suficientes para análise de variação.")
                            st.write("Colunas necessárias:", ['timestamp', 'above_ma*'])
                            st.write("Colunas disponíveis:", list(ma_tracking.columns))
                else:
                    st.error("Não foi possível processar os dados de médias móveis")
            
        except Exception as e:
            st.error(f"Erro na análise de amplitude: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def analyze_government_data(self):
        """
        Analisa dados governamentais com análises aumentadas e explicações detalhadas.
        """
        st.markdown("### 🏛️ Análise de Dados Governamentais")
        
        try:
            # Carregar dados
            current_year = datetime.now().year
            cot_data = self.fetcher.load_data_file(f"gov/cftc_cot_bitcoin_{current_year}.csv")
            treasury_data = self.fetcher.load_data_file(f"gov/treasury_yields_{current_year}.csv")
            
            st.info(f"Arquivos consultados: cftc_cot_bitcoin_{current_year}.csv, treasury_yields_{current_year}.csv")
            
            # Processar dados COT
            if cot_data is not None:
                cot_data = self._process_api_data(cot_data)
                if not cot_data.empty:
                    st.success(f"Dados COT processados com sucesso. Colunas: {list(cot_data.columns)}")
                    
                    date_col = 'timestamp' if 'timestamp' in cot_data.columns else None
                    long_col = 'longs' if 'longs' in cot_data.columns else None
                    short_col = 'shorts' if 'shorts' in cot_data.columns else None
                    open_interest_col = 'open_interest' if 'open_interest' in cot_data.columns else None
                    
                    if date_col and long_col and short_col:
                        with st.expander("📊 Análise dos Relatórios COT (Commitment of Traders)", expanded=True):
                            st.markdown("#### 📈 Análise COT (Commitment of Traders)")
                            for col in [long_col, short_col]:
                                cot_data[col] = pd.to_numeric(cot_data[col], errors='coerce')
                            
                            # Adicionar open_interest se disponível
                            if open_interest_col:
                                cot_data[open_interest_col] = pd.to_numeric(cot_data[open_interest_col], errors='coerce')
                            
                            # Adicionar colunas derivadas para análise mais profunda
                            cot_data['net_position'] = cot_data[long_col] - cot_data[short_col]
                            cot_data['long_percent'] = (cot_data[long_col] / (cot_data[long_col] + cot_data[short_col])) * 100
                            
                            # Média móvel de 4 semanas (aproximadamente 1 mês)
                            cot_data['net_position_ma4'] = cot_data['net_position'].rolling(window=4).mean()
                            
                            # Calcular mudanças semanais
                            cot_data['net_position_change'] = cot_data['net_position'].diff()
                            cot_data['longs_change'] = cot_data[long_col].diff()
                            cot_data['shorts_change'] = cot_data[short_col].diff()
                            
                            # Calcular Z-score para identificar extremos
                            if len(cot_data) >= 12:  # Usamos pelo menos 12 pontos para uma análise razoável
                                window = min(12, len(cot_data))
                                cot_data['net_position_zscore'] = (
                                    cot_data['net_position'] - cot_data['net_position'].rolling(window).mean()
                                ) / cot_data['net_position'].rolling(window).std()
                            
                            cot_data = cot_data.dropna(subset=[date_col, long_col, short_col, 'net_position'])
                            cot_data = cot_data.sort_values(by=date_col)

                            if len(cot_data) >= 2:                                
                                # Gráfico 1: Long vs Short Positions
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=cot_data[date_col], 
                                    y=cot_data[long_col], 
                                    name='Posições Long (Lev Money)', 
                                    marker_color=self.color_palette['bullish']
                                ))
                                fig.add_trace(go.Bar(
                                    x=cot_data[date_col], 
                                    y=cot_data[short_col], 
                                    name='Posições Short (Lev Money)', 
                                    marker_color=self.color_palette['bearish']
                                ))
                                
                                fig.update_layout(
                                    title="Posições Long vs Short (Lev Money) - Bitcoin Futures", 
                                    barmode='group', 
                                    xaxis_title="Data", 
                                    yaxis_title="Número de Contratos",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Gráfico 2: Net Position com MA4
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=cot_data[date_col], 
                                    y=cot_data['net_position'], 
                                    mode='lines+markers', 
                                    name='Posição Líquida (Lev Money)', 
                                    line=dict(color=self.color_palette['primary'], width=2)
                                ))
                                
                                # Adicionar média móvel se tivermos pontos suficientes
                                if 'net_position_ma4' in cot_data.columns and not cot_data['net_position_ma4'].isna().all():
                                    fig.add_trace(go.Scatter(
                                        x=cot_data[date_col], 
                                        y=cot_data['net_position_ma4'], 
                                        mode='lines', 
                                        name='MM4 da Posição Líquida', 
                                        line=dict(color=self.color_palette['secondary'], width=2, dash='dot')
                                    ))
                                
                                fig.update_layout(
                                    title="Posição Líquida (Lev Money) - Bitcoin Futures", 
                                    xaxis_title="Data", 
                                    yaxis_title="Contratos (Net Long)", 
                                    hovermode='x unified'
                                )
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Gráfico 3: Percentual Long
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=cot_data[date_col], 
                                    y=cot_data['long_percent'], 
                                    mode='lines+markers', 
                                    name='% Long', 
                                    line=dict(color='purple', width=2),
                                    hovertemplate="%{y:.1f}%<extra></extra>"
                                ))
                                
                                # Adicionar linhas de referência
                                fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.7)
                                fig.add_hline(y=60, line_dash="dot", line_color="green", opacity=0.5)
                                fig.add_hline(y=40, line_dash="dot", line_color="red", opacity=0.5)
                                
                                fig.update_layout(
                                    title="Percentual Long (Lev Money) - Bitcoin Futures", 
                                    xaxis_title="Data", 
                                    yaxis_title="% Long", 
                                    hovermode='x unified',
                                    yaxis=dict(range=[0, 100])
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Gráfico 4: Mudanças semanais
                                if 'net_position_change' in cot_data.columns and not cot_data['net_position_change'].isna().all():
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        x=cot_data[date_col], 
                                        y=cot_data['net_position_change'], 
                                        name='Mudança Semanal na Posição Líquida', 
                                        marker_color=np.where(cot_data['net_position_change'] >= 0, 
                                                            self.color_palette['bullish'], 
                                                            self.color_palette['bearish'])
                                    ))
                                    
                                    fig.update_layout(
                                        title="Mudança Semanal na Posição Líquida (Lev Money)", 
                                        xaxis_title="Data", 
                                        yaxis_title="Mudança em Contratos", 
                                        hovermode='x unified'
                                    )
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Estatísticas e métricas atuais
                                latest_long = float(cot_data[long_col].iloc[-1])
                                latest_short = float(cot_data[short_col].iloc[-1])
                                latest_net = float(cot_data['net_position'].iloc[-1])
                                latest_long_pct = float(cot_data['long_percent'].iloc[-1])
                                
                                prev_long = float(cot_data[long_col].iloc[-2]) if len(cot_data) > 1 else 0
                                prev_short = float(cot_data[short_col].iloc[-2]) if len(cot_data) > 1 else 0
                                prev_net = float(cot_data['net_position'].iloc[-2]) if len(cot_data) > 1 else 0
                                
                                # Z-score para identificar extremos
                                market_sentiment = ""
                                if 'net_position_zscore' in cot_data.columns and not pd.isna(cot_data['net_position_zscore'].iloc[-1]):
                                    z_score = float(cot_data['net_position_zscore'].iloc[-1])
                                    if z_score > 2:
                                        market_sentiment = "Extremo Otimismo (Contrarian Bearish)"
                                    elif z_score > 1:
                                        market_sentiment = "Otimismo Alto"
                                    elif z_score > 0:
                                        market_sentiment = "Levemente Otimista"
                                    elif z_score > -1:
                                        market_sentiment = "Levemente Pessimista"
                                    elif z_score > -2:
                                        market_sentiment = "Pessimismo Alto"
                                    else:
                                        market_sentiment = "Extremo Pessimismo (Contrarian Bullish)"
                                else:
                                    # Alternativa se não temos Z-score
                                    if latest_long_pct > 70:
                                        market_sentiment = "Extremo Otimismo (Contrarian Bearish)"
                                    elif latest_long_pct > 60:
                                        market_sentiment = "Otimismo Alto"
                                    elif latest_long_pct > 50:
                                        market_sentiment = "Levemente Otimista"
                                    elif latest_long_pct > 40:
                                        market_sentiment = "Levemente Pessimista"
                                    elif latest_long_pct > 30:
                                        market_sentiment = "Pessimismo Alto"
                                    else:
                                        market_sentiment = "Extremo Pessimismo (Contrarian Bullish)"
                                
                                # Mostrar métricas em colunas
                                col1, col2, col3, col4 = st.columns(4)
                                with col1: 
                                    st.metric("Longs (Lev Money)", f"{int(latest_long):,}", f"{int(latest_long - prev_long):+,}")
                                with col2: 
                                    st.metric("Shorts (Lev Money)", f"{int(latest_short):,}", f"{int(latest_short - prev_short):+,}")
                                with col3: 
                                    st.metric("Net (Lev Money)", f"{int(latest_net):,}", f"{int(latest_net - prev_net):+,}")
                                with col4: 
                                    st.metric("Long %", f"{latest_long_pct:.1f}%", market_sentiment)
                                
                                # Análise detalhada
                                st.markdown("""
                                **Análise do Relatório COT (Commitment of Traders)**
                                
                                O relatório COT (Commitment of Traders) publicado pela CFTC (Commodity Futures Trading Commission) mostra as posições de diferentes tipos de traders nos mercados futuros, incluindo Bitcoin.
                                
                                **Principais insights:**
                                """)
                                
                                # Tendência de longo prazo
                                long_term_trend = ""
                                if len(cot_data) >= 8:
                                    recent_trend = cot_data['net_position'].iloc[-8:].values
                                    if np.all(np.diff(recent_trend[-4:]) > 0):
                                        long_term_trend = "Posições líquidas em **tendência de alta consistente** nas últimas 4 semanas, indicando momentum de compra crescente."
                                    elif np.all(np.diff(recent_trend[-4:]) < 0):
                                        long_term_trend = "Posições líquidas em **tendência de baixa consistente** nas últimas 4 semanas, indicando momentum de venda crescente."
                                    elif cot_data['net_position'].iloc[-1] > cot_data['net_position'].iloc[-8]:
                                        long_term_trend = "Posições líquidas majoritariamente mais altas que há 8 semanas, sugerindo **tendência de alta de médio prazo** com alguma volatilidade."
                                    elif cot_data['net_position'].iloc[-1] < cot_data['net_position'].iloc[-8]:
                                        long_term_trend = "Posições líquidas majoritariamente mais baixas que há 8 semanas, sugerindo **tendência de baixa de médio prazo** com alguma volatilidade."
                                    else:
                                        long_term_trend = "Posições líquidas estáveis sem tendência clara de médio prazo."
                                
                                # Sentimento contrarian
                                contrarian_signal = ""
                                if market_sentiment in ["Extremo Otimismo (Contrarian Bearish)", "Extremo Pessimismo (Contrarian Bullish)"]:
                                    if "Bearish" in market_sentiment:
                                        contrarian_signal = "Os fundos alavancados (Leveraged Funds) estão **extremamente otimistas**, o que historicamente pode ser um indicador contrário (bearish) quando este grupo atinge extremos."
                                    else:
                                        contrarian_signal = "Os fundos alavancados (Leveraged Funds) estão **extremamente pessimistas**, o que historicamente pode ser um indicador contrário (bullish) quando este grupo atinge extremos."
                                
                                # Mudanças recentes significativas
                                recent_changes = ""
                                if len(cot_data) >= 2:
                                    last_change = cot_data['net_position_change'].iloc[-1]
                                    if abs(last_change) > 1000:  # Um limiar arbitrário para mudanças significativas
                                        direction = "aumento" if last_change > 0 else "diminuição"
                                        recent_changes = f"Houve uma **{direction} significativa** nas posições líquidas na última semana, indicando uma possível mudança de sentimento ou reação a notícias de mercado."
                                
                                # Montagem da análise final
                                analysis_points = [
                                    f"- **Sentimento atual:** {market_sentiment}",
                                    f"- **Percentual Long:** {latest_long_pct:.1f}% (acima de 50% indica predominância de posições compradas)",
                                ]
                                
                                if long_term_trend:
                                    analysis_points.append(f"- **Tendência de médio prazo:** {long_term_trend}")
                                if contrarian_signal:
                                    analysis_points.append(f"- **Sinal contrarian:** {contrarian_signal}")
                                if recent_changes:
                                    analysis_points.append(f"- **Mudanças recentes:** {recent_changes}")
                                
                                # Adicionar interpretação final
                                if latest_net > 0:
                                    if latest_net > 5000:
                                        analysis_points.append("- **Interpretação:** Os grandes especuladores (Leveraged Funds) têm uma posição líquida **fortemente comprada**, o que geralmente reflete otimismo em relação ao Bitcoin, mas também pode representar um risco se houver uma liquidação em massa.")
                                    else:
                                        analysis_points.append("- **Interpretação:** Os grandes especuladores têm uma posição líquida **moderadamente comprada**, indicando otimismo cauteloso.")
                                else:
                                    if latest_net < -5000:
                                        analysis_points.append("- **Interpretação:** Os grandes especuladores têm uma posição líquida **fortemente vendida**, indicando pessimismo em relação ao Bitcoin, mas também criando potencial para um short squeeze se o mercado subir.")
                                    else:
                                        analysis_points.append("- **Interpretação:** Os grandes especuladores têm uma posição líquida **moderadamente vendida**, indicando algum pessimismo, mas não em níveis extremos.")
                                
                                st.info("\n".join(analysis_points))
                            else:
                                st.error("Dados COT não contêm as colunas necessárias/válidas ou não há dados suficientes.")
                                st.write("Colunas necessárias após normalização:", ['timestamp', 'longs', 'shorts'])
                                st.write("Colunas disponíveis:", list(cot_data.columns))
                    else:
                        missing_cols = []
                        if not date_col: missing_cols.append("'timestamp' (date, Report_Date...)")
                        if not long_col: missing_cols.append("'longs' (Lev_Money_Long, NonComm_Long...)")
                        if not short_col: missing_cols.append("'shorts' (Lev_Money_Short, NonComm_Short...)")
                        st.error(f"Colunas necessárias não encontradas nos dados COT: {', '.join(missing_cols)}")
                        st.write("Colunas disponíveis após normalização:", list(cot_data.columns))
                else:
                    st.error("Não foi possível processar os dados COT ou o arquivo está vazio.")
            else:
                 st.warning(f"Arquivo gov/cftc_cot_bitcoin_{current_year}.csv não encontrado ou vazio.")

            # Processar dados do Tesouro
            if treasury_data is not None:
                treasury_data = self._process_api_data(treasury_data)
                if not treasury_data.empty:
                    st.success(f"Dados do Tesouro processados com sucesso. Colunas: {list(treasury_data.columns)}")
                    
                    date_col = next((col for col in ['date', 'timestamp', 'time', 'report_date'] 
                                     if col in treasury_data.columns), None)
                    yields_cols = [col for col in treasury_data.columns 
                                  if ('yield' in col.lower() or 'yr' in col.lower() or 'year' in col.lower() or 'month' in col.lower())
                                  and col != date_col]
                    
                    if date_col and yields_cols:
                        with st.expander("📈 Análise de Rendimentos do Tesouro e Implicações Econômicas", expanded=True):
                            st.markdown("#### 📈 Análise Avançada de Rendimentos do Tesouro")
                            
                            # Converter colunas numéricas e timestamp
                            for col in yields_cols:
                                treasury_data[col] = pd.to_numeric(treasury_data[col], errors='coerce')
                            if treasury_data[date_col].dtype == 'object':
                                treasury_data[date_col] = pd.to_datetime(treasury_data[date_col], errors='coerce')
                            treasury_data = treasury_data.dropna(subset=[date_col, *yields_cols])

                            if len(treasury_data) > 0:
                                treasury_data = treasury_data.sort_values(by=date_col)
                                
                                # ANÁLISE 1: Curva de Rendimentos ao Longo do Tempo
                                fig = go.Figure()
                                for col in yields_cols:
                                    col_name = col.replace('_', ' ').replace('yield', '').replace('yr', ' year').title().strip()
                                    if not col_name:
                                        col_name = col
                                    
                                    fig.add_trace(go.Scatter(
                                        x=treasury_data[date_col],
                                        y=treasury_data[col],
                                        mode='lines',
                                        name=col_name
                                    ))
                                
                                fig.update_layout(
                                    title="Curva de Rendimentos do Tesouro ao Longo do Tempo",
                                    xaxis_title="Data",
                                    yaxis_title="Rendimento (%)",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # ANÁLISE 2: Curva de Rendimentos Atual
                                latest_data = treasury_data.iloc[-1]
                                
                                tenors = []
                                yields = []
                                tenor_map = {
                                    '1 month': 1/12, '3 month': 3/12, '6 month': 6/12,
                                    '1 year': 1, '2 year': 2, '3 year': 3, '5 year': 5,
                                    '7 year': 7, '10 year': 10, '20 year': 20, '30 year': 30
                                }
                                for col in yields_cols:
                                    clean_col = col.lower().replace('_', ' ').replace('yield', '').strip()
                                    tenor = tenor_map.get(clean_col)
                                    if tenor is None and ('yr' in clean_col or 'year' in clean_col or 'month' in clean_col):
                                        try:
                                            num = float(re.findall(r'\d+\.?\d*', clean_col)[0])
                                            if 'month' in clean_col: tenor = num / 12
                                            else: tenor = num
                                        except: tenor = None
                                    
                                    if tenor is not None and not pd.isna(latest_data[col]):
                                        tenors.append(tenor)
                                        yields.append(float(latest_data[col]))
                                
                                if tenors and yields:
                                    tenor_yield = sorted(zip(tenors, yields))
                                    tenors, yields = zip(*tenor_yield)
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=tenors,
                                        y=yields,
                                        mode='lines+markers',
                                        name='Curva de Rendimentos Atual',
                                        line=dict(color=self.color_palette['primary'], width=2)
                                    ))
                                    
                                    # Se tivermos dados históricos suficientes, adicionar uma curva anterior para comparação
                                    if len(treasury_data) > 20:  # Digamos, 20 pontos para ter uma boa separação temporal
                                        historical_data = treasury_data.iloc[-20]
                                        historical_tenors = []
                                        historical_yields = []
                                        
                                        for col in yields_cols:
                                            clean_col = col.lower().replace('_', ' ').replace('yield', '').strip()
                                            tenor = tenor_map.get(clean_col)
                                            if tenor is None and ('yr' in clean_col or 'year' in clean_col or 'month' in clean_col):
                                                try:
                                                    num = float(re.findall(r'\d+\.?\d*', clean_col)[0])
                                                    if 'month' in clean_col: tenor = num / 12
                                                    else: tenor = num
                                                except: tenor = None
                                            
                                            if tenor is not None and not pd.isna(historical_data[col]):
                                                historical_tenors.append(tenor)
                                                historical_yields.append(float(historical_data[col]))
                                        
                                        if historical_tenors and historical_yields:
                                            historical = sorted(zip(historical_tenors, historical_yields))
                                            historical_tenors, historical_yields = zip(*historical)
                                            
                                            fig.add_trace(go.Scatter(
                                                x=historical_tenors,
                                                y=historical_yields,
                                                mode='lines+markers',
                                                name=f'Curva {historical_data[date_col].strftime("%Y-%m-%d")}',
                                                line=dict(color=self.color_palette['secondary'], width=2, dash='dot')
                                            ))
                                    
                                    fig.update_layout(
                                        title=f"Curva de Rendimentos Atual ({latest_data[date_col].strftime('%Y-%m-%d')})",
                                        xaxis_title="Anos",
                                        yaxis_title="Rendimento (%)",
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # ANÁLISE 3: Cálculo da Inclinação da Curva (Yield Curve Steepness)
                                    # Vamos usar 10Y-2Y como medida clássica de inclinação
                                    
                                    # Primeiro, identificar as colunas mais próximas de 2Y e 10Y
                                    tenor_col_map = {}
                                    for tenor_target in [2, 10]:
                                        best_col = None
                                        min_diff = float('inf')
                                        for tenor, yield_val in tenor_yield:
                                            diff = abs(tenor - tenor_target)
                                            if diff < min_diff:
                                                min_diff = diff
                                                # Encontrar a coluna original correspondente a este tenor
                                                for col in yields_cols:
                                                    clean_col = col.lower().replace('_', ' ').replace('yield', '').strip()
                                                    if tenor_target == 2 and ('2 year' in clean_col or '2 yr' in clean_col or '2-year' in clean_col):
                                                        best_col = col
                                                        break
                                                    elif tenor_target == 10 and ('10 year' in clean_col or '10 yr' in clean_col or '10-year' in clean_col):
                                                        best_col = col
                                                        break
                                                if best_col is None:  # Se não encontrou correspondência exata
                                                    for col in yields_cols:
                                                        t = None
                                                        clean_col = col.lower().replace('_', ' ').replace('yield', '').strip()
                                                        t = tenor_map.get(clean_col)
                                                        if t is None and ('yr' in clean_col or 'year' in clean_col):
                                                            try:
                                                                num = float(re.findall(r'\d+\.?\d*', clean_col)[0])
                                                                t = num
                                                            except: pass
                                                        if t is not None and abs(t - tenor_target) < 0.01:  # Muito próximo
                                                            best_col = col
                                                            break
                                        tenor_col_map[tenor_target] = best_col
                                    
                                    # Calcular a inclinação
                                    steepness_analysis = []
                                    if tenor_col_map[2] and tenor_col_map[10]:
                                        # Adicionar coluna de inclinação ao dataframe
                                        treasury_data['yield_curve_steepness'] = treasury_data[tenor_col_map[10]] - treasury_data[tenor_col_map[2]]
                                        
                                        # Gráfico da inclinação ao longo do tempo
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=treasury_data[date_col],
                                            y=treasury_data['yield_curve_steepness'],
                                            mode='lines',
                                            name='Inclinação da Curva (10Y-2Y)',
                                            line=dict(color='purple', width=2)
                                        ))
                                        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
                                        
                                        fig.update_layout(
                                            title="Inclinação da Curva de Rendimentos (10Y-2Y)",
                                            xaxis_title="Data",
                                            yaxis_title="Diferença de Rendimento (%)",
                                            hovermode='x unified'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Análise da inclinação atual
                                        current_steepness = treasury_data['yield_curve_steepness'].iloc[-1]
                                        
                                        # Determinar o regime da curva
                                        if current_steepness > 0.5:
                                            curve_regime = "Fortemente Positiva"
                                            economic_implication = "Expectativa de crescimento econômico forte, possível indicação de pressão inflacionária"
                                            crypto_implication = "Geralmente positivo para criptomoedas se acompanhado de crescimento real sem aperto monetário excessivo"
                                        elif current_steepness > 0:
                                            curve_regime = "Moderadamente Positiva"
                                            economic_implication = "Expectativa de crescimento econômico moderado e inflação controlada"
                                            crypto_implication = "Neutro a positivo para criptomoedas, ambiente de crescimento com política monetária ainda acomodativa"
                                        elif current_steepness > -0.5:
                                            curve_regime = "Achatada a Levemente Invertida"
                                            economic_implication = "Possível desaceleração econômica, mercado precificando cortes de juros futuros"
                                            crypto_implication = "Misto - negativo a curto prazo (risco de recessão), potencialmente positivo a médio prazo (cortes de juros)"
                                        else:
                                            curve_regime = "Fortemente Invertida"
                                            economic_implication = "Forte sinal de recessão futura, mercado antecipando significativo relaxamento monetário"
                                            crypto_implication = "Risco elevado a curto prazo, mas potencial de alta no médio prazo com o afrouxamento monetário esperado"
                                        
                                        steepness_analysis.extend([
                                            f"- **Inclinação atual (10Y-2Y):** {current_steepness:.3f}%",
                                            f"- **Regime da curva:** {curve_regime}",
                                            f"- **Implicação econômica:** {economic_implication}",
                                            f"- **Implicação para criptomoedas:** {crypto_implication}"
                                        ])
                                        
                                        # Verificar se há inversão e quanto tempo durou
                                        if current_steepness < 0:
                                            inversion_periods = (treasury_data['yield_curve_steepness'] < 0).sum()
                                            if inversion_periods > 1:
                                                # Verificar se a inversão é contínua
                                                continuous_inversion = True
                                                for i in range(len(treasury_data) - inversion_periods, len(treasury_data)):
                                                    if treasury_data['yield_curve_steepness'].iloc[i] >= 0:
                                                        continuous_inversion = False
                                                        break
                                                
                                                if continuous_inversion:
                                                    steepness_analysis.append(f"- **Alerta de Inversão:** A curva está invertida por {inversion_periods} períodos consecutivos. Historicamente, inversões sustentadas da curva de rendimentos precedem recessões.")
                                                else:
                                                    steepness_analysis.append(f"- **Nota de Inversão:** A curva está atualmente invertida, mas a inversão não tem sido contínua. {inversion_periods} períodos de inversão foram observados recentemente.")
                                    
                                    # ANÁLISE 4: Velocidade de mudança
                                    if len(treasury_data) > 5:  # Precisamos de alguns pontos para calcular mudanças
                                        # Olhar para mudanças nos rendimentos de 2Y e 10Y
                                        for tenor_key in [2, 10]:
                                            if tenor_key in tenor_col_map and tenor_col_map[tenor_key]:
                                                col = tenor_col_map[tenor_key]
                                                recent_change = treasury_data[col].iloc[-1] - treasury_data[col].iloc[-5]
                                                pct_change = (recent_change / treasury_data[col].iloc[-5]) * 100 if treasury_data[col].iloc[-5] != 0 else 0
                                                
                                                direction = "subido" if recent_change > 0 else "caído"
                                                magnitude = "significativamente" if abs(pct_change) > 5 else "moderadamente" if abs(pct_change) > 2 else "levemente"
                                                
                                                steepness_analysis.append(f"- **Mudança recente no rendimento de {tenor_key}Y:** Tem {direction} {magnitude} ({pct_change:.1f}%)")
                                    
                                    # Exibir a análise de inclinação
                                    if steepness_analysis:
                                        st.info("**Análise da Inclinação da Curva de Rendimentos:**\n\n" + "\n".join(steepness_analysis))
                                    
                                    # Explicação educativa sobre a curva de rendimentos
                                    st.markdown("""
                                    **Interpretação da Curva de Rendimentos e Implicações para Criptomoedas:**
                                    
                                    A curva de rendimentos do Tesouro dos EUA é um indicador econômico crucial que mostra a relação entre as taxas de juros e os prazos de vencimento dos títulos do governo.
                                    
                                    **Formas da curva e seus significados:**
                                    
                                    1. **Curva normal (inclinação positiva):**
                                       - Rendimentos de longo prazo maiores que os de curto prazo
                                       - Indica expectativa de crescimento econômico saudável
                                       - Geralmente positivo para ativos de risco, incluindo criptomoedas
                                    
                                    2. **Curva achatada:**
                                       - Pequena diferença entre rendimentos de curto e longo prazo
                                       - Indica incerteza sobre perspectivas econômicas
                                       - Sugere possível fim de ciclo de alta ou mudança na política monetária
                                    
                                    3. **Curva invertida (inclinação negativa):**
                                       - Rendimentos de curto prazo maiores que os de longo prazo
                                       - Forte preditor histórico de recessões futuras
                                       - Geralmente sinal de cautela para ativos de risco
                                    
                                    **Como isso afeta o Bitcoin e criptomoedas:**
                                    
                                    - **Liquidez e políticas monetárias:** Curvas de rendimento refletem expectativas de política monetária; políticas expansionistas tendem a beneficiar ativos como Bitcoin
                                    - **Alocação de risco:** Em ambientes de curva normal, investidores tendem a aumentar exposição a ativos de risco
                                    - **Indicador antecipado:** A inversão da curva geralmente precede períodos de turbulência, que podem afetar criptomoedas num primeiro momento, mas criar condições para valorização posterior com políticas monetárias mais flexíveis
                                    
                                    A correlação entre taxas de juros e preços de criptomoedas se tornou mais forte nos últimos anos, com o Bitcoin reagindo de maneira mais próxima às expectativas de mercado sobre ações do Federal Reserve e condições macroeconômicas.
                                    """)
                            else:
                                st.error("Dados do Tesouro não contêm as colunas necessárias/válidas ou não há dados suficientes.")
                                st.write("Colunas necessárias:", ['date/timestamp', 'yield_columns'])
                                st.write("Colunas disponíveis:", list(treasury_data.columns))
                    else:
                        st.error("Colunas necessárias não encontradas nos dados do Tesouro")
                        st.write("Colunas necessárias:", ['date/timestamp', 'yield_columns'])
                        st.write("Colunas disponíveis:", list(treasury_data.columns))
                else:
                    st.error("Não foi possível processar os dados do Tesouro")
            
        except Exception as e:
            st.error(f"Erro na análise governamental: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def analyze_ohlc_data(self, symbol="BTCUSDT"):
        """
        Analisa dados OHLC com indicadores técnicos e análise de padrões.
        """
        st.markdown("### 📊 Análise de Dados OHLC")
        
        try:
            # Carregar dados
            spot_data = self.fetcher.load_data_file(f"ohlc/binance_spot_{symbol}_daily.csv")
            futures_data = self.fetcher.load_data_file(f"ohlc/binance_futures_{symbol}_daily.csv")
            
            st.info(f"Arquivos consultados: binance_spot_{symbol}_daily.csv, binance_futures_{symbol}_daily.csv")
            
            # Processar dados spot
            if spot_data is not None:
                spot_data = self._process_api_data(spot_data)
                if not spot_data.empty:
                    st.success(f"Dados spot processados com sucesso. Colunas: {list(spot_data.columns)}")
                else:
                    st.error("Não foi possível processar os dados spot")
                    spot_data = None # Set to None if empty
            
            # Processar dados futures
            if futures_data is not None:
                futures_data = self._process_api_data(futures_data)
                if not futures_data.empty:
                    st.success(f"Dados futures processados com sucesso. Colunas: {list(futures_data.columns)}")
                else:
                    st.error("Não foi possível processar os dados futures")
                    futures_data = None # Set to None if empty
            
            if spot_data is not None and futures_data is not None:
                required_cols = ['timestamp', 'open', 'high', 'low', 'close']
                col_mappings = {
                    'timestamp': ['timestamp', 'date', 'time', 'Date', 'Time'],
                    'open': ['open', 'open_price', 'opening_price', 'Open'],
                    'high': ['high', 'high_price', 'highest_price', 'High'],
                    'low': ['low', 'low_price', 'lowest_price', 'Low'],
                    'close': ['close', 'close_price', 'closing_price', 'Close']
                }
                
                def find_column(df, possible_names):
                    return next((col for col in possible_names if col in df.columns), None)
                
                spot_cols = {req: find_column(spot_data, options) for req, options in col_mappings.items()}
                futures_cols = {req: find_column(futures_data, options) for req, options in col_mappings.items()}
                
                vol_col_spot = find_column(spot_data, ['volume', 'Volume']) if spot_data is not None else None
                vol_col_futures = find_column(futures_data, ['volume', 'Volume']) if futures_data is not None else None
                
                if all(spot_cols.values()) and all(futures_cols.values()):
                    # Convert data types and drop NaNs
                    for df, cols in [(spot_data, spot_cols), (futures_data, futures_cols)]:
                        for col_key, col_name in cols.items():
                            if col_key == 'timestamp':
                                if df[col_name].dtype == 'object':
                                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                            else:
                                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                        df.dropna(subset=list(cols.values()), inplace=True)
                    
                    # Converter volume também se existir
                    if vol_col_spot and vol_col_spot in spot_data.columns:
                        spot_data[vol_col_spot] = pd.to_numeric(spot_data[vol_col_spot], errors='coerce')
                    if vol_col_futures and vol_col_futures in futures_data.columns:
                        futures_data[vol_col_futures] = pd.to_numeric(futures_data[vol_col_futures], errors='coerce')
                    
                    if spot_data.empty or futures_data.empty:
                         st.error("Dados OHLC insuficientes após limpeza.")
                         return

                    with st.expander("📈 Preços Spot e Indicadores Técnicos", expanded=True):
                        # Preparar dados para análise técnica
                        df_tech = spot_data.copy()
                        df_tech = df_tech.rename(columns={
                            spot_cols['timestamp']: 'timestamp',
                            spot_cols['open']: 'open',
                            spot_cols['high']: 'high',
                            spot_cols['low']: 'low',
                            spot_cols['close']: 'close'
                        })
                        if vol_col_spot:
                            df_tech = df_tech.rename(columns={vol_col_spot: 'volume'})
                        
                        # Calcular indicadores técnicos
                        # 1. RSI
                        rsi = RSIIndicator(close=df_tech['close'], window=14)
                        df_tech['rsi'] = rsi.rsi()
                        
                        # 2. MACD
                        macd = MACD(close=df_tech['close'])
                        df_tech['macd'] = macd.macd()
                        df_tech['macd_signal'] = macd.macd_signal()
                        df_tech['macd_histogram'] = macd.macd_diff()
                        
                        # 3. Bollinger Bands
                        bollinger = BollingerBands(close=df_tech['close'], window=20, window_dev=2)
                        df_tech['bb_upper'] = bollinger.bollinger_hband()
                        df_tech['bb_middle'] = bollinger.bollinger_mavg()
                        df_tech['bb_lower'] = bollinger.bollinger_lband()
                        
                        # 4. Médias Móveis
                        df_tech['sma_50'] = SMAIndicator(close=df_tech['close'], window=50).sma_indicator()
                        df_tech['sma_200'] = SMAIndicator(close=df_tech['close'], window=200).sma_indicator()
                        df_tech['ema_20'] = EMAIndicator(close=df_tech['close'], window=20).ema_indicator()
                        
                        # 5. On-Balance Volume (OBV) se o volume estiver disponível
                        if 'volume' in df_tech.columns:
                            obv = OnBalanceVolumeIndicator(close=df_tech['close'], volume=df_tech['volume'])
                            df_tech['obv'] = obv.on_balance_volume()
                        
                        # Identificar padrões de candlestick (simplificado)
                        # Doji
                        df_tech['doji'] = ((abs(df_tech['close'] - df_tech['open']) / (df_tech['high'] - df_tech['low'])) < 0.1) & \
                                        ((df_tech['high'] - df_tech['low']) > (2 * abs(df_tech['close'] - df_tech['open'])))
                        
                        # Hammer (high-low range > 3x open-close range, small upper shadow)
                        df_tech['hammer'] = ((df_tech['high'] - df_tech['low']) > (3 * abs(df_tech['close'] - df_tech['open']))) & \
                                         ((df_tech['high'] - df_tech['close']) < (0.3 * (df_tech['high'] - df_tech['low']))) & \
                                         ((df_tech['open'] - df_tech['low']) > (0.6 * (df_tech['high'] - df_tech['low'])))
                        
                        # Shooting Star (high-low range > 3x open-close range, small lower shadow)
                        df_tech['shooting_star'] = ((df_tech['high'] - df_tech['low']) > (3 * abs(df_tech['close'] - df_tech['open']))) & \
                                                ((df_tech['close'] - df_tech['low']) < (0.3 * (df_tech['high'] - df_tech['low']))) & \
                                                ((df_tech['high'] - df_tech['open']) > (0.6 * (df_tech['high'] - df_tech['low'])))
                        
                        # Gráfico de preços spot com indicadores
                        fig = make_subplots(rows=3, cols=1, 
                                          shared_xaxes=True,
                                          vertical_spacing=0.05,
                                          row_heights=[0.6, 0.2, 0.2],
                                          subplot_titles=("Preço e Indicadores", "Volume", "RSI (14)"))
                        
                        # Candlestick chart
                        fig.add_trace(
                            go.Candlestick(
                                x=df_tech['timestamp'],
                                open=df_tech['open'],
                                high=df_tech['high'],
                                low=df_tech['low'],
                                close=df_tech['close'],
                                name="OHLC",
                                increasing_line_color=self.color_palette['bullish'],
                                decreasing_line_color=self.color_palette['bearish']
                            ),
                            row=1, col=1
                        )
                        
                        # Adicionar Bollinger Bands
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['bb_upper'],
                                line=dict(color='rgba(250, 128, 114, 0.7)', width=1),
                                name="BB Superior",
                                hoverinfo='skip'
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['bb_middle'],
                                line=dict(color='rgba(184, 134, 11, 0.7)', width=1, dash='dash'),
                                name="BB Média (SMA20)",
                                hoverinfo='skip'
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['bb_lower'],
                                line=dict(color='rgba(152, 251, 152, 0.7)', width=1),
                                name="BB Inferior",
                                hoverinfo='skip',
                                fill='tonexty',
                                fillcolor='rgba(230, 230, 250, 0.2)'
                            ),
                            row=1, col=1
                        )
                        
                        # Adicionar SMAs/EMAs
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['sma_50'],
                                line=dict(color='rgba(255, 69, 0, 0.7)', width=1.5),
                                name="SMA 50"
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['sma_200'],
                                line=dict(color='rgba(0, 0, 139, 0.7)', width=1.5),
                                name="SMA 200"
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['ema_20'],
                                line=dict(color='rgba(138, 43, 226, 0.7)', width=1.5, dash='dot'),
                                name="EMA 20"
                            ),
                            row=1, col=1
                        )
                        
                        # Adicionar padrões de candlestick como marcadores
                        doji_df = df_tech[df_tech['doji']]
                        hammer_df = df_tech[df_tech['hammer']]
                        star_df = df_tech[df_tech['shooting_star']]
                        
                        # Adicionar marcadores para Doji
                        if not doji_df.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=doji_df['timestamp'],
                                    y=doji_df['low'] * 0.995,  # Slightly below the candle
                                    mode='markers',
                                    marker=dict(symbol='star', size=8, color='purple'),
                                    name='Doji',
                                    hoverinfo='text',
                                    hovertext='Doji: Indecisão no mercado'
                                ),
                                row=1, col=1
                            )
                        
                        # Adicionar marcadores para Hammer
                        if not hammer_df.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=hammer_df['timestamp'],
                                    y=hammer_df['low'] * 0.99,  # Slightly below the candle
                                    mode='markers',
                                    marker=dict(symbol='triangle-up', size=10, color='green'),
                                    name='Hammer',
                                    hoverinfo='text',
                                    hovertext='Hammer: Possível reversão de baixa'
                                ),
                                row=1, col=1
                            )
                        
                        # Adicionar marcadores para Shooting Star
                        if not star_df.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=star_df['timestamp'],
                                    y=star_df['high'] * 1.01,  # Slightly above the candle
                                    mode='markers',
                                    marker=dict(symbol='triangle-down', size=10, color='red'),
                                    name='Shooting Star',
                                    hoverinfo='text',
                                    hovertext='Shooting Star: Possível reversão de alta'
                                ),
                                row=1, col=1
                            )
                        
                        # Volume
                        if 'volume' in df_tech.columns:
                            colors = np.where(df_tech['close'] >= df_tech['open'], self.color_palette['bullish'], self.color_palette['bearish'])
                            fig.add_trace(
                                go.Bar(
                                    x=df_tech['timestamp'],
                                    y=df_tech['volume'],
                                    marker_color=colors,
                                    name="Volume",
                                    opacity=0.8
                                ),
                                row=2, col=1
                            )
                            
                            # Adicionar média móvel do volume
                            vol_sma = df_tech['volume'].rolling(window=20).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=df_tech['timestamp'],
                                    y=vol_sma,
                                    line=dict(color='rgba(100, 149, 237, 0.7)', width=1.5),
                                    name="Volume SMA 20"
                                ),
                                row=2, col=1
                            )
                        
                        # RSI
                        fig.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['rsi'],
                                line=dict(color='rgba(0, 128, 0, 1)'),
                                name="RSI (14)"
                            ),
                            row=3, col=1
                        )
                        
                        # Adicionar linhas de referência para RSI
                        fig.add_shape(
                            type="line", line_color="red", line_width=1, line_dash="dash",
                            x0=df_tech['timestamp'].iloc[0], x1=df_tech['timestamp'].iloc[-1], y0=70, y1=70,
                            row=3, col=1
                        )
                        
                        fig.add_shape(
                            type="line", line_color="green", line_width=1, line_dash="dash",
                            x0=df_tech['timestamp'].iloc[0], x1=df_tech['timestamp'].iloc[-1], y0=30, y1=30,
                            row=3, col=1
                        )
                        
                        fig.add_shape(
                            type="line", line_color="gray", line_width=1, line_dash="dot",
                            x0=df_tech['timestamp'].iloc[0], x1=df_tech['timestamp'].iloc[-1], y0=50, y1=50,
                            row=3, col=1
                        )
                        
                        # Atualizar layout
                        fig.update_layout(
                            title=f"{symbol} - Análise Técnica",
                            xaxis_rangeslider_visible=False,
                            xaxis_title="Data",
                            yaxis_title="Preço",
                            yaxis3_title="RSI",
                            legend=dict(orientation="h", y=1.02),
                            height=800,
                            hovermode='x unified'
                        )
                        
                        # Atualizar ranges do RSI e outras configurações
                        fig.update_yaxes(range=[0, 100], row=3, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Análise do MACD em um gráfico separado
                        fig_macd = make_subplots(rows=1, cols=1)
                        
                        fig_macd.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['macd'],
                                line=dict(color='blue', width=1.5),
                                name="MACD Line"
                            )
                        )
                        
                        fig_macd.add_trace(
                            go.Scatter(
                                x=df_tech['timestamp'],
                                y=df_tech['macd_signal'],
                                line=dict(color='red', width=1.5),
                                name="Signal Line"
                            )
                        )
                        
                        fig_macd.add_trace(
                            go.Bar(
                                x=df_tech['timestamp'],
                                y=df_tech['macd_histogram'],
                                marker=dict(
                                    color=np.where(df_tech['macd_histogram'] >= 0, 'green', 'red'),
                                    line=dict(color='rgba(0, 0, 0, 0)', width=0)
                                ),
                                name="Histogram"
                            )
                        )
                        
                        fig_macd.update_layout(
                            title="Indicador MACD",
                            xaxis_title="Data",
                            yaxis_title="MACD",
                            hovermode='x unified',
                            legend=dict(orientation="h", y=1.1)
                        )
                        
                        st.plotly_chart(fig_macd, use_container_width=True)
                        
                        # Análise atual dos indicadores e sinais
                        st.markdown("#### 📊 Análise dos Indicadores Técnicos")
                        
                        # Pegar últimos valores dos indicadores
                        last_close = df_tech['close'].iloc[-1]
                        last_sma50 = df_tech['sma_50'].iloc[-1]
                        last_sma200 = df_tech['sma_200'].iloc[-1]
                        last_ema20 = df_tech['ema_20'].iloc[-1]
                        last_rsi = df_tech['rsi'].iloc[-1]
                        last_macd = df_tech['macd'].iloc[-1]
                        last_macd_signal = df_tech['macd_signal'].iloc[-1]
                        last_macd_hist = df_tech['macd_histogram'].iloc[-1]
                        last_bb_upper = df_tech['bb_upper'].iloc[-1]
                        last_bb_lower = df_tech['bb_lower'].iloc[-1]
                        last_bb_width = (last_bb_upper - last_bb_lower) / df_tech['bb_middle'].iloc[-1]
                        
                        # Sinais de tendência
                        trend_signals = []
                        if last_close > last_sma50:
                            trend_signals.append({"sinal": "Bullish", "descrição": "Preço acima da SMA 50", "força": "Moderado"})
                        else:
                            trend_signals.append({"sinal": "Bearish", "descrição": "Preço abaixo da SMA 50", "força": "Moderado"})
                            
                        if last_close > last_sma200:
                            trend_signals.append({"sinal": "Bullish", "descrição": "Preço acima da SMA 200", "força": "Forte"})
                        else:
                            trend_signals.append({"sinal": "Bearish", "descrição": "Preço abaixo da SMA 200", "força": "Forte"})
                            
                        if last_sma50 > last_sma200:
                            trend_signals.append({"sinal": "Bullish", "descrição": "Golden Cross (SMA50 acima da SMA200)", "força": "Forte"})
                        elif last_sma50 < last_sma200:
                            trend_signals.append({"sinal": "Bearish", "descrição": "Death Cross (SMA50 abaixo da SMA200)", "força": "Forte"})
                            
                        if last_close > last_ema20:
                            trend_signals.append({"sinal": "Bullish", "descrição": "Preço acima da EMA 20", "força": "Fraco"})
                        else:
                            trend_signals.append({"sinal": "Bearish", "descrição": "Preço abaixo da EMA 20", "força": "Fraco"})
                        
                        # Sinais de momentum
                        momentum_signals = []
                        if last_rsi > 70:
                            momentum_signals.append({"sinal": "Bearish", "descrição": "RSI em sobrecompra (> 70)", "força": "Moderado"})
                        elif last_rsi < 30:
                            momentum_signals.append({"sinal": "Bullish", "descrição": "RSI em sobrevenda (< 30)", "força": "Moderado"})
                        elif last_rsi > 50:
                            momentum_signals.append({"sinal": "Bullish", "descrição": "RSI acima de 50 (momentum positivo)", "força": "Fraco"})
                        else:
                            momentum_signals.append({"sinal": "Bearish", "descrição": "RSI abaixo de 50 (momentum negativo)", "força": "Fraco"})
                            
                        if last_macd > last_macd_signal:
                            momentum_signals.append({"sinal": "Bullish", "descrição": "MACD acima da linha de sinal", "força": "Moderado"})
                        else:
                            momentum_signals.append({"sinal": "Bearish", "descrição": "MACD abaixo da linha de sinal", "força": "Moderado"})
                            
                        if last_macd_hist > 0 and df_tech['macd_histogram'].iloc[-2] <= 0:
                            momentum_signals.append({"sinal": "Bullish", "descrição": "Cruzamento MACD recente (histograma positivo)", "força": "Forte"})
                        elif last_macd_hist < 0 and df_tech['macd_histogram'].iloc[-2] >= 0:
                            momentum_signals.append({"sinal": "Bearish", "descrição": "Cruzamento MACD recente (histograma negativo)", "força": "Forte"})
                            
                        # Sinais de volatilidade
                        volatility_signals = []
                        if last_close > last_bb_upper:
                            volatility_signals.append({"sinal": "Bearish", "descrição": "Preço acima da banda superior de Bollinger", "força": "Moderado"})
                        elif last_close < last_bb_lower:
                            volatility_signals.append({"sinal": "Bullish", "descrição": "Preço abaixo da banda inferior de Bollinger", "força": "Moderado"})
                            
                        # Analisar largura das bandas (comparando com média recente)
                        avg_bb_width = pd.Series([
                            (df_tech['bb_upper'].iloc[i] - df_tech['bb_lower'].iloc[i]) / df_tech['bb_middle'].iloc[i]
                            for i in range(-20, 0)
                        ]).mean()
                        
                        if last_bb_width < avg_bb_width * 0.8:
                            volatility_signals.append({"sinal": "Neutro", "descrição": "Contração das Bandas de Bollinger (possível movimento brusco em breve)", "força": "Moderado"})
                        elif last_bb_width > avg_bb_width * 1.2:
                            volatility_signals.append({"sinal": "Neutro", "descrição": "Expansão das Bandas de Bollinger (alta volatilidade)", "força": "Moderado"})
                        
                        # Resumir resultados em tabelas
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("Sinais de Tendência")
                            trend_df = pd.DataFrame(trend_signals)
                            st.table(trend_df)
                            
                        with col2:
                            st.subheader("Sinais de Momentum")
                            momentum_df = pd.DataFrame(momentum_signals)
                            st.table(momentum_df)
                            
                        with col3:
                            st.subheader("Sinais de Volatilidade")
                            volatility_df = pd.DataFrame(volatility_signals)
                            st.table(volatility_df)
                        
                        # Resumo da análise técnica
                        bullish_signals = sum(1 for signal in trend_signals + momentum_signals + volatility_signals if signal["sinal"] == "Bullish")
                        bearish_signals = sum(1 for signal in trend_signals + momentum_signals + volatility_signals if signal["sinal"] == "Bearish")
                        
                        # Calcular pontuação ponderada com base na força do sinal
                        force_weights = {"Forte": 3, "Moderado": 2, "Fraco": 1}
                        bullish_score = sum(force_weights[signal["força"]] for signal in trend_signals + momentum_signals + volatility_signals if signal["sinal"] == "Bullish")
                        bearish_score = sum(force_weights[signal["força"]] for signal in trend_signals + momentum_signals + volatility_signals if signal["sinal"] == "Bearish")
                        
                        # Determinar viés geral
                        if bullish_score > bearish_score * 1.5:
                            overall_bias = "Fortemente Bullish"
                        elif bullish_score > bearish_score:
                            overall_bias = "Moderadamente Bullish"
                        elif bearish_score > bullish_score * 1.5:
                            overall_bias = "Fortemente Bearish"
                        elif bearish_score > bullish_score:
                            overall_bias = "Moderadamente Bearish"
                        else:
                            overall_bias = "Neutro"
                        
                        # Exibir resumo
                        st.markdown(f"""
                        ### 📝 Resumo da Análise Técnica
                        
                        - **Viés Geral:** {overall_bias}
                        - **Sinais Bullish:** {bullish_signals} (Pontuação ponderada: {bullish_score})
                        - **Sinais Bearish:** {bearish_signals} (Pontuação ponderada: {bearish_score})
                        - **Último Preço:** ${last_close:.2f}
                        
                        **Níveis Técnicos Importantes:**
                        - SMA 50: ${last_sma50:.2f}
                        - SMA 200: ${last_sma200:.2f}
                        - EMA 20: ${last_ema20:.2f}
                        - Banda Superior Bollinger: ${last_bb_upper:.2f}
                        - Banda Inferior Bollinger: ${last_bb_lower:.2f}
                        
                        **Indicadores de Momentum:**
                        - RSI (14): {last_rsi:.2f}
                        - MACD: {last_macd:.4f}
                        - Linha de Sinal MACD: {last_macd_signal:.4f}
                        - Histograma MACD: {last_macd_hist:.4f}
                        """)
                        
                        # Educational content about technical analysis
                        with st.expander("📚 Entendendo a Análise Técnica", expanded=False):
                            st.markdown("""
                            ### 📚 Fundamentos da Análise Técnica para Criptomoedas
                            
                            A análise técnica é um método para prever movimentos de preços com base em padrões históricos, usando gráficos e indicadores estatísticos.
                            
                            #### 🔍 Principais Indicadores:
                            
                            **1. Médias Móveis (MAs):**
                            - **SMA (Simple Moving Average):** Média aritmética dos preços em um período específico.
                            - **EMA (Exponential Moving Average):** Dá mais peso aos preços recentes.
                            - **Cruzamentos:** Golden Cross (SMA 50 cruza acima da SMA 200) é bullish; Death Cross (SMA 50 cruza abaixo da SMA 200) é bearish.
                            
                            **2. RSI (Relative Strength Index):**
                            - Oscilador de momentum que mede a velocidade e magnitude das mudanças de preço.
                            - Escala de 0-100: Valores acima de 70 indicam sobrecompra; abaixo de 30 indicam sobrevenda.
                            - Divergências entre RSI e preço podem sinalizar reversões potenciais.
                            
                            **3. MACD (Moving Average Convergence Divergence):**
                            - Combina duas EMAs (geralmente 12 e 26 períodos) com uma linha de sinal (geralmente EMA de 9 períodos).
                            - Cruzamentos da linha MACD acima/abaixo da linha de sinal indicam potenciais entradas/saídas.
                            - O histograma mostra a diferença entre MACD e sua linha de sinal.
                            
                            **4. Bandas de Bollinger:**
                            - Consiste em uma SMA central (geralmente 20 períodos) e bandas superior/inferior (geralmente 2 desvios padrão).
                            - Preços tendem a retornar à média; tocar as bandas pode indicar extremos.
                            - Contração das bandas sugere baixa volatilidade (geralmente precede grandes movimentos).
                            
                            #### 📊 Padrões de Candlestick:
                            
                            - **Doji:** Candle com corpo pequeno (abertura ≈ fechamento), mostrando indecisão.
                            - **Hammer (Martelo):** Candle com sombra inferior longa e corpo pequeno no topo, geralmente sinal de reversão em baixas.
                            - **Shooting Star (Estrela Cadente):** Candle com sombra superior longa e corpo pequeno na base, geralmente sinal de reversão em altas.
                            
                            #### ⚠️ Limitações da Análise Técnica:
                            
                            - Não considera fundamentos ou eventos externos.
                            - Pode gerar falsos sinais em mercados altamente voláteis.
                            - Funciona melhor em combinação com análise fundamental e análise on-chain.
                            - Diferentes timeframes podem gerar sinais conflitantes.
                            
                            #### 💡 Melhores Práticas:
                            
                            - Use múltiplos indicadores para confirmação.
                            - Considere o contexto de mercado mais amplo.
                            - Adapte sua estratégia com base no timeframe e volatilidade.
                            - Estabeleça regras claras para gerenciamento de risco.
                            """)

                    with st.expander("📉 Análise de Dados Futures e Base", expanded=True):
                        # Gráfico de preços futures
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=futures_data[futures_cols['timestamp']],
                            open=futures_data[futures_cols['open']],
                            high=futures_data[futures_cols['high']],
                            low=futures_data[futures_cols['low']],
                            close=futures_data[futures_cols['close']],
                            name='Futures',
                            increasing_line_color=self.color_palette['bullish'],
                            decreasing_line_color=self.color_palette['bearish']
                        ))
                        fig.update_layout(
                            title=f"{symbol} - Preços Futures",
                            xaxis_title="Data", 
                            yaxis_title="Preço", 
                            hovermode='x unified',
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calcular e mostrar base
                        merged = pd.merge(
                            spot_data[[spot_cols['timestamp'], spot_cols['close']]].rename(
                                columns={spot_cols['close']: 'spot_close', spot_cols['timestamp']: 'timestamp'}),
                            futures_data[[futures_cols['timestamp'], futures_cols['close']]].rename(
                                columns={futures_cols['close']: 'futures_close', futures_cols['timestamp']: 'timestamp'}),
                            on='timestamp',
                            how='inner'
                        )
                        
                        if not merged.empty:
                            # Calcular base em percentual
                            merged['base'] = ((merged['futures_close'] - merged['spot_close']) / 
                                            merged['spot_close'].replace(0, 1) * 100)
                            
                            # Calcular estatísticas adicionais
                            # 1. Média móvel da base
                            merged['base_ma7'] = merged['base'].rolling(window=7).mean()
                            merged['base_ma30'] = merged['base'].rolling(window=30).mean()
                            
                            # 2. Volatilidade da base (desvio padrão móvel)
                            merged['base_std7'] = merged['base'].rolling(window=7).std()
                            
                            # 3. Z-score da base (para identificar extremos)
                            if len(merged) >= 60:  # Precisa de pelo menos 60 pontos para Z-score significativo
                                lookback = 60
                                merged['base_zscore'] = (
                                    merged['base'] - merged['base'].rolling(lookback).mean()
                                ) / merged['base'].rolling(lookback).std()
                            
                            # Gráfico de base mais detalhado
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=merged['timestamp'],
                                y=merged['base'],
                                mode='lines', 
                                name='Base %',
                                line=dict(color=self.color_palette['primary'], width=2)
                            ))
                            
                            # Adicionar médias móveis
                            fig.add_trace(go.Scatter(
                                x=merged['timestamp'],
                                y=merged['base_ma7'],
                                mode='lines', 
                                name='MM7 da Base',
                                line=dict(color=self.color_palette['secondary'], width=1.5)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=merged['timestamp'],
                                y=merged['base_ma30'],
                                mode='lines', 
                                name='MM30 da Base',
                                line=dict(color=self.color_palette['tertiary'], width=1.5, dash='dot')
                            ))
                            
                            # Adicionar área sombreada para volatilidade
                            fig.add_trace(go.Scatter(
                                x=merged['timestamp'],
                                y=merged['base_ma7'] + merged['base_std7'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=merged['timestamp'],
                                y=merged['base_ma7'] - merged['base_std7'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(173, 216, 230, 0.2)',
                                name='±1 Desvio Padrão',
                                hoverinfo='skip'
                            ))
                            
                            fig.update_layout(
                                title=f"{symbol} - Base (Futures - Spot) %",
                                xaxis_title="Data", 
                                yaxis_title="Base %", 
                                hovermode='x unified',
                                legend=dict(orientation="h", y=1.1)
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            # Adicionar linhas de referência para extremos históricos se tivermos dados suficientes
                            if len(merged) > 30:
                                percentile_95 = merged['base'].quantile(0.95)
                                percentile_05 = merged['base'].quantile(0.05)
                                
                                fig.add_hline(y=percentile_95, line_dash="dot", line_color="red", opacity=0.5)
                                fig.add_hline(y=percentile_05, line_dash="dot", line_color="green", opacity=0.5)
                                
                                fig.add_annotation(
                                    x=merged['timestamp'].iloc[-1],
                                    y=percentile_95,
                                    text="95º Percentil",
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(color="red")
                                )
                                
                                fig.add_annotation(
                                    x=merged['timestamp'].iloc[-1],
                                    y=percentile_05,
                                    text="5º Percentil",
                                    showarrow=False,
                                    yshift=-10,
                                    font=dict(color="green")
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Estatísticas recentes
                            latest_spot = float(spot_data[spot_cols['close']].iloc[-1])
                            latest_futures = float(futures_data[futures_cols['close']].iloc[-1])
                            latest_base = (latest_futures - latest_spot) / latest_spot * 100 if latest_spot else 0
                            latest_base_ma7 = float(merged['base_ma7'].iloc[-1]) if 'base_ma7' in merged and not pd.isna(merged['base_ma7'].iloc[-1]) else float('nan')
                            latest_base_std7 = float(merged['base_std7'].iloc[-1]) if 'base_std7' in merged and not pd.isna(merged['base_std7'].iloc[-1]) else float('nan')
                            latest_base_zscore = float(merged['base_zscore'].iloc[-1]) if 'base_zscore' in merged and not pd.isna(merged['base_zscore'].iloc[-1]) else float('nan')
                            
                            # Determinar regime atual da base
                            base_regime = ""
                            if latest_base > 0.5:
                                base_regime = "Contango Significativo"
                                base_implication = "Mercado otimista, expectativa de preços mais altos. Estratégia carry trade positiva."
                            elif latest_base > 0:
                                base_regime = "Contango Leve"
                                base_implication = "Viés levemente otimista no mercado."
                            elif latest_base > -0.5:
                                base_regime = "Backwardation Leve"
                                base_implication = "Viés levemente pessimista, possível demanda por liquidação."
                            else:
                                base_regime = "Backwardation Significativo"
                                base_implication = "Mercado pessimista ou escassez na oferta spot. Pode indicar condições de sobrevenda."
                            
                            # Avaliar extremos via Z-score (se disponível)
                            extreme_signal = ""
                            if not pd.isna(latest_base_zscore):
                                if latest_base_zscore > 2:
                                    extreme_signal = "⚠️ Base extremamente alta (>2σ). Potencial oportunidade de arbitragem ou sinal de excesso de otimismo."
                                elif latest_base_zscore < -2:
                                    extreme_signal = "⚠️ Base extremamente baixa (<-2σ). Potencial oportunidade de arbitragem reversa ou sinal de pessimismo excessivo."
                            
                            # Métricas em colunas
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: 
                                st.metric("Último Preço Spot", f"${latest_spot:.2f}")
                            with col2: 
                                st.metric("Último Preço Futures", f"${latest_futures:.2f}")
                            with col3: 
                                st.metric("Base Atual", f"{latest_base:.2f}%", base_regime)
                            with col4: 
                                if not pd.isna(latest_base_std7):
                                    st.metric("Volatilidade Base (7d)", f"{latest_base_std7:.2f}%")
                                elif not pd.isna(latest_base_zscore):
                                    st.metric("Z-Score da Base", f"{latest_base_zscore:.2f}")
                                else:
                                    st.metric("Média Base (7d)", f"{latest_base_ma7:.2f}%")
                            
                            # Análise detalhada
                            st.info(f"""
                            **Análise da Base (Futures - Spot):**
                            
                            - **Regime atual:** {base_regime}
                            - **Implicação:** {base_implication}
                            {f"- **{extreme_signal}**" if extreme_signal else ""}
                            - **Volatilidade recente (7d):** {latest_base_std7:.2f}%
                            """)
                            
                            # Adicionar explicação educativa
                            with st.expander("📚 Entendendo o Significado da Base", expanded=False):
                                st.markdown("""
                                ### 📚 Interpretação da Base (Futures - Spot)
                                
                                A "base" é a diferença percentual entre o preço dos contratos futuros e o preço spot. Esta métrica oferece insights valiosos sobre o sentimento do mercado e as condições de oferta/demanda.
                                
                                #### 🔍 Regimes de Base:
                                
                                **1. Contango (Base Positiva):**
                                - Ocorre quando preços futuros > preços spot
                                - **Interpretação:** Geralmente indica expectativa de preços mais altos no futuro
                                - **Implicações para traders:** Oportunidade para estratégias de "cash and carry" (comprar spot, vender futuro, lucrar com a convergência)
                                - **Por que acontece:** Custo de carregamento, expectativa de alta, demanda por alavancagem long
                                
                                **2. Backwardation (Base Negativa):**
                                - Ocorre quando preços futuros < preços spot
                                - **Interpretação:** Pode indicar escassez atual de oferta ou pessimismo sobre preços futuros
                                - **Implicações para traders:** Oportunidade para arbitragem reversa
                                - **Por que acontece:** Demanda incomum por liquidação imediata, escassez de oferta, mercado fortemente bearish
                                
                                #### 📊 Métricas Importantes:
                                
                                - **Base Absoluta:** Magnitude da diferença percentual entre futures e spot
                                - **Volatilidade da Base:** Instabilidade na relação, geralmente aumenta em períodos de incerteza
                                - **Z-Score da Base:** Quantifica quão extrema é a base atual em relação à média histórica (valores >2 ou <-2 indicam extremos)
                                
                                #### 💡 Estratégias Baseadas na Base:
                                
                                - **Arbitragem Estatística:** Negociar quando a base atinge extremos estatísticos (alto Z-score)
                                - **Hedge com Base:** Usar a base para ajustar estratégias de hedge
                                - **Carry Trade:** Capturar a convergência da base próximo ao vencimento dos futuros
                                
                                #### ⚠️ Considerações Importantes:
                                
                                - A base tende a convergir para zero à medida que o contrato futuro se aproxima do vencimento
                                - Mudanças bruscas na base podem sinalizar mudanças fundamentais no mercado
                                - Diferentes exchanges podem ter bases diferentes, criando oportunidades de arbitragem entre plataformas
                                """)
                        else:
                            st.warning("Não foi possível calcular a base (sem dados correspondentes).")
                else:
                    st.error("Colunas necessárias não encontradas nos dados OHLC após limpeza")
                    st.write("Colunas necessárias:", required_cols)
                    st.write("Colunas encontradas (Spot):", spot_cols)
                    st.write("Colunas encontradas (Futures):", futures_cols)
            
        except Exception as e:
            st.error(f"Erro na análise OHLC: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def analyze_risk_metrics(self, symbol="BTCUSDT"):
        """
        Analisa métricas de risco com interpretações avançadas.
        """
        st.markdown("### ⚠️ Análise de Métricas de Risco")
        
        try:
            # Carregar dados
            correlations = self.fetcher.load_data_file(f"risk/correlations_{symbol}.csv")
            var_data = self.fetcher.load_data_file(f"risk/var_{symbol}.csv")
            
            st.info(f"Arquivos consultados: risk/correlations_{symbol}.csv, risk/var_{symbol}.csv")
            
            # Processar dados de correlação
            if correlations is not None:
                correlations = self._process_api_data(correlations)
                if not correlations.empty:
                    st.success(f"Dados de correlação processados com sucesso. Colunas: {list(correlations.columns)}")
                    
                    pair_col = next((col for col in ['Pair', 'pair', 'asset1', 'asset'] if col in correlations.columns), None)
                    counter_col = next((col for col in ['CounterPair', 'counter_pair', 'asset2', 'counter_asset'] 
                                       if col in correlations.columns), None)
                    corr_col = next((col for col in ['Correlation', 'correlation', 'corr', 'value'] 
                                     if col in correlations.columns), None)
                    
                    if pair_col and counter_col and corr_col:
                        with st.expander("🔄 Análise de Correlações entre Ativos", expanded=True):
                            st.markdown("#### 🔄 Análise de Correlações")
                            
                            try:
                                corr_df = correlations.rename(columns={
                                    pair_col: 'Pair',
                                    counter_col: 'CounterPair',
                                    corr_col: 'Correlation'
                                })
                                corr_df['Correlation'] = pd.to_numeric(corr_df['Correlation'], errors='coerce')
                                corr_df.dropna(subset=['Pair', 'CounterPair', 'Correlation'], inplace=True)

                                if not corr_df.empty:
                                    # Criar matriz de correlação
                                    corr_matrix = pd.pivot_table(
                                        corr_df,
                                        values='Correlation',
                                        index='Pair',
                                        columns='CounterPair',
                                        fill_value=np.nan # Use NaN instead of 1.0 for missing
                                    )
                                    # Ensure diagonal is 1.0
                                    for idx in corr_matrix.index:
                                        if idx in corr_matrix.columns:
                                            corr_matrix.loc[idx, idx] = 1.0
                                    
                                    # Criar heatmap melhorado
                                    fig = px.imshow(
                                        corr_matrix,
                                        title="Mapa de Correlações entre Ativos",
                                        color_continuous_scale='RdBu_r',
                                        zmin=-1, zmax=1, 
                                        aspect='auto',
                                        labels=dict(x="Ativo", y="Ativo", color="Correlação")
                                    )
                                    
                                    # Melhorar o layout do gráfico
                                    fig.update_layout(
                                        height=600,
                                        hovermode='closest',
                                        coloraxis_colorbar=dict(
                                            title="Correlação",
                                            tickvals=[-1, -0.5, 0, 0.5, 1],
                                            ticktext=["-1.0", "-0.5", "0", "0.5", "1.0"]
                                        )
                                    )
                                    
                                    # Adicionar anotações com os valores no heatmap se não for muito grande
                                    if len(corr_matrix) <= 15 and len(corr_matrix.columns) <= 15:
                                        annotations = []
                                        for i, idx in enumerate(corr_matrix.index):
                                            for j, col in enumerate(corr_matrix.columns):
                                                if not pd.isna(corr_matrix.loc[idx, col]):
                                                    annotations.append(
                                                        dict(
                                                            x=col,
                                                            y=idx,
                                                            text=str(round(corr_matrix.loc[idx, col], 2)),
                                                            showarrow=False,
                                                            font=dict(
                                                                color="white" if abs(corr_matrix.loc[idx, col]) > 0.7 else "black"
                                                            )
                                                        )
                                                    )
                                        fig.update_layout(annotations=annotations)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Análise de Clusters (Agrupamento Hierárquico)
                                    st.markdown("#### 🧮 Análise de Clusters de Correlação")
                                    
                                    # Calcular clusters de correlação (se temos pelo menos 3 ativos)
                                    if len(corr_matrix) >= 3:
                                        # Preencher valores NaN com médias para a análise de cluster
                                        corr_matrix_filled = corr_matrix.fillna(corr_matrix.mean(axis=0))
                                        
                                        # Calcular distâncias de correlação (1 - abs(corr))
                                        corr_dist = 1 - np.abs(corr_matrix_filled.values)
                                        
                                        # Agrupar hierarquicamente
                                        from scipy.cluster.hierarchy import linkage, dendrogram
                                        Z = linkage(corr_dist, 'ward')
                                        
                                        # Plotar dendrograma
                                        fig = go.Figure()
                                        
                                        # Converter o dendrograma para um formato plotável no Plotly
                                        dendro_leaves = dendrogram(Z, labels=corr_matrix.index.tolist(), no_plot=True)
                                        
                                        # Extrair as coordenadas x, y
                                        dendro_leaves_idx = dendro_leaves['leaves']
                                        dendro_labels = [corr_matrix.index[i] for i in dendro_leaves_idx]
                                        
                                        xlocs = dendro_leaves['icoord']
                                        ylocs = dendro_leaves['dcoord']
                                        
                                        # Desenhar as linhas do dendrograma
                                        for i, (xi, yi) in enumerate(zip(xlocs, ylocs)):
                                            # Escalar x para melhor visualização
                                            xi_scaled = [(x - 5) / 10 for x in xi]
                                            fig.add_trace(go.Scatter(
                                                x=xi_scaled, 
                                                y=yi, 
                                                mode='lines',
                                                line=dict(color='black'),
                                                hoverinfo='skip',
                                                showlegend=False
                                            ))
                                        
                                        # Adicionar rótulos
                                        for i, label in enumerate(dendro_labels):
                                            fig.add_trace(go.Scatter(
                                                x=[i],
                                                y=[0],
                                                mode='text',
                                                text=label,
                                                textposition='bottom center',
                                                textfont=dict(size=12),
                                                hoverinfo='text',
                                                hovertext=f'Cluster: {label}',
                                                showlegend=False
                                            ))
                                        
                                        fig.update_layout(
                                            title='Dendrograma de Clusters de Correlação',
                                            xaxis=dict(
                                                showticklabels=False,
                                                title='Ativos',
                                                zeroline=False,
                                                showgrid=False
                                            ),
                                            yaxis=dict(
                                                title='Distância de Correlação',
                                                zeroline=False
                                            ),
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Análise de clusters para o uso específico do portfólio
                                        st.info("""
                                        **Interpretação do Dendrograma:**
                                        
                                        O dendrograma acima agrupa ativos que têm comportamentos semelhantes em termos de correlação. 
                                        
                                        - Ativos no mesmo "ramo" tendem a se mover juntos
                                        - Ramos que se conectam em baixas alturas indicam correlações fortes
                                        - Para diversificação de portfólio, selecione ativos de diferentes clusters (ramos distantes)
                                        
                                        Esta visualização é especialmente útil para identificar grupos de ativos que poderiam ser tratados como uma classe para fins de alocação de portfólio.
                                        """)
                                    
                                    # Correlações mais significativas
                                    st.markdown("#### 🔍 Correlações Mais Significativas")
                                    
                                    # Adicionar uma coluna de magnitude da correlação para ordenação
                                    corr_df['abs_correlation'] = abs(corr_df['Correlation'])
                                    
                                    # Remover auto-correlações e duplicatas
                                    filtered_corr = corr_df[corr_df['Pair'] != corr_df['CounterPair']].drop_duplicates(subset=['abs_correlation'])
                                    
                                    # Ordenar por magnitude de correlação
                                    top_correlations = filtered_corr.nlargest(10, 'abs_correlation')
                                    
                                    # Separar em positivas e negativas
                                    pos_corr = top_correlations[top_correlations['Correlation'] > 0].nlargest(5, 'Correlation')
                                    neg_corr = top_correlations[top_correlations['Correlation'] < 0].nsmallest(5, 'Correlation')
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("Correlações Positivas Mais Fortes")
                                        for _, row in pos_corr.iterrows():
                                            st.markdown(
                                                f"**{row['Pair']} vs {row['CounterPair']}**: "
                                                f"<span style='color:green'>{row['Correlation']:.2f}</span>",
                                                unsafe_allow_html=True
                                            )
                                    with col2:
                                        st.subheader("Correlações Negativas Mais Fortes")
                                        for _, row in neg_corr.iterrows():
                                            st.markdown(
                                                f"**{row['Pair']} vs {row['CounterPair']}**: "
                                                f"<span style='color:red'>{row['Correlation']:.2f}</span>",
                                                unsafe_allow_html=True
                                            )
                                    
                                    # Análise de correlações específicas com Bitcoin
                                    if symbol.upper() in corr_df['Pair'].values:
                                        st.markdown(f"#### 🔄 Correlações com {symbol}")
                                        btc_correlations = corr_df[corr_df['Pair'] == symbol.upper()].sort_values(by='Correlation', ascending=False)
                                        
                                        # Gráfico de barras para correlações com BTC
                                        if len(btc_correlations) > 1:  # Precisa de pelo menos 2 correlações
                                            fig = go.Figure()
                                            
                                            # Usar cores diferentes para correlações positivas e negativas
                                            colors = np.where(btc_correlations['Correlation'] >= 0, 'green', 'red')
                                            
                                            fig.add_trace(go.Bar(
                                                x=btc_correlations['CounterPair'],
                                                y=btc_correlations['Correlation'],
                                                marker_color=colors,
                                                text=btc_correlations['Correlation'].round(2),
                                                textposition='auto'
                                            ))
                                            
                                            fig.update_layout(
                                                title=f"Correlações com {symbol}",
                                                xaxis_title="Ativo",
                                                yaxis_title="Correlação",
                                                yaxis=dict(range=[-1, 1]),
                                                height=400
                                            )
                                            
                                            # Adicionar linha de referência em zero
                                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Adicionar análise de diversificação de portfólio
                                    st.markdown("#### 💼 Implicações para Diversificação de Portfólio")
                                    
                                    # Avaliar a qualidade de diversificação da carteira de ativos
                                    avg_corr = filtered_corr['Correlation'].mean()
                                    avg_abs_corr = filtered_corr['abs_correlation'].mean()
                                    
                                    # Determinar o status de diversificação
                                    if avg_abs_corr < 0.3:
                                        diversification_status = "Excelente"
                                        diversification_desc = "Os ativos apresentam correlações muito baixas entre si, indicando excelente diversificação."
                                    elif avg_abs_corr < 0.5:
                                        diversification_status = "Boa"
                                        diversification_desc = "Os ativos apresentam correlações moderadas, oferecendo boa diversificação."
                                    elif avg_abs_corr < 0.7:
                                        diversification_status = "Moderada"
                                        diversification_desc = "Correlações moderadamente altas indicam diversificação limitada."
                                    else:
                                        diversification_status = "Pobre"
                                        diversification_desc = "Correlações muito altas sugerem baixa diversificação e exposição a riscos similares."
                                    
                                    # Encontrar pares com correlação próxima a zero (± 0.1)
                                    zero_corr_pairs = filtered_corr[(filtered_corr['Correlation'] > -0.1) & (filtered_corr['Correlation'] < 0.1)]
                                    zero_corr_pairs = zero_corr_pairs.nsmallest(3, 'abs_correlation')
                                    
                                    # Recomendações
                                    st.info(f"""
                                    **Análise de Diversificação:**
                                    
                                    - **Status de Diversificação:** {diversification_status}
                                    - **Correlação Média:** {avg_corr:.3f}
                                    - **Correlação Média Absoluta:** {avg_abs_corr:.3f}
                                    - **Interpretação:** {diversification_desc}
                                    
                                    **Oportunidades de Diversificação:**
                                    """)
                                    
                                    if not zero_corr_pairs.empty:
                                        for _, row in zero_corr_pairs.iterrows():
                                            st.markdown(f"- Pares com correlação próxima a zero: **{row['Pair']}** e **{row['CounterPair']}** ({row['Correlation']:.3f})")
                                    else:
                                        st.markdown("- Nenhum par com correlação próxima a zero foi encontrado nos dados.")
                                    
                                    if avg_abs_corr > 0.6:
                                        st.markdown("- 🚨 **Alerta:** Correlações geralmente altas indicam que o conjunto de ativos pode não oferecer proteção adequada em períodos de estresse de mercado.")
                                    
                                    # Conteúdo educativo
                                    with st.expander("📚 Entendendo Correlações e Diversificação", expanded=False):
                                        st.markdown("""
                                        ### 📚 Guia de Correlações e Diversificação de Portfólio
                                        
                                        #### 🔍 O que é Correlação?
                                        
                                        A correlação mede o grau em que dois ativos se movem juntos ao longo do tempo. Os valores variam de -1 a +1:
                                        
                                        - **+1.0:** Correlação positiva perfeita (movimento idêntico)
                                        - **0.0:** Nenhuma correlação (movimento independente)
                                        - **-1.0:** Correlação negativa perfeita (movimento em direções opostas)
                                        
                                        #### 💼 Importância para Construção de Portfólio:
                                        
                                        1. **Redução de Risco:**
                                           - Combinar ativos com baixa ou negativa correlação reduz a volatilidade geral do portfólio
                                           - A teoria moderna de portfólio demonstra que portfólios diversificados podem melhorar o retorno ajustado ao risco
                                        
                                        2. **Tipos de Correlações Úteis:**
                                           - **Correlações Negativas:** Oferecem a melhor proteção durante quedas de mercado
                                           - **Correlações Próximas a Zero:** Proporcionam boa diversificação sem sacrificar retornos
                                           - **Correlações Variáveis:** Ativos que mudam de correlação em diferentes regimes de mercado podem ser valiosos
                                        
                                        3. **Armadilhas a Evitar:**
                                           - Correlações tendem a aumentar durante crises ("tudo se correlaciona em pânico")
                                           - Correlações históricas não garantem comportamento futuro
                                           - Dados de curto prazo podem não capturar relações de longo prazo
                                        
                                        #### 🛠️ Aplicação Prática em Criptomoedas:
                                        
                                        - **Entre Criptomoedas:** Muitas altcoins têm alta correlação com Bitcoin; procure projetos em nichos distintos
                                        - **Com Outras Classes:** Monitore correlações com ouro, ações tech, índices de commodities
                                        - **Durante Ciclos:** As correlações em cripto tendem a mudar durante diferentes fases do ciclo de mercado
                                        
                                        #### 🔄 Dinâmica de Correlações:
                                        
                                        - Correlações não são estáticas - podem mudar drasticamente com mudanças macro
                                        - Reavalie regularmente a estrutura de correlação do seu portfólio
                                        - Considere correlações condicionais (como os ativos se correlacionam durante quedas vs. altas)
                                        """)
                                else:
                                    st.warning("Dados de correlação insuficientes após limpeza.")
                            
                            except Exception as e:
                                st.error(f"Erro ao criar matriz de correlação: {str(e)}")
                                st.write("Exemplo de linha:", correlations.iloc[0] if len(correlations) > 0 else None)
                    else:
                        st.error("Formato de dados de correlação não suportado ou colunas não encontradas")
                        st.write("Colunas esperadas:", ['Pair/asset1', 'CounterPair/asset2', 'Correlation/corr'])
                        st.write("Colunas disponíveis:", list(correlations.columns))
                else:
                    st.error("Não foi possível processar os dados de correlação")
            
            # Processar dados de VaR
            if var_data is not None:
                var_data = self._process_api_data(var_data)
                if not var_data.empty:
                    st.success(f"Dados de VaR processados com sucesso. Colunas: {list(var_data.columns)}")
                    
                    # Usar nomes normalizados
                    timestamp_col = 'timestamp' if 'timestamp' in var_data.columns else None
                    var_99_col = 'var_99' if 'var_99' in var_data.columns else None
                    var_95_col = 'var_95' if 'var_95' in var_data.columns else None
                    var_90_col = 'var_90' if 'var_90' in var_data.columns else None # Suporte opcional para 90%
                    worst_loss_col = 'worst_loss' if 'worst_loss' in var_data.columns else None # Perda histórica pior
                    
                    # Construir dict de colunas encontradas
                    var_cols = {}
                    if var_99_col: var_cols['var_99'] = var_99_col
                    if var_95_col: var_cols['var_95'] = var_95_col
                    if var_90_col: var_cols['var_90'] = var_90_col
                    if worst_loss_col: var_cols['worst_loss'] = worst_loss_col
                    
                    if timestamp_col and var_cols:
                        with st.expander("📉 Análise de Value at Risk (VaR) e Risco de Cauda", expanded=True):
                            st.markdown("#### 📉 Análise de Value at Risk (VaR)")
                            
                            # Converter colunas numéricas
                            for col_key, col_name in var_cols.items():
                                var_data[col_name] = pd.to_numeric(var_data[col_name], errors='coerce')
                            if var_data[timestamp_col].dtype == 'object':
                                var_data[timestamp_col] = pd.to_datetime(var_data[timestamp_col], errors='coerce')
                            var_data = var_data.dropna(subset=[timestamp_col, *var_cols.values()])

                            if len(var_data) > 0:
                                var_data = var_data.sort_values(by=timestamp_col)
                                
                                # Adicionar colunas derivadas para análise
                                if 'var_99' in var_cols and 'var_95' in var_cols:
                                    var_data['tail_risk_ratio'] = var_data[var_cols['var_99']] / var_data[var_cols['var_95']]
                                
                                # Calcular médias móveis
                                for col_key, col_name in var_cols.items():
                                    if col_key in ['var_99', 'var_95']:  # Apenas para os principais VaRs
                                        var_data[f'{col_key}_ma14'] = var_data[col_name].rolling(window=14).mean()
                                
                                # Calcular Z-scores para identificar extremos
                                if len(var_data) >= 30:
                                    for col_key, col_name in var_cols.items():
                                        if col_key in ['var_99', 'var_95']:
                                            var_data[f'{col_key}_zscore'] = (
                                                var_data[col_name] - var_data[col_name].rolling(30).mean()
                                            ) / var_data[col_name].rolling(30).std()
                                
                                if len(var_data) >= 2:
                                    # Exibir métricas atuais
                                    metric_cols = st.columns(len(var_cols))
                                    i = 0
                                    for var_type, col in var_cols.items():
                                        with metric_cols[i]:
                                            current = float(var_data[col].iloc[-1])
                                            previous = float(var_data[col].iloc[-2])
                                            delta = current - previous
                                            display_name = var_type.upper().replace('_', ' ') # Ex: VAR 99
                                            st.metric(display_name, f"{current:.2f}%", f"{delta:.2f}%")
                                        i += 1
                                else:
                                    st.subheader("Métricas de Risco Atuais")
                                    for var_type, col in var_cols.items():
                                        current = float(var_data[col].iloc[-1])
                                        display_name = var_type.upper().replace('_', ' ')
                                        st.metric(display_name, f"{current:.2f}%")
                                
                                # Gráfico principal de VaR
                                fig = go.Figure()
                                
                                for var_type, col in var_cols.items():
                                    if var_type in ['var_99', 'var_95', 'var_90']:  # Apenas mostrar VaRs principais
                                        display_name = var_type.upper().replace('_', ' ')
                                        
                                        # Escolher cores diferentes para cada nível de VaR
                                        color = 'red' if var_type == 'var_99' else 'orange' if var_type == 'var_95' else 'yellow'
                                        
                                        fig.add_trace(go.Scatter(
                                            x=var_data[timestamp_col], 
                                            y=var_data[col], 
                                            mode='lines', 
                                            name=display_name,
                                            line=dict(color=color, width=2)
                                        ))
                                        
                                        # Adicionar média móvel para VaR 99%
                                        if var_type == 'var_99' and f'{var_type}_ma14' in var_data.columns:
                                            fig.add_trace(go.Scatter(
                                                x=var_data[timestamp_col],
                                                y=var_data[f'{var_type}_ma14'],
                                                mode='lines',
                                                name=f"{display_name} MA14",
                                                line=dict(color=color, width=1, dash='dot')
                                            ))
                                
                                # Adicionar dados de pior perda se disponíveis
                                if 'worst_loss' in var_cols:
                                    fig.add_trace(go.Scatter(
                                        x=var_data[timestamp_col], 
                                        y=var_data[var_cols['worst_loss']], 
                                        mode='markers', 
                                        name='Pior Perda',
                                        marker=dict(color='purple', size=8)
                                    ))
                                
                                fig.update_layout(
                                    title="Histórico de Value at Risk (VaR)",
                                    xaxis_title="Data", 
                                    yaxis_title="VaR (%)", 
                                    hovermode='x unified',
                                    legend=dict(orientation="h", y=1.1)
                                )
                                
                                # Adicionar linha para nível de alerta
                                fig.add_hline(y=5, line_dash="dot", line_color="red", opacity=0.7)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Gráfico de razão de risco de cauda (tail risk ratio)
                                if 'tail_risk_ratio' in var_data.columns:
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=var_data[timestamp_col],
                                        y=var_data['tail_risk_ratio'],
                                        mode='lines',
                                        name='Razão de Risco de Cauda',
                                        line=dict(color='purple', width=2)
                                    ))
                                    
                                    fig.update_layout(
                                        title="Razão de Risco de Cauda (VaR 99% / VaR 95%)",
                                        xaxis_title="Data",
                                        yaxis_title="Razão",
                                        hovermode='x unified'
                                    )
                                    
                                    # Adicionar linha de referência para média histórica
                                    avg_ratio = var_data['tail_risk_ratio'].mean()
                                    fig.add_hline(y=avg_ratio, line_dash="dash", line_color="blue", opacity=0.7)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Explicar o significado da razão
                                    if avg_ratio > 1.3:
                                        tail_risk_desc = "alto, indicando distribuição com caudas gordas significativas (eventos extremos mais prováveis que numa distribuição normal)"
                                    elif avg_ratio > 1.15:
                                        tail_risk_desc = "moderado, indicando algum excesso de curtose (caudas mais pesadas que uma distribuição normal)"
                                    else:
                                        tail_risk_desc = "baixo, aproximando-se de uma distribuição normal"
                                    
                                    st.info(f"""
                                    **Análise da Razão de Risco de Cauda:**
                                    
                                    A razão VaR 99% / VaR 95% é um indicador de "gordura" das caudas da distribuição de retornos.
                                    
                                    - **Valor Atual:** {var_data['tail_risk_ratio'].iloc[-1]:.3f}
                                    - **Média Histórica:** {avg_ratio:.3f}
                                    - **Interpretação:** O risco de cauda é {tail_risk_desc}
                                    
                                    Valores mais altos indicam maior probabilidade de movimentos extremos de preço.
                                    """)
                                
                                # Análise mais profunda de VaR
                                current_var99 = float(var_data[var_cols['var_99']].iloc[-1]) if 'var_99' in var_cols else float('nan')
                                current_var95 = float(var_data[var_cols['var_95']].iloc[-1]) if 'var_95' in var_cols else float('nan')
                                
                                # Determinar nível de risco
                                risk_level = ""
                                if not pd.isna(current_var99):
                                    if current_var99 > 7.5:
                                        risk_level = "Extremamente Alto"
                                        risk_implication = "Volatilidade severa; sugerindo condições de crise ou pânico no mercado"
                                    elif current_var99 > 5:
                                        risk_level = "Muito Alto"
                                        risk_implication = "Volatilidade significativamente elevada; possível período de estresse de mercado"
                                    elif current_var99 > 3.5:
                                        risk_level = "Alto"
                                        risk_implication = "Volatilidade acima da média; possibilidade aumentada de movimentos mais bruscos"
                                    elif current_var99 > 2.5:
                                        risk_level = "Moderado"
                                        risk_implication = "Volatilidade normal para o mercado de criptomoedas"
                                    else:
                                        risk_level = "Baixo"
                                        risk_implication = "Volatilidade abaixo da média; mercado relativamente calmo"
                                
                                # Verificar mudanças recentes no VaR
                                var_trend = ""
                                if len(var_data) >= 7 and 'var_99' in var_cols:
                                    week_ago_var = float(var_data[var_cols['var_99']].iloc[-7]) if len(var_data) >= 7 else float('nan')
                                    if not pd.isna(week_ago_var) and not pd.isna(current_var99):
                                        pct_change = ((current_var99 - week_ago_var) / week_ago_var) * 100
                                        
                                        if pct_change > 30:
                                            var_trend = "🚨 Aumento expressivo de risco nas últimas semanas (> 30%)"
                                        elif pct_change > 15:
                                            var_trend = "⚠️ Aumento moderado de risco nas últimas semanas (> 15%)"
                                        elif pct_change < -30:
                                            var_trend = "✅ Redução expressiva de risco nas últimas semanas (> 30%)"
                                        elif pct_change < -15:
                                            var_trend = "👍 Redução moderada de risco nas últimas semanas (> 15%)"
                                        else:
                                            var_trend = "Estabilidade relativa no perfil de risco nas últimas semanas"
                                
                                # Z-score do VaR para identificar extremos 
                                var_zscore_alert = ""
                                if 'var_99_zscore' in var_data.columns and not pd.isna(var_data['var_99_zscore'].iloc[-1]):
                                    var_zscore = var_data['var_99_zscore'].iloc[-1]
                                    if var_zscore > 2:
                                        var_zscore_alert = "🚨 Alerta: VaR atual está extremamente elevado em relação à média histórica (> 2σ)"
                                    elif var_zscore < -2:
                                        var_zscore_alert = "📊 Nota: VaR atual está extremamente baixo em relação à média histórica (< -2σ)"
                                
                                # Exibir análise final
                                st.info(f"""
                                **Análise Detalhada de Risco:**
                                
                                - **Nível de Risco Atual:** {risk_level}
                                - **VaR 99%:** {current_var99:.2f}% (perda máxima diária esperada com 99% de confiança)
                                - **VaR 95%:** {current_var95:.2f}% (perda máxima diária esperada com 95% de confiança)
                                - **Implicação:** {risk_implication}
                                - **Tendência Recente:** {var_trend}
                                {f"- **{var_zscore_alert}**" if var_zscore_alert else ""}
                                """)
                                
                                # Conteúdo educativo
                                with st.expander("📚 Entendendo o Value at Risk (VaR)", expanded=False):
                                    st.markdown("""
                                    ### 📚 Value at Risk (VaR) e Análise de Risco em Criptomoedas
                                    
                                    #### 🔍 O que é Value at Risk (VaR)?
                                    
                                    O Value at Risk (VaR) é uma metodologia estatística que quantifica o nível máximo de perda esperada em um período específico, com um determinado nível de confiança.
                                    
                                    **Por exemplo:**
                                    - Um VaR 99% de 5% significa que há apenas 1% de chance de uma perda diária maior que 5%
                                    - Um VaR 95% de 3% significa que há apenas 5% de chance de uma perda diária maior que 3%
                                    
                                    #### 📊 Interpretação de Diferentes Níveis de VaR:
                                    
                                    1. **VaR 99% vs. VaR 95%:**
                                       - VaR 99% é mais conservador e captura eventos mais extremos
                                       - A diferença entre VaR 99% e 95% indica a "gordura" da cauda da distribuição
                                    
                                    2. **Razão de Risco de Cauda:**
                                       - Calculada como VaR 99% / VaR 95%
                                       - Valores próximos a 1.3 ou maiores indicam caudas gordas significativas
                                       - Quanto maior a razão, maior a probabilidade de eventos extremos
                                    
                                    #### ⚠️ Limitações do VaR:
                                    
                                    - Não captura a magnitude das perdas além do limiar de confiança
                                    - Baseia-se em dados históricos, que podem não prever eventos sem precedentes
                                    - Pode subestimar risco em períodos de baixa volatilidade ou em ativos com distribuições não-normais
                                    
                                    #### 💹 Aplicações Práticas em Criptomoedas:
                                    
                                    - **Dimensionamento de Posição:** Ajuste o tamanho da posição com base no VaR atual
                                    - **Stop Loss Dinâmico:** Use múltiplos do VaR para definir stops adequados ao regime de volatilidade
                                    - **Alavancagem Adaptativa:** Reduza alavancagem em períodos de VaR elevado
                                    - **Hedge Condicional:** Implemente hedges quando o VaR ultrapassar limiares predefinidos
                                    
                                    #### 📈 VaR e Ciclos de Mercado:
                                    
                                    - O VaR tende a aumentar durante quedas de mercado e reduzir em períodos de baixa volatilidade
                                    - Aumentos súbitos no VaR geralmente precedem ou coincidem com correções significativas
                                    - Períodos prolongados de VaR baixo podem indicar complacência e risco acumulado
                                    
                                    #### 🛡️ Complementando o VaR:
                                    
                                    Para uma análise de risco mais robusta, complemente o VaR com:
                                    - **Stress Testing:** Simulações de cenários extremos específicos
                                    - **Expected Shortfall (CVaR):** Média das perdas além do VaR
                                    - **Drawdown Analysis:** Análise da magnitude e duração de quedas consecutivas
                                    """)
                            else:
                                 st.error("Dados de VaR não contêm as colunas necessárias/válidas ou não há dados suficientes após limpeza.")
                                 st.write("Colunas necessárias após normalização:", ['timestamp', 'var_99', 'var_95'])
                                 st.write("Colunas disponíveis:", list(var_data.columns))
                    else:
                        # Mensagem de erro mais específica
                        missing_cols = []
                        if not timestamp_col: missing_cols.append("'timestamp' (ou 'worstDate')")
                        if not var_cols: missing_cols.append("'var_99' ou 'var_95' (ou '99% VaR', '95% VaR')")
                        st.error(f"Colunas necessárias não encontradas nos dados VaR: {', '.join(missing_cols)}")
                        st.write("Colunas disponíveis após normalização:", list(var_data.columns))
                else:
                    st.error("Não foi possível processar os dados de VaR ou o arquivo está vazio.")
            else:
                 st.warning(f"Arquivo risk/var_{symbol}.csv não encontrado ou vazio.")
            
        except Exception as e:
            st.error(f"Erro na análise de risco: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def analyze_summary_data(self, symbol="BTCUSDT"):
        """
        Analisa dados resumidos com visualizações e interpretações aprimoradas.
        """
        st.markdown("### 📑 Análise de Dados Resumidos")
        
        try:
            # Carregar dados
            funding = self.fetcher.load_data_file(f"summary/binance_futures_funding_{symbol}.csv")
            metrics = self.fetcher.load_data_file(f"summary/binance_futures_metrics_{symbol}.csv")
            oi_historical = self.fetcher.load_data_file(f"summary/binance_options_oi_{symbol}.csv")
            
            st.info(f"Arquivos consultados: binance_futures_funding_{symbol}.csv, binance_futures_metrics_{symbol}.csv, binance_options_oi_{symbol}.csv")
            
            # Processar dados de funding
            if funding is not None:
                funding = self._process_api_data(funding)
                if not funding.empty:
                    st.success(f"Dados de funding processados com sucesso. Colunas: {list(funding.columns)}")
                    
                    timestamp_col = next((col for col in ['timestamp', 'date', 'time', 'Date', 'unix_timestamp'] 
                                         if col in funding.columns), None)
                    funding_col = next((col for col in funding.columns 
                                      if any(term in col.lower() for term in ['funding', 'funding_rate', 'last_funding_rate'])), None)
                    
                    if timestamp_col and funding_col:
                        with st.expander("💸 Análise de Funding Rate e Implicações", expanded=True):
                            st.markdown("#### 💸 Análise de Funding Rate")
                            
                            funding[funding_col] = pd.to_numeric(funding[funding_col], errors='coerce')
                            if 'unix' in timestamp_col.lower():
                                funding['timestamp'] = pd.to_datetime(funding[timestamp_col], unit='ms', errors='coerce')
                                timestamp_col = 'timestamp'
                            elif funding[timestamp_col].dtype == 'object':
                                 funding[timestamp_col] = pd.to_datetime(funding[timestamp_col], errors='coerce')
                            funding = funding.dropna(subset=[timestamp_col, funding_col])

                            if len(funding) > 0:
                                funding = funding.sort_values(by=timestamp_col)
                                funding_values = funding[funding_col].astype(float)
                                
                                # Assegurar que o funding está em formato percentual
                                if funding_values.abs().max() < 0.1:
                                    funding[funding_col] = funding_values * 100
                                
                                # Adicionar análises derivadas
                                # 1. Médias móveis
                                funding['funding_ma8'] = funding[funding_col].rolling(window=8).mean()  # ~1 dia (8 períodos = 8 horas * 8 = 64 horas)
                                funding['funding_ma24'] = funding[funding_col].rolling(window=24).mean()  # ~3 dias
                                funding['funding_ma72'] = funding[funding_col].rolling(window=72).mean()  # ~9 dias
                                
                                # 2. Volatilidade (desvio padrão)
                                funding['funding_std24'] = funding[funding_col].rolling(window=24).std()
                                
                                # 3. Soma acumulada de funding (total pago/recebido)
                                funding['funding_cumsum'] = funding[funding_col].cumsum()
                                
                                # 4. Regime de funding (classificar se está em período positivo, negativo ou neutro)
                                funding['funding_regime'] = pd.cut(
                                    funding['funding_ma24'],
                                    bins=[-float('inf'), -0.01, 0.01, float('inf')],
                                    labels=['Negativo', 'Neutro', 'Positivo']
                                )
                                
                                # Gráfico principal com médias móveis
                                fig = go.Figure()
                                
                                # Adicionar áreas sombreadas para regimes
                                # Primeiro encontrar os regimes e seus limites
                                current_regime = None
                                regime_changes = []
                                
                                for idx, row in funding.iterrows():
                                    regime = row['funding_regime']
                                    if regime != current_regime:
                                        regime_changes.append((idx, row[timestamp_col], regime))
                                        current_regime = regime
                                
                                # Adicionar as áreas sombreadas para cada regime se tivermos pelo menos uma mudança
                                if len(regime_changes) > 1:
                                    for i in range(len(regime_changes) - 1):
                                        start_idx, start_time, regime = regime_changes[i]
                                        end_idx, end_time, _ = regime_changes[i + 1]
                                        
                                        # Definir cores baseadas no regime
                                        if regime == 'Positivo':
                                            color = 'rgba(0, 255, 0, 0.1)'  # Verde transparente
                                        elif regime == 'Negativo':
                                            color = 'rgba(255, 0, 0, 0.1)'  # Vermelho transparente
                                        else:
                                            color = 'rgba(128, 128, 128, 0.1)'  # Cinza transparente
                                        
                                        fig.add_shape(
                                            type="rect",
                                            x0=start_time,
                                            x1=end_time,
                                            y0=funding[funding_col].min(),
                                            y1=funding[funding_col].max(),
                                            fillcolor=color,
                                            line=dict(width=0),
                                            layer="below"
                                        )
                                    
                                    # Adicionar a última área até o final dos dados
                                    start_idx, start_time, regime = regime_changes[-1]
                                    
                                    # Definir cores baseadas no regime
                                    if regime == 'Positivo':
                                        color = 'rgba(0, 255, 0, 0.1)'  # Verde transparente
                                    elif regime == 'Negativo':
                                        color = 'rgba(255, 0, 0, 0.1)'  # Vermelho transparente
                                    else:
                                        color = 'rgba(128, 128, 128, 0.1)'  # Cinza transparente
                                    
                                    fig.add_shape(
                                        type="rect",
                                        x0=start_time,
                                        x1=funding[timestamp_col].iloc[-1],
                                        y0=funding[funding_col].min(),
                                        y1=funding[funding_col].max(),
                                        fillcolor=color,
                                        line=dict(width=0),
                                        layer="below"
                                    )
                                
                                # Linha principal de funding
                                fig.add_trace(go.Scatter(
                                    x=funding[timestamp_col], 
                                    y=funding[funding_col], 
                                    mode='lines', 
                                    name='Funding Rate (%)', 
                                    line=dict(color=self.color_palette['primary'], width=1.5)
                                ))
                                
                                # Adicionar médias móveis
                                fig.add_trace(go.Scatter(
                                    x=funding[timestamp_col],
                                    y=funding['funding_ma8'],
                                    mode='lines',
                                    name='MM8 (~1 dia)',
                                    line=dict(color=self.color_palette['secondary'], width=1.5)
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=funding[timestamp_col],
                                    y=funding['funding_ma24'],
                                    mode='lines',
                                    name='MM24 (~3 dias)',
                                    line=dict(color=self.color_palette['tertiary'], width=1.5, dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} - Histórico de Funding Rate",
                                    xaxis_title="Data", 
                                    yaxis_title="Funding Rate (%)", 
                                    hovermode='x unified',
                                    legend=dict(orientation="h", y=1.1)
                                )
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Gráfico de soma acumulada de funding
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=funding[timestamp_col],
                                    y=funding['funding_cumsum'],
                                    mode='lines',
                                    name='Funding Acumulado',
                                    fill='tozeroy',
                                    line=dict(color='purple', width=2)
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} - Funding Acumulado (Soma)",
                                    xaxis_title="Data",
                                    yaxis_title="Funding Acumulado (%)",
                                    hovermode='x unified'
                                )
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Estatísticas detalhadas
                                last_value = float(funding[funding_col].iloc[-1])
                                mean_24h = float(funding['funding_ma8'].iloc[-1]) if len(funding) >= 8 else float('nan')
                                mean_72h = float(funding['funding_ma24'].iloc[-1]) if len(funding) >= 24 else float('nan')
                                accumulated = float(funding['funding_cumsum'].iloc[-1])
                                std_dev = float(funding['funding_std24'].iloc[-1]) if len(funding) > 24 else float('nan')
                                
                                # Estatísticas históricas do funding
                                funding_stats = {}
                                if len(funding) > 0:
                                    funding_stats = {
                                        'min': float(funding[funding_col].min()),
                                        'max': float(funding[funding_col].max()),
                                        'mean': float(funding[funding_col].mean()),
                                        'median': float(funding[funding_col].median()),
                                        'std': float(funding[funding_col].std()),
                                        'positive_pct': float((funding[funding_col] > 0).mean() * 100),
                                        'negative_pct': float((funding[funding_col] < 0).mean() * 100)
                                    }
                                
                                # Exibir estatísticas em colunas
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    color = "green" if last_value > 0 else "red" if last_value < 0 else "gray"
                                    st.markdown(f"**Atual:** <span style='color:{color}'>{last_value:.4f}%</span>", unsafe_allow_html=True)
                                with col2:
                                    if not pd.isna(mean_24h):
                                        color = "green" if mean_24h > 0 else "red" if mean_24h < 0 else "gray"
                                        st.markdown(f"**Média 8 períodos (~24h):** <span style='color:{color}'>{mean_24h:.4f}%</span>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("**Média 24h:** N/A")
                                with col3:
                                    if not pd.isna(accumulated):
                                         color = "green" if accumulated > 0 else "red" if accumulated < 0 else "gray"
                                         st.markdown(f"**Acumulado:** <span style='color:{color}'>{accumulated:.4f}%</span>", unsafe_allow_html=True)
                                    else:
                                         st.markdown("**Acumulado:** N/A")
                                with col4:
                                    if not pd.isna(std_dev):
                                         st.markdown(f"**Volatilidade 24 períodos:** {std_dev:.4f}%" )
                                    else:
                                         st.markdown("**Volatilidade:** N/A")
                                
                                # Análise do regime atual de funding
                                current_regime = funding['funding_regime'].iloc[-1]
                                regime_duration = 1  # Iniciar com pelo menos 1 período
                                
                                # Contar por quantos períodos estamos no regime atual
                                for i in range(len(funding) - 2, -1, -1):
                                    if funding['funding_regime'].iloc[i] == current_regime:
                                        regime_duration += 1
                                    else:
                                        break
                                
                                # Interpretar o regime atual
                                regime_meaning = ""
                                if current_regime == 'Positivo':
                                    if last_value > 0.05:
                                        regime_meaning = "Mercado fortemente otimista com excesso potencial de alavancagem long"
                                    else:
                                        regime_meaning = "Mercado moderadamente otimista"
                                elif current_regime == 'Negativo':
                                    if last_value < -0.05:
                                        regime_meaning = "Mercado fortemente pessimista com excesso potencial de alavancagem short"
                                    else:
                                        regime_meaning = "Mercado moderadamente pessimista"
                                else:
                                    regime_meaning = "Mercado em equilíbrio/neutralidade"
                                
                                # Calcular médias históricas para comparação
                                if funding_stats:
                                    historical_context = ""
                                    if last_value > funding_stats['mean'] + funding_stats['std']:
                                        historical_context = "Funding atual está significativamente acima da média histórica (> 1σ)"
                                    elif last_value < funding_stats['mean'] - funding_stats['std']:
                                        historical_context = "Funding atual está significativamente abaixo da média histórica (< -1σ)"
                                    elif abs(last_value - funding_stats['mean']) < funding_stats['std'] * 0.2:
                                        historical_context = "Funding atual está próximo da média histórica"
                                    else:
                                        historical_context = f"Funding atual está dentro de 1 desvio padrão da média histórica ({funding_stats['mean']:.4f}%)"
                                
                                # Análise de momentum
                                momentum = ""
                                if len(funding) >= 8:
                                    recent_trend = [funding[funding_col].iloc[-i] for i in range(1, min(9, len(funding) + 1))]
                                    
                                    if all(x >= y for x, y in zip(recent_trend, recent_trend[1:])):
                                        momentum = "em tendência de alta consistente nos últimos períodos"
                                    elif all(x <= y for x, y in zip(recent_trend, recent_trend[1:])):
                                        momentum = "em tendência de baixa consistente nos últimos períodos"
                                    elif recent_trend[0] > recent_trend[-1]:
                                        momentum = "com tendência geral de alta, mas com alguma variabilidade"
                                    elif recent_trend[0] < recent_trend[-1]:
                                        momentum = "com tendência geral de baixa, mas com alguma variabilidade"
                                    else:
                                        momentum = "sem tendência clara, apresentando movimento lateral"
                                
                                # Exibir análise detalhada
                                st.info(f"""
                                **Análise do Regime de Funding:**
                                
                                - **Regime Atual:** {current_regime} (duração: {regime_duration} períodos)
                                - **Interpretação:** {regime_meaning}
                                - **Contexto Histórico:** {historical_context if funding_stats else "Dados históricos insuficientes"}
                                - **Momentum:** Funding está {momentum if momentum else "sem dados suficientes para análise de tendência"}
                                
                                **Estatísticas Históricas:**
                                - Média: {funding_stats.get('mean', 'N/A'):.4f}%
                                - Mediana: {funding_stats.get('median', 'N/A'):.4f}%
                                - Mínimo: {funding_stats.get('min', 'N/A'):.4f}%
                                - Máximo: {funding_stats.get('max', 'N/A'):.4f}%
                                - % do tempo positivo: {funding_stats.get('positive_pct', 'N/A'):.1f}%
                                - % do tempo negativo: {funding_stats.get('negative_pct', 'N/A'):.1f}%
                                """)
                                
                                # Conteúdo educativo
                                with st.expander("📚 Entendendo o Funding Rate em Futuros Perpétuos", expanded=False):
                                    st.markdown("""
                                    ### 📚 Guia de Funding Rate em Futuros Perpétuos
                                    
                                    #### 🔍 O que é Funding Rate?
                                    
                                    O funding rate é um mecanismo usado nos contratos de futuros perpétuos (sem data de vencimento) para manter o preço dos futuros alinhado com o preço do mercado à vista (spot).
                                    
                                    **Mecanismo básico:**
                                    - Quando o preço futuro > preço spot: Funding rate é **positivo** (posições long pagam para short)
                                    - Quando o preço futuro < preço spot: Funding rate é **negativo** (posições short pagam para long)
                                    
                                    O pagamento geralmente ocorre a cada 8 horas (3 vezes ao dia) nas principais exchanges.
                                    
                                    #### 📊 Interpretação de Valores:
                                    
                                    1. **Funding Rate Positivo:**
                                       - Indica que o mercado está predominantemente comprado (long)
                                       - Sugerindo viés de alta (bullish) entre os traders
                                       - Valores extremamente altos (>0.05% por período) podem indicar excesso de alavancagem long
                                    
                                    2. **Funding Rate Negativo:**
                                       - Indica que o mercado está predominantemente vendido (short)
                                       - Sugerindo viés de baixa (bearish) entre os traders
                                       - Valores extremamente baixos (<-0.05% por período) podem indicar excesso de alavancagem short
                                    
                                    3. **Funding Rate próximo a zero:**
                                       - Indica equilíbrio entre posições compradas e vendidas
                                       - Preço dos futuros próximo ao preço spot
                                    
                                    #### 💡 Aplicações Estratégicas:
                                    
                                    - **Indicador de Sentimento:** Representa o posicionamento predominante no mercado
                                    - **Oportunidades de Arbitragem:** Capitalizar em funding rates extremos enquanto neutraliza exposição direcional
                                    - **Sinal Contrário:** Valores extremos frequentemente precedem reversões de mercado
                                    - **Cash & Carry Strategy:** Short no futuro + long no spot para capturar funding negativo
                                    
                                    #### ⚠️ Sinais de Alerta:
                                    
                                    - **Funding rates consistentemente altos:** Potencial excesso de otimismo e alavancagem
                                    - **Divergências funding vs preço:** Quando o preço sobe mas o funding cai (ou vice-versa)
                                    - **Extremos históricos:** Valores fora de 2-3 desvios padrão da média histórica
                                    
                                    #### 🔄 Relação com Volatilidade:
                                    
                                    - Períodos de alta volatilidade tendem a ter funding rates mais extremos
                                    - A volatilidade do próprio funding rate é indicador de incerteza no mercado
                                    - Valores estáveis geralmente indicam consenso sobre a direção do mercado
                                    """)
                            else:
                                st.error("Dados de funding insuficientes após limpeza.")
                    else:
                        st.error("Colunas necessárias não encontradas nos dados de funding")
                        st.write("Colunas necessárias:", ['timestamp/date', 'funding_rate'])
                        st.write("Colunas disponíveis:", list(funding.columns) if not funding.empty else "Nenhuma")
                else:
                    st.error("Não foi possível processar os dados de funding")
            
            # Processar dados de métricas
            if metrics is not None:
                metrics = self._process_api_data(metrics)
                if not metrics.empty:
                    st.success(f"Dados de métricas processados com sucesso. Colunas: {list(metrics.columns)}")
                    
                    timestamp_col = next((col for col in ['timestamp', 'date', 'time', 'Date', 'unix_timestamp'] 
                                         if col in metrics.columns), None)
                    volume_col = next((col for col in metrics.columns if 'volume' in col.lower()), None)
                    oi_col = next((col for col in metrics.columns 
                                  if any(term in col.lower() for term in ['interest', 'oi', 'open_interest'])), None)
                    
                    if timestamp_col and (volume_col or oi_col):
                        with st.expander("📊 Análise de Volume e Open Interest", expanded=True):
                            st.markdown("#### 📊 Métricas de Futuros")
                            
                            if volume_col: metrics[volume_col] = pd.to_numeric(metrics[volume_col], errors='coerce')
                            if oi_col: metrics[oi_col] = pd.to_numeric(metrics[oi_col], errors='coerce')
                            if 'unix' in timestamp_col.lower():
                                metrics['timestamp'] = pd.to_datetime(metrics[timestamp_col], unit='ms', errors='coerce')
                                timestamp_col = 'timestamp'
                            elif metrics[timestamp_col].dtype == 'object':
                                metrics[timestamp_col] = pd.to_datetime(metrics[timestamp_col], errors='coerce')
                            
                            required_metric_cols = [timestamp_col]
                            if volume_col: required_metric_cols.append(volume_col)
                            if oi_col: required_metric_cols.append(oi_col)
                            metrics = metrics.dropna(subset=required_metric_cols)

                            if len(metrics) > 0:
                                metrics = metrics.sort_values(by=timestamp_col)
                                
                                # Adicionar métricas derivadas
                                # 1. Médias móveis
                                if volume_col:
                                    metrics[f'{volume_col}_ma7'] = metrics[volume_col].rolling(window=7).mean()
                                    metrics[f'{volume_col}_ma30'] = metrics[volume_col].rolling(window=30).mean()
                                
                                if oi_col:
                                    metrics[f'{oi_col}_ma7'] = metrics[oi_col].rolling(window=7).mean()
                                    metrics[f'{oi_col}_ma30'] = metrics[oi_col].rolling(window=30).mean()
                                
                                # 2. Volume/OI Ratio (se ambos existirem)
                                if volume_col and oi_col:
                                    metrics['vol_oi_ratio'] = metrics[volume_col] / metrics[oi_col].replace(0, np.nan)
                                    metrics['vol_oi_ratio_ma7'] = metrics['vol_oi_ratio'].rolling(window=7).mean()
                                
                                # 3. Mudança diária em OI
                                if oi_col:
                                    metrics[f'{oi_col}_daily_change'] = metrics[oi_col].pct_change().fillna(0) * 100
                                    metrics[f'{oi_col}_daily_change_ma7'] = metrics[f'{oi_col}_daily_change'].rolling(window=7).mean()
                                
                                # Gráfico principal: Volume e OI
                                if volume_col and oi_col:
                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                    
                                    # Adicionar barras de volume
                                    fig.add_trace(
                                        go.Bar(
                                            x=metrics[timestamp_col], 
                                            y=metrics[volume_col], 
                                            name="Volume", 
                                            marker_color='rgba(0, 0, 255, 0.3)',
                                            opacity=0.7
                                        ), 
                                        secondary_y=False
                                    )
                                    
                                    # Adicionar média móvel de volume
                                    fig.add_trace(
                                        go.Scatter(
                                            x=metrics[timestamp_col],
                                            y=metrics[f'{volume_col}_ma7'],
                                            name="Volume MM7",
                                            line=dict(color='blue', width=1.5, dash='dot')
                                        ),
                                        secondary_y=False
                                    )
                                    
                                    # Adicionar linha de Open Interest
                                    fig.add_trace(
                                        go.Scatter(
                                            x=metrics[timestamp_col], 
                                            y=metrics[oi_col], 
                                            name="Open Interest", 
                                            line=dict(color='rgb(255, 0, 0)', width=2)
                                        ), 
                                        secondary_y=True
                                    )
                                    
                                    # Adicionar média móvel de OI
                                    fig.add_trace(
                                        go.Scatter(
                                            x=metrics[timestamp_col],
                                            y=metrics[f'{oi_col}_ma7'],
                                            name="OI MM7",
                                            line=dict(color='rgba(255, 0, 0, 0.5)', width=1.5, dash='dot')
                                        ),
                                        secondary_y=True
                                    )
                                    
                                    fig.update_layout(
                                        title=f"{symbol} - Volume e Open Interest",
                                        hovermode='x unified',
                                        xaxis_title="Data",
                                        legend=dict(orientation="h", y=1.1)
                                    )
                                    fig.update_yaxes(title_text="Volume", secondary_y=False)
                                    fig.update_yaxes(title_text="Open Interest", secondary_y=True)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Gráfico 2: Volume/OI Ratio
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics['vol_oi_ratio'],
                                        mode='lines',
                                        name='Volume/OI Ratio',
                                        line=dict(color='purple', width=1.5)
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics['vol_oi_ratio_ma7'],
                                        mode='lines',
                                        name='Vol/OI Ratio MM7',
                                        line=dict(color='rgba(128, 0, 128, 0.5)', width=2, dash='dot')
                                    ))
                                    
                                    # Adicionar linha para média histórica
                                    avg_ratio = metrics['vol_oi_ratio'].mean()
                                    fig.add_hline(y=avg_ratio, line_dash="dash", line_color="gray")
                                    
                                    fig.update_layout(
                                        title="Ratio Volume/Open Interest",
                                        xaxis_title="Data",
                                        yaxis_title="Ratio",
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif volume_col:
                                    # Apenas gráfico de volume se não temos OI
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=metrics[timestamp_col], 
                                        y=metrics[volume_col], 
                                        name="Volume", 
                                        marker_color='rgba(0, 0, 255, 0.3)'
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics[f'{volume_col}_ma7'],
                                        name="Volume MM7",
                                        line=dict(color='blue', width=1.5)
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics[f'{volume_col}_ma30'],
                                        name="Volume MM30",
                                        line=dict(color='navy', width=1.5, dash='dot')
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{symbol} - Volume de Negociação",
                                        xaxis_title="Data", 
                                        yaxis_title="Volume",
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif oi_col:
                                    # Apenas gráfico de OI se não temos volume
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col], 
                                        y=metrics[oi_col], 
                                        name="Open Interest", 
                                        line=dict(color='rgb(255, 0, 0)', width=2)
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics[f'{oi_col}_ma7'],
                                        name="OI MM7",
                                        line=dict(color='rgba(255, 0, 0, 0.5)', width=1.5)
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics[f'{oi_col}_ma30'],
                                        name="OI MM30",
                                        line=dict(color='darkred', width=1.5, dash='dot')
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{symbol} - Open Interest",
                                        xaxis_title="Data", 
                                        yaxis_title="Open Interest",
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Gráfico adicional: Mudança diária em OI
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=metrics[timestamp_col],
                                        y=metrics[f'{oi_col}_daily_change'],
                                        name="Mudança Diária em OI (%)",
                                        marker_color=np.where(metrics[f'{oi_col}_daily_change'] >= 0, 'green', 'red')
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=metrics[timestamp_col],
                                        y=metrics[f'{oi_col}_daily_change_ma7'],
                                        name="MM7 da Mudança",
                                        line=dict(color='purple', width=1.5)
                                    ))
                                    
                                    fig.update_layout(
                                        title="Mudança Diária no Open Interest (%)",
                                        xaxis_title="Data",
                                        yaxis_title="Mudança (%)",
                                        hovermode='x unified'
                                    )
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Estatísticas e métricas atuais
                                if len(metrics) >= 2:
                                    col1, col2, col3 = st.columns(3)
                                    
                                    if volume_col:
                                        latest_vol = float(metrics[volume_col].iloc[-1])
                                        prev_vol = float(metrics[volume_col].iloc[-2])
                                        vol_pct_change = ((latest_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
                                        avg_vol_7d = float(metrics[f'{volume_col}_ma7'].iloc[-1]) if f'{volume_col}_ma7' in metrics.columns and not pd.isna(metrics[f'{volume_col}_ma7'].iloc[-1]) else 0
                                        vol_vs_avg = ((latest_vol / avg_vol_7d) - 1) * 100 if avg_vol_7d else 0
                                        
                                        with col1: 
                                            st.metric(
                                                "Volume Atual", 
                                                f"{latest_vol:,.0f}", 
                                                f"{vol_pct_change:.1f}% vs ontem | {vol_vs_avg:.1f}% vs MM7"
                                            )
                                    
                                    if oi_col:
                                        latest_oi = float(metrics[oi_col].iloc[-1])
                                        prev_oi = float(metrics[oi_col].iloc[-2])
                                        oi_pct_change = ((latest_oi - prev_oi) / prev_oi * 100) if prev_oi else 0
                                        avg_oi_7d = float(metrics[f'{oi_col}_ma7'].iloc[-1]) if f'{oi_col}_ma7' in metrics.columns and not pd.isna(metrics[f'{oi_col}_ma7'].iloc[-1]) else 0
                                        oi_vs_avg = ((latest_oi / avg_oi_7d) - 1) * 100 if avg_oi_7d else 0
                                        
                                        with col2: 
                                            st.metric(
                                                "Open Interest Atual", 
                                                f"{latest_oi:,.0f}", 
                                                f"{oi_pct_change:.1f}% vs ontem | {oi_vs_avg:.1f}% vs MM7"
                                            )
                                    
                                    if volume_col and oi_col:
                                        latest_ratio = float(metrics['vol_oi_ratio'].iloc[-1])
                                        prev_ratio = float(metrics['vol_oi_ratio'].iloc[-2])
                                        ratio_change = latest_ratio - prev_ratio
                                        avg_ratio = float(metrics['vol_oi_ratio'].mean())
                                        ratio_vs_avg = ((latest_ratio / avg_ratio) - 1) * 100 if avg_ratio else 0
                                        
                                        with col3: 
                                            st.metric(
                                                "Ratio Volume/OI", 
                                                f"{latest_ratio:.2f}", 
                                                f"{ratio_change:+.2f} vs ontem | {ratio_vs_avg:.1f}% vs média"
                                            )
                                
                                # Análise de tendências e momentum
                                if oi_col and len(metrics) >= 7:
                                    oi_trend = ""
                                    oi_7d_change = ((metrics[oi_col].iloc[-1] - metrics[oi_col].iloc[-7]) / metrics[oi_col].iloc[-7]) * 100 if metrics[oi_col].iloc[-7] != 0 else 0
                                    
                                    if oi_7d_change > 15:
                                        oi_trend = "Aumento muito significativo no OI (>15% em 7 dias), indicando forte entrada de novos participantes e possível continuação da tendência atual"
                                    elif oi_7d_change > 5:
                                        oi_trend = "Aumento moderado no OI (>5% em 7 dias), sugerindo interesse crescente no mercado"
                                    elif oi_7d_change < -15:
                                        oi_trend = "Queda expressiva no OI (>15% em 7 dias), indicando saídas significativas e possível mudança de tendência"
                                    elif oi_7d_change < -5:
                                        oi_trend = "Queda moderada no OI (>5% em 7 dias), sugerindo redução do interesse no mercado"
                                    else:
                                        oi_trend = "OI relativamente estável na última semana, indicando equilíbrio entre entradas e saídas"
                                    
                                    # Analisar padrão de Volume e OI juntos (se ambos disponíveis)
                                    vol_oi_pattern = ""
                                    if volume_col:
                                        vol_7d_change = ((metrics[volume_col].iloc[-1] - metrics[volume_col].iloc[-7]) / metrics[volume_col].iloc[-7]) * 100 if metrics[volume_col].iloc[-7] != 0 else 0
                                        
                                        if vol_7d_change > 0 and oi_7d_change > 0:
                                            vol_oi_pattern = "Volume e OI aumentando juntos - geralmente confirma a tendência atual (bullish em alta, bearish em baixa)"
                                        elif vol_7d_change > 0 and oi_7d_change < 0:
                                            vol_oi_pattern = "Volume aumentando mas OI caindo - pode indicar fechamento de posições e potencial reversão"
                                        elif vol_7d_change < 0 and oi_7d_change > 0:
                                            vol_oi_pattern = "Volume diminuindo mas OI aumentando - possível sinal de esvaziamento de momentum"
                                        elif vol_7d_change < 0 and oi_7d_change < 0:
                                            vol_oi_pattern = "Volume e OI diminuindo juntos - típico de mercados em contração ou perda de interesse"
                                    
                                    # Análise do ratio Volume/OI (se disponível)
                                    ratio_analysis = ""
                                    if 'vol_oi_ratio' in metrics.columns:
                                        current_ratio = metrics['vol_oi_ratio'].iloc[-1]
                                        avg_historical_ratio = metrics['vol_oi_ratio'].mean()
                                        
                                        if current_ratio > avg_historical_ratio * 1.5:
                                            ratio_analysis = "Ratio Volume/OI significativamente acima da média histórica, indicando alta atividade de rotatividade/especulação"
                                        elif current_ratio < avg_historical_ratio * 0.5:
                                            ratio_analysis = "Ratio Volume/OI significativamente abaixo da média histórica, indicando baixa rotatividade e possível acumulação/distribuição"
                                    
                                    st.info(f"""
                                    **Análise de Tendências em Volume e Open Interest:**
                                    
                                    - **Tendência de OI:** {oi_trend}
                                    {f"- **Padrão Volume-OI:** {vol_oi_pattern}" if vol_oi_pattern else ""}
                                    {f"- **Análise do Ratio:** {ratio_analysis}" if ratio_analysis else ""}
                                    
                                    **Interpretação:**
                                    
                                    O Open Interest representa o total de contratos abertos no mercado e é um indicador importante de participação e interesse.
                                    Aumentos no OI indicam novos participantes/dinheiro entrando no mercado, enquanto quedas indicam fechamento de posições.
                                    
                                    O volume representa a atividade total de negociação e é útil para confirmar a força de movimentos de preço.
                                    A relação entre volume, OI e preço pode revelar padrões ocultos de acumulação/distribuição e fornecer insights sobre potenciais reversões.
                                    """)
                                    
                                    # Conteúdo educativo
                                    with st.expander("📚 Entendendo Volume, Open Interest e suas Implicações", expanded=False):
                                        st.markdown("""
                                        ### 📚 Guia de Volume, Open Interest e suas Relações
                                        
                                        #### 🔍 Definições Básicas:
                                        
                                        **1. Volume:**
                                        - Mede a quantidade total de contratos negociados em um período
                                        - Representa o fluxo, a atividade de negociação
                                        - É resetado a cada período (não cumulativo)
                                        
                                        **2. Open Interest (OI):**
                                        - Representa o número total de contratos em aberto
                                        - Medida de estoque, não de fluxo
                                        - Aumenta quando novos contratos são criados e diminui quando posições são fechadas
                                        
                                        #### 📊 Padrões Chave da Relação Volume-OI-Preço:
                                        
                                        1. **Preço Subindo, OI Subindo, Volume Subindo:**
                                           - Padrão mais forte de tendência de alta
                                           - Indica novas posições compradas sendo estabelecidas
                                           - Alta convicção, aumenta a probabilidade de continuação
                                        
                                        2. **Preço Caindo, OI Subindo, Volume Subindo:**
                                           - Padrão forte de tendência de baixa
                                           - Indica novas posições vendidas sendo estabelecidas
                                           - Momentum bearish saudável
                                        
                                        3. **Preço Subindo, OI Caindo, Volume Alto:**
                                           - Possível short squeeze
                                           - Traders em posição vendida estão fechando posições
                                           - Pode indicar rally temporário em vez de mudança de tendência
                                        
                                        4. **Preço Caindo, OI Caindo, Volume Alto:**
                                           - Possível liquidação de posições compradas
                                           - Traders em posição comprada estão saindo do mercado
                                           - Pode ser estágio final de um movimento de queda
                                        
                                        5. **Preço Se Movendo, Volume Baixo, OI Estável:**
                                           - Movimento de preço não confiável
                                           - Baixa convicção e participação
                                           - Alto risco de reversão
                                        
                                        #### 💹 Volume/OI Ratio:
                                        
                                        - **Ratio Alto:** Indica alta rotatividade em relação ao tamanho do mercado
                                           - Característica de mercados especulativos/voláteis
                                           - Comum em topos locais ou períodos de aumento de volatilidade
                                        
                                        - **Ratio Baixo:** Sugere baixa rotatividade em relação ao tamanho do mercado
                                           - Pode indicar compromisso de longo prazo dos participantes
                                           - Comum em fases de acumulação/distribuição
                                        
                                        #### 🔄 Ciclos de Mercado e OI:
                                        
                                        - **Início de Tendência:** OI começa a crescer gradualmente
                                        - **Tendência Estabelecida:** OI cresce consistentemente
                                        - **Fase Final/Exaustão:** OI atinge picos extremos
                                        - **Consolidação/Correção:** OI diminui ou se estabiliza
                                        
                                        #### ⚠️ Sinais de Alerta:
                                        
                                        - **Divergências:** Preço sobe para novos topos mas OI não confirma
                                        - **Extremos de OI:** Níveis historicamente altos de OI podem indicar saturação
                                        - **Quedas abruptas no OI:** Podem sinalizar grandes players saindo do mercado
                                        - **Volume em declínio com OI estável:** Participantes existentes segurando posições, mas novos não estão entrando
                                        """)
                            else:
                                 st.error("Dados de métricas insuficientes após limpeza.")
                    else:
                        st.error("Colunas necessárias não encontradas nos dados de métricas")
                        st.write("Colunas necessárias:", ['timestamp/date', 'volume', 'open_interest'])
                        st.write("Colunas disponíveis:", list(metrics.columns) if not metrics.empty else "Nenhuma")
                else:
                    st.error("Não foi possível processar os dados de métricas")
            
            # Processar dados de open interest histórico de opções
            if oi_historical is not None:
                oi_historical = self._process_api_data(oi_historical)
                if not oi_historical.empty:
                    st.success(f"Dados de OI histórico de opções processados com sucesso. Colunas: {list(oi_historical.columns)}")
                    
                    # Verificar colunas necessárias para OI histórico
                    timestamp_col_oi = next((col for col in ['timestamp', 'date', 'time'] if col in oi_historical.columns), None)
                    calls_col_oi = next((col for col in ['total_calls_oi', 'calls', 'call_oi'] if col in oi_historical.columns), None)
                    puts_col_oi = next((col for col in ['total_puts_oi', 'puts', 'put_oi'] if col in oi_historical.columns), None)
                    
                    if timestamp_col_oi and calls_col_oi and puts_col_oi:
                        with st.expander("📜 Análise do Mercado de Opções", expanded=True):
                            st.markdown("#### 📜 Análise Histórica de Open Interest de Opções")
                            
                            # Converter colunas numéricas e timestamp
                            oi_historical[calls_col_oi] = pd.to_numeric(oi_historical[calls_col_oi], errors='coerce')
                            oi_historical[puts_col_oi] = pd.to_numeric(oi_historical[puts_col_oi], errors='coerce')
                            if oi_historical[timestamp_col_oi].dtype == 'object':
                                 oi_historical[timestamp_col_oi] = pd.to_datetime(oi_historical[timestamp_col_oi], errors='coerce')
                            oi_historical = oi_historical.dropna(subset=[timestamp_col_oi, calls_col_oi, puts_col_oi])

                            if len(oi_historical) > 0:
                                oi_historical = oi_historical.sort_values(by=timestamp_col_oi)
                                
                                # Calcular métricas derivadas
                                # 1. Put/Call Ratio
                                oi_historical['pc_ratio'] = oi_historical[puts_col_oi] / oi_historical[calls_col_oi].replace(0, 1) # Evitar divisão por zero
                                
                                # 2. Total OI
                                oi_historical['total_oi'] = oi_historical[calls_col_oi] + oi_historical[puts_col_oi]
                                
                                # 3. % Calls e % Puts
                                oi_historical['pct_calls'] = (oi_historical[calls_col_oi] / oi_historical['total_oi'] * 100).replace(np.nan, 0)
                                oi_historical['pct_puts'] = (oi_historical[puts_col_oi] / oi_historical['total_oi'] * 100).replace(np.nan, 0)
                                
                                # 4. Médias móveis
                                oi_historical['pc_ratio_ma7'] = oi_historical['pc_ratio'].rolling(window=7).mean()
                                oi_historical['total_oi_ma7'] = oi_historical['total_oi'].rolling(window=7).mean()
                                
                                # 5. Mudanças diárias
                                oi_historical['total_oi_daily_change'] = oi_historical['total_oi'].pct_change() * 100
                                oi_historical['pc_ratio_daily_change'] = oi_historical['pc_ratio'].diff()
                                
                                # 6. Benchmark para classificar o Put/Call ratio
                                # Classificar o P/C ratio em percentis históricos
                                if len(oi_historical) >= 30:  # Precisamos de dados suficientes
                                    oi_historical['pc_ratio_percentile'] = oi_historical['pc_ratio'].rolling(window=30, min_periods=10).apply(
                                        lambda x: stats.percentileofscore(x, x.iloc[-1])
                                    )

                                # Gráfico 1: Total OI Calls vs Puts
                                fig = go.Figure()
                                
                                # Adicionar área para Calls
                                fig.add_trace(go.Scatter(
                                    x=oi_historical[timestamp_col_oi], 
                                    y=oi_historical[calls_col_oi], 
                                    mode='lines', 
                                    name='Total Calls OI', 
                                    line=dict(color='green', width=2),
                                    fill='tozeroy'
                                ))
                                
                                # Adicionar área para Puts
                                fig.add_trace(go.Scatter(
                                    x=oi_historical[timestamp_col_oi], 
                                    y=oi_historical[puts_col_oi], 
                                    mode='lines', 
                                    name='Total Puts OI', 
                                    line=dict(color='red', width=2),
                                    fill='tozeroy'
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} - Open Interest Total de Opções (Calls vs Puts)",
                                    xaxis_title="Data", 
                                    yaxis_title="Open Interest", 
                                    hovermode='x unified',
                                    legend=dict(orientation="h", y=1.1)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Gráfico 2: Total OI com % Calls/Puts
                                fig = make_subplots(specs=[[{"secondary_y": True}]])
                                
                                # Adicionar linha para Total OI
                                fig.add_trace(
                                    go.Scatter(
                                        x=oi_historical[timestamp_col_oi],
                                        y=oi_historical['total_oi'],
                                        name='Total OI',
                                        line=dict(color='blue', width=2)
                                    ),
                                    secondary_y=False
                                )
                                
                                # Adicionar linha para MA7 do Total OI
                                fig.add_trace(
                                    go.Scatter(
                                        x=oi_historical[timestamp_col_oi],
                                        y=oi_historical['total_oi_ma7'],
                                        name='Total OI MA7',
                                        line=dict(color='blue', width=1.5, dash='dot')
                                    ),
                                    secondary_y=False
                                )
                                
                                # Adicionar linha para % Calls
                                fig.add_trace(
                                    go.Scatter(
                                        x=oi_historical[timestamp_col_oi],
                                        y=oi_historical['pct_calls'],
                                        name='% Calls',
                                        line=dict(color='green', width=1.5)
                                    ),
                                    secondary_y=True
                                )
                                
                                fig.update_layout(
                                    title=f"{symbol} - Open Interest Total e % de Calls",
                                    xaxis_title="Data",
                                    hovermode='x unified',
                                    legend=dict(orientation="h", y=1.1)
                                )
                                
                                fig.update_yaxes(title_text="Open Interest Total", secondary_y=False)
                                fig.update_yaxes(title_text="% de Calls", range=[0, 100], secondary_y=True)
                                
                                # Adicionar linha de referência para 50%
                                fig.add_hline(y=50, line_dash="dash", line_color="gray", secondary_y=True)
                                
                                st.plotly_chart(fig, use_container_width=True)

                                # Gráfico 3: Put/Call Ratio Histórico com zonas
                                fig = go.Figure()
                                
                                # Adicionar zonas de referência
                                fig.add_shape(
                                    type="rect",
                                    x0=oi_historical[timestamp_col_oi].min(),
                                    x1=oi_historical[timestamp_col_oi].max(),
                                    y0=0.8,
                                    y1=0.5,
                                    fillcolor="rgba(0, 255, 0, 0.1)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                                
                                fig.add_shape(
                                    type="rect",
                                    x0=oi_historical[timestamp_col_oi].min(),
                                    x1=oi_historical[timestamp_col_oi].max(),
                                    y0=0.5,
                                    y1=0,
                                    fillcolor="rgba(0, 255, 0, 0.2)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                                
                                fig.add_shape(
                                    type="rect",
                                    x0=oi_historical[timestamp_col_oi].min(),
                                    x1=oi_historical[timestamp_col_oi].max(),
                                    y0=1.2,
                                    y1=0.8,
                                    fillcolor="rgba(255, 165, 0, 0.1)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                                
                                fig.add_shape(
                                    type="rect",
                                    x0=oi_historical[timestamp_col_oi].min(),
                                    x1=oi_historical[timestamp_col_oi].max(),
                                    y0=2,
                                    y1=1.2,
                                    fillcolor="rgba(255, 0, 0, 0.1)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                                
                                # Adicionar linha de P/C Ratio
                                fig.add_trace(go.Scatter(
                                    x=oi_historical[timestamp_col_oi],
                                    y=oi_historical['pc_ratio'],
                                    mode='lines',
                                    name='Put/Call Ratio',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Adicionar média móvel do P/C Ratio
                                fig.add_trace(go.Scatter(
                                    x=oi_historical[timestamp_col_oi],
                                    y=oi_historical['pc_ratio_ma7'],
                                    mode='lines',
                                    name='P/C Ratio MA7',
                                    line=dict(color='purple', width=1.5, dash='dot')
                                ))
                                
                                # Adicionar linhas de referência
                                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
                                fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.7)
                                fig.add_hline(y=1.2, line_dash="dot", line_color="red", opacity=0.7)
                                
                                fig.update_layout(
                                    title=f"{symbol} - Histórico do Put/Call Ratio",
                                    xaxis_title="Data",
                                    yaxis_title="Ratio",
                                    hovermode='x unified',
                                    legend=dict(orientation="h", y=1.1)
                                )
                                
                                # Adicionar anotações para zonas
                                fig.add_annotation(
                                    x=oi_historical[timestamp_col_oi].iloc[-1],
                                    y=0.3,
                                    text="Extremo Bullish",
                                    showarrow=False,
                                    font=dict(color="green")
                                )
                                
                                fig.add_annotation(
                                    x=oi_historical[timestamp_col_oi].iloc[-1],
                                    y=0.7,
                                    text="Bullish",
                                    showarrow=False,
                                    font=dict(color="green")
                                )
                                
                                fig.add_annotation(
                                    x=oi_historical[timestamp_col_oi].iloc[-1],
                                    y=1.1,
                                    text="Neutro",
                                    showarrow=False,
                                    font=dict(color="gray")
                                )
                                
                                fig.add_annotation(
                                    x=oi_historical[timestamp_col_oi].iloc[-1],
                                    y=1.6,
                                    text="Bearish",
                                    showarrow=False,
                                    font=dict(color="red")
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)

                                # Métricas atuais
                                latest_calls = int(oi_historical[calls_col_oi].iloc[-1])
                                latest_puts = int(oi_historical[puts_col_oi].iloc[-1])
                                latest_pc_ratio = oi_historical['pc_ratio'].iloc[-1]
                                latest_total_oi = oi_historical['total_oi'].iloc[-1]
                                latest_pct_calls = oi_historical['pct_calls'].iloc[-1]
                                
                                # Determinar o sentimento com base no Put/Call ratio
                                if latest_pc_ratio > 1.3:
                                    sentiment = "Fortemente Bearish (Possível Contrarian Bullish)"
                                    sentiment_color = "red"
                                elif latest_pc_ratio > 1.1:
                                    sentiment = "Bearish"
                                    sentiment_color = "red"
                                elif latest_pc_ratio > 0.9:
                                    sentiment = "Neutro"
                                    sentiment_color = "gray"
                                elif latest_pc_ratio > 0.6:
                                    sentiment = "Bullish"
                                    sentiment_color = "green"
                                else:
                                    sentiment = "Fortemente Bullish (Possível Contrarian Bearish)"
                                    sentiment_color = "green"
                                
                                # Interpretar tendência do put/call ratio
                                pc_trend = ""
                                if len(oi_historical) >= 7:
                                    pc_7d_ago = oi_historical['pc_ratio'].iloc[-7]
                                    pc_change = latest_pc_ratio - pc_7d_ago
                                    
                                    if pc_change > 0.2:
                                        pc_trend = "Movimento significativo em direção bearish (aumento no P/C ratio)"
                                    elif pc_change > 0.1:
                                        pc_trend = "Movimento moderado em direção bearish"
                                    elif pc_change < -0.2:
                                        pc_trend = "Movimento significativo em direção bullish (queda no P/C ratio)"
                                    elif pc_change < -0.1:
                                        pc_trend = "Movimento moderado em direção bullish"
                                    else:
                                        pc_trend = "Estável, sem mudança significativa na última semana"
                                
                                # Apresentar em colunas
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Put/Call Ratio Atual", f"{latest_pc_ratio:.2f}")
                                    st.markdown(f"<span style='color:{sentiment_color}'>**Sentimento**: {sentiment}</span>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.metric("Total Calls OI", f"{latest_calls:,}")
                                    st.metric("% de Calls", f"{latest_pct_calls:.1f}%")
                                
                                with col3:
                                    st.metric("Total Puts OI", f"{latest_puts:,}")
                                    st.metric("% de Puts", f"{100 - latest_pct_calls:.1f}%")
                                
                                # Adicionar análise percentil se disponível
                                percentile_analysis = ""
                                if 'pc_ratio_percentile' in oi_historical.columns and not pd.isna(oi_historical['pc_ratio_percentile'].iloc[-1]):
                                    percentile = oi_historical['pc_ratio_percentile'].iloc[-1]
                                    
                                    if percentile > 80:
                                        percentile_desc = f"P/C ratio atual está no {percentile:.0f}º percentil histórico (extremamente alto)"
                                    elif percentile > 60:
                                        percentile_desc = f"P/C ratio atual está no {percentile:.0f}º percentil histórico (acima da média)"
                                    elif percentile > 40:
                                        percentile_desc = f"P/C ratio atual está no {percentile:.0f}º percentil histórico (na média)"
                                    elif percentile > 20:
                                        percentile_desc = f"P/C ratio atual está no {percentile:.0f}º percentil histórico (abaixo da média)"
                                    else:
                                        percentile_desc = f"P/C ratio atual está no {percentile:.0f}º percentil histórico (extremamente baixo)"
                                    
                                    percentile_analysis = f"**Análise de Percentil:** {percentile_desc}"
                                
                                # Análise final
                                st.info(f"""
                                **Análise do Mercado de Opções:**
                                
                                - **Sentimento Atual:** {sentiment}
                                - **Tendência do Put/Call Ratio:** {pc_trend if pc_trend else "Dados insuficientes para análise de tendência"}
                                {f"- {percentile_analysis}" if percentile_analysis else ""}
                                - **Total OI:** {latest_total_oi:,.0f} contratos
                                
                                **Interpretação do Put/Call Ratio:**
                                
                                - **> 1.2:** Geralmente bearish, mas pode ser contrarian bullish em extremos
                                - **0.8 - 1.2:** Zona neutro/equilibrada
                                - **< 0.8:** Geralmente bullish, mas pode ser contrarian bearish em extremos
                                
                                O mercado de opções oferece uma visão das expectativas dos traders institucionais, que tendem a ser mais sofisticados. Valores extremos no P/C ratio frequentemente funcionam como indicadores contrários.
                                """)
                                
                                # Conteúdo educativo
                                with st.expander("📚 Entendendo o Mercado de Opções e o Put/Call Ratio", expanded=False):
                                    st.markdown("""
                                    ### 📚 Guia do Mercado de Opções e do Put/Call Ratio
                                    
                                    #### 🔍 O que são Opções?
                                    
                                    Opções são contratos financeiros derivativos que dão ao titular o direito, mas não a obrigação, de comprar (call) ou vender (put) um ativo a um preço predeterminado dentro de um período específico.
                                    
                                    **Características chave:**
                                    - **Call Options:** Direito de COMPRAR o ativo subjacente
                                    - **Put Options:** Direito de VENDER o ativo subjacente
                                    - **Strike Price:** Preço predeterminado para exercício
                                    - **Expiration Date:** Data de vencimento do contrato
                                    
                                    #### 📊 O que é o Put/Call Ratio?
                                    
                                    O Put/Call Ratio (P/C Ratio) é uma métrica que compara o volume ou open interest de opções de venda (puts) em relação às opções de compra (calls).
                                    
                                    **Cálculo:** P/C Ratio = Total de Puts / Total de Calls
                                    
                                    #### 💹 Interpretação Tradicional vs. Contrária:
                                    
                                    1. **Interpretação Tradicional:**
                                       - **P/C Ratio Alto (>1.2):** Sentimento bearish, traders esperam queda
                                       - **P/C Ratio Neutro (0.8-1.2):** Sentimento equilibrado
                                       - **P/C Ratio Baixo (<0.8):** Sentimento bullish, traders esperam alta
                                    
                                    2. **Interpretação Contrária:**
                                       - **P/C Ratio Extremamente Alto:** Possível pessimismo excessivo, potencial sinal de compra
                                       - **P/C Ratio Extremamente Baixo:** Possível otimismo excessivo, potencial sinal de venda
                                    
                                    #### 🛠️ Estratégias Utilizando o P/C Ratio:
                                    
                                    - **Identificação de Extremos:** Usar percentis históricos para identificar valores anormais
                                    - **Divergências:** Comparar P/C ratio com movimento de preço para identificar divergências
                                    - **Análise de Tendência:** Observar a direção do P/C ratio para confirmar ou antecipar mudanças
                                    - **Combinação com Outros Indicadores:** Usar em conjunto com análise técnica tradicional
                                    
                                    #### ⚠️ Limitações a Considerar:
                                    
                                    - Nem todas as opções são especulativas (muitas são usadas para hedge)
                                    - O tamanho dos contratos não é considerado (um grande trader pode desequilibrar o ratio)
                                    - Diferentes vencimentos e strikes podem contar histórias diferentes
                                    - Extremos frequentemente ocorrem próximos a eventos programados
                                    
                                    #### 🔄 P/C Ratio em Diferentes Timeframes:
                                    
                                    - **Diário:** Mais volátil, útil para traders de curto prazo
                                    - **Semanal:** Útil para identificar mudanças de sentimento de médio prazo
                                    - **Mensal:** Melhor para análise de tendências de longo prazo
                                    
                                    #### 💡 Melhores Práticas de Uso:
                                    
                                    - Compare valores atuais com médias históricas e extremos
                                    - Use percentis para contextualizar os valores
                                    - Considere o P/C ratio como uma ferramenta complementar, não isolada
                                    - Preste atenção especial a mudanças na direção do ratio, não apenas aos valores absolutos
                                    """)
                            else:
                                 st.error("Dados de OI histórico insuficientes após limpeza.")
                                 st.write("Colunas necessárias:", ['timestamp', 'total_calls_oi', 'total_puts_oi'])
                                 st.write("Colunas disponíveis:", list(oi_historical.columns))
                    else:
                        st.error("Colunas necessárias não encontradas nos dados de OI histórico de opções")
                        st.write("Colunas necessárias:", ['timestamp', 'total_calls_oi', 'total_puts_oi'])
                        st.write("Colunas disponíveis:", list(oi_historical.columns))
                else:
                    st.error("Não foi possível processar os dados de OI histórico de opções")
            
        except Exception as e:
            st.error(f"Erro na análise de dados resumidos: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def analyze_all_data(self, symbol="BTCUSDT"):
        """
        Executa todas as análises em uma única chamada e apresenta um dashboard completo.
        
        Args:
            symbol: Símbolo da criptomoeda para análise
        """
        st.title(f"🔍 Dashboard Completo de Análise de Mercado para {symbol}")
        
        # Resumo executivo
        with st.expander("📋 Resumo Executivo", expanded=True):
            st.markdown("### 📋 Resumo Executivo do Mercado")
            st.markdown("""
            Este dashboard fornece uma análise abrangente das condições atuais do mercado de criptomoedas.
            Cada seção pode ser expandida para visualizar análises detalhadas, gráficos interativos e métricas essenciais.
            
            **Principais componentes:**
            
            1. **Amplitude de Mercado:** Visualize a força geral do mercado, analisando o número de criptomoedas em máximas e mínimas históricas.
            
            2. **Dados Governamentais:** Analise dados COT (Commitment of Traders) e rendimentos do Tesouro para compreender correlações macro.
            
            3. **Preços e Análise Técnica:** Explore preços históricos com indicadores técnicos avançados.
            
            4. **Métricas de Risco:** Avalie correlações entre ativos e métricas VaR para gerenciamento de risco.
            
            5. **Dados de Derivativos:** Acompanhe funding rates, volume, open interest e métricas do mercado de opções.
            
            Cada seção inclui conteúdo educacional para ajudar a compreender os dados apresentados.
            """)
        
        # Executar todas as análises
        try:
            with st.spinner("Analisando dados de amplitude de mercado..."):
                self.analyze_breadth_data()
            
            with st.spinner("Analisando dados governamentais..."):
                self.analyze_government_data()
            
            with st.spinner(f"Analisando dados OHLC para {symbol}..."):
                self.analyze_ohlc_data(symbol)
            
            with st.spinner(f"Analisando métricas de risco para {symbol}..."):
                self.analyze_risk_metrics(symbol)
            
            with st.spinner(f"Analisando dados resumidos para {symbol}..."):
                self.analyze_summary_data(symbol)
                
        except Exception as e:
            st.error(f"Erro ao executar análises: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def generate_market_report(self, symbol="BTCUSDT", include_sections=None):
        """
        Gera um relatório de mercado completo com as análises selecionadas.
        
        Args:
            symbol: Símbolo da criptomoeda para análise
            include_sections: Lista de seções a incluir (None = todas)
        
        Returns:
            String formatada em Markdown com o relatório completo
        """
        if include_sections is None:
            include_sections = ["breadth", "government", "ohlc", "risk", "summary"]
        
        # Iniciar o relatório
        report = f"""
        # Relatório de Mercado: {symbol}
        **Data de geração:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        ## Resumo Executivo
        
        Este relatório fornece uma análise abrangente das condições atuais do mercado de criptomoedas com foco em {symbol}.
        As análises incluem métricas de amplitude de mercado, dados governamentais, análise técnica,
        métricas de risco e dados de derivativos.
        
        ---
        
        """
        
        # Coletar dados para cada seção
        sections_data = {}
        
        # Amplitude de mercado
        if "breadth" in include_sections:
            try:
                highs_lows = self.fetcher.load_data_file("breadth/52wk_highs_lows.csv")
                ma_tracking = self.fetcher.load_data_file("breadth/moving_average_tracking.csv")
                
                if highs_lows is not None:
                    highs_lows = self._process_api_data(highs_lows)
                    if not highs_lows.empty and all(col in highs_lows.columns for col in ['timestamp', 'new_highs', 'new_lows']):
                        for col in ['new_highs', 'new_lows', 'total']:
                            if col in highs_lows.columns:
                                highs_lows[col] = pd.to_numeric(highs_lows[col], errors='coerce')
                        if highs_lows['timestamp'].dtype == 'object':
                            highs_lows['timestamp'] = pd.to_datetime(highs_lows['timestamp'], errors='coerce')
                        highs_lows = highs_lows.dropna(subset=['timestamp', 'new_highs', 'new_lows'])
                        highs_lows = highs_lows.sort_values('timestamp')
                        
                        # Calcular métricas
                        if len(highs_lows) >= 2:
                            current_highs = highs_lows['new_highs'].iloc[-1]
                            current_lows = highs_lows['new_lows'].iloc[-1]
                            prev_highs = highs_lows['new_highs'].iloc[-2]
                            prev_lows = highs_lows['new_lows'].iloc[-2]
                            
                            delta_highs = current_highs - prev_highs
                            delta_lows = current_lows - prev_lows
                            
                            # Calcular índice de força
                            highs_lows['strength_index'] = (highs_lows['new_highs'] - highs_lows['new_lows']) / \
                                                        (highs_lows['new_highs'] + highs_lows['new_lows']).replace(0, 1)
                            current_strength = highs_lows['strength_index'].iloc[-1]
                            
                            # Determinar fase de mercado
                            if current_strength > 0.5:
                                market_phase = "Extremamente Forte"
                            elif current_strength > 0.2:
                                market_phase = "Forte"
                            elif current_strength > -0.2:
                                market_phase = "Neutro"
                            elif current_strength > -0.5:
                                market_phase = "Fraco"
                            else:
                                market_phase = "Extremamente Fraco"
                            
                            sections_data["breadth"] = {
                                "current_highs": current_highs,
                                "current_lows": current_lows,
                                "delta_highs": delta_highs,
                                "delta_lows": delta_lows,
                                "strength_index": current_strength,
                                "market_phase": market_phase
                            }
                
                if ma_tracking is not None:
                    ma_tracking = self._process_api_data(ma_tracking)
                    ma_cols = [col for col in ma_tracking.columns if 'above_ma' in col.lower()]
                    
                    if not ma_tracking.empty and 'timestamp' in ma_tracking.columns and ma_cols:
                        for col in ma_cols:
                            ma_tracking[col] = pd.to_numeric(ma_tracking[col], errors='coerce')
                        if ma_tracking['timestamp'].dtype == 'object':
                            ma_tracking['timestamp'] = pd.to_datetime(ma_tracking['timestamp'], errors='coerce')
                        ma_tracking = ma_tracking.dropna(subset=[*ma_cols, 'timestamp'])
                        ma_tracking = ma_tracking.sort_values('timestamp')
                        
                        if len(ma_tracking) >= 2:
                            ma_values = {}
                            for col in ma_cols:
                                ma_period = col.split('_')[-1].upper()
                                ma_values[f"above_ma{ma_period}"] = float(ma_tracking[col].iloc[-1])
                            
                            sections_data.setdefault("breadth", {})
                            sections_data["breadth"].update(ma_values)
            except Exception as e:
                print(f"Erro ao coletar dados de amplitude: {str(e)}")
        
        # Dados governamentais
        if "government" in include_sections:
            try:
                current_year = datetime.now().year
                cot_data = self.fetcher.load_data_file(f"gov/cftc_cot_bitcoin_{current_year}.csv")
                treasury_data = self.fetcher.load_data_file(f"gov/treasury_yields_{current_year}.csv")
                
                if cot_data is not None:
                    cot_data = self._process_api_data(cot_data)
                    if not cot_data.empty:
                        date_col = 'timestamp' if 'timestamp' in cot_data.columns else None
                        long_col = 'longs' if 'longs' in cot_data.columns else None
                        short_col = 'shorts' if 'shorts' in cot_data.columns else None
                        
                        if date_col and long_col and short_col:
                            for col in [long_col, short_col]:
                                cot_data[col] = pd.to_numeric(cot_data[col], errors='coerce')
                            cot_data = cot_data.dropna(subset=[date_col, long_col, short_col])
                            cot_data = cot_data.sort_values(by=date_col)
                            
                            if len(cot_data) >= 2:
                                latest_long = float(cot_data[long_col].iloc[-1])
                                latest_short = float(cot_data[short_col].iloc[-1])
                                latest_net = latest_long - latest_short
                                
                                sections_data["government"] = {
                                    "cot_longs": latest_long,
                                    "cot_shorts": latest_short,
                                    "cot_net": latest_net
                                }
                
                if treasury_data is not None:
                    treasury_data = self._process_api_data(treasury_data)
                    if not treasury_data.empty:
                        date_col = next((col for col in ['date', 'timestamp', 'time', 'report_date'] 
                                        if col in treasury_data.columns), None)
                        yields_cols = [col for col in treasury_data.columns 
                                    if ('yield' in col.lower() or 'yr' in col.lower() or 'year' in col.lower() or 'month' in col.lower())
                                    and col != date_col]
                        
                        if date_col and yields_cols:
                            for col in yields_cols:
                                treasury_data[col] = pd.to_numeric(treasury_data[col], errors='coerce')
                            if treasury_data[date_col].dtype == 'object':
                                treasury_data[date_col] = pd.to_datetime(treasury_data[date_col], errors='coerce')
                            treasury_data = treasury_data.dropna(subset=[date_col, *yields_cols])
                            treasury_data = treasury_data.sort_values(by=date_col)
                            
                            if len(treasury_data) > 0:
                                # Procurar colunas para 2Y e 10Y
                                two_year_col = next((col for col in yields_cols if '2' in col and ('year' in col.lower() or 'yr' in col.lower())), None)
                                ten_year_col = next((col for col in yields_cols if '10' in col and ('year' in col.lower() or 'yr' in col.lower())), None)
                                
                                if two_year_col and ten_year_col:
                                    latest_2y = float(treasury_data[two_year_col].iloc[-1])
                                    latest_10y = float(treasury_data[ten_year_col].iloc[-1])
                                    curve_steepness = latest_10y - latest_2y
                                    
                                    sections_data.setdefault("government", {})
                                    sections_data["government"].update({
                                        "yield_2y": latest_2y,
                                        "yield_10y": latest_10y,
                                        "curve_steepness": curve_steepness
                                    })
            except Exception as e:
                print(f"Erro ao coletar dados governamentais: {str(e)}")
        
        # Dados OHLC
        if "ohlc" in include_sections:
            try:
                spot_data = self.fetcher.load_data_file(f"ohlc/binance_spot_{symbol}_daily.csv")
                
                if spot_data is not None:
                    spot_data = self._process_api_data(spot_data)
                    if not spot_data.empty:
                        col_mappings = {
                            'timestamp': ['timestamp', 'date', 'time', 'Date', 'Time'],
                            'open': ['open', 'open_price', 'opening_price', 'Open'],
                            'high': ['high', 'high_price', 'highest_price', 'High'],
                            'low': ['low', 'low_price', 'lowest_price', 'Low'],
                            'close': ['close', 'close_price', 'closing_price', 'Close']
                        }
                        
                        def find_column(df, possible_names):
                            return next((col for col in possible_names if col in df.columns), None)
                        
                        spot_cols = {req: find_column(spot_data, options) for req, options in col_mappings.items()}
                        
                        if all(spot_cols.values()):
                            # Convert data types
                            for col_key, col_name in spot_cols.items():
                                if col_key == 'timestamp':
                                    if spot_data[col_name].dtype == 'object':
                                        spot_data[col_name] = pd.to_datetime(spot_data[col_name], errors='coerce')
                                else:
                                    spot_data[col_name] = pd.to_numeric(spot_data[col_name], errors='coerce')
                            
                            spot_data = spot_data.dropna(subset=list(spot_cols.values()))
                            spot_data = spot_data.sort_values(by=spot_cols['timestamp'])
                            
                            if len(spot_data) > 0:
                                # Get latest price data
                                latest_price = float(spot_data[spot_cols['close']].iloc[-1])
                                
                                # Calculate some technical indicators
                                # RSI
                                delta = spot_data[spot_cols['close']].diff()
                                gain = delta.where(delta > 0, 0)
                                loss = -delta.where(delta < 0, 0)
                                avg_gain = gain.rolling(window=14).mean()
                                avg_loss = loss.rolling(window=14).mean()
                                rs = avg_gain / avg_loss.replace(0, 0.001)
                                rsi = 100 - (100 / (1 + rs))
                                latest_rsi = float(rsi.iloc[-1])
                                
                                # Moving Averages
                                sma50 = spot_data[spot_cols['close']].rolling(window=50).mean()
                                sma200 = spot_data[spot_cols['close']].rolling(window=200).mean()
                                
                                latest_sma50 = float(sma50.iloc[-1])
                                latest_sma200 = float(sma200.iloc[-1])
                                
                                # Price vs MAs
                                price_vs_sma50 = (latest_price / latest_sma50 - 1) * 100
                                price_vs_sma200 = (latest_price / latest_sma200 - 1) * 100
                                
                                # Determine trend
                                if latest_price > latest_sma50 and latest_sma50 > latest_sma200:
                                    trend = "Forte tendência de alta"
                                elif latest_price > latest_sma50 and latest_sma50 < latest_sma200:
                                    trend = "Tendência de alta de curto prazo, tendência de baixa de longo prazo"
                                elif latest_price < latest_sma50 and latest_sma50 > latest_sma200:
                                    trend = "Correção de curto prazo em tendência de alta"
                                else:
                                    trend = "Tendência de baixa"
                                
                                sections_data["ohlc"] = {
                                    "latest_price": latest_price,
                                    "rsi": latest_rsi,
                                    "sma50": latest_sma50,
                                    "sma200": latest_sma200,
                                    "price_vs_sma50": price_vs_sma50,
                                    "price_vs_sma200": price_vs_sma200,
                                    "trend": trend
                                }
            except Exception as e:
                print(f"Erro ao coletar dados OHLC: {str(e)}")
        
        # Métricas de risco
        if "risk" in include_sections:
            try:
                var_data = self.fetcher.load_data_file(f"risk/var_{symbol}.csv")
                
                if var_data is not None:
                    var_data = self._process_api_data(var_data)
                    if not var_data.empty:
                        timestamp_col = 'timestamp' if 'timestamp' in var_data.columns else None
                        var_99_col = 'var_99' if 'var_99' in var_data.columns else None
                        var_95_col = 'var_95' if 'var_95' in var_data.columns else None
                        
                        var_cols = {}
                        if var_99_col: var_cols['var_99'] = var_99_col
                        if var_95_col: var_cols['var_95'] = var_95_col
                        
                        if timestamp_col and var_cols:
                            for col_key, col_name in var_cols.items():
                                var_data[col_name] = pd.to_numeric(var_data[col_name], errors='coerce')
                            if var_data[timestamp_col].dtype == 'object':
                                var_data[timestamp_col] = pd.to_datetime(var_data[timestamp_col], errors='coerce')
                            var_data = var_data.dropna(subset=[timestamp_col, *var_cols.values()])
                            var_data = var_data.sort_values(by=timestamp_col)
                            
                            if len(var_data) > 0:
                                risk_data = {}
                                for var_type, col in var_cols.items():
                                    current_var = float(var_data[col].iloc[-1])
                                    risk_data[var_type] = current_var
                                
                                # Determine risk level
                                if 'var_99' in risk_data:
                                    var99 = risk_data['var_99']
                                    if var99 > 7.5:
                                        risk_level = "Extremamente Alto"
                                    elif var99 > 5:
                                        risk_level = "Muito Alto"
                                    elif var99 > 3.5:
                                        risk_level = "Alto"
                                    elif var99 > 2.5:
                                        risk_level = "Moderado"
                                    else:
                                        risk_level = "Baixo"
                                    
                                    risk_data['risk_level'] = risk_level
                                
                                sections_data["risk"] = risk_data
            except Exception as e:
                print(f"Erro ao coletar dados de risco: {str(e)}")
        
        # Dados resumidos
        if "summary" in include_sections:
            try:
                funding = self.fetcher.load_data_file(f"summary/binance_futures_funding_{symbol}.csv")
                oi_historical = self.fetcher.load_data_file(f"summary/binance_options_oi_{symbol}.csv")
                
                summary_data = {}
                
                if funding is not None:
                    funding = self._process_api_data(funding)
                    if not funding.empty:
                        timestamp_col = next((col for col in ['timestamp', 'date', 'time', 'Date', 'unix_timestamp'] 
                                            if col in funding.columns), None)
                        funding_col = next((col for col in funding.columns 
                                        if any(term in col.lower() for term in ['funding', 'funding_rate', 'last_funding_rate'])), None)
                        
                        if timestamp_col and funding_col:
                            funding[funding_col] = pd.to_numeric(funding[funding_col], errors='coerce')
                            if 'unix' in timestamp_col.lower():
                                funding['timestamp'] = pd.to_datetime(funding[timestamp_col], unit='ms', errors='coerce')
                                timestamp_col = 'timestamp'
                            elif funding[timestamp_col].dtype == 'object':
                                funding[timestamp_col] = pd.to_datetime(funding[timestamp_col], errors='coerce')
                            
                            funding = funding.dropna(subset=[timestamp_col, funding_col])
                            funding = funding.sort_values(by=timestamp_col)
                            
                            if len(funding) > 0:
                                funding_values = funding[funding_col].astype(float)
                                if funding_values.abs().max() < 0.1:
                                    funding[funding_col] = funding_values * 100
                                
                                last_value = float(funding[funding_col].iloc[-1])
                                mean_24h = float(funding[funding_col].rolling(window=8).mean().iloc[-1]) if len(funding) >= 8 else float('nan')
                                
                                summary_data['funding_rate'] = last_value
                                summary_data['funding_ma8'] = mean_24h
                                
                                # Determine funding regime
                                if last_value > 0.05:
                                    funding_regime = "Fortemente Positivo (Mercado Otimista)"
                                elif last_value > 0:
                                    funding_regime = "Levemente Positivo"
                                elif last_value > -0.05:
                                    funding_regime = "Levemente Negativo"
                                else:
                                    funding_regime = "Fortemente Negativo (Mercado Pessimista)"
                                
                                summary_data['funding_regime'] = funding_regime
                
                if oi_historical is not None:
                    oi_historical = self._process_api_data(oi_historical)
                    if not oi_historical.empty:
                        timestamp_col_oi = next((col for col in ['timestamp', 'date', 'time'] if col in oi_historical.columns), None)
                        calls_col_oi = next((col for col in ['total_calls_oi', 'calls', 'call_oi'] if col in oi_historical.columns), None)
                        puts_col_oi = next((col for col in ['total_puts_oi', 'puts', 'put_oi'] if col in oi_historical.columns), None)
                        
                        if timestamp_col_oi and calls_col_oi and puts_col_oi:
                            oi_historical[calls_col_oi] = pd.to_numeric(oi_historical[calls_col_oi], errors='coerce')
                            oi_historical[puts_col_oi] = pd.to_numeric(oi_historical[puts_col_oi], errors='coerce')
                            if oi_historical[timestamp_col_oi].dtype == 'object':
                                oi_historical[timestamp_col_oi] = pd.to_datetime(oi_historical[timestamp_col_oi], errors='coerce')
                            
                            oi_historical = oi_historical.dropna(subset=[timestamp_col_oi, calls_col_oi, puts_col_oi])
                            oi_historical = oi_historical.sort_values(by=timestamp_col_oi)
                            
                            if len(oi_historical) > 0:
                                oi_historical['pc_ratio'] = oi_historical[puts_col_oi] / oi_historical[calls_col_oi].replace(0, 1)
                                oi_historical['pct_calls'] = (oi_historical[calls_col_oi] / (oi_historical[calls_col_oi] + oi_historical[puts_col_oi]) * 100).replace(np.nan, 0)
                                
                                latest_pc_ratio = float(oi_historical['pc_ratio'].iloc[-1])
                                latest_pct_calls = float(oi_historical['pct_calls'].iloc[-1])
                                
                                summary_data['put_call_ratio'] = latest_pc_ratio
                                summary_data['pct_calls'] = latest_pct_calls
                                
                                # Determine options sentiment
                                if latest_pc_ratio > 1.3:
                                    options_sentiment = "Fortemente Bearish (Possível Contrarian Bullish)"
                                elif latest_pc_ratio > 1.1:
                                    options_sentiment = "Bearish"
                                elif latest_pc_ratio > 0.9:
                                    options_sentiment = "Neutro"
                                elif latest_pc_ratio > 0.6:
                                    options_sentiment = "Bullish"
                                else:
                                    options_sentiment = "Fortemente Bullish (Possível Contrarian Bearish)"
                                
                                summary_data['options_sentiment'] = options_sentiment
                
                if summary_data:
                    sections_data["summary"] = summary_data
            except Exception as e:
                print(f"Erro ao coletar dados resumidos: {str(e)}")
        
        # Montagem do relatório
        
        # Amplitude de mercado
        if "breadth" in sections_data:
            breadth_data = sections_data["breadth"]
            
            report += """
            ## Amplitude de Mercado
            
            """
            
            if "current_highs" in breadth_data and "current_lows" in breadth_data:
                report += f"""
                ### Novas Máximas e Mínimas
                
                - **Novas Máximas**: {int(breadth_data['current_highs'])} ({int(breadth_data['delta_highs']):+d} vs. período anterior)
                - **Novas Mínimas**: {int(breadth_data['current_lows'])} ({int(breadth_data['delta_lows']):+d} vs. período anterior)
                - **Índice de Força**: {breadth_data['strength_index']:.2f}
                - **Fase do Mercado**: {breadth_data['market_phase']}
                
                """
            
            if any(key.startswith("above_ma") for key in breadth_data.keys()):
                report += """
                ### Médias Móveis
                
                """
                
                for key, value in breadth_data.items():
                    if key.startswith("above_ma"):
                        ma_period = key.replace("above_ma", "")
                        report += f"- **Acima da MA{ma_period}**: {value:.1f}%\n"
                
                # Determinar viés com base nas médias móveis
                if "above_ma50" in breadth_data and "above_ma200" in breadth_data:
                    ma50 = breadth_data["above_ma50"]
                    ma200 = breadth_data["above_ma200"]
                    
                    if ma50 > 80 and ma200 > 80:
                        ma_bias = "Bull Market Forte"
                    elif ma50 > ma200 and ma50 > 60:
                        ma_bias = "Bull Market"
                    elif ma50 > ma200:
                        ma_bias = "Início de Alta/Recuperação"
                    elif ma50 < 20 and ma200 < 20:
                        ma_bias = "Bear Market Forte"
                    elif ma50 < ma200 and ma50 < 40:
                        ma_bias = "Bear Market"
                    elif ma50 < ma200:
                        ma_bias = "Início de Baixa/Correção"
                    else:
                        ma_bias = "Mercado Neutro/Indeciso"
                    
                    report += f"""
                    - **Viés de Mercado**: {ma_bias}
                    - **Diferença MA50-MA200**: {ma50 - ma200:.1f}%
                    """
        
        # Dados governamentais
        if "government" in sections_data:
            gov_data = sections_data["government"]
            
            report += """
            ## Dados Governamentais
            
            """
            
            if "cot_longs" in gov_data:
                report += f"""
                ### Análise COT (Commitment of Traders)
                
                - **Posições Long**: {int(gov_data['cot_longs']):,}
                - **Posições Short**: {int(gov_data['cot_shorts']):,}
                - **Posição Líquida**: {int(gov_data['cot_net']):+,}
                
                """
            
            if "yield_2y" in gov_data:
                report += f"""
                ### Rendimentos do Tesouro
                
                - **Rendimento 2 Anos**: {gov_data['yield_2y']:.2f}%
                - **Rendimento 10 Anos**: {gov_data['yield_10y']:.2f}%
                - **Inclinação da Curva (10Y-2Y)**: {gov_data['curve_steepness']:.2f}%
                
                """
                
                # Adicionar avaliação da curva
                steepness = gov_data['curve_steepness']
                if steepness > 0.5:
                    curve_regime = "Fortemente Positiva"
                    economic_implication = "Expectativa de crescimento econômico forte"
                elif steepness > 0:
                    curve_regime = "Moderadamente Positiva"
                    economic_implication = "Expectativa de crescimento econômico moderado"
                elif steepness > -0.5:
                    curve_regime = "Achatada a Levemente Invertida"
                    economic_implication = "Possível desaceleração econômica"
                else:
                    curve_regime = "Fortemente Invertida"
                    economic_implication = "Forte sinal de recessão futura"
                
                report += f"""
                - **Regime da Curva**: {curve_regime}
                - **Implicação Econômica**: {economic_implication}
                """
        
        # Dados OHLC
        if "ohlc" in sections_data:
            ohlc_data = sections_data["ohlc"]
            
            report += f"""
            ## Análise Técnica de {symbol}
            
            - **Último Preço**: ${ohlc_data['latest_price']:.2f}
            - **RSI (14)**: {ohlc_data['rsi']:.2f}
            - **SMA 50**: ${ohlc_data['sma50']:.2f} ({ohlc_data['price_vs_sma50']:+.2f}%)
            - **SMA 200**: ${ohlc_data['sma200']:.2f} ({ohlc_data['price_vs_sma200']:+.2f}%)
            - **Tendência**: {ohlc_data['trend']}
            
            """
            
            # Adicionar avaliação de RSI
            rsi = ohlc_data['rsi']
            if rsi > 70:
                rsi_eval = "Região de sobrecompra (>70) - possível exaustão de alta"
            elif rsi < 30:
                rsi_eval = "Região de sobrevenda (<30) - possível exaustão de baixa"
            elif rsi > 50:
                rsi_eval = "Acima de 50 - momentum de alta"
            else:
                rsi_eval = "Abaixo de 50 - momentum de baixa"
            
            report += f"- **Avaliação de RSI**: {rsi_eval}\n"
        
        # Métricas de risco
        if "risk" in sections_data:
            risk_data = sections_data["risk"]
            
            report += """
            ## Análise de Risco
            
            ### Métricas VaR (Value at Risk)
            
            """
            
            # Mostrar métricas de VaR
            for var_type, value in risk_data.items():
                if var_type not in ['risk_level']:
                    report += f"- **{var_type.upper()}**: {value:.2f}%\n"
            
            if 'risk_level' in risk_data:
                report += f"""
                - **Nível de Risco**: {risk_data['risk_level']}
                
                """
        
        # Dados resumidos
        if "summary" in sections_data:
            summary_data = sections_data["summary"]
            
            report += """
            ## Análise de Derivativos
            
            """
            
            if "funding_rate" in summary_data:
                report += f"""
                ### Funding Rate
                
                - **Funding Rate Atual**: {summary_data['funding_rate']:.4f}%
                - **Média de 8 períodos (~24h)**: {summary_data['funding_ma8']:.4f}%
                - **Regime de Funding**: {summary_data['funding_regime']}
                
                """
            
            if "put_call_ratio" in summary_data:
                report += f"""
                ### Análise de Opções
                
                - **Put/Call Ratio**: {summary_data['put_call_ratio']:.2f}
                - **% de Calls**: {summary_data['pct_calls']:.1f}%
                - **Sentimento do Mercado de Opções**: {summary_data['options_sentiment']}
                
                """
        
        # Conclusão
        report += """
        ## Conclusão e Recomendações
        
        """
        
        # Avaliar o sentimento geral com base nos dados coletados
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        if "breadth" in sections_data:
            breadth_data = sections_data["breadth"]
            total_signals += 2
            
            # Índice de força
            if breadth_data.get("strength_index", 0) > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # MA50 vs MA200
            if "above_ma50" in breadth_data and "above_ma200" in breadth_data:
                if breadth_data["above_ma50"] > breadth_data["above_ma200"]:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
        
        if "government" in sections_data:
            gov_data = sections_data["government"]
            
            # Posição líquida COT
            if "cot_net" in gov_data:
                total_signals += 1
                if gov_data["cot_net"] > 0:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # Curva de rendimentos
            if "curve_steepness" in gov_data:
                total_signals += 1
                if gov_data["curve_steepness"] > 0:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
        
        if "ohlc" in sections_data:
            ohlc_data = sections_data["ohlc"]
            total_signals += 3
            
            # Preço vs SMA50
            if ohlc_data["price_vs_sma50"] > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Preço vs SMA200
            if ohlc_data["price_vs_sma200"] > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # RSI
            if ohlc_data["rsi"] > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if "summary" in sections_data:
            summary_data = sections_data["summary"]
            
            # Funding Rate
            if "funding_rate" in summary_data:
                total_signals += 1
                if summary_data["funding_rate"] > 0:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # Put/Call Ratio (contrarian)
            if "put_call_ratio" in summary_data:
                total_signals += 1
                if summary_data["put_call_ratio"] > 1.2:
                    bullish_signals += 1  # Contrarian (muitas puts = excesso de pessimismo)
                elif summary_data["put_call_ratio"] < 0.8:
                    bearish_signals += 1  # Contrarian (muitas calls = excesso de otimismo)
                else:
                    # Neutro, dar meio ponto para cada
                    bullish_signals += 0.5
                    bearish_signals += 0.5
        
        # Calcular o sentimento geral
        if total_signals > 0:
            bullish_pct = bullish_signals / total_signals * 100
            bearish_pct = bearish_signals / total_signals * 100
            
            if bullish_pct > 70:
                market_bias = "Fortemente Bullish"
            elif bullish_pct > 50:
                market_bias = "Moderadamente Bullish"
            elif bearish_pct > 70:
                market_bias = "Fortemente Bearish"
            elif bearish_pct > 50:
                market_bias = "Moderadamente Bearish"
            else:
                market_bias = "Neutro"
            
            report += f"""
            ### Resumo do Sentimento de Mercado
            
            - **Sinais Bullish**: {bullish_signals:.1f} ({bullish_pct:.1f}%)
            - **Sinais Bearish**: {bearish_signals:.1f} ({bearish_pct:.1f}%)
            - **Viés de Mercado**: {market_bias}
            
            """
        
        report += """
        ### Observações Finais
        
        Este relatório fornece uma visão abrangente das condições atuais do mercado, baseada em diversos indicadores e métricas. 
        É importante complementar esta análise com uma avaliação dos fundamentos e fatores de risco específicos antes de tomar decisões de investimento.
        
        As condições de mercado podem mudar rapidamente, e a diversificação continua sendo uma estratégia essencial de gestão de risco.
        """
        
        return report    
    def detect_market_regimes(self, symbol="BTCUSDT", lookback_days=30):
        """
        Detecta regimes de mercado com base em múltiplos indicadores.
        
        Args:
            symbol: Símbolo da criptomoeda para análise
            lookback_days: Período de lookback para análise
            
        Returns:
            Dict com os regimes detectados
        """
        regimes = {
            "volatility": None,  # "alta", "baixa", "média", "em_expansão", "em_contração"
            "trend": None,       # "alta", "baixa", "lateral", "iniciando_alta", "iniciando_baixa"
            "liquidity": None,   # "alta", "baixa", "média", "em_aumento", "em_diminuição"
            "sentiment": None,   # "otimista", "pessimista", "neutro", "extremo_otimismo", "extremo_pessimismo"
            "cycle": None        # "acumulação", "markup", "distribuição", "markdown"
        }
        
        try:
            # Carregar e processar dados OHLC
            spot_data = self.fetcher.load_data_file(f"ohlc/binance_spot_{symbol}_daily.csv")
            
            if spot_data is not None:
                spot_data = self._process_api_data(spot_data)
                if not spot_data.empty:
                    col_mappings = {
                        'timestamp': ['timestamp', 'date', 'time', 'Date', 'Time'],
                        'open': ['open', 'open_price', 'opening_price', 'Open'],
                        'high': ['high', 'high_price', 'highest_price', 'High'],
                        'low': ['low', 'low_price', 'lowest_price', 'Low'],
                        'close': ['close', 'close_price', 'closing_price', 'Close'],
                        'volume': ['volume', 'Volume']
                    }
                    
                    def find_column(df, possible_names):
                        return next((col for col in possible_names if col in df.columns), None)
                    
                    spot_cols = {req: find_column(spot_data, options) for req, options in col_mappings.items()}
                    
                    if all(key in spot_cols and spot_cols[key] is not None for key in ['timestamp', 'open', 'high', 'low', 'close']):
                        # Convert data types
                        for col_key, col_name in spot_cols.items():
                            if col_key == 'timestamp':
                                if spot_data[col_name].dtype == 'object':
                                    spot_data[col_name] = pd.to_datetime(spot_data[col_name], errors='coerce')
                            else:
                                spot_data[col_name] = pd.to_numeric(spot_data[col_name], errors='coerce')
                        
                        # Drop NaNs for required columns
                        required_cols = [spot_cols[k] for k in ['timestamp', 'open', 'high', 'low', 'close']]
                        spot_data = spot_data.dropna(subset=required_cols)
                        
                        if len(spot_data) > 0:
                            # Sort by timestamp
                            spot_data = spot_data.sort_values(by=spot_cols['timestamp'])
                            
                            # Get recent data (lookback period)
                            lookback_date = spot_data[spot_cols['timestamp']].iloc[-1] - pd.Timedelta(days=lookback_days)
                            recent_data = spot_data[spot_data[spot_cols['timestamp']] >= lookback_date]
                            
                            if len(recent_data) > 5:  # Ensure we have enough data points
                                # 1. Analisar Volatilidade
                                # Calcular ATR (Average True Range)
                                tr1 = recent_data[spot_cols['high']] - recent_data[spot_cols['low']]
                                tr2 = abs(recent_data[spot_cols['high']] - recent_data[spot_cols['close']].shift(1))
                                tr3 = abs(recent_data[spot_cols['low']] - recent_data[spot_cols['close']].shift(1))
                                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                                atr = tr.rolling(window=14).mean()
                                
                                # Normalizar ATR como % do preço
                                normalized_atr = atr / recent_data[spot_cols['close']] * 100
                                
                                current_atr = normalized_atr.iloc[-1]
                                avg_atr = normalized_atr.mean()
                                atr_change = (current_atr / normalized_atr.iloc[-14] - 1) * 100 if len(normalized_atr) >= 14 else 0
                                
                                # Determinar regime de volatilidade
                                if current_atr > avg_atr * 1.5:
                                    vol_regime = "alta"
                                elif current_atr < avg_atr * 0.5:
                                    vol_regime = "baixa"
                                else:
                                    vol_regime = "média"
                                
                                # Verificar se volatilidade está em expansão ou contração
                                if atr_change > 20:
                                    vol_regime = "em_expansão"
                                elif atr_change < -20:
                                    vol_regime = "em_contração"
                                
                                regimes['volatility'] = vol_regime
                                
                                # 2. Analisar Tendência
                                # Calcular médias móveis
                                recent_data['sma50'] = recent_data[spot_cols['close']].rolling(window=min(50, len(recent_data))).mean()
                                recent_data['sma200'] = recent_data[spot_cols['close']].rolling(window=min(200, len(recent_data))).mean()
                                
                                # Calcular direção recente
                                price_sma50_ratio = recent_data[spot_cols['close']].iloc[-1] / recent_data['sma50'].iloc[-1]
                                sma50_sma200_ratio = recent_data['sma50'].iloc[-1] / recent_data['sma200'].iloc[-1] if not pd.isna(recent_data['sma200'].iloc[-1]) else 1
                                
                                # Detectar movimentos laterais (baixa volatilidade + preço próximo à média)
                                is_sideways = vol_regime in ["baixa", "média"] and 0.95 < price_sma50_ratio < 1.05
                                
                                # Detectar crossovers (sinais de mudanças de tendência)
                                if len(recent_data) >= 3:
                                    price_above_sma50 = recent_data[spot_cols['close']] > recent_data['sma50']
                                    price_cross_above = price_above_sma50.iloc[-1] and not price_above_sma50.iloc[-2]
                                    price_cross_below = not price_above_sma50.iloc[-1] and price_above_sma50.iloc[-2]
                                    
                                    if price_cross_above:
                                        trend_regime = "iniciando_alta"
                                    elif price_cross_below:
                                        trend_regime = "iniciando_baixa"
                                    elif is_sideways:
                                        trend_regime = "lateral"
                                    elif price_sma50_ratio > 1.05 and sma50_sma200_ratio > 1.01:
                                        trend_regime = "alta"
                                    elif price_sma50_ratio < 0.95 and sma50_sma200_ratio < 0.99:
                                        trend_regime = "baixa"
                                    else:
                                        trend_regime = "lateral"
                                else:
                                    trend_regime = "indefinido"
                                
                                regimes['trend'] = trend_regime
                                
                                # 3. Analisar Liquidez/Volume
                                if 'volume' in spot_cols and spot_cols['volume'] is not None:
                                    recent_volume = recent_data[spot_cols['volume']]
                                    avg_volume = recent_volume.mean()
                                    current_volume = recent_volume.iloc[-1]
                                    volume_change = (recent_volume.tail(7).mean() / recent_volume.head(7).mean() - 1) * 100 if len(recent_volume) >= 14 else 0
                                    
                                    # Determinar regime de liquidez
                                    if current_volume > avg_volume * 1.5:
                                        liquidity_regime = "alta"
                                    elif current_volume < avg_volume * 0.5:
                                        liquidity_regime = "baixa"
                                    else:
                                        liquidity_regime = "média"
                                    
                                    # Verificar tendência de volume
                                    if volume_change > 20:
                                        liquidity_regime = "em_aumento"
                                    elif volume_change < -20:
                                        liquidity_regime = "em_diminuição"
                                    
                                    regimes['liquidity'] = liquidity_regime
                                
                                # 4. Tentar analisar sentimento
                                # Calcular RSI
                                delta = recent_data[spot_cols['close']].diff()
                                gain = delta.where(delta > 0, 0)
                                loss = -delta.where(delta < 0, 0)
                                avg_gain = gain.rolling(window=14).mean()
                                avg_loss = loss.rolling(window=14).mean()
                                rs = avg_gain / avg_loss.replace(0, 0.001)
                                rsi = 100 - (100 / (1 + rs))
                                
                                current_rsi = rsi.iloc[-1]
                                
                                # Determinar regime de sentimento
                                if current_rsi > 80:
                                    sentiment_regime = "extremo_otimismo"
                                elif current_rsi > 60:
                                    sentiment_regime = "otimista"
                                elif current_rsi < 20:
                                    sentiment_regime = "extremo_pessimismo"
                                elif current_rsi < 40:
                                    sentiment_regime = "pessimista"
                                else:
                                    sentiment_regime = "neutro"
                                
                                regimes['sentiment'] = sentiment_regime
                                
                                # 5. Tentar identificar ciclo de mercado (Wyckoff/Livermore)
                                if trend_regime in ["alta", "iniciando_alta"] and sentiment_regime in ["pessimista", "extremo_pessimismo"]:
                                    cycle_regime = "acumulação"
                                elif trend_regime == "alta" and sentiment_regime in ["otimista", "extremo_otimismo"]:
                                    cycle_regime = "markup"
                                elif trend_regime in ["lateral", "iniciando_baixa"] and sentiment_regime in ["otimista", "extremo_otimismo"]:
                                    cycle_regime = "distribuição"
                                elif trend_regime == "baixa" and sentiment_regime in ["pessimista", "extremo_pessimismo"]:
                                    cycle_regime = "markdown"
                                else:
                                    cycle_regime = "indefinido"
                                
                                regimes['cycle'] = cycle_regime
        except Exception as e:
            print(f"Erro ao detectar regimes de mercado: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return regimes
    
    def suggest_trading_strategies(self, symbol="BTCUSDT"):
        """
        Sugere estratégias de trading com base na análise do mercado atual.
        
        Args:
            symbol: Símbolo da criptomoeda para análise
            
        Returns:
            Dict com estratégias recomendadas
        """
        # Detectar regimes de mercado
        regimes = self.detect_market_regimes(symbol)
        
        strategies = {
            "short_term": [],  # Estratégias de curto prazo (dias)
            "medium_term": [], # Estratégias de médio prazo (semanas)
            "long_term": [],   # Estratégias de longo prazo (meses)
            "risk_level": "médio", # Nível de risco geral
            "explanation": {}  # Explicações para as recomendações
        }
        
        # Estratégias de curto prazo com base nos regimes
        # Volatilidade
        if regimes["volatility"] == "alta":
            strategies["short_term"].append("Estratégias de swing trading com proteção de stop-loss")
            strategies["short_term"].append("Operações com opções aproveitando valor elevado de volatilidade implícita")
        elif regimes["volatility"] == "baixa":
            strategies["short_term"].append("Range trading (comprar no suporte, vender na resistência)")
            strategies["short_term"].append("Estratégias de arbitragem entre exchanges")
        elif regimes["volatility"] == "em_expansão":
            strategies["short_term"].append("Breakout trading com confirmação de volume")
            strategies["short_term"].append("Posicionamento direcional com base na tendência principal")
        
        # Tendência
        if regimes["trend"] == "alta":
            strategies["short_term"].append("Comprar em pullbacks para a média móvel de 20 períodos")
            strategies["medium_term"].append("Posições long com gerenciamento de risco dinâmico")
        elif regimes["trend"] == "baixa":
            strategies["short_term"].append("Vender rallies para a média móvel de 20 períodos")
            strategies["medium_term"].append("Estratégias de hedge para proteção de portfólio")
        elif regimes["trend"] == "lateral":
            strategies["short_term"].append("Operações bidirecionais nos limites do range")
            strategies["medium_term"].append("Estratégias de acumulação nos suportes técnicos")
        elif regimes["trend"] == "iniciando_alta":
            strategies["short_term"].append("Entradas agressivas na confirmação de reversão")
            strategies["medium_term"].append("Aumento gradual de exposição direcional")
        elif regimes["trend"] == "iniciando_baixa":
            strategies["short_term"].append("Redução de exposição e implementação de hedge")
            strategies["medium_term"].append("Estratégias de proteção via opções")
        
        # Liquidez
        if regimes["liquidity"] == "alta":
            strategies["short_term"].append("Trading de alta frequência aproveitando a profundidade do mercado")
        elif regimes["liquidity"] == "baixa":
            strategies["short_term"].append("Cautela com operações que exigem alta liquidez")
            strategies["risk_level"] = "alto" # Aumentar nível de risco
        elif regimes["liquidity"] == "em_diminuição":
            strategies["short_term"].append("Reduzir tamanho das posições e aumentar cautela")
        
        # Sentimento
        if regimes["sentiment"] == "extremo_otimismo":
            strategies["short_term"].append("Estratégias contrarian de venda com stop ajustado")
            strategies["medium_term"].append("Implementação de hedge por conta do excesso de otimismo")
        elif regimes["sentiment"] == "extremo_pessimismo":
            strategies["short_term"].append("Estratégias contrarian de compra com risco controlado")
            strategies["medium_term"].append("Acumulação de ativos de qualidade com desconto")
        
        # Ciclo de mercado
        if regimes["cycle"] == "acumulação":
            strategies["medium_term"].append("Acumulação estratégica em ativos fundamentalmente sólidos")
            strategies["long_term"].append("Posicionamento para o próximo ciclo de alta")
        elif regimes["cycle"] == "markup":
            strategies["medium_term"].append("Manter exposição direcional com stops seguindo a tendência")
            strategies["long_term"].append("Posicionamento em ativos de alta beta para aproveitar a força do mercado")
        elif regimes["cycle"] == "distribuição":
            strategies["medium_term"].append("Realização parcial de lucros e redução de exposição")
            strategies["long_term"].append("Diversificação para ativos menos correlacionados")
        elif regimes["cycle"] == "markdown":
            strategies["medium_term"].append("Posições defensivas com foco em preservação de capital")
            strategies["long_term"].append("Construção gradual de posições em ativos de qualidade a preços descontados")
        
        # Criar explicações
        strategies["explanation"] = {
            "market_regimes": {
                "volatility": self._explain_volatility_regime(regimes["volatility"]),
                "trend": self._explain_trend_regime(regimes["trend"]),
                "liquidity": self._explain_liquidity_regime(regimes["liquidity"]),
                "sentiment": self._explain_sentiment_regime(regimes["sentiment"]),
                "cycle": self._explain_cycle_regime(regimes["cycle"])
            },
            "risk_assessment": self._generate_risk_assessment(regimes)
        }
        
        return strategies
    
    def _explain_volatility_regime(self, regime):
        """Explica o regime de volatilidade atual."""
        explanations = {
            "alta": "O mercado está em um período de alta volatilidade, com movimentos amplos e rápidos. Isso cria oportunidades para trading direcional, mas também aumenta o risco.",
            "baixa": "O mercado está em um período de baixa volatilidade, com movimentos contidos. Estratégias de range trading tendem a ser mais eficazes.",
            "média": "O mercado está com volatilidade normal, permitindo abordagens balanceadas de trading.",
            "em_expansão": "A volatilidade está aumentando, o que frequentemente precede movimentos significativos de preço. Atenção para breakouts e mudanças de tendência.",
            "em_contração": "A volatilidade está diminuindo, o que frequentemente precede períodos de consolidação ou uma possível explosão de volatilidade em breve."
        }
        return explanations.get(regime, "Regime de volatilidade não determinado.")
    
    def _explain_trend_regime(self, regime):
        """Explica o regime de tendência atual."""
        explanations = {
            "alta": "O mercado está em tendência de alta clara, com preços acima das médias móveis de curto e longo prazo. Estratégias de compra em pullbacks são geralmente eficazes.",
            "baixa": "O mercado está em tendência de baixa clara, com preços abaixo das médias móveis de curto e longo prazo. Estratégias de venda em rallies são geralmente eficazes.",
            "lateral": "O mercado está em movimento lateral, sem direção clara. Estratégias de negociação dentro de um range tendem a funcionar melhor.",
            "iniciando_alta": "O mercado está mostrando sinais de início de tendência de alta. Crossovers de médias móveis e suportes sendo estabelecidos são características deste regime.",
            "iniciando_baixa": "O mercado está mostrando sinais de início de tendência de baixa. Quebras de suporte e resistências sendo estabelecidas são características deste regime."
        }
        return explanations.get(regime, "Regime de tendência não determinado.")
    
    def _explain_liquidity_regime(self, regime):
        """Explica o regime de liquidez atual."""
        explanations = {
            "alta": "O mercado está com alta liquidez, permitindo a execução de ordens maiores com menor slippage. Favorável para estratégias que dependem de execução rápida.",
            "baixa": "O mercado está com baixa liquidez, o que pode resultar em maior slippage e execução mais difícil. Cautela é recomendada para operações de grande volume.",
            "média": "O mercado está com liquidez normal, permitindo a execução da maioria das estratégias sem preocupações excessivas com slippage.",
            "em_aumento": "A liquidez do mercado está aumentando, o que geralmente sinaliza maior interesse e participação. Isso pode preceder movimentos mais expressivos.",
            "em_diminuição": "A liquidez do mercado está diminuindo, o que pode sinalizar redução de interesse e participação. Cautela é recomendada."
        }
        return explanations.get(regime, "Regime de liquidez não determinado.")
    
    def _explain_sentiment_regime(self, regime):
        """Explica o regime de sentimento atual."""
        explanations = {
            "extremo_otimismo": "O mercado está em um estado de extremo otimismo, frequentemente indicativo de topos de mercado. Abordagens contrarian de venda podem ser consideradas.",
            "otimista": "O mercado está otimista, com momentum positivo. Favorável para estratégias direcionais de compra, mas com atenção para sinais de exaustão.",
            "neutro": "O sentimento do mercado está equilibrado, sem excesso de otimismo ou pessimismo. Momento ideal para análise objetiva.",
            "pessimista": "O mercado está pessimista, com momentum negativo. Pode oferecer oportunidades de compra em ativos de qualidade.",
            "extremo_pessimismo": "O mercado está em estado de extremo pessimismo, frequentemente indicativo de fundos de mercado. Abordagens contrarian de compra podem ser consideradas."
        }
        return explanations.get(regime, "Regime de sentimento não determinado.")
    
    def _explain_cycle_regime(self, regime):
        """Explica o regime de ciclo atual."""
        explanations = {
            "acumulação": "O mercado está na fase de acumulação, onde investidores institucionais e informados começam a comprar enquanto o preço ainda está deprimido e o sentimento é negativo. Esta fase geralmente ocorre após um período de queda significativa.",
            "markup": "O mercado está na fase de markup (alta), onde o preço sobe consistentemente e o sentimento melhora. Esta é a fase onde a tendência de alta se estabelece claramente.",
            "distribuição": "O mercado está na fase de distribuição, onde investidores informados começam a vender suas posições enquanto o sentimento ainda é positivo. Esta fase geralmente ocorre após um período de alta significativa.",
            "markdown": "O mercado está na fase de markdown (baixa), onde o preço cai consistentemente e o sentimento deteriora. Esta é a fase onde a tendência de baixa se estabelece claramente."
        }
        return explanations.get(regime, "Ciclo de mercado não determinado.")
    
    def _generate_risk_assessment(self, regimes):
        """Gera uma avaliação de risco com base nos regimes de mercado."""
        risk_factors = []
        
        # Avaliar risco com base na volatilidade
        if regimes["volatility"] in ["alta", "em_expansão"]:
            risk_factors.append("Alta volatilidade aumenta o risco de movimentos adversos rápidos")
        elif regimes["volatility"] == "em_contração":
            risk_factors.append("Contração de volatilidade pode preceder movimento explosivo em qualquer direção")
        
        # Avaliar risco com base na tendência
        if regimes["trend"] in ["iniciando_alta", "iniciando_baixa"]:
            risk_factors.append("Mudança de tendência pode criar falsos sinais inicialmente")
        elif regimes["trend"] == "lateral":
            risk_factors.append("Mercado sem direção clara pode criar operações de baixa convicção")
        
        # Avaliar risco com base na liquidez
        if regimes["liquidity"] in ["baixa", "em_diminuição"]:
            risk_factors.append("Baixa liquidez aumenta o risco de slippage e dificuldade de saída")
        
        # Avaliar risco com base no sentimento
        if regimes["sentiment"] in ["extremo_otimismo", "extremo_pessimismo"]:
            risk_factors.append("Sentimento extremo pode criar reversões bruscas de curto prazo")
        
        # Avaliar risco com base no ciclo
        if regimes["cycle"] in ["distribuição", "markdown"]:
            risk_factors.append("Fase de distribuição/queda tende a apresentar rallies falsos e armadilhas")
        
        # Gerar avaliação final
        if len(risk_factors) >= 3:
            risk_level = "alto"
            risk_advice = "Recomenda-se cautela, redução do tamanho das posições e proteção via stops ou hedge"
        elif len(risk_factors) >= 1:
            risk_level = "médio"
            risk_advice = "Mantenha disciplina no gerenciamento de risco e defina níveis claros de stop loss"
        else:
            risk_level = "baixo"
            risk_advice = "Condições favoráveis para operações dentro da sua estratégia habitual"
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "risk_advice": risk_advice
        }

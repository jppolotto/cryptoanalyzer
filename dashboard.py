"""
Interactive dashboard for cryptocurrency analysis and ranking.
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np
from crypto_scorer import CryptoScorer, json_serializable
from crypto_api_fetcher import CryptoAPIFetcher
from crypto_data_analysis import CryptoDataAnalyzer

# Set page configuration
st.set_page_config(
    page_title="Crypto Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Custom CSS
st.markdown("""
<style>
    /* Estilos existentes */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        color: #000000;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .recommendation-buy {
        color: #2E7D32;
        font-weight: bold;
    }
    .recommendation-hold {
        color: #F57C00;
        font-weight: bold;
    }
    .recommendation-sell {
        color: #D32F2F;
        font-weight: bold;
    }
    .risk-conservative {
        color: #2E7D32;
        font-weight: bold;
    }
    .risk-moderate {
        color: #F57C00;
        font-weight: bold;
    }
    .risk-aggressive {
        color: #D32F2F;
        font-weight: bold;
    }
    
    /* Novo estilo para forçar texto preto apenas dentro dos cards */
    .card .st-emotion-cache-10trblm.e1nzilvr1,
    .card [class*="st-emotion-cache-"] {
        color: #000000 !important;
    }
    
    /* Exceções para elementos que devem manter suas cores originais mesmo dentro dos cards */
    .card .recommendation-buy,
    .card .recommendation-hold,
    .card .recommendation-sell,
    .card .risk-conservative,
    .card .risk-moderate,
    .card .risk-aggressive,
    .card .metric-label {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

def json_serializable(obj):
    """Função para tornar objetos complexos serializáveis para JSON."""
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, (datetime, date)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)

def convert_timestamps(data):
    """Converte timestamps em strings para objetos datetime e vice-versa."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'timestamp' and isinstance(value, str):
                try:
                    data[key] = pd.to_datetime(value)
                except:
                    pass
            elif isinstance(value, (dict, list)):
                data[key] = convert_timestamps(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                data[i] = convert_timestamps(item)
    return data

class CryptoDashboard:
    """Class to create and manage the cryptocurrency analysis dashboard."""
    
    def __init__(self, data_dir='data', analysis_dir='analysis'):
        """
        Initialize the dashboard.
        
        Args:
            data_dir: Directory containing cryptocurrency data
            analysis_dir: Directory containing analysis results
        """
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        self.rankings = None
        self.top_cryptos = None
        self.global_metrics = None
        self.fear_greed = None
        self.historical_data = {}
        self.detailed_analyses = {}
        self.data_loader = None
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(f"{analysis_dir}/detailed", exist_ok=True)
        os.makedirs(f"{analysis_dir}/patterns", exist_ok=True)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load cryptocurrency data and analysis results."""
        try:
            # Load rankings
            rankings_file = f"{self.analysis_dir}/crypto_rankings.csv"
            if os.path.exists(rankings_file):
                self.rankings = pd.read_csv(rankings_file)
                print(f"Loaded rankings for {len(self.rankings)} cryptocurrencies")
            else:
                print("Rankings data not found")
                
            # Load top cryptocurrencies
            top_cryptos_file = f"{self.data_dir}/top_cryptos.csv"
            if os.path.exists(top_cryptos_file):
                self.top_cryptos = pd.read_csv(top_cryptos_file)
                print(f"Loaded data for {len(self.top_cryptos)} cryptocurrencies")
            else:
                print("Top cryptocurrencies data not found")
                
            # Load global metrics
            global_metrics_file = f"{self.data_dir}/global_metrics.csv"
            if os.path.exists(global_metrics_file):
                self.global_metrics = pd.read_csv(global_metrics_file).iloc[0].to_dict()
                print("Loaded global market metrics")
            else:
                print("Global metrics data not found")
                
            # Load Fear & Greed Index
            fear_greed_file = f"{self.data_dir}/fear_greed_index.csv"
            if os.path.exists(fear_greed_file):
                self.fear_greed = pd.read_csv(fear_greed_file).iloc[0].to_dict()
                print("Loaded Fear & Greed Index")
            else:
                print("Fear & Greed Index data not found")
                
            # Load historical data for each cryptocurrency
            if self.rankings is not None:
                for _, row in self.rankings.iterrows():
                    symbol = row['symbol']
                    daily_file = f"{self.data_dir}/{symbol.lower()}_historical_daily.csv"
                    
                    if os.path.exists(daily_file):
                        df = pd.read_csv(daily_file)
                        # Converter timestamp para datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        self.historical_data[symbol] = df
                
                print(f"Loaded historical data for {len(self.historical_data)} cryptocurrencies")
                
            # Load detailed analyses
            detailed_dir = f"{self.analysis_dir}/detailed"
            if os.path.exists(detailed_dir):
                for symbol in self.rankings['symbol'].unique() if self.rankings is not None else []:
                    analysis_file = f"{detailed_dir}/{symbol.lower()}_analysis.json"
                    if os.path.exists(analysis_file):
                        try:
                            with open(analysis_file, 'r') as f:
                                analysis_data = json.load(f)
                                # Converter timestamps
                                analysis_data = convert_timestamps(analysis_data)
                                self.detailed_analyses[symbol] = analysis_data
                        except json.JSONDecodeError:
                            print(f"Error loading detailed analysis for {symbol}")
                
                print(f"Loaded detailed analyses for {len(self.detailed_analyses)} cryptocurrencies")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def format_large_number(self, number, prefix=''):
        """
        Formata números grandes em formato mais legível (K, M, B).
        """
        if not isinstance(number, (int, float)):
            return f"{prefix}0"
            
        if number >= 1_000_000_000:
            return f"{prefix}{number/1_000_000_000:.2f}B"
        elif number >= 1_000_000:
            return f"{prefix}{number/1_000_000:.2f}M"
        elif number >= 1_000:
            return f"{prefix}{number/1_000:.2f}K"
        return f"{prefix}{number:.2f}"
    
    def get_recommendation_html(self, recommendation):
        """Get HTML for recommendation with appropriate styling."""
        if recommendation in ['Strong Buy', 'Buy']:
            return f'<span class="recommendation-buy">{recommendation}</span>'
        elif recommendation == 'Hold':
            return f'<span class="recommendation-hold">{recommendation}</span>'
        else:
            return f'<span class="recommendation-sell">{recommendation}</span>'
    
    def get_risk_profile_html(self, risk_profile):
        """Get HTML for risk profile with appropriate styling."""
        if risk_profile == 'Conservative':
            return f'<span class="risk-conservative">{risk_profile}</span>'
        elif risk_profile == 'Moderate':
            return f'<span class="risk-moderate">{risk_profile}</span>'
        else:
            return f'<span class="risk-aggressive">{risk_profile}</span>'
    
    def create_price_chart(self, symbol):
        """Create price chart for a cryptocurrency."""
        if symbol not in self.historical_data:
            return None
        
        df = self.historical_data[symbol].copy()
        df = df.sort_values('timestamp')
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Add volume bars
        if 'volume_24h' in df.columns:
            fig.add_trace(go.Bar(
                x=df['timestamp'],
                y=df['volume_24h'] / df['volume_24h'].max() * df['price'].max() * 0.3,
                name='Volume',
                marker=dict(color='rgba(30, 136, 229, 0.3)')
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price and Volume',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            template='plotly_white'
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        return fig
    
    def create_score_components_chart(self, symbol):
        """Create score components chart for a cryptocurrency."""
        if self.rankings is None or symbol not in self.rankings['symbol'].values:
            return None
        
        crypto_data = self.rankings[self.rankings['symbol'] == symbol].iloc[0]
        
        # Create data for radar chart
        categories = ['Technical', 'Fundamental', 'Sentiment']
        values = [
            crypto_data.get('technical_score', 0),
            crypto_data.get('fundamental_score', 0),
            crypto_data.get('sentiment_score', 0)
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=symbol,
            line=dict(color='#1E88E5'),
            fillcolor='rgba(30, 136, 229, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title=f'{symbol} Score Components',
            height=400,
            template='plotly_white'
        )
        
        return fig

    def display_chart_patterns(self, symbol):
        """Exibe os padrões de gráfico identificados para uma criptomoeda."""
        if symbol not in self.detailed_analyses:
            st.warning("Dados de análise técnica não disponíveis.")
            return
            
        analysis = self.detailed_analyses[symbol]
        if 'pattern_data' not in analysis:
            st.warning("Padrões de gráfico não disponíveis.")
            return
            
        patterns = analysis['pattern_data']
        
        # Criar colunas para os padrões
        cols = st.columns(3)
        pattern_count = 0
        
        for pattern_name, is_detected in patterns.items():
            if pattern_name != 'pattern_strength' and is_detected:
                col = cols[pattern_count % 3]
                with col:
                    # Determinar cor baseado na força do padrão
                    pattern_strength = patterns.get('pattern_strength', 0)
                    if pattern_strength >= 8:
                        color = '#4CAF50'  # Verde
                    elif pattern_strength >= 6:
                        color = '#8BC34A'  # Verde claro
                    elif pattern_strength >= 4:
                        color = '#FFC107'  # Amarelo
                    else:
                        color = '#FF9800'  # Laranja
                    
                    # Criar card para o padrão
                    pattern_display_name = pattern_name.replace('_', ' ').title()
                    st.markdown(f"""
                    <div class='card' style='border-left: 4px solid {color};'>
                        <h4>{pattern_display_name}</h4>
                        <p>Força: {pattern_strength}/10</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    pattern_count += 1
        
        if pattern_count == 0:
            st.info("Nenhum padrão de gráfico significativo identificado.")

    def create_technical_indicators_chart(self, symbol):
        """Create a chart showing technical indicators for a cryptocurrency."""
        if symbol not in self.detailed_analyses:
            st.warning("Dados de análise técnica não disponíveis.")
            return
            
        analysis = self.detailed_analyses[symbol]
        if 'technical_data' not in analysis:  # Mudança aqui: 'technical_indicators' -> 'technical_data'
            st.warning("Indicadores técnicos não disponíveis.")
            return
            
        technical_data = analysis['technical_data']  # Mudança aqui: usar technical_data
        
        # Criar DataFrame com os indicadores
        indicator_data = [
            {
                'indicator': 'RSI',
                'value': technical_data.get('rsi_14', 0),
                'signal': 'Bullish' if technical_data.get('rsi_14', 0) < 30 else 'Bearish' if technical_data.get('rsi_14', 0) > 70 else 'Neutral',
                'strength': abs(50 - technical_data.get('rsi_14', 50)) / 50 * 10
            },
            {
                'indicator': 'MACD',
                'value': technical_data.get('macd', 0),
                'signal': 'Bullish' if technical_data.get('macd', 0) > technical_data.get('macd_signal', 0) else 'Bearish',
                'strength': abs(technical_data.get('macd', 0) - technical_data.get('macd_signal', 0))
            },
            {
                'indicator': 'Stochastic',
                'value': technical_data.get('stoch_k', 0),
                'signal': 'Bullish' if technical_data.get('stoch_k', 0) < 20 else 'Bearish' if technical_data.get('stoch_k', 0) > 80 else 'Neutral',
                'strength': abs(50 - technical_data.get('stoch_k', 50)) / 50 * 10
            },
            {
                'indicator': 'MFI',
                'value': technical_data.get('mfi', 0),
                'signal': 'Bullish' if technical_data.get('mfi', 0) < 20 else 'Bearish' if technical_data.get('mfi', 0) > 80 else 'Neutral',
                'strength': abs(50 - technical_data.get('mfi', 50)) / 50 * 10
            },
            {
                'indicator': 'ADX',
                'value': technical_data.get('adx', 0),
                'signal': 'Bullish' if technical_data.get('adx', 0) > 25 else 'Neutral',
                'strength': min(technical_data.get('adx', 0) / 10, 10)
            }
        ]
        
        df = pd.DataFrame(indicator_data)
        
        # Verificar se o DataFrame está vazio
        if df.empty:
            st.warning("Não há dados de indicadores técnicos disponíveis.")
            return
        
        # Criar gráfico de barras
        fig = go.Figure()
        
        # Adicionar barras para cada sinal
        for signal in ['Bullish', 'Bearish', 'Neutral']:
            signal_data = df[df['signal'] == signal]
            if not signal_data.empty:
                fig.add_trace(go.Bar(
                    x=signal_data['indicator'],
                    y=signal_data['strength'],
                    name=signal,
                    marker_color={
                        'Bullish': '#2E7D32',
                        'Bearish': '#D32F2F',
                        'Neutral': '#F57C00'
                    }[signal],
                    text=signal_data['value'].apply(lambda x: f"{x:.2f}"),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Valor: %{text}<br>Força: %{y:.1f}<extra></extra>'
                ))
        
        # Atualizar layout
        fig.update_layout(
            title='Indicadores Técnicos',
            xaxis_title='Indicador',
            yaxis_title='Força do Sinal',
            barmode='group',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def display_sentiment_analysis(self, symbol):
        """Exibe a análise de sentimento para uma criptomoeda."""
        if symbol not in self.detailed_analyses:
            st.warning("Dados de análise de sentimento não disponíveis.")
            return
            
        analysis = self.detailed_analyses[symbol]
        if 'sentiment_data' not in analysis:
            st.warning("Análise de sentimento não disponível.")
            return
            
        sentiment = analysis['sentiment_data']
        
        # Criar layout em colunas
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Exibir métricas principais
            st.metric("Sentimento Geral", 
                     f"{sentiment.get('overall_sentiment', 'N/A')}",
                     delta=f"{sentiment.get('sentiment_change', 0)}%")
                     
            st.metric("Confiança", 
                     f"{sentiment.get('confidence', 0)}%")
                     
            st.metric("Volume de Menções", 
                     f"{sentiment.get('mention_volume', 0)}")
        
        with col2:
            # Criar gráfico de radar para distribuição de sentimento
            categories = ['Positivo', 'Neutro', 'Negativo']
            values = [
                sentiment.get('positive_percentage', 0),
                sentiment.get('neutral_percentage', 0),
                sentiment.get('negative_percentage', 0)
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Distribuição de Sentimento',
                line_color='#1E88E5'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Exibir insights de sentimento
        st.subheader("Insights de Sentimento")
        
        # Gerar insights baseados nos dados disponíveis
        insights = []
        
        # Análise do sentimento geral
        overall = sentiment.get('overall_sentiment', '').lower()
        if overall in ['positivo', 'muito positivo']:
            insights.append("O sentimento geral do mercado está otimista, indicando possível tendência de alta.")
        elif overall in ['negativo', 'muito negativo']:
            insights.append("O sentimento geral está pessimista, sugerindo cautela nas operações.")
        
        # Análise da mudança de sentimento
        sentiment_change = sentiment.get('sentiment_change', 0)
        if sentiment_change > 5:
            insights.append(f"Houve uma melhora significativa no sentimento (+{sentiment_change}%), sinalizando possível reversão positiva.")
        elif sentiment_change < -5:
            insights.append(f"Houve uma deterioração no sentimento ({sentiment_change}%), indicando possível pressão de venda.")
        
        # Análise do volume de menções
        mention_volume = sentiment.get('mention_volume', 0)
        if mention_volume > 1000:
            insights.append(f"Alto volume de menções ({mention_volume}) indica forte interesse do mercado.")
        
        # Análise da distribuição de sentimento
        positive_pct = sentiment.get('positive_percentage', 0)
        negative_pct = sentiment.get('negative_percentage', 0)
        if positive_pct > 60:
            insights.append(f"Predominância de sentimento positivo ({positive_pct:.1f}%) sugere otimismo do mercado.")
        elif negative_pct > 60:
            insights.append(f"Alta proporção de sentimento negativo ({negative_pct:.1f}%) indica preocupação dos investidores.")
        
        if not insights:
            insights = [
                "O mercado apresenta sentimento misto sem tendência clara.",
                "Volume de menções dentro da média histórica.",
                "Distribuição equilibrada entre sentimentos positivos e negativos."
            ]
        
        for insight in insights:
                st.markdown(f"- {insight}")
            
        # Exibir fontes de dados
        st.subheader("Fontes de Dados")
        
        # Gerar fontes de dados baseadas em análise típica
        default_sources = {
            'Twitter/X': f"{mention_volume * 0.4:.0f}",
            'Reddit': f"{mention_volume * 0.3:.0f}",
            'Telegram': f"{mention_volume * 0.2:.0f}",
            'Notícias': f"{mention_volume * 0.1:.0f}"
        }
        
        data_sources = sentiment.get('data_sources', default_sources)
        
        for source, count in data_sources.items():
            st.markdown(f"- {source}: {count} menções")

    def create_ranking_table(self, n=10):
        """Create ranking table for top N cryptocurrencies."""
        if self.rankings is None or self.rankings.empty:
            return None
        
        # Get top N cryptocurrencies
        top_n = self.rankings.head(n).copy()
        
        # Select and rename columns
        cols = [
            'rank', 'symbol', 'name', 'price_usd', 'percent_change_24h',
            'total_score', 'recommendation', 'risk_profile'
        ]
        
        # Ensure all columns exist
        for col in cols:
            if col not in top_n.columns:
                top_n[col] = 'N/A'
        
        table_data = top_n[cols].copy()
        
        # Format columns
        table_data['price_usd'] = table_data['price_usd'].apply(
            lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
        )
        table_data['percent_change_24h'] = table_data['percent_change_24h'].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "N/A"
        )
        table_data['total_score'] = table_data['total_score'].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x
        )
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Rank', 'Symbol', 'Name', 'Price', '24h Change', 'Score', 'Recommendation', 'Risk Profile'],
                fill_color='#0D47A1',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    table_data['rank'],
                    table_data['symbol'],
                    table_data['name'],
                    table_data['price_usd'],
                    table_data['percent_change_24h'],
                    table_data['total_score'],
                    table_data['recommendation'],
                    table_data['risk_profile']
                ],
                fill_color=[
                    ['#f9f9f9', '#ffffff'] * (n // 2 + 1)
                ],
                align='left',
                font=dict(color='black', size=12),
                height=30
            )
        )])
        
        fig.update_layout(
            title=f'Top {n} Cryptocurrencies by Score',
            height=80 + 30 * n,
            margin=dict(l=0, r=0, t=30, b=0),
            template='plotly_white'
        )
        
        return fig
    
    def create_score_distribution_chart(self):
        """Create score distribution chart for all cryptocurrencies."""
        if self.rankings is None or self.rankings.empty:
            return None
        
        # Ensure total_score column exists and contains numeric values
        if 'total_score' not in self.rankings.columns:
            return None
            
        # Filter out non-numeric values
        valid_scores = self.rankings[pd.to_numeric(self.rankings['total_score'], errors='coerce').notnull()]
        if valid_scores.empty:
            return None
        
        # Create histogram
        fig = px.histogram(
            valid_scores,
            x='total_score',
            nbins=20,
            color_discrete_sequence=['#1E88E5'],
            opacity=0.8
        )
        
        # Update layout
        fig.update_layout(
            title='Distribution of Cryptocurrency Scores',
            xaxis_title='Total Score',
            yaxis_title='Number of Cryptocurrencies',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_risk_profile_chart(self):
        """Create risk profile distribution chart."""
        if self.rankings is None or self.rankings.empty or 'risk_profile' not in self.rankings.columns:
            return None
        
        # Count cryptocurrencies by risk profile
        risk_counts = self.rankings['risk_profile'].value_counts().reset_index()
        risk_counts.columns = ['Risk Profile', 'Count']
        
        # Create pie chart
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Risk Profile',
            color='Risk Profile',
            color_discrete_map={
                'Conservative': '#2E7D32',
                'Moderate': '#F57C00',
                'Aggressive': '#D32F2F',
                'Unknown': '#9E9E9E'
            },
            hole=0.4
        )
        
        # Update layout
        fig.update_layout(
            title='Cryptocurrency Distribution by Risk Profile',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_recommendation_chart(self):
        """Create recommendation distribution chart."""
        if self.rankings is None or self.rankings.empty or 'recommendation' not in self.rankings.columns:
            return None
        
        # Count cryptocurrencies by recommendation
        rec_counts = self.rankings['recommendation'].value_counts().reset_index()
        rec_counts.columns = ['Recommendation', 'Count']
        
        # Create pie chart
        fig = px.pie(
            rec_counts,
            values='Count',
            names='Recommendation',
            color='Recommendation',
            color_discrete_map={
                'Strong Buy': '#1B5E20',
                'Buy': '#2E7D32',
                'Hold': '#F57C00',
                'Sell': '#D32F2F',
                'Strong Sell': '#B71C1C'
            },
            hole=0.4
        )
        
        # Update layout
        fig.update_layout(
            title='Cryptocurrency Distribution by Recommendation',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def render_dashboard(self):
        """Render the main dashboard interface."""
        st.title("Crypto Analyzer Dashboard")
        
        # Sidebar navigation
        st.sidebar.title("Navegação")
        page = st.sidebar.radio(
            "Selecione uma página:",
            ["Visão Geral do Mercado", "Rankings", "Análise Detalhada", 
             "Comparação", "Análise Preditiva", "Detecção de Padrões", 
             "Relatório de Mercado", "Extensões"]
        )
        
        if page == "Visão Geral do Mercado":
            self.render_market_overview()
        elif page == "Rankings":
            self.render_rankings_page()
        elif page == "Análise Detalhada":
            self.render_detailed_analysis()
        elif page == "Comparação":
            self.render_comparison_page()
        elif page == "Análise Preditiva":
            self.render_predictive_analysis()
        elif page == "Detecção de Padrões":
            self.render_pattern_detection()
        elif page == "Relatório de Mercado":
            self.render_market_report()
        elif page == "Extensões":
            self.render_extensions_page()
    
    def render_extensions_page(self):
        """Renderiza a página de extensões com funcionalidades adicionais."""
        st.markdown('<h2 class="sub-header">Extensões</h2>', unsafe_allow_html=True)
        
        # Criar tabs para diferentes funcionalidades
        tab1, tab2, tab3 = st.tabs(["Sistema de Alertas", "Previsões ML", "Dados CryptoDataDownload"])
        
        with tab1:
            try:
                from dashboard_extensions import DashboardExtensions
                extensions = DashboardExtensions(data_dir=self.data_dir, analysis_dir=self.analysis_dir)
                
                # Renderizar painel de login
                extensions.render_login_panel()
                
                # Renderizar dashboard de alertas se estiver logado
                if st.session_state.get('logged_in', False):
                    extensions.render_alert_dashboard()
                
            except ImportError:
                st.error("Módulo de extensões de alertas não encontrado.")
                st.info("Verifique se o arquivo dashboard_extensions.py está presente no diretório.")
        
        with tab2:
            try:
                from dashboard_extensions import DashboardExtensions
                extensions = DashboardExtensions(data_dir=self.data_dir, analysis_dir=self.analysis_dir)
                
                # Renderizar dashboard de previsões ML
                extensions.render_ml_predictions_dashboard()
                
            except ImportError:
                st.error("Módulo de extensões de ML não encontrado.")
                st.info("Verifique se o arquivo dashboard_extensions.py está presente no diretório.")
        
        with tab3:
            # Renderizar aba de dados CryptoDataDownload
            self.render_crypto_data_tab()
    
    def render_crypto_data_tab(self):
        """Renderiza a aba de dados do CryptoDataDownload."""
        st.markdown('<h3 class="sub-header">Dados CryptoDataDownload</h3>', unsafe_allow_html=True)
        
        try:
            # Inicializar o fetcher
            fetcher = CryptoAPIFetcher(data_dir=self.data_dir)
            
            # Inicializar o analisador
            analyzer = CryptoDataAnalyzer(fetcher)
            
            # Criar colunas para as opções
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Campo para API Token
                api_token = st.text_input("API Token CryptoDataDownload", type="password")
                
                # Seleção de símbolos
                symbols = st.multiselect(
                    "Selecione as criptomoedas",
                    ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"],
                    default=["BTCUSDT", "ETHUSDT"]
                )
            
            with col2:
                # Botão para buscar dados
                if st.button("Buscar Dados") and api_token:
                    with st.spinner("Buscando dados..."):
                        fetcher.api_token = api_token
                        results = fetcher.fetch_all_data(symbols=symbols)
                        
                        if results['success']:
                            st.success(f"Dados baixados com sucesso! {results['saved_files']} arquivos salvos.")
                        else:
                            st.error("Erro ao buscar alguns dados. Verifique os logs para mais detalhes.")
        
            # Adicionar seleção de tipo de análise
            analysis_type = st.selectbox(
                "Selecione o tipo de análise",
                ["Amplitude do Mercado", "Dados Governamentais", "Dados OHLC", 
                 "Métricas de Risco", "Dados Resumidos", "Visualizar Arquivos"]
            )
            
            # Executar a análise selecionada
            if analysis_type == "Amplitude do Mercado":
                analyzer.analyze_breadth_data()
            elif analysis_type == "Dados Governamentais":
                analyzer.analyze_government_data()
            elif analysis_type == "Dados OHLC":
                symbol = st.selectbox("Selecione a criptomoeda", symbols)
                analyzer.analyze_ohlc_data(symbol)
            elif analysis_type == "Métricas de Risco":
                symbol = st.selectbox("Selecione a criptomoeda", symbols)
                analyzer.analyze_risk_metrics(symbol)
            elif analysis_type == "Dados Resumidos":
                symbol = st.selectbox("Selecione a criptomoeda", symbols)
                analyzer.analyze_summary_data(symbol)
            elif analysis_type == "Visualizar Arquivos":
                self.render_file_viewer(fetcher)
        
        except Exception as e:
            import traceback
            st.error(f"Erro ao renderizar dados: {str(e)}")
            st.text(traceback.format_exc())
            st.info("Verifique se todos os módulos necessários estão instalados e configurados corretamente.")

    def render_file_viewer(self, fetcher):
        """Renderiza o visualizador de arquivos."""
        st.markdown('<h4>Arquivos Disponíveis</h4>', unsafe_allow_html=True)
        
        files = fetcher.get_available_files()
        
        # Criar tabs para cada categoria
        categories = list(files.keys())
        if categories:
            tabs = st.tabs(categories)
            
            for tab, category in zip(tabs, categories):
                with tab:
                    if files[category]:
                        # Criar DataFrame com os arquivos
                        df = pd.DataFrame(files[category])
                        
                        # Adicionar coluna de ações
                        for idx, file in df.iterrows():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"""
                                **{file['name']}**  
                                Tamanho: {file['size']/1024:.1f} KB  
                                Modificado: {file['modified']}
                                """)
                            
                            with col2:
                                # Botão para visualizar dados
                                if st.button("Visualizar", key=f"view_{category}_{idx}"):
                                    data = fetcher.load_data_file(file['path'])
                                    if data is not None:
                                        st.dataframe(data)
                                        
                                        # Opções de visualização se houver dados numéricos
                                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                                        if len(numeric_cols) > 0:
                                            st.markdown("### Visualização")
                                            plot_type = st.selectbox(
                                                "Tipo de gráfico",
                                                ["Linha", "Barra", "Dispersão"],
                                                key=f"plot_{category}_{idx}"
                                            )
                                            
                                            x_col = st.selectbox("Coluna X", data.columns, key=f"x_{category}_{idx}")
                                            y_col = st.selectbox("Coluna Y", numeric_cols, key=f"y_{category}_{idx}")
                                            
                                            if plot_type == "Linha":
                                                fig = px.line(data, x=x_col, y=y_col)
                                            elif plot_type == "Barra":
                                                fig = px.bar(data, x=x_col, y=y_col)
                                            else:
                                                fig = px.scatter(data, x=x_col, y=y_col)
                                            
                                            st.plotly_chart(fig)
                    else:
                        st.info("Nenhum arquivo disponível nesta categoria.")
    
    def render_market_overview(self):
        """Render market overview page."""
        st.markdown('<h2 class="sub-header">Market Overview</h2>', unsafe_allow_html=True)
        
        # Global metrics
        if self.global_metrics is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{self.format_large_number(self.global_metrics.get("total_market_cap_usd", 0))}</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="metric-label">Total Market Cap</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{self.format_large_number(self.global_metrics.get("total_volume_24h_usd", 0))}</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="metric-label">24h Trading Volume</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{self.global_metrics.get("btc_dominance", 0):.2f}%</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="metric-label">Bitcoin Dominance</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Fear & Greed Index
        if self.fear_greed is not None:
            st.markdown('<h3 class="sub-header">Market Sentiment</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Determine color based on value
                value = self.fear_greed.get('value', 50)
                if value <= 25:
                    color = '#D32F2F'  # Extreme Fear (Red)
                elif value <= 40:
                    color = '#FF9800'  # Fear (Orange)
                elif value <= 60:
                    color = '#FFC107'  # Neutral (Yellow)
                elif value <= 75:
                    color = '#8BC34A'  # Greed (Light Green)
                else:
                    color = '#4CAF50'  # Extreme Greed (Green)
                
                st.markdown(
                    f'<div class="metric-value" style="color: {color};">{value}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="metric-label">{self.fear_greed.get("value_classification", "Neutral")}</div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                # Create gauge chart for Fear & Greed Index
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fear & Greed Index"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 25], 'color': '#FFCDD2'},
                            {'range': [25, 40], 'color': '#FFECB3'},
                            {'range': [40, 60], 'color': '#FFF9C4'},
                            {'range': [60, 75], 'color': '#DCEDC8'},
                            {'range': [75, 100], 'color': '#C8E6C9'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': value
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        # Top Cryptocurrencies Table
        st.markdown('<h3 class="sub-header">Top Cryptocurrencies</h3>', unsafe_allow_html=True)
        
        if self.rankings is not None and not self.rankings.empty:
            ranking_table = self.create_ranking_table(n=10)
            if ranking_table:
                st.plotly_chart(ranking_table, use_container_width=True)
        else:
            st.info("No cryptocurrency ranking data available. Please run the analysis first.")
        
        # Market Statistics
        st.markdown('<h3 class="sub-header">Market Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score Distribution
            score_dist_chart = self.create_score_distribution_chart()
            if score_dist_chart:
                st.plotly_chart(score_dist_chart, use_container_width=True)
            else:
                st.info("Score distribution data not available.")
        
        with col2:
            # Risk Profile Distribution
            risk_chart = self.create_risk_profile_chart()
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
            else:
                st.info("Risk profile data not available.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Recommendation Distribution
            rec_chart = self.create_recommendation_chart()
            if rec_chart:
                st.plotly_chart(rec_chart, use_container_width=True)
            else:
                st.info("Recommendation data not available.")
        
        with col2:
            # Price vs. Score Scatter Plot
            if (self.rankings is not None and not self.rankings.empty and 
                'market_cap_usd' in self.rankings.columns and 
                'total_score' in self.rankings.columns and
                'volume_24h_usd' in self.rankings.columns and
                'risk_profile' in self.rankings.columns):
                
                # Check if data contains numeric values
                valid_data = self.rankings[
                    pd.to_numeric(self.rankings['market_cap_usd'], errors='coerce').notnull() &
                    pd.to_numeric(self.rankings['total_score'], errors='coerce').notnull() &
                    pd.to_numeric(self.rankings['volume_24h_usd'], errors='coerce').notnull()
                ]
                
                if not valid_data.empty:
                    fig = px.scatter(
                        valid_data,
                        x='market_cap_usd',
                        y='total_score',
                        color='risk_profile',
                        size='volume_24h_usd',
                        hover_name='name',
                        hover_data=['symbol', 'price_usd', 'recommendation'],
                        log_x=True,
                        color_discrete_map={
                            'Conservative': '#2E7D32',
                            'Moderate': '#F57C00',
                            'Aggressive': '#D32F2F',
                            'Unknown': '#9E9E9E'
                        }
                    )
                    
                    fig.update_layout(
                        title='Score vs. Market Cap',
                        xaxis_title='Market Cap (USD, log scale)',
                        yaxis_title='Total Score',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient valid data for scatter plot.")
            else:
                st.info("Market cap and score data not available.")
    
    def render_rankings_page(self):
        """Render cryptocurrency rankings page."""
        st.markdown('<h2 class="sub-header">Cryptocurrency Rankings</h2>', unsafe_allow_html=True)
        
        if self.rankings is None or self.rankings.empty:
            st.warning("No ranking data available.")
            st.info("Run the analysis to generate cryptocurrency rankings.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Risk profile filter
            if 'risk_profile' in self.rankings.columns:
                risk_profiles = ['All'] + sorted(self.rankings['risk_profile'].unique().tolist())
                selected_risk = st.selectbox('Risk Profile', risk_profiles)
            else:
                selected_risk = 'All'
                st.info("Risk profile data not available.")
        
        with col2:
            # Recommendation filter
            if 'recommendation' in self.rankings.columns:
                recommendations = ['All'] + sorted(self.rankings['recommendation'].unique().tolist())
                selected_rec = st.selectbox('Recommendation', recommendations)
            else:
                selected_rec = 'All'
                st.info("Recommendation data not available.")
        
        with col3:
            # Score threshold
            if 'total_score' in self.rankings.columns:
                min_score = st.slider('Minimum Score', 0, 100, 0)
            else:
                min_score = 0
                st.info("Score data not available.")
        
        # Apply filters
        filtered_rankings = self.rankings.copy()
        
        if selected_risk != 'All' and 'risk_profile' in filtered_rankings.columns:
            filtered_rankings = filtered_rankings[filtered_rankings['risk_profile'] == selected_risk]
            
        if selected_rec != 'All' and 'recommendation' in filtered_rankings.columns:
            filtered_rankings = filtered_rankings[filtered_rankings['recommendation'] == selected_rec]
            
        if min_score > 0 and 'total_score' in filtered_rankings.columns:
            # Convert to numeric, replacing non-numeric values with NaN
            filtered_rankings['total_score'] = pd.to_numeric(filtered_rankings['total_score'], errors='coerce')
            # Filter out NaN and values below min_score
            filtered_rankings = filtered_rankings[filtered_rankings['total_score'] >= min_score]
        
        # Sort options
        sort_options = {
            'Score (High to Low)': ('total_score', False),
            'Score (Low to High)': ('total_score', True),
            'Market Cap (High to Low)': ('market_cap_usd', False),
            'Market Cap (Low to High)': ('market_cap_usd', True),
            '24h Change (High to Low)': ('percent_change_24h', False),
            '24h Change (Low to High)': ('percent_change_24h', True)
        }
        
        sort_by = st.selectbox('Sort By', list(sort_options.keys()))
        sort_col, sort_asc = sort_options[sort_by]
        
        if sort_col in filtered_rankings.columns:
            # Convert to numeric, replacing non-numeric values with NaN
            filtered_rankings[sort_col] = pd.to_numeric(filtered_rankings[sort_col], errors='coerce')
            # Sort, with NaN values at the end
            filtered_rankings = filtered_rankings.sort_values(
                sort_col, 
                ascending=sort_asc,
                na_position='last'
            ).reset_index(drop=True)
            filtered_rankings['rank'] = filtered_rankings.index + 1
        
        # Display results
        st.markdown(f"<h3>Found {len(filtered_rankings)} cryptocurrencies</h3>", unsafe_allow_html=True)
        
        # Create table
        if not filtered_rankings.empty:
            # Select and rename columns
            cols = [
                'rank', 'symbol', 'name', 'price_usd', 'market_cap_usd', 'percent_change_24h',
                'technical_score', 'fundamental_score', 'sentiment_score', 'total_score',
                'recommendation', 'risk_profile'
            ]
            
            # Ensure all columns exist
            for col in cols:
                if col not in filtered_rankings.columns:
                    filtered_rankings[col] = 'N/A'
            
            table_data = filtered_rankings[cols].copy()
            
            # Format columns
            table_data['price_usd'] = table_data['price_usd'].apply(
                lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x)
            )
            table_data['market_cap_usd'] = table_data['market_cap_usd'].apply(
                lambda x: self.format_large_number(x) if isinstance(x, (int, float)) else str(x)
            )
            table_data['percent_change_24h'] = table_data['percent_change_24h'].apply(
                lambda x: f"{x:+.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "N/A"
            )
            table_data['technical_score'] = table_data['technical_score'].apply(
                lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
            )
            table_data['fundamental_score'] = table_data['fundamental_score'].apply(
                lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
            )
            table_data['sentiment_score'] = table_data['sentiment_score'].apply(
                lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
            )
            table_data['total_score'] = table_data['total_score'].apply(
                lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
            )
            
            # Create table
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=[
                        'Rank', 'Symbol', 'Name', 'Price', 'Market Cap', '24h Change',
                        'Technical', 'Fundamental', 'Sentiment', 'Total Score',
                        'Recommendation', 'Risk Profile'
                    ],
                    fill_color='#0D47A1',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[
                        table_data['rank'],
                        table_data['symbol'],
                        table_data['name'],
                        table_data['price_usd'],
                        table_data['market_cap_usd'],
                        table_data['percent_change_24h'],
                        table_data['technical_score'],
                        table_data['fundamental_score'],
                        table_data['sentiment_score'],
                        table_data['total_score'],
                        table_data['recommendation'],
                        table_data['risk_profile']
                    ],
                    fill_color=[
                        ['#f9f9f9', '#ffffff'] * (len(filtered_rankings) // 2 + 1)
                    ],
                    align='left',
                    font=dict(color='black', size=12),
                    height=30
                )
            )])
            
            fig.update_layout(
                height=80 + 30 * min(len(filtered_rankings), 20),
                margin=dict(l=0, r=0, t=0, b=0),
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show more button
            if len(filtered_rankings) > 20:
                show_all = st.checkbox('Show all results')
                
                if show_all:
                    # Create table for all results
                    fig = go.Figure(data=[go.Table(
                        header=dict(
                            values=[
                                'Rank', 'Symbol', 'Name', 'Price', 'Market Cap', '24h Change',
                                'Technical', 'Fundamental', 'Sentiment', 'Total Score',
                                'Recommendation', 'Risk Profile'
                            ],
                            fill_color='#0D47A1',
                            align='left',
                            font=dict(color='white', size=12)
                        ),
                        cells=dict(
                            values=[
                                table_data['rank'],
                                table_data['symbol'],
                                table_data['name'],
                                table_data['price_usd'],
                                table_data['market_cap_usd'],
                                table_data['percent_change_24h'],
                                table_data['technical_score'],
                                table_data['fundamental_score'],
                                table_data['sentiment_score'],
                                table_data['total_score'],
                                table_data['recommendation'],
                                table_data['risk_profile']
                            ],
                            fill_color=[
                                ['#f9f9f9', '#ffffff'] * (len(filtered_rankings) // 2 + 1)
                            ],
                            align='left',
                            font=dict(color='black', size=12),
                            height=30
                        )
                    )])
                    
                    fig.update_layout(
                        height=80 + 30 * len(filtered_rankings),
                        margin=dict(l=0, r=0, t=0, b=0),
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cryptocurrencies match the selected filters.")
    
    def render_detailed_analysis(self):
        """Renderiza a página de análise detalhada para uma criptomoeda selecionada."""
        st.markdown('<h2 class="sub-header">Análise Detalhada de Criptomoeda</h2>', unsafe_allow_html=True)
        
        if self.rankings is None or self.rankings.empty:
            st.warning("Não há dados de criptomoedas disponíveis.")
            st.info("Execute a análise completa para gerar os dados.")
            return
        
        # Seletor de criptomoeda
        symbols = sorted(self.rankings['symbol'].unique().tolist())
        selected_symbol = st.selectbox('Selecione uma Criptomoeda', symbols)
        
        if not selected_symbol:
            st.warning("Por favor, selecione uma criptomoeda.")
            return
        
        # Obtenha dados da criptomoeda
        crypto_data = self.rankings[self.rankings['symbol'] == selected_symbol].iloc[0]
        
        # Verificar se há análise detalhada disponível
        if selected_symbol not in self.detailed_analyses:
            st.warning(f"Análise detalhada não disponível para {selected_symbol}.")
            st.info("Execute a análise completa para gerar os dados.")
            
            # Oferecer a opção de gerar apenas para esta criptomoeda
            if st.button(f"Gerar análise para {selected_symbol}"):
                try:
                    scorer = CryptoScorer()
                    with st.spinner(f"Gerando análise para {selected_symbol}..."):
                        scorer.generate_detailed_analysis(selected_symbol)
                        st.success("Análise gerada! Recarregando dados...")
                        # Recarregar dados
                        self._load_data()
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro ao gerar análise: {str(e)}")
            return
        
        detailed_analysis = self.detailed_analyses[selected_symbol]
        
        # Exibir cabeçalho
        st.markdown(f"<h1>{crypto_data['name']} ({crypto_data['symbol']})</h1>", unsafe_allow_html=True)
        
        # Exibir recomendação atual
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price = crypto_data.get('price_usd', 0)
            change_24h = crypto_data.get('percent_change_24h', 0)
            st.metric(
                "Preço Atual", 
                f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A",
                f"{change_24h:.2f}%" if isinstance(change_24h, (int, float)) else None
            )
            
        with col2:
            total_score = crypto_data.get('total_score', 0)
            st.metric(
                "Pontuação Total", 
                f"{total_score:.1f}/100" if isinstance(total_score, (int, float)) else "N/A", 
                delta=None
            )
            
        with col3:
            recommendation = crypto_data.get('recommendation', 'N/A')
            rec_color = {
                'Strong Buy': 'green',
                'Buy': 'lightgreen',
                'Hold': 'orange',
                'Sell': 'red',
                'Strong Sell': 'darkred'
            }.get(recommendation, 'black')
            
            st.markdown(
                f"<h3 style='color: {rec_color};'>Recomendação: {recommendation}</h3>",
                unsafe_allow_html=True
            )
            
        # Tese de investimento
        st.subheader("Tese de Investimento")
        st.markdown(f"<div class='card'>{detailed_analysis.get('investment_thesis', 'Não disponível')}</div>", unsafe_allow_html=True)
        
        # Avaliação de risco
        st.subheader("Avaliação de Risco")
        st.markdown(f"<div class='card'>{detailed_analysis.get('risk_assessment', 'Não disponível')}</div>", unsafe_allow_html=True)
        
        # Adicionar visualização de padrões de gráfico
        st.markdown('<h3 class="sub-header">Padrões e Sinais Técnicos</h3>', unsafe_allow_html=True)
        self.display_chart_patterns(selected_symbol)
        
        # Adicionar radar chart melhorado para componentes de pontuação
        st.markdown('<h3 class="sub-header">Análise Multiponto</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart ampliado
            if selected_symbol in self.detailed_analyses:
                analysis = self.detailed_analyses[selected_symbol]
                
                # Obter componentes da pontuação técnica
                tech_components = {}
                if 'technical_data' in analysis and 'score_components' in analysis['technical_data']:
                    tech_components = analysis['technical_data']['score_components']
                
                if tech_components:
                    # Criar radar chart expandido
                    categories = [
                        'Tendência', 'Momentum', 'Volatilidade', 
                        'Suporte/Resistência', 'Volume', 'Ichimoku',
                        'OBV', 'ADX', 'PSAR', 'Cruzamento MA'
                    ]
                    
                    values = [
                        tech_components.get('trend_score', 0),
                        tech_components.get('momentum_score', 0),
                        tech_components.get('volatility_score', 0),
                        tech_components.get('support_resistance_score', 0),
                        tech_components.get('volume_score', 0),
                        tech_components.get('ichimoku_score', 0),
                        tech_components.get('obv_score', 0),
                        tech_components.get('adx_score', 0),
                        tech_components.get('psar_score', 0),
                        tech_components.get('ma_cross_score', 0)
                    ]
                    
                    # Normalizar para escala percentual
                    max_scores = [20, 20, 10, 10, 10, 10, 5, 5, 5, 5]
                    normalized_values = []
                    for i in range(len(values)):
                        if i < len(max_scores) and max_scores[i] > 0:
                            normalized_values.append(values[i]/max_scores[i]*100)
                        else:
                            normalized_values.append(0)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values,
                        theta=categories,
                        fill='toself',
                        name=selected_symbol,
                        line=dict(color='#1E88E5'),
                        fillcolor='rgba(30, 136, 229, 0.3)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        showlegend=False,
                        title=f'Componentes Técnicos Detalhados',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dados de componentes técnicos não disponíveis.")
        
        with col2:
            # Componentes fundamentais e de sentimento
            fund_components = {}
            sent_components = {}
            
            if 'fundamental_data' in analysis and 'score_components' in analysis['fundamental_data']:
                fund_components = analysis['fundamental_data']['score_components']
            
            if 'sentiment_data' in analysis and 'score_components' in analysis['sentiment_data']:
                sent_components = analysis['sentiment_data']['score_components']
            
            if fund_components and sent_components:
                # Componentes fundamentais
                fund_categories = [
                    'Market Cap', 'Volume/MCap', 'Rank', 
                    'Estabilidade', 'Crescimento', 'Posição de Mercado'
                ]
                
                fund_values = [
                    fund_components.get('market_cap_score', 0),
                    fund_components.get('volume_mcap_score', 0),
                    fund_components.get('rank_score', 0),
                    fund_components.get('stability_score', 0),
                    fund_components.get('growth_score', 0),
                    fund_components.get('market_position_score', 0)
                ]
                
                # Componentes de sentimento
                sent_categories = [
                    'Momentum', 'Sentimento de Mercado', 
                    'Desempenho Relativo', 'Alinhamento de Tendência'
                ]
                
                sent_values = [
                    sent_components.get('momentum_score', 0),
                    sent_components.get('market_sentiment_score', 0),
                    sent_components.get('relative_score', 0),
                    sent_components.get('alignment_score', 0)
                ]
                
                # Combinar para um gráfico de barras
                all_categories = fund_categories + sent_categories
                all_values = fund_values + sent_values
                bar_colors = ['#1E88E5'] * len(fund_categories) + ['#FFC107'] * len(sent_categories)
                
                # Criar gráfico de barras
                fig = go.Figure(data=[
                    go.Bar(
                        x=all_categories,
                        y=all_values,
                        marker_color=bar_colors
                    )
                ])
                
                fig.update_layout(
                    title='Componentes Fundamentais e de Sentimento',
                    xaxis_tickangle=-45,
                    yaxis=dict(
                        title='Pontuação',
                        range=[0, max(all_values) * 1.1] if all_values else [0, 10]
                    ),
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados de componentes fundamentais ou de sentimento não disponíveis.")
        
        # Indicadores técnicos detalhados
        st.subheader("Análise Técnica Detalhada")
        
        tech_chart = self.create_technical_indicators_chart(selected_symbol)
        if tech_chart:
            st.plotly_chart(tech_chart, use_container_width=True)
        
        # Display sentiment analysis
        self.display_sentiment_analysis(selected_symbol)
        
        # Disclaimer
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666; margin-top: 2rem;">
        <strong>Aviso:</strong> As análises apresentadas são baseadas em dados históricos e indicadores técnicos.
        Investimentos em criptomoedas envolvem alto risco e resultados passados não garantem resultados futuros.
        Sempre faça sua própria pesquisa antes de tomar decisões de investimento.
        </div>
        """, unsafe_allow_html=True)

    def render_comparison_page(self):
        """Render cryptocurrency comparison page."""
        st.markdown('<h2 class="page-header">Comparar Criptomoedas</h2>', unsafe_allow_html=True)
        
        # Verificar se temos dados de rankings
        if self.rankings is None or self.rankings.empty:
            st.warning("Nenhum dado de criptomoeda disponível. Execute a análise completa primeiro.")
            return
        
        # Criar duas colunas para seleção de criptomoedas
        col1, col2 = st.columns(2)
        
        # Lista de criptomoedas disponíveis
        available_cryptos = sorted(self.rankings['symbol'].unique().tolist())
        
        with col1:
            symbol1 = st.selectbox(
                'Selecione a primeira criptomoeda',
                available_cryptos,
                key='crypto1'
            )
        
        with col2:
            # Filtrar a primeira criptomoeda selecionada
            remaining_cryptos = [c for c in available_cryptos if c != symbol1]
            symbol2 = st.selectbox(
                'Selecione a segunda criptomoeda',
                remaining_cryptos,
                key='crypto2'
            )
        
        if not (symbol1 and symbol2):
            st.warning("Por favor, selecione duas criptomoedas para comparar.")
            return
        
        try:
            # Obter dados das criptomoedas selecionadas
            crypto1_data = self.rankings[self.rankings['symbol'] == symbol1].iloc[0]
            crypto2_data = self.rankings[self.rankings['symbol'] == symbol2].iloc[0]
            
            # Exibir informações básicas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"<h3>{symbol1}</h3>", unsafe_allow_html=True)
                st.markdown(f"Preço Atual: {self.format_large_number(crypto1_data['price_usd'], '$')}")
                st.markdown(f"Market Cap: {self.format_large_number(crypto1_data['market_cap_usd'], '$')}")
                st.markdown(f"Volume 24h: {self.format_large_number(crypto1_data['volume_24h_usd'], '$')}")
                st.markdown(f"Variação 24h: {crypto1_data['percent_change_24h']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"<h3>{symbol2}</h3>", unsafe_allow_html=True)
                st.markdown(f"Preço Atual: {self.format_large_number(crypto2_data['price_usd'], '$')}")
                st.markdown(f"Market Cap: {self.format_large_number(crypto2_data['market_cap_usd'], '$')}")
                st.markdown(f"Volume 24h: {self.format_large_number(crypto2_data['volume_24h_usd'], '$')}")
                st.markdown(f"Variação 24h: {crypto2_data['percent_change_24h']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Score comparison
            st.markdown('<h3 class="sub-header">Comparação de Scores</h3>', unsafe_allow_html=True)
            
            categories = ['Technical', 'Fundamental', 'Sentiment', 'Total']
            values1 = [
                crypto1_data.get('technical_score', 0),
                crypto1_data.get('fundamental_score', 0),
                crypto1_data.get('sentiment_score', 0),
                crypto1_data.get('total_score', 0)
            ]
            
            values2 = [
                crypto2_data.get('technical_score', 0),
                crypto2_data.get('fundamental_score', 0),
                crypto2_data.get('sentiment_score', 0),
                crypto2_data.get('total_score', 0)
            ]
            
            # Convert to numeric
            values1 = [float(v) if isinstance(v, (int, float)) else 0 for v in values1]
            values2 = [float(v) if isinstance(v, (int, float)) else 0 for v in values2]
            
            fig = go.Figure()
            
            # Add bars for first cryptocurrency
            fig.add_trace(go.Bar(
                x=categories,
                y=values1,
                name=crypto1_data['symbol'],
                marker_color='#1E88E5'
            ))
            
            # Add bars for second cryptocurrency
            fig.add_trace(go.Bar(
                x=categories,
                y=values2,
                name=crypto2_data['symbol'],
                marker_color='#FFC107'
            ))
            
            # Update layout
            fig.update_layout(
                title='Score Comparison',
                xaxis_title='Score Category',
                yaxis_title='Score (0-100)',
                barmode='group',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation and risk profile
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h4>Investment Profile</h4>", unsafe_allow_html=True)
                recommendation1 = crypto1_data.get('recommendation', 'N/A')
                risk_profile1 = crypto1_data.get('risk_profile', 'N/A')
                st.markdown(
                    f"<div>Recommendation: {self.get_recommendation_html(recommendation1)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div>Risk Profile: {self.get_risk_profile_html(risk_profile1)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h4>Investment Profile</h4>", unsafe_allow_html=True)
                recommendation2 = crypto2_data.get('recommendation', 'N/A')
                risk_profile2 = crypto2_data.get('risk_profile', 'N/A')
                st.markdown(
                    f"<div>Recommendation: {self.get_recommendation_html(recommendation2)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div>Risk Profile: {self.get_risk_profile_html(risk_profile2)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Price history comparison
            st.markdown('<h3 class="sub-header">Price History Comparison</h3>', unsafe_allow_html=True)
            
            # Check if historical data is available
            if symbol1 in self.historical_data and symbol2 in self.historical_data:
                df1 = self.historical_data[symbol1].copy()
                df2 = self.historical_data[symbol2].copy()
                
                # Sort by timestamp
                df1 = df1.sort_values('timestamp')
                df2 = df2.sort_values('timestamp')
                
                # Normalize prices for comparison (first day = 100)
                if 'price' in df1.columns and 'price' in df2.columns and not df1.empty and not df2.empty:
                    df1['normalized_price'] = df1['price'] / df1['price'].iloc[0] * 100
                    df2['normalized_price'] = df2['price'] / df2['price'].iloc[0] * 100
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    # Add price lines
                    fig.add_trace(go.Scatter(
                        x=df1['timestamp'],
                        y=df1['normalized_price'],
                        mode='lines',
                        name=f"{symbol1} (normalized)",
                        line=dict(color='#1E88E5', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df2['timestamp'],
                        y=df2['normalized_price'],
                        mode='lines',
                        name=f"{symbol2} (normalized)",
                        line=dict(color='#FFC107', width=2)
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Normalized Price Comparison (Starting Value = 100)',
                        xaxis_title='Date',
                        yaxis_title='Normalized Price',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500,
                        template='plotly_white'
                    )
                    
                    # Add range selector
                    fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Price data not available in the historical data.")
            else:
                st.warning("Historical price data not available for one or both cryptocurrencies.")
            
            # Correlation analysis
            st.markdown('<h3 class="sub-header">Correlation Analysis</h3>', unsafe_allow_html=True)
            
            if symbol1 in self.historical_data and symbol2 in self.historical_data:
                df1 = self.historical_data[symbol1].copy()
                df2 = self.historical_data[symbol2].copy()
                
                if 'price' in df1.columns and 'price' in df2.columns and not df1.empty and not df2.empty:
                    # Calculate daily returns
                    df1['daily_return'] = df1['price'].pct_change()
                    df2['daily_return'] = df2['price'].pct_change()
                    
                    # Merge dataframes on timestamp
                    merged_df = pd.merge(
                        df1[['timestamp', 'daily_return']],
                        df2[['timestamp', 'daily_return']],
                        on='timestamp',
                        suffixes=(f'_{symbol1}', f'_{symbol2}')
                    )
                    
                    # Calculate correlation
                    correlation = merged_df[f'daily_return_{symbol1}'].corr(merged_df[f'daily_return_{symbol2}'])
                    
                    # Create scatter plot
                    fig = px.scatter(
                        merged_df,
                        x=f'daily_return_{symbol1}',
                        y=f'daily_return_{symbol2}',
                        trendline='ols',
                        labels={
                            f'daily_return_{symbol1}': f'{symbol1} Daily Return',
                            f'daily_return_{symbol2}': f'{symbol2} Daily Return'
                        },
                        title=f'Return Correlation: {correlation:.4f}'
                    )
                    
                    fig.update_layout(
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    if correlation > 0.8:
                        st.markdown(f"""
                        <p><strong>Strong Positive Correlation ({correlation:.4f})</strong>: {symbol1} and {symbol2} tend to move strongly together, suggesting similar market drivers and limited diversification benefit when holding both.</p>
                        """, unsafe_allow_html=True)
                    elif correlation > 0.5:
                        st.markdown(f"""
                        <p><strong>Moderate Positive Correlation ({correlation:.4f})</strong>: {symbol1} and {symbol2} generally move in the same direction, but with some independent movements, offering partial diversification benefits.</p>
                        """, unsafe_allow_html=True)
                    elif correlation > 0.2:
                        st.markdown(f"""
                        <p><strong>Weak Positive Correlation ({correlation:.4f})</strong>: {symbol1} and {symbol2} have a slight tendency to move together, but largely move independently, offering good diversification benefits.</p>
                        """, unsafe_allow_html=True)
                    elif correlation > -0.2:
                        st.markdown(f"""
                        <p><strong>No Significant Correlation ({correlation:.4f})</strong>: {symbol1} and {symbol2} move independently of each other, providing excellent diversification benefits.</p>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <p><strong>Negative Correlation ({correlation:.4f})</strong>: {symbol1} and {symbol2} tend to move in opposite directions, providing strong diversification benefits and potential hedging opportunities.</p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Insufficient price data for correlation analysis.")
            else:
                st.warning("Historical price data not available for correlation analysis.")
            
        except Exception as e:
            st.error(f"Erro ao carregar dados das criptomoedas: {str(e)}")
            st.info("Tente executar a análise completa para atualizar os dados.")

    def render_predictive_analysis(self):
        """Renderiza a página de análise preditiva."""
        st.markdown('<h2 class="sub-header">Análise Preditiva</h2>', unsafe_allow_html=True)
        
        if self.rankings is None or self.rankings.empty:
            st.warning("Não há dados de criptomoedas disponíveis.")
            st.info("Execute a análise completa para gerar os dados.")
            return
        
        # Seletor de criptomoeda
        symbols = sorted(self.rankings['symbol'].unique().tolist())
        selected_symbol = st.selectbox('Selecione uma Criptomoeda', symbols)
        
        if not selected_symbol:
            st.warning("Por favor, selecione uma criptomoeda.")
            return
        
        # Verificar se existem dados de previsão
        has_predictions = (
            selected_symbol in self.detailed_analyses and
            'predictions' in self.detailed_analyses[selected_symbol] and
            'predicted_price_7d' in self.detailed_analyses[selected_symbol]['predictions'] and
            'predicted_price_30d' in self.detailed_analyses[selected_symbol]['predictions'] and
            'prediction_confidence' in self.detailed_analyses[selected_symbol]['predictions']
        )
        
        if not has_predictions:
            st.warning("Dados de previsão não disponíveis para esta criptomoeda.")
            st.info("Execute a análise preditiva para gerar previsões.")
            
            # Oferecer a opção de gerar previsões
            if st.button("Gerar Previsões"):
                try:
                    generated_preds = self.generate_predictions(selected_symbol)
                    if generated_preds:
                        st.success("Previsões geradas! Recarregando dados...")
                        self._load_data()
                        st.rerun()
                    else:
                        st.error("Não foi possível gerar previsões. Verifique se há dados históricos suficientes ou se ocorreram erros.")
                except Exception as e:
                    st.error(f"Erro ao gerar previsões: {str(e)}")
                    import traceback
                    traceback.print_exc()
            return
        
        predictions = self.detailed_analyses[selected_symbol].get('predictions', {})
        
        # Exibir métricas preditivas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            predicted_price_7d = predictions.get('predicted_price_7d', 0)
            price_change_7d = predictions.get('price_change_7d', 0)
            
            if isinstance(predicted_price_7d, (int, float)) and isinstance(price_change_7d, (int, float)):
                st.metric("Previsão de Preço (7 dias)", 
                        f"${predicted_price_7d:.2f}",
                        delta=f"{price_change_7d:.2f}%")
            else:
                st.metric("Previsão de Preço (7 dias)", "N/A")
        
        with col2:
            predicted_price_30d = predictions.get('predicted_price_30d', 0)
            price_change_30d = predictions.get('price_change_30d', 0)
            
            if isinstance(predicted_price_30d, (int, float)) and isinstance(price_change_30d, (int, float)):
                st.metric("Previsão de Preço (30 dias)", 
                        f"${predicted_price_30d:.2f}",
                        delta=f"{price_change_30d:.2f}%")
            else:
                st.metric("Previsão de Preço (30 dias)", "N/A")
        
        with col3:
            prediction_confidence = predictions.get('prediction_confidence', 0)
            
            if isinstance(prediction_confidence, (int, float)):
                st.metric("Confiança da Previsão", 
                        f"{prediction_confidence:.1f}%")
            else:
                st.metric("Confiança da Previsão", "N/A")
        
        # Gráfico de previsão
        if selected_symbol in self.historical_data:
            df = self.historical_data[selected_symbol].copy()
            
            if 'price' in df.columns and not df.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    mode='lines',
                    name='Preço Histórico',
                    line=dict(color='#1E88E5')
                ))
                
                # Adicionar previsões (se disponíveis como lista de preços)
                if 'predicted_prices' in predictions and isinstance(predictions.get('predicted_prices'), list):
                    predicted_prices_list = predictions['predicted_prices']
                    if predicted_prices_list:
                         pred_dates = pd.date_range(start=df['timestamp'].max(), periods=len(predicted_prices_list) + 1, freq='D')[1:]
                         fig.add_trace(go.Scatter(
                            x=pred_dates,
                            y=predicted_prices_list,
                            mode='lines',
                            name='Previsão (Detalhada)',
                            line=dict(color='#FFC107', dash='dash')
                        ))
                # Se não houver lista de preços, usar os valores 7d e 30d
                elif 'predicted_price_7d' in predictions and 'predicted_price_30d' in predictions:
                    current_price = df['price'].iloc[-1] if not df.empty else 0
                    pred_dates = pd.date_range(start=df['timestamp'].max(), periods=31, freq='D')[1:]

                    price_7d = predictions.get('predicted_price_7d', current_price)
                    price_30d = predictions.get('predicted_price_30d', current_price)

                    pred_prices_interp = []
                    days_to_forecast = 30
                    for i in range(1, days_to_forecast + 1):
                        if i <= 7:
                            interp_price = current_price + (price_7d - current_price) * i / 7
                        else:
                            interp_price = price_7d + (price_30d - price_7d) * (i - 7) / (days_to_forecast - 7)
                        pred_prices_interp.append(interp_price)

                    fig.add_trace(go.Scatter(
                        x=pred_dates[:days_to_forecast],
                        y=pred_prices_interp,
                        mode='lines',
                        name='Previsão (7d/30d)',
                        line=dict(color='#FFA000', dash='dot')
                    ))

                fig.update_layout(
                    title='Previsão de Preço',
                    xaxis_title='Data',
                    yaxis_title='Preço (USD)',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dados de preço não disponíveis no histórico.")
        else:
            st.warning("Dados históricos não disponíveis para esta criptomoeda.")
        
        # Exibir insights preditivos
        st.subheader("Insights Preditivos")
        predictive_insights = predictions.get('predictive_insights')
        if predictive_insights and isinstance(predictive_insights, list):
            for insight in predictive_insights:
                st.markdown(f"- {insight}")
        else:
            st.info("Insights preditivos não disponíveis para esta criptomoeda.")
        
        # Exibir fatores de influência
        st.subheader("Fatores de Influência")
        influence_factors = predictions.get('influence_factors')
        if influence_factors and isinstance(influence_factors, dict):
            for factor, impact in influence_factors.items():
                st.markdown(f"- {factor}: {impact}")
        else:
            st.info("Fatores de influência não disponíveis para esta criptomoeda.")
        
        # Disclaimer
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666; margin-top: 2rem;">
        <strong>Aviso:</strong> As previsões apresentadas são baseadas em modelos matemáticos e análise técnica.
        Previsões financeiras possuem limitações inerentes e não garantem resultados futuros.
        Use estas informações apenas como uma ferramenta complementar em sua análise.
        </div>
        """, unsafe_allow_html=True)
            
    def render_pattern_detection(self):
        """Renderiza a página de detecção de padrões."""
        st.markdown('<h2 class="sub-header">Detecção de Padrões</h2>', unsafe_allow_html=True)
        
        if self.rankings is None or self.rankings.empty:
            st.warning("Não há dados de criptomoedas disponíveis.")
            st.info("Execute a análise completa para gerar os dados.")
            return
        
        # Filtros
        st.subheader("Filtrar por Padrões")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_bullish = st.checkbox("Padrões Bullish", value=True)
        
        with col2:
            show_bearish = st.checkbox("Padrões Bearish", value=True)
        
        with col3:
            show_divergences = st.checkbox("Divergências", value=True)
        
        # Criar listas de padrões
        bullish_patterns = ['inverse_head_and_shoulders', 'double_bottom', 'bullish_flag', 'ascending_triangle']
        bearish_patterns = ['head_and_shoulders', 'double_top', 'bearish_flag', 'descending_triangle']
        
        # Inicializar lista para armazenar resultados
        pattern_results = []
        
        # Verificar se o diretório de padrões existe
        patterns_dir = f"{self.analysis_dir}/patterns"
        if not os.path.exists(patterns_dir):
            os.makedirs(patterns_dir, exist_ok=True)
            st.warning("Diretório de padrões não encontrado. Criado novo diretório.")
            st.info("Execute a análise de padrões para detectar padrões nas criptomoedas.")
            return
        
        # Verificar padrões para todas as criptomoedas
        for _, row in self.rankings.iterrows():
            symbol = row['symbol']
            name = row.get('name', symbol)
            price = row.get('price_usd', 0)
            change_24h = row.get('percent_change_24h', 0)
            
            # Verificar se temos arquivos de padrões
            pattern_file = f"{patterns_dir}/{symbol.lower()}_patterns.json"
            divergence_file = f"{patterns_dir}/{symbol.lower()}_divergences.json"
            
            has_pattern = False
            detected_patterns = []
            divergences = []
            
            # Verificar padrões
            if os.path.exists(pattern_file):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern_data = json.load(f)
                    
                    # Verificar padrões bullish
                    if show_bullish:
                        for pattern in bullish_patterns:
                            if pattern_data.get(pattern, False):
                                has_pattern = True
                                pattern_name = pattern.replace('_', ' ').title()
                                detected_patterns.append({
                                    'name': pattern_name,
                                    'type': 'bullish',
                                    'strength': pattern_data.get('pattern_strength', 0)
                                })
                    
                    # Verificar padrões bearish
                    if show_bearish:
                        for pattern in bearish_patterns:
                            if pattern_data.get(pattern, False):
                                has_pattern = True
                                pattern_name = pattern.replace('_', ' ').title()
                                detected_patterns.append({
                                    'name': pattern_name,
                                    'type': 'bearish',
                                    'strength': pattern_data.get('pattern_strength', 0)
                                })
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Erro ao processar arquivo de padrões para {symbol}: {str(e)}")
            
            # Verificar divergências
            if show_divergences and os.path.exists(divergence_file):
                try:
                    with open(divergence_file, 'r') as f:
                        divergence_data = json.load(f)
                    
                    if divergence_data.get('bullish_divergence', False):
                        has_pattern = True
                        divergences.append({
                            'type': 'bullish',
                            'strength': divergence_data.get('bullish_divergence_strength', 0)
                        })
                    
                    if divergence_data.get('bearish_divergence', False):
                        has_pattern = True
                        divergences.append({
                            'type': 'bearish',
                            'strength': divergence_data.get('bearish_divergence_strength', 0)
                        })
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Erro ao processar arquivo de divergências para {symbol}: {str(e)}")
            
            # Se encontrou algum padrão ou divergência, adicionar aos resultados
            if has_pattern:
                pattern_results.append({
                    'symbol': symbol,
                    'name': name,
                    'price': price,
                    'change_24h': change_24h,
                    'patterns': detected_patterns,
                    'divergences': divergences
                })
        
        # Exibir resultados
        if pattern_results:
            st.subheader(f"Encontradas {len(pattern_results)} criptomoedas com padrões")
            
            # Criar tabela expandível para cada criptomoeda
            for idx, result in enumerate(pattern_results):
                price_str = f"${result['price']:.2f}" if isinstance(result['price'], (int, float)) else "N/A"
                change_str = f"{result['change_24h']:.2f}%" if isinstance(result['change_24h'], (int, float)) else "N/A"
                
                with st.expander(f"{result['symbol']} - {result['name']} ({price_str}, {change_str})"):
                    col1, col2 = st.columns(2)
                    
                    # Padrões
                    with col1:
                        if result['patterns']:
                            st.subheader("Padrões Detectados:")
                            for pattern in result['patterns']:
                                color = 'green' if pattern['type'] == 'bullish' else 'red'
                                st.markdown(
                                    f"<span style='color: {color};'>✓ {pattern['name']} (Força: {pattern['strength']}/10)</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown("Nenhum padrão detectado")
                    
                    # Divergências
                    with col2:
                        if result['divergences']:
                            st.subheader("Divergências Detectadas:")
                            for div in result['divergences']:
                                color = 'green' if div['type'] == 'bullish' else 'red'
                                st.markdown(
                                    f"<span style='color: {color};'>✓ Divergência {div['type'].title()} (Força: {div['strength']}/10)</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown("Nenhuma divergência detectada")
                    
                    # Link para análise detalhada
                    st.markdown(f"[Ver Análise Detalhada →](#)")
        else:
            st.info("Nenhum padrão encontrado com os filtros selecionados.")
            
            # Oferecer a opção de gerar padrões
            if st.button("Analisar Padrões"):
                try:
                    scorer = CryptoScorer()
                    with st.spinner("Analisando padrões em todas as criptomoedas..."):
                        scorer.analyze_patterns()
                        st.success("Análise de padrões concluída! Recarregando dados...")
                        # Recarregar dados
                        self._load_data()
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro ao analisar padrões: {str(e)}")
    
    def render_market_report(self):
        """
        Renderiza a página de relatório de mercado com métricas globais e análises.
        """
        st.title("📊 Relatório de Mercado")

        # Obtém métricas do mercado
        metrics = self.get_market_metrics()

        # Cria colunas para métricas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Capitalização Total do Mercado",
                self.format_large_number(metrics.get('total_market_cap', 0), prefix="$") # Usar .get() para segurança
            )

        with col2:
            st.metric(
                "Volume 24h",
                self.format_large_number(metrics.get('total_volume_24h', 0), prefix="$") # Usar .get() para segurança
            )

        with col3:
            st.metric(
                "Dominância do Bitcoin",
                f"{metrics.get('btc_dominance', 0):.2f}%" # Usar .get() para segurança
            )

        st.divider()

        # Verifica se existe um relatório salvo
        report_path = f"{self.analysis_dir}/market_report.md"
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                # Verificar se o conteúdo do relatório é a mensagem de erro padrão
                if report_content.strip() == "Error generating report. See logs for details.":
                     st.warning("O último relatório gerado continha erros. Tente gerar novamente.")
                     # Forçar a exibição do botão para gerar novo relatório
                     os.remove(report_path) # Remover o arquivo de erro para tentar gerar de novo
                     st.rerun() # Rerodar para mostrar o botão
                else:
                    st.markdown(report_content)
            except Exception as e:
                st.error(f"Erro ao ler relatório: {str(e)}")
                st.info("Gere um novo relatório para visualizar as análises de mercado.")
        else:
            st.warning("Nenhum relatório de mercado encontrado.")
            if st.button("Gerar Novo Relatório"):
                with st.spinner("Gerando relatório de mercado..."):
                    try:
                        # --- CORREÇÃO AQUI ---
                        # Remover o argumento analysis_dir
                        scorer = CryptoScorer(data_dir=self.data_dir)
                        # --- FIM DA CORREÇÃO ---

                        if scorer.scores is None or scorer.scores.empty:
                             scorer.score_all_cryptocurrencies()

                        report_content = scorer.generate_report()

                        if report_content == "Error generating report. See logs for details.":
                             st.error("Erro interno ao gerar relatório. Verifique os logs do console para detalhes.")
                        else:
                            st.success("Relatório gerado com sucesso!")
                            st.rerun()

                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        st.error(f"Erro ao gerar relatório: [{error_type}] {error_msg}")
                        # import traceback
                        # st.text_area("Traceback:", traceback.format_exc(), height=200)

        st.divider()
        st.caption("""
        ⚠️ **Aviso**: Este relatório é gerado automaticamente com base em dados históricos e indicadores técnicos.
        Não deve ser considerado como recomendação de investimento. Sempre faça sua própria pesquisa e análise
        antes de tomar decisões de investimento.
        """)

    def run_scoring_analysis(self):
        """Executa análise de pontuação usando o CryptoScorer."""
        try:
            # --- CORREÇÃO AQUI ---
            # Remover o argumento analysis_dir
            scorer = CryptoScorer(data_dir=self.data_dir)
            # --- FIM DA CORREÇÃO ---

            with st.spinner('Executando análise completa de criptomoedas...'):
                # Gerar rankings
                self.rankings = scorer.score_all_cryptocurrencies()
                if self.rankings is not None:
                    # Gerar análises detalhadas para cada criptomoeda
                    for symbol in self.rankings['symbol'].unique():
                        scorer.generate_detailed_analysis(symbol)

                        # Gerar previsões para as principais criptomoedas
                        if symbol in self.rankings.head(20)['symbol'].values:
                            self.generate_predictions(symbol)

                    # Analisar padrões técnicos
                    self.analyze_patterns()

                    # Gerar relatório de mercado
                    scorer.generate_report()

                    st.success("Análise concluída com sucesso! Recarregando dados...")
                    # Recarregar dados para atualizar o dashboard
                    self._load_data()
                    st.rerun()
                else:
                    st.error("Erro ao gerar rankings. Verifique os logs para mais detalhes.")

        except Exception as e:
            st.error(f"Erro ao executar análise: {str(e)}")
            st.info("Verifique se todos os dados necessários estão disponíveis.")

    def get_market_metrics(self):
        """
        Obtém métricas globais do mercado de criptomoedas.
        """
        try:
            if self.rankings is None or self.rankings.empty:
                return {
                    'total_market_cap': 0,
                    'total_volume_24h': 0,
                    'btc_dominance': 0
                }
            
            # Converte colunas para numérico se necessário
            for col in ['market_cap_usd', 'volume_24h_usd']:
                if col in self.rankings.columns:
                    self.rankings[col] = pd.to_numeric(self.rankings[col], errors='coerce')
            
            # Calcula métricas principais usando os dados já carregados
            total_market_cap = self.rankings['market_cap_usd'].sum() if 'market_cap_usd' in self.rankings.columns else 0
            total_volume_24h = self.rankings['volume_24h_usd'].sum() if 'volume_24h_usd' in self.rankings.columns else 0
            
            # Calcula dominância do Bitcoin
            btc_dominance = 0
            if 'market_cap_usd' in self.rankings.columns:
                btc_data = self.rankings[self.rankings['symbol'] == 'BTC']
                if not btc_data.empty and total_market_cap > 0:
                    btc_market_cap = btc_data['market_cap_usd'].iloc[0]
                    btc_dominance = (btc_market_cap / total_market_cap * 100)
            
            return {
                'total_market_cap': total_market_cap,
                'total_volume_24h': total_volume_24h,
                'btc_dominance': btc_dominance
            }
        except Exception as e:
            print(f"Erro ao obter métricas do mercado: {str(e)}")
            return {
                'total_market_cap': 0,
                'total_volume_24h': 0,
                'btc_dominance': 0
            }

    def generate_predictions(self, symbol):
        """Gera previsões para uma criptomoeda específica."""
        try:
            if symbol not in self.historical_data:
                print(f"No historical data available for {symbol}")
                return None
                
            df = self.historical_data[symbol].copy()
            
            # Garantir que temos dados suficientes
            if len(df) < 30:
                print(f"Insufficient data for {symbol}")
                return None
                
            # Preparar dados para previsão
            df = df.sort_values('timestamp')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Calcular médias móveis
            df['ma7'] = df['price'].rolling(window=7).mean()
            df['ma30'] = df['price'].rolling(window=30).mean()
            
            # Calcular RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Evitar divisão por zero no RSI
            loss = np.where(loss == 0, 1e-10, loss) # Adicionado para evitar divisão por zero
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Previsão para 7 dias
            last_price = df['price'].iloc[-1]
            last_ma7 = df['ma7'].iloc[-1]
            last_ma30 = df['ma30'].iloc[-1]
            last_rsi = df['rsi'].iloc[-1]
            
            # Verificar se os valores são NaN antes de usar
            if pd.isna(last_price) or pd.isna(last_ma7) or pd.isna(last_ma30) or pd.isna(last_rsi):
                print(f"Cannot generate predictions for {symbol} due to NaN values in indicators.")
                return None

            # Lógica simples de previsão baseada em tendências
            if last_price > last_ma7 and last_ma7 > last_ma30:
                trend = "bullish"
            elif last_price < last_ma7 and last_ma7 < last_ma30:
                trend = "bearish"
            else:
                trend = "neutral"
                
            # Calcular previsões
            if trend == "bullish":
                predicted_price_7d = last_price * 1.05  # +5%
                predicted_price_30d = last_price * 1.15  # +15%
            elif trend == "bearish":
                predicted_price_7d = last_price * 0.95  # -5%
                predicted_price_30d = last_price * 0.85  # -15%
            else:
                predicted_price_7d = last_price * 1.02  # +2%
                predicted_price_30d = last_price * 1.06  # +6%
                
            # Ajustar previsões baseado no RSI
            if last_rsi > 70:  # Sobrecomprado
                predicted_price_7d *= 0.98
                predicted_price_30d *= 0.95
            elif last_rsi < 30:  # Sobrevendido
                predicted_price_7d *= 1.02
                predicted_price_30d *= 1.05
                
            # Calcular mudança percentual
            price_change_7d = ((predicted_price_7d - last_price) / last_price) * 100 if last_price != 0 else 0
            price_change_30d = ((predicted_price_30d - last_price) / last_price) * 100 if last_price != 0 else 0
            
            # Calcular confiança da previsão
            confidence = 70  # Base
            if trend == "bullish" or trend == "bearish":
                confidence += 10
            if last_rsi < 30 or last_rsi > 70:
                confidence += 5
            confidence = min(confidence, 95) # Limitar confiança máxima

            prediction_data = {
                'predicted_price_7d': predicted_price_7d,
                'predicted_price_30d': predicted_price_30d,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'prediction_confidence': confidence,
                'trend': trend,
                'last_rsi': last_rsi
            }
                
            # Atualizar dados da criptomoeda em memória E SALVAR NO ARQUIVO
            if symbol in self.detailed_analyses:
                self.detailed_analyses[symbol]['predictions'] = prediction_data
                
                # --- INÍCIO DO CÓDIGO ADICIONADO ---
                # Salvar o dicionário atualizado de volta no arquivo JSON
                analysis_file = f"{self.analysis_dir}/detailed/{symbol.lower()}_analysis.json"
                try:
                    with open(analysis_file, 'w') as f:
                        json.dump(self.detailed_analyses[symbol], f, indent=4, default=json_serializable)
                    print(f"Detailed analysis with predictions saved for {symbol}")
                except Exception as e:
                    print(f"Error saving detailed analysis with predictions for {symbol}: {str(e)}")
                # --- FIM DO CÓDIGO ADICIONADO ---
                
            return prediction_data # Retornar os dados de previsão calculados
            
        except Exception as e:
            print(f"Error generating predictions for {symbol}: {e}")
            # Adicionar stack trace para depuração
            import traceback
            traceback.print_exc()
            return None

    def analyze_patterns(self):
        """Analisa padrões técnicos para todas as criptomoedas."""
        try:
            if self.rankings is None or self.rankings.empty:
                print("No cryptocurrency data available")
                return
                
            patterns_dir = f"{self.analysis_dir}/patterns"
            os.makedirs(patterns_dir, exist_ok=True)
            
            for _, row in self.rankings.iterrows():
                symbol = row['symbol']
                if symbol not in self.historical_data:
                    continue
                    
                df = self.historical_data[symbol].copy()
                df = df.sort_values('timestamp')
                
                # Inicializar resultados
                patterns = {
                    'head_and_shoulders': False,
                    'inverse_head_and_shoulders': False,
                    'double_top': False,
                    'double_bottom': False,
                    'bullish_flag': False,
                    'bearish_flag': False,
                    'ascending_triangle': False,
                    'descending_triangle': False,
                    'pattern_strength': 0
                }
                
                # Verificar padrões de cabeça e ombros
                if len(df) >= 100:
                    # Calcular médias móveis
                    df['ma20'] = df['price'].rolling(window=20).mean()
                    df['ma50'] = df['price'].rolling(window=50).mean()
                    
                    # Verificar tendência
                    last_price = df['price'].iloc[-1]
                    last_ma20 = df['ma20'].iloc[-1]
                    last_ma50 = df['ma50'].iloc[-1]
                    
                    # Verificar padrões
                    if last_price < last_ma20 and last_ma20 < last_ma50:
                        patterns['head_and_shoulders'] = True
                        patterns['pattern_strength'] += 3
                    elif last_price > last_ma20 and last_ma20 > last_ma50:
                        patterns['inverse_head_and_shoulders'] = True
                        patterns['pattern_strength'] += 3
                        
                    # Verificar topos e fundos duplos
                    peaks = df['price'].rolling(window=5, center=True).max()
                    troughs = df['price'].rolling(window=5, center=True).min()
                    
                    if len(peaks) >= 2 and abs(peaks.iloc[-1] - peaks.iloc[-2]) / peaks.iloc[-2] < 0.02:
                        patterns['double_top'] = True
                        patterns['pattern_strength'] += 2
                        
                    if len(troughs) >= 2 and abs(troughs.iloc[-1] - troughs.iloc[-2]) / troughs.iloc[-2] < 0.02:
                        patterns['double_bottom'] = True
                        patterns['pattern_strength'] += 2
                        
                    # Verificar bandeiras
                    if df['price'].iloc[-5:].std() / df['price'].iloc[-5:].mean() < 0.01:
                        if df['price'].iloc[-1] > df['price'].iloc[-5]:
                            patterns['bullish_flag'] = True
                            patterns['pattern_strength'] += 1
                        else:
                            patterns['bearish_flag'] = True
                            patterns['pattern_strength'] += 1
                            
                    # Verificar triângulos
                    if len(df) >= 30:
                        high = df['price'].rolling(window=10).max()
                        low = df['price'].rolling(window=10).min()
                        
                        if high.iloc[-1] < high.iloc[-2] and low.iloc[-1] > low.iloc[-2]:
                            patterns['ascending_triangle'] = True
                            patterns['pattern_strength'] += 2
                        elif high.iloc[-1] > high.iloc[-2] and low.iloc[-1] < low.iloc[-2]:
                            patterns['descending_triangle'] = True
                            patterns['pattern_strength'] += 2
                
                # Salvar padrões
                pattern_file = f"{patterns_dir}/{symbol.lower()}_patterns.json"
                with open(pattern_file, 'w') as f:
                    json.dump(patterns, f, indent=4, default=json_serializable)
                    
                # Verificar divergências
                divergences = {
                    'bullish_divergence': False,
                    'bearish_divergence': False,
                    'bullish_divergence_strength': 0,
                    'bearish_divergence_strength': 0
                }
                
                # Calcular RSI
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Verificar divergências
                if len(df) >= 30:
                    # Verificar divergência de alta
                    if (df['price'].iloc[-1] < df['price'].iloc[-2] and 
                        df['rsi'].iloc[-1] > df['rsi'].iloc[-2]):
                        divergences['bullish_divergence'] = True
                        divergences['bullish_divergence_strength'] = 5
                        
                    # Verificar divergência de baixa
                    if (df['price'].iloc[-1] > df['price'].iloc[-2] and 
                        df['rsi'].iloc[-1] < df['rsi'].iloc[-2]):
                        divergences['bearish_divergence'] = True
                        divergences['bearish_divergence_strength'] = 5
                
                # Salvar divergências
                divergence_file = f"{patterns_dir}/{symbol.lower()}_divergences.json"
                with open(divergence_file, 'w') as f:
                    json.dump(divergences, f, indent=4, default=json_serializable)
                    
            print("Pattern analysis completed successfully")
            
        except Exception as e:
            print(f"Error in pattern analysis: {e}")

# Run the dashboard if script is executed directly
if __name__ == "__main__":
    try:
        # Tenta importar CryptoScorer
        from crypto_scorer import CryptoScorer
        dashboard = CryptoDashboard()
        dashboard.render_dashboard()
        
    except ImportError:
        st.error("Módulo CryptoScorer não encontrado.")
        st.info("""
        Este dashboard depende do módulo CryptoScorer que não está disponível.
        
        Verifique se o arquivo crypto_scorer.py está no mesmo diretório que este script
        ou no Python path.
        """)
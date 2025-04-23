"""
Extensions for the cryptocurrency dashboard to add customizable alerts and ML predictions.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from alert_system import CryptoAlertSystem
from ml_analyzer import CryptoMLAnalyzer
from crypto_data_analysis import CryptoDataAnalyzer

class CryptoDataFetcher:
    """Classe para buscar e fornecer dados para análise."""
    
    def __init__(self, data_dir='data'):
        """
        Inicializa o fetcher de dados.
        
        Args:
            data_dir: Diretório contendo os dados
        """
        self.data_dir = data_dir
    
    def load_data_file(self, relative_path):
        """
        Carrega um arquivo de dados do diretório de dados.
        
        Args:
            relative_path: Caminho relativo do arquivo dentro do diretório de dados
            
        Returns:
            DataFrame com os dados ou None se o arquivo não existir
        """
        try:
            file_path = os.path.join(self.data_dir, relative_path)
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            return None
        except Exception as e:
            st.error(f"Erro ao carregar arquivo {relative_path}: {str(e)}")
            return None

class CryptoAnalysisExtension:
    """Classe para integrar análises avançadas do CryptoDataAnalyzer."""
    
    def __init__(self, data_dir='data', analysis_dir='analysis'):
        """
        Inicializa a extensão de análise.
        
        Args:
            data_dir: Diretório contendo dados de criptomoedas
            analysis_dir: Diretório contendo resultados de análise
        """
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        self.fetcher = CryptoDataFetcher(data_dir)
        self.analyzer = CryptoDataAnalyzer(self.fetcher)
    
    def _validate_data_availability(self):
        """
        Valida a disponibilidade dos dados necessários para as análises.
        
        Returns:
            dict: Dicionário com status de disponibilidade de cada tipo de dado
        """
        data_status = {
            'breadth': {
                'available': False,
                'files': ['breadth/52wk_highs_lows.csv', 'breadth/moving_average_tracking.csv'],
                'message': ''
            },
            'government': {
                'available': False,
                'files': [
                    f'gov/cftc_cot_bitcoin_{datetime.now().year}.csv',
                    f'gov/treasury_yields_{datetime.now().year}.csv'
                ],
                'message': ''
            },
            'ohlc': {
                'available': False,
                'files': [
                    'ohlc/binance_spot_BTCUSDT_daily.csv',
                    'ohlc/binance_futures_BTCUSDT_daily.csv'
                ],
                'message': ''
            }
        }
        
        # Verificar cada tipo de dado
        for data_type, info in data_status.items():
            missing_files = []
            for file in info['files']:
                file_path = os.path.join(self.data_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                elif os.path.getsize(file_path) == 0:
                    missing_files.append(f"{file} (vazio)")
            
            if not missing_files:
                info['available'] = True
                info['message'] = "✅ Dados disponíveis"
            else:
                info['message'] = f"❌ Arquivos faltando ou vazios: {', '.join(missing_files)}"
        
        return data_status
    
    def render_analysis_dashboard(self):
        """Renderiza o dashboard de análise avançada."""
        st.markdown("## 📊 Análise Avançada de Mercado")
        
        # Validar disponibilidade dos dados
        data_status = self._validate_data_availability()
        
        # Adicionar informações sobre a última atualização dos dados
        self._render_data_status()
        
        # Criar tabs para diferentes tipos de análise
        analysis_tabs = st.tabs([
            "Análise de Amplitude",
            "Análise Governamental",
            "Análise OHLC"
        ])
        
        try:
            with analysis_tabs[0]:
                if data_status['breadth']['available']:
                    with st.spinner("Carregando análise de amplitude do mercado..."):
                        self.analyzer.analyze_breadth_data()
                else:
                    st.error(data_status['breadth']['message'])
                    st.info("Verifique se os arquivos de dados necessários estão disponíveis no diretório correto.")
                
            with analysis_tabs[1]:
                if data_status['government']['available']:
                    with st.spinner("Carregando análise de dados governamentais..."):
                        self.analyzer.analyze_government_data()
                else:
                    st.error(data_status['government']['message'])
                    st.info("Verifique se os arquivos de dados governamentais estão disponíveis no diretório correto.")
                
            with analysis_tabs[2]:
                if data_status['ohlc']['available']:
                    with st.spinner("Carregando análise OHLC..."):
                        # Permitir seleção do par de trading
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            symbol = st.selectbox(
                                "Selecione o Par de Trading",
                                ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"],
                                index=0
                            )
                        
                        with col2:
                            # Botão para atualizar dados
                            if st.button("🔄 Atualizar Dados"):
                                st.experimental_rerun()
                        
                        self.analyzer.analyze_ohlc_data(symbol)
                else:
                    st.error(data_status['ohlc']['message'])
                    st.info("Verifique se os arquivos de dados OHLC estão disponíveis no diretório correto.")
        
        except Exception as e:
            st.error(f"Erro ao carregar análises: {str(e)}")
            st.info("Tente atualizar a página ou verificar a conexão com os dados")
            import traceback
            st.code(traceback.format_exc(), language="python")
    
    def _render_data_status(self):
        """Renderiza informações sobre o status dos dados."""
        try:
            # Verificar última atualização dos arquivos
            breadth_file = os.path.join(self.data_dir, "breadth/52wk_highs_lows.csv")
            gov_file = os.path.join(self.data_dir, f"gov/cftc_cot_bitcoin_{datetime.now().year}.csv")
            ohlc_file = os.path.join(self.data_dir, "ohlc/binance_spot_BTCUSDT_daily.csv")
            
            last_updates = []
            
            if os.path.exists(breadth_file):
                last_updates.append(("Dados de Amplitude", datetime.fromtimestamp(os.path.getmtime(breadth_file))))
            
            if os.path.exists(gov_file):
                last_updates.append(("Dados Governamentais", datetime.fromtimestamp(os.path.getmtime(gov_file))))
            
            if os.path.exists(ohlc_file):
                last_updates.append(("Dados OHLC", datetime.fromtimestamp(os.path.getmtime(ohlc_file))))
            
            if last_updates:
                with st.expander("ℹ️ Status dos Dados", expanded=False):
                    cols = st.columns(len(last_updates))
                    for i, (data_type, last_update) in enumerate(last_updates):
                        with cols[i]:
                            st.metric(
                                f"{data_type}",
                                f"Última Atualização",
                                f"{last_update.strftime('%d/%m/%Y %H:%M')}"
                            )
            else:
                st.warning("⚠️ Não foi possível verificar o status dos dados. Alguns recursos podem não estar disponíveis.")
        
        except Exception as e:
            st.warning(f"⚠️ Erro ao verificar status dos dados: {str(e)}")

class DashboardExtensions:
    """Class to add extended functionality to the cryptocurrency dashboard."""
    
    def __init__(self, data_dir='data', analysis_dir='analysis'):
        """
        Initialize dashboard extensions.
        
        Args:
            data_dir: Directory containing cryptocurrency data
            analysis_dir: Directory containing analysis results
        """
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        self.alert_system = CryptoAlertSystem(data_dir, analysis_dir)
        self.ml_analyzer = CryptoMLAnalyzer(data_dir, analysis_dir)
        self.analysis_extension = CryptoAnalysisExtension(data_dir, analysis_dir)
        
        # Initialize session state variables
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'alert_success' not in st.session_state:
            st.session_state.alert_success = None
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Create Alerts"
        if 'prediction_timeframe' not in st.session_state:
            st.session_state.prediction_timeframe = '7d'
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'default'
        if 'show_confidence' not in st.session_state:
            st.session_state.show_confidence = True
    
    def render_login_panel(self):
        """Render user login panel for personalized alerts."""
        st.markdown('<h2 class="sub-header">User Authentication</h2>', unsafe_allow_html=True)
        
        # Mostrar formulário de login se não estiver logado
        if not st.session_state.logged_in:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_email = st.text_input("Digite seu email para alertas personalizados:", 
                                          placeholder="usuario@exemplo.com")
            
            with col2:
                if st.button("Login"):
                    if user_email and '@' in user_email:
                        st.session_state.user_id = user_email
                        st.session_state.logged_in = True
                        st.session_state.username = user_email
                        st.success(f"Logado como {user_email}")
                        st.experimental_rerun()
                    else:
                        st.error("Por favor, digite um email válido")
        else:
            # Mostrar informações do usuário e botão de logout
            st.info(f"Logado como: {st.session_state.username}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.logged_in = False
                st.session_state.username = None
                st.success("Logout realizado com sucesso")
                st.experimental_rerun()
    
    def render_alert_dashboard(self):
        """Render the alerts dashboard."""
        st.markdown('<h2 class="sub-header">Crypto Alerts Dashboard</h2>', unsafe_allow_html=True)
        
        # Check if user is logged in
        if not st.session_state.logged_in:
            st.warning("Please login to use the alerts system")
            return
        
        # Create tabs
        tabs = ["Create Alerts", "My Alerts", "Alert Settings"]
        
        if 'current_tab' in st.session_state:
            selected_tab = st.radio("Select Tab", tabs, index=tabs.index(st.session_state.current_tab))
            st.session_state.current_tab = selected_tab
        else:
            selected_tab = st.radio("Select Tab", tabs)
            st.session_state.current_tab = selected_tab
        
        # Render selected tab
        if selected_tab == "Create Alerts":
            self.render_create_alerts_tab()
        elif selected_tab == "My Alerts":
            self.render_my_alerts_tab()
        elif selected_tab == "Alert Settings":
            self.render_alert_settings_tab()
    
    def render_create_alerts_tab(self):
        """Render the create alerts tab."""
        st.markdown('<h3 class="sub-header">Create New Alert</h3>', unsafe_allow_html=True)
        
        # Show alert success message if exists
        if st.session_state.alert_success:
            st.success(st.session_state.alert_success)
            st.session_state.alert_success = None
        
        # Load cryptocurrencies
        cryptos_file = f"{self.data_dir}/top_cryptos.csv"
        if os.path.exists(cryptos_file):
            cryptos_df = pd.read_csv(cryptos_file)
            symbols = sorted(cryptos_df['symbol'].tolist())
        else:
            symbols = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX", "MATIC", "LINK"]
        
        # Select cryptocurrency
        selected_symbol = st.selectbox("Select Cryptocurrency", symbols)
        
        # Alert type
        alert_types = [
            "Price Alert", 
            "Percent Change Alert", 
            "Technical Indicator Alert", 
            "Score Alert",
            "ML Prediction Alert"
        ]
        
        selected_alert_type = st.selectbox("Alert Type", alert_types)
        
        # Show form based on alert type
        if selected_alert_type == "Price Alert":
            self._render_price_alert_form(selected_symbol)
            
        elif selected_alert_type == "Percent Change Alert":
            self._render_percent_change_alert_form(selected_symbol)
            
        elif selected_alert_type == "Technical Indicator Alert":
            self._render_technical_alert_form(selected_symbol)
            
        elif selected_alert_type == "Score Alert":
            self._render_score_alert_form(selected_symbol)
            
        elif selected_alert_type == "ML Prediction Alert":
            self._render_ml_prediction_alert_form(selected_symbol)
    
    def _render_price_alert_form(self, symbol):
        """Render form for price alerts."""
        # Get current price if available
        current_price = self._get_current_price(symbol)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show current price if available
            if current_price:
                st.info(f"Current price: ${current_price:.2f}")
            
            # Input for target price
            target_price = st.number_input(
                "Target Price (USD)", 
                min_value=0.0,
                value=float(current_price) if current_price else 0.0,
                step=0.01
            )
        
        with col2:
            # Direction
            direction = st.radio(
                "Alert me when price goes:",
                ["above", "below"],
                horizontal=True
            )
            
            # Email checkbox
            notify_email = st.checkbox("Send email notification", value=True)
        
        # Submit button
        if st.button("Create Price Alert"):
            if target_price <= 0:
                st.error("Please enter a valid target price")
            else:
                alert_id = self.alert_system.add_price_alert(
                    user_id=st.session_state.user_id,
                    symbol=symbol,
                    target_price=target_price,
                    direction=direction,
                    notify_email=st.session_state.user_id if notify_email else None
                )
                
                if alert_id:
                    st.session_state.alert_success = f"Price alert created successfully (ID: {alert_id})"
                    st.experimental_rerun()
                else:
                    st.error("Failed to create alert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_percent_change_alert_form(self, symbol):
        """Render form for percent change alerts."""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input for percent change
            percent_change = st.number_input(
                "Percent Change (%)", 
                min_value=0.1,
                value=5.0,
                step=0.1
            )
            
            # Direction
            change_direction = st.radio(
                "Direction:",
                ["Increase", "Decrease"],
                horizontal=True
            )
            
            # Adjust sign based on direction
            percent_change = percent_change if change_direction == "Increase" else -percent_change
        
        with col2:
            # Time period
            time_period = st.radio(
                "Time Period:",
                ["1h", "24h", "7d"],
                horizontal=True
            )
            
            # Email checkbox
            notify_email = st.checkbox("Send email notification", value=True)
        
        # Submit button
        if st.button("Create Percent Change Alert"):
            alert_id = self.alert_system.add_percent_change_alert(
                user_id=st.session_state.user_id,
                symbol=symbol,
                percent_change=percent_change,
                time_period=time_period,
                notify_email=st.session_state.user_id if notify_email else None
            )
            
            if alert_id:
                st.session_state.alert_success = f"Percent change alert created successfully (ID: {alert_id})"
                st.experimental_rerun()
            else:
                st.error("Failed to create alert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_technical_alert_form(self, symbol):
        """Render form for technical indicator alerts."""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Technical indicator
            indicator = st.selectbox(
                "Technical Indicator:", 
                ["rsi", "macd", "ema_cross", "sma_cross", "bollinger"]
            )
        
        with col2:
            # Condition based on indicator
            if indicator in ["rsi", "macd"]:
                condition = st.selectbox(
                    "Condition:", 
                    ["above", "below"]
                )
                value = st.number_input(
                    "Value:", 
                    value=70 if indicator == "rsi" else 0,
                    step=0.1
                )
            elif indicator in ["ema_cross", "sma_cross"]:
                condition = st.selectbox(
                    "Condition:", 
                    ["crosses_above", "crosses_below"]
                )
                value = None
            elif indicator == "bollinger":
                condition = st.selectbox(
                    "Condition:", 
                    ["above", "below"]
                )
                value = None
            
            # Email checkbox
            notify_email = st.checkbox("Send email notification", value=True)
        
        # Submit button
        if st.button("Create Technical Alert"):
            alert_id = self.alert_system.add_technical_indicator_alert(
                user_id=st.session_state.user_id,
                symbol=symbol,
                indicator=indicator,
                condition=condition,
                value=value,
                notify_email=st.session_state.user_id if notify_email else None
            )
            
            if alert_id:
                st.session_state.alert_success = f"Technical indicator alert created successfully (ID: {alert_id})"
                st.experimental_rerun()
            else:
                st.error("Failed to create alert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_score_alert_form(self, symbol):
        """Render form for score alerts."""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score type
            score_type = st.selectbox(
                "Score Type:", 
                ["technical", "fundamental", "sentiment", "total"]
            )
            
            # Target score
            target_score = st.slider(
                "Target Score:", 
                min_value=0,
                max_value=100,
                value=70,
                step=1
            )
        
        with col2:
            # Direction
            direction = st.radio(
                "Alert me when score goes:",
                ["above", "below"],
                horizontal=True
            )
            
            # Email checkbox
            notify_email = st.checkbox("Send email notification", value=True)
        
        # Submit button
        if st.button("Create Score Alert"):
            alert_id = self.alert_system.add_score_alert(
                user_id=st.session_state.user_id,
                symbol=symbol,
                score_type=score_type,
                target_score=target_score,
                direction=direction,
                notify_email=st.session_state.user_id if notify_email else None
            )
            
            if alert_id:
                st.session_state.alert_success = f"Score alert created successfully (ID: {alert_id})"
                st.experimental_rerun()
            else:
                st.error("Failed to create alert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_ml_prediction_alert_form(self, symbol):
        """Render form for ML prediction alerts."""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction type
            prediction_type = st.selectbox(
                "Prediction Type:", 
                ["price_change", "trend", "confidence"]
            )
        
        with col2:
            # Threshold based on prediction type
            if prediction_type == "price_change":
                threshold = st.number_input(
                    "Percent Change Threshold:", 
                    min_value=0.1,
                    value=10.0,
                    step=0.1
                )
                direction = st.radio(
                    "Alert me when predicted change is:",
                    ["above", "below"],
                    horizontal=True
                )
            elif prediction_type == "trend":
                threshold = st.selectbox(
                    "Target Trend:", 
                    ["Bullish", "Bearish", "Neutral"]
                )
                direction = "equal"  # Not used but needed for API
            elif prediction_type == "confidence":
                threshold = st.slider(
                    "Confidence Threshold (%):", 
                    min_value=0,
                    max_value=100,
                    value=80,
                    step=1
                )
                direction = st.radio(
                    "Alert me when confidence is:",
                    ["above", "below"],
                    horizontal=True
                )
            
            # Email checkbox
            notify_email = st.checkbox("Send email notification", value=True)
        
        # Submit button
        if st.button("Create ML Prediction Alert"):
            alert_id = self.alert_system.add_ml_prediction_alert(
                user_id=st.session_state.user_id,
                symbol=symbol,
                prediction_type=prediction_type,
                threshold=threshold,
                direction=direction if prediction_type != "trend" else None,
                notify_email=st.session_state.user_id if notify_email else None
            )
            
            if alert_id:
                st.session_state.alert_success = f"ML prediction alert created successfully (ID: {alert_id})"
                st.experimental_rerun()
            else:
                st.error("Failed to create alert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_my_alerts_tab(self):
        """Render the my alerts tab."""
        st.markdown('<h3 class="sub-header">My Alerts</h3>', unsafe_allow_html=True)
        
        # Get user alerts
        user_alerts = self.alert_system.get_user_alerts(st.session_state.user_id)
        
        if not user_alerts:
            st.info("You don't have any alerts yet. Create your first alert in the 'Create Alerts' tab.")
            return
        
        # Group alerts by type
        grouped_alerts = {}
        for alert in user_alerts:
            alert_type = alert['type']
            if alert_type not in grouped_alerts:
                grouped_alerts[alert_type] = []
            grouped_alerts[alert_type].append(alert)
        
        # Display alerts by type
        for alert_type, alerts in grouped_alerts.items():
            st.markdown(f'<h4>{alert_type.replace("_", " ").title()} Alerts</h4>', unsafe_allow_html=True)
            
            for alert in alerts:
                # Create a card for each alert
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    # Alert details based on type
                    if alert_type == 'price':
                        st.markdown(f"**{alert['symbol']}**: Price {alert['direction']} ${alert['target_price']:.2f}")
                    elif alert_type == 'percent_change':
                        direction = "increase" if alert['percent_change'] > 0 else "decrease"
                        st.markdown(f"**{alert['symbol']}**: {abs(alert['percent_change']):.1f}% {direction} in {alert['time_period']}")
                    elif alert_type == 'technical':
                        if alert['indicator'] in ['rsi', 'macd'] and alert['value'] is not None:
                            st.markdown(f"**{alert['symbol']}**: {alert['indicator'].upper()} {alert['condition']} {alert['value']}")
                        else:
                            condition_text = alert['condition'].replace('crosses_', '')
                            if alert['indicator'] == 'ema_cross':
                                st.markdown(f"**{alert['symbol']}**: EMA-12 crosses {condition_text} EMA-26")
                            elif alert['indicator'] == 'sma_cross':
                                st.markdown(f"**{alert['symbol']}**: SMA-7 crosses {condition_text} SMA-30")
                            elif alert['indicator'] == 'bollinger':
                                band = "upper" if alert['condition'] == 'above' else "lower"
                                st.markdown(f"**{alert['symbol']}**: Price {alert['condition']} {band} Bollinger Band")
                    elif alert_type == 'score':
                        st.markdown(f"**{alert['symbol']}**: {alert['score_type'].capitalize()} score {alert['direction']} {alert['target_score']}")
                    elif alert_type == 'ml_prediction':
                        if alert['prediction_type'] == 'price_change':
                            st.markdown(f"**{alert['symbol']}**: Predicted price change {alert['direction']} {alert['threshold']}%")
                        elif alert['prediction_type'] == 'trend':
                            st.markdown(f"**{alert['symbol']}**: Predicted trend is {alert['threshold']}")
                        elif alert['prediction_type'] == 'confidence':
                            st.markdown(f"**{alert['symbol']}**: Prediction confidence {alert['direction']} {alert['threshold']}%")
                
                with col2:
                    # Status
                    if alert.get('triggered', False):
                        st.markdown(f"<span style='color:green;'>Triggered: {alert.get('triggered_at', 'N/A')}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:orange;'>Waiting...</span>", unsafe_allow_html=True)
                    
                    # Notification status
                    if alert.get('notify_email'):
                        st.markdown(f"Email notification: Enabled")
                    else:
                        st.markdown(f"Email notification: Disabled")
                
                with col3:
                    # Remove button
                    if st.button("Remove", key=f"remove_{alert['id']}"):
                        if self.alert_system.remove_alert(alert['id']):
                            st.success("Alert removed")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to remove alert")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Manual refresh button
        if st.button("Check Alerts Now"):
            triggered_alerts = self.alert_system.check_all_alerts()
            if triggered_alerts:
                st.success(f"Triggered {len(triggered_alerts)} alerts! Please refresh to see the updated status.")
            else:
                st.info("No alerts triggered at this time")
    
    def render_alert_settings_tab(self):
        """Render the alert settings tab."""
        st.markdown('<h3 class="sub-header">Email Notification Settings</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Email settings form
        with st.form("email_settings_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                smtp_server = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587)
                username = st.text_input("SMTP Username", placeholder="your_email@gmail.com")
            
            with col2:
                password = st.text_input("SMTP Password", type="password")
                sender_email = st.text_input("Sender Email", placeholder="your_email@gmail.com")
            
            submit_button = st.form_submit_button("Save Email Settings")
            
            if submit_button:
                if smtp_server and smtp_port and username and password and sender_email:
                    if self.alert_system.set_email_settings(
                        smtp_server=smtp_server,
                        smtp_port=smtp_port,
                        username=username,
                        password=password,
                        sender_email=sender_email
                    ):
                        st.success("Email settings saved successfully")
                    else:
                        st.error("Failed to save email settings")
                else:
                    st.error("Please fill all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<h3 class="sub-header">Alert System Status</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Show system status
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Active Alerts", len(self.alert_system.user_alerts))
            st.metric("Your Alerts", len(self.alert_system.get_user_alerts(st.session_state.user_id)))
        
        with col2:
            # Check if email settings are configured
            if self.alert_system.email_settings:
                st.success("Email notifications are configured")
            else:
                st.warning("Email notifications are not configured")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_ml_predictions_dashboard(self):
        """Render the ML predictions dashboard."""
        st.markdown('<h2 class="sub-header">Machine Learning Predictions</h2>', unsafe_allow_html=True)
        
        # Load cryptocurrencies
        cryptos_file = f"{self.data_dir}/top_cryptos.csv"
        if os.path.exists(cryptos_file):
            cryptos_df = pd.read_csv(cryptos_file)
            symbols = sorted(cryptos_df['symbol'].tolist())
        else:
            symbols = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX", "MATIC", "LINK"]
        
        # Create tabs
        tabs = ["Price Predictions", "Model Performance", "Trend Analysis"]
        selected_tab = st.radio("Select Tab", tabs, horizontal=True)
        
        # Select cryptocurrency
        selected_symbol = st.selectbox("Select Cryptocurrency", symbols)
        
        # Render selected tab
        if selected_tab == "Price Predictions":
            self.render_price_predictions_tab(selected_symbol)
        elif selected_tab == "Model Performance":
            self.render_model_performance_tab(selected_symbol)
        elif selected_tab == "Trend Analysis":
            self.render_trend_analysis_tab(selected_symbol)
    
    def render_price_predictions_tab(self, symbol):
        """Render the price predictions tab."""
        st.markdown('<h3 class="sub-header">Price Predictions</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Prediction options
            prediction_days = st.slider("Prediction Days", min_value=7, max_value=90, value=30, step=1)
            model_type = st.radio("Model Type", ["linear", "rf", "gb"])
            
            if st.button("Generate Prediction"):
                with st.spinner(f"Generating price prediction for {symbol}..."):
                    # Generate prediction
                    prediction_df = self.ml_analyzer.predict_future_prices(
                        symbol=symbol,
                        days=prediction_days,
                        model_type=model_type
                    )
                    
                    if prediction_df is not None:
                        st.success("Prediction generated successfully!")
                    else:
                        st.error("Failed to generate prediction")
        
        with col2:
            # Check if prediction visualization exists
            vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_{model_type}_prediction.png"
            
            if os.path.exists(vis_path):
                # Show visualization
                st.image(vis_path)
            else:
                # Generate visualization
                vis_path = self.ml_analyzer.visualize_predictions(
                    symbol=symbol,
                    model_type=model_type,
                    days=prediction_days
                )
                
                if vis_path and os.path.exists(vis_path):
                    st.image(vis_path)
                else:
                    st.info("No prediction visualization available. Click 'Generate Prediction' to create one.")
        
        # Show prediction details
        prediction_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_{model_type}_predictions.csv"
        
        if os.path.exists(prediction_file):
            st.markdown('<h4>Predicted Prices</h4>', unsafe_allow_html=True)
            
            # Load prediction data
            prediction_df = pd.read_csv(prediction_file)
            prediction_df['timestamp'] = pd.to_datetime(prediction_df['timestamp'])
            
            # Display in table form
            st.dataframe(prediction_df.style.format({
                'predicted_price': '${:.2f}'
            }))
        
        # Show price trend summary
        trend_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_trend_prediction.json"
        
        if os.path.exists(trend_file):
            with open(trend_file, 'r') as f:
                trend_data = json.load(f)
            
            st.markdown('<h4>Trend Prediction Summary</h4>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_change = trend_data.get('price_change_percent', 0)
                color = 'green' if price_change > 0 else 'red'
                st.markdown(f"<div style='font-size: 1.5rem; font-weight: bold; color: {color};'>{price_change:.2f}%</div>", unsafe_allow_html=True)
                st.markdown("Predicted Change")
            
            with col2:
                trend = trend_data.get('trend', 'Neutral')
                strength = trend_data.get('trend_strength', 'Neutral')
                color = 'green' if trend == 'Bullish' else 'red' if trend == 'Bearish' else 'orange'
                st.markdown(f"<div style='font-size: 1.5rem; font-weight: bold; color: {color};'>{trend} ({strength})</div>", unsafe_allow_html=True)
                st.markdown("Predicted Trend")
            
            with col3:
                confidence = trend_data.get('confidence', 0)
                st.markdown(f"<div style='font-size: 1.5rem; font-weight: bold;'>{confidence:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("Prediction Confidence")
    
    def render_model_performance_tab(self, symbol):
        """Render the model performance tab."""
        st.markdown('<h3 class="sub-header">Model Performance Comparison</h3>', unsafe_allow_html=True)
        
        # Generate model comparison
        if st.button("Compare Model Performance"):
            with st.spinner(f"Comparing model performance for {symbol}..."):
                comparison_df = self.ml_analyzer.compare_model_performances(symbol)
                
                if comparison_df is not None:
                    st.success("Model comparison completed successfully!")
                else:
                    st.error("Failed to compare models")
        
        # Check if comparison visualization exists
        vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_model_comparison.png"
        
        if os.path.exists(vis_path):
            # Show visualization
            st.image(vis_path)
        else:
            st.info("No model comparison available. Click 'Compare Model Performance' to create one.")
        
        # Show all models comparison
        all_models_vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_all_models_comparison.png"
        
        if os.path.exists(all_models_vis_path):
            st.markdown('<h4>Price Predictions Across All Models</h4>', unsafe_allow_html=True)
            st.image(all_models_vis_path)
        
        # Show comparison table
        comparison_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_model_comparison.csv"
        
        if os.path.exists(comparison_file):
            st.markdown('<h4>Model Performance Metrics</h4>', unsafe_allow_html=True)
            
            # Load comparison data
            comparison_df = pd.read_csv(comparison_file, index_col=0)
            
            # Display in table form
            st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen')
                                     .highlight_max(axis=0, subset=['R²'], color='lightgreen')
                                     .format({
                                         'RMSE': '{:.4f}',
                                         'MAE': '{:.4f}',
                                         'MAPE (%)': '{:.2f}',
                                         'R²': '{:.4f}'
                                     }))
            
            # Add explanation
            st.markdown("""
            **Metrics Explanation:**
            - **RMSE (Root Mean Square Error)**: Lower is better, measures the square root of the average squared difference between predicted and actual values.
            - **MAE (Mean Absolute Error)**: Lower is better, measures the average absolute difference between predicted and actual values.
            - **MAPE (Mean Absolute Percentage Error)**: Lower is better, measures the average percentage difference between predicted and actual values.
            - **R² (R-squared)**: Higher is better, measures how well the model explains the variance in the data.
            """)
    
    def render_trend_analysis_tab(self, symbol):
        """Render the trend analysis tab."""
        st.markdown('<h3 class="sub-header">Trend Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Trend analysis options
            prediction_days = st.slider("Prediction Horizon", min_value=7, max_value=90, value=30, step=1)
            
            if st.button("Analyze Trend"):
                with st.spinner(f"Analyzing trend for {symbol}..."):
                    # Generate trend prediction
                    trend_prediction = self.ml_analyzer.predict_price_trends(
                        symbol=symbol,
                        days=prediction_days
                    )
                    
                    if trend_prediction:
                        st.success("Trend analysis completed successfully!")
                    else:
                        st.error("Failed to analyze trend")
        
        # Show trend prediction
        trend_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_trend_prediction.json"
        
        if os.path.exists(trend_file):
            with open(trend_file, 'r') as f:
                trend_data = json.load(f)
            
            with col2:
                # Create trend summary card
                st.markdown('<div class="card" style="padding: 1rem;">', unsafe_allow_html=True)
                
                # Trend info
                trend = trend_data.get('trend', 'Neutral')
                strength = trend_data.get('trend_strength', 'Neutral')
                confidence = trend_data.get('confidence', 0)
                price_change = trend_data.get('price_change_percent', 0)
                current_price = trend_data.get('current_price', 0)
                predicted_price = trend_data.get('predicted_price', 0)
                best_model = trend_data.get('best_model', 'Unknown')
                
                # Determine color based on trend
                color = 'green' if trend == 'Bullish' else 'red' if trend == 'Bearish' else 'orange'
                
                # Display trend summary
                st.markdown(f"<h2 style='color: {color};'>{trend} {strength} Trend</h2>", unsafe_allow_html=True)
                st.markdown(f"<p>Confidence: <b>{confidence:.1f}%</b> (based on {trend_data.get('model_agreement', 0)}/3 model agreement)</p>", unsafe_allow_html=True)
                
                # Display price info
                st.markdown(f"<p>Current Price: <b>${current_price:.2f}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p>Predicted Price ({prediction_days} days): <b>${predicted_price:.2f}</b></p>", unsafe_allow_html=True)
                
                # Display price change
                change_color = 'green' if price_change > 0 else 'red'
                st.markdown(f"<p>Predicted Change: <span style='color: {change_color};'><b>{price_change:+.2f}%</b></span></p>", unsafe_allow_html=True)
                
                # Display best model
                st.markdown(f"<p>Best Performing Model: <b>{best_model}</b></p>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Show all models comparison visualization
        all_models_vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_all_models_comparison.png"
        
        if os.path.exists(all_models_vis_path):
            st.markdown('<h4>Model Predictions Comparison</h4>', unsafe_allow_html=True)
            st.image(all_models_vis_path)
        
        # Show trend analysis for multiple cryptocurrencies
        st.markdown('<h4>Market-Wide Trend Analysis</h4>', unsafe_allow_html=True)
        
        # Check if we have predictions for multiple cryptocurrencies
        all_predictions_file = f"{self.analysis_dir}/ml/predictions/all_cryptocurrencies_predictions.csv"
        
        if os.path.exists(all_predictions_file):
            # Load predictions
            predictions_df = pd.read_csv(all_predictions_file)
            
            if len(predictions_df) > 0:
                # Create trend distribution pie chart
                trend_counts = predictions_df['trend'].value_counts().reset_index()
                trend_counts.columns = ['Trend', 'Count']
                
                fig_trend = px.pie(
                    trend_counts, 
                    values='Count', 
                    names='Trend',
                    title='Trend Distribution',
                    color='Trend',
                    color_discrete_map={
                        'Bullish': 'green',
                        'Neutral': 'orange',
                        'Bearish': 'red'
                    }
                )
                
                st.plotly_chart(fig_trend)
                
                # Create table of all cryptocurrency predictions
                st.markdown('<h4>Trend Predictions for Top Cryptocurrencies</h4>', unsafe_allow_html=True)
                
                # Sort by price change (descending)
                predictions_df = predictions_df.sort_values('price_change_percent', ascending=False)
                
                # Display in table form
                st.dataframe(predictions_df[[
                    'symbol', 'name', 'trend', 'trend_strength', 'price_change_percent', 'confidence', 'best_model'
                ]].style.format({
                    'price_change_percent': '{:+.2f}%',
                    'confidence': '{:.1f}%'
                }).background_gradient(
                    subset=['price_change_percent'],
                    cmap='RdYlGn',
                    vmin=-20,
                    vmax=20
                ))
        else:
            if st.button("Analyze Market-Wide Trends"):
                with st.spinner("Analyzing trends for top cryptocurrencies..."):
                    # Analyze top 10 cryptocurrencies
                    predictions = self.ml_analyzer.analyze_all_cryptocurrencies(top_n=10)
                    
                    if predictions is not None:
                        st.success("Market-wide trend analysis completed successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to analyze market-wide trends")
    
    def _get_current_price(self, symbol):
        """Helper method to get current price of a cryptocurrency."""
        try:
            # Try to get from rankings first
            rankings_file = f"{self.analysis_dir}/crypto_rankings.csv"
            if os.path.exists(rankings_file):
                rankings_df = pd.read_csv(rankings_file)
                crypto_data = rankings_df[rankings_df['symbol'] == symbol]
                if not crypto_data.empty:
                    return crypto_data.iloc[0]['price_usd']
            
            # If not found in rankings, try top_cryptos
            cryptos_file = f"{self.data_dir}/top_cryptos.csv"
            if os.path.exists(cryptos_file):
                cryptos_df = pd.read_csv(cryptos_file)
                crypto_data = cryptos_df[cryptos_df['symbol'] == symbol]
                if not crypto_data.empty:
                    return crypto_data.iloc[0]['price_usd']
            
            # If not found in top_cryptos, try historical data
            if symbol in self.alert_system.historical_data:
                df = self.alert_system.historical_data[symbol]
                if not df.empty:
                    return df.iloc[-1]['price']
            
            return None
            
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None


def extend_dashboard(crypto_dashboard):
    """
    Extend the existing cryptocurrency dashboard with new features.
    
    Args:
        crypto_dashboard: Existing CryptoDashboard instance
    """
    # Initialize session state variables if they don't exist
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'alert_success' not in st.session_state:
        st.session_state.alert_success = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Create Alerts"
    if 'prediction_timeframe' not in st.session_state:
        st.session_state.prediction_timeframe = '7d'
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'default'
    if 'show_confidence' not in st.session_state:
        st.session_state.show_confidence = True
    
    # Create extensions
    extensions = DashboardExtensions()
    
    # Add new pages to sidebar
    st.sidebar.title("Advanced Features")
    
    advanced_page = st.sidebar.radio(
        "Select Advanced Feature",
        ["Análise Avançada", "Alerts System", "ML Predictions"]
    )
    
    # Render selected page
    if advanced_page == "Análise Avançada":
        # Render advanced analysis dashboard
        extensions.analysis_extension.render_analysis_dashboard()
    
    elif advanced_page == "Alerts System":
        # Render login panel
        extensions.render_login_panel()
        
        # Render alerts dashboard if logged in
        if st.session_state.logged_in:
            extensions.render_alert_dashboard()
    
    elif advanced_page == "ML Predictions":
        # Render ML predictions dashboard
        extensions.render_ml_predictions_dashboard()
"""
Alert system for cryptocurrency analysis.
Generates customized alerts based on user preferences and market conditions.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alerts.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("alert_system")

class CryptoAlertSystem:
    """Class to manage cryptocurrency alerts based on configurable triggers."""
    
    def __init__(self, data_dir='data', analysis_dir='analysis'):
        """
        Initialize the alert system.
        
        Args:
            data_dir: Directory containing cryptocurrency data
            analysis_dir: Directory containing analysis results
        """
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        self.rankings = None
        self.historical_data = {}
        self.user_alerts = {}
        self.triggered_alerts = []
        
        # Load alert configuration if exists
        self.config_file = "config/alerts.json"
        self.email_settings = {}
        
        # Create alert directories
        os.makedirs("alerts", exist_ok=True)
        os.makedirs("alerts/logs", exist_ok=True)
        os.makedirs("alerts/visualizations", exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Load data
        self._load_data()
    
    def _load_config(self):
        """Load alert system configuration."""
        logger.info(f"Carregando configuração de alertas de {self.config_file}")
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Load email settings
                if 'email' in config:
                    self.email_settings = config['email']
                    logger.info("Loaded email configuration")
                
                # Load user alerts
                if 'alerts' in config:
                    self.user_alerts = config['alerts']
                    logger.info(f"Loaded {len(self.user_alerts)} user alerts")
                
                logger.info(f"Configuração carregada com sucesso: {len(self.user_alerts)} alertas configurados")
                logger.debug(f"Alertas carregados: {list(self.user_alerts.keys())}")
                
                return True
            else:
                logger.warning("Configuration file not found")
                self.user_alerts = {}
                return False
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.error(traceback.format_exc())
            self.user_alerts = {}
            return False
    
    def _save_config(self):
        """Save alert system configuration."""
        logger.info("Salvando configuração de alertas")
        
        try:
            config = {
                'email': self.email_settings,
                'alerts': self.user_alerts
            }
            
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_data(self):
        """Load cryptocurrency data for alert processing."""
        try:
            # Load rankings
            rankings_file = f"{self.analysis_dir}/crypto_rankings.csv"
            if os.path.exists(rankings_file):
                self.rankings = pd.read_csv(rankings_file)
                logger.info(f"Loaded rankings for {len(self.rankings)} cryptocurrencies")
            else:
                logger.warning("Rankings data not found")
            
            # Load historical data for top cryptocurrencies
            if self.rankings is not None:
                for _, row in self.rankings.iterrows():
                    symbol = row['symbol']
                    daily_file = f"{self.data_dir}/{symbol.lower()}_historical_daily.csv"
                    
                    if os.path.exists(daily_file):
                        self.historical_data[symbol] = pd.read_csv(daily_file)
                        # Convert timestamp to datetime
                        self.historical_data[symbol]['timestamp'] = pd.to_datetime(
                            self.historical_data[symbol]['timestamp']
                        )
                
                logger.info(f"Loaded historical data for {len(self.historical_data)} cryptocurrencies")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def set_email_settings(self, smtp_server, smtp_port, username, password, sender_email):
        """
        Set email settings for alert notifications.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            sender_email: Sender email address
            
        Returns:
            True if settings are saved successfully, False otherwise
        """
        try:
            self.email_settings = {
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'username': username,
                'password': password,
                'sender_email': sender_email
            }
            
            # Save configuration
            self._save_config()
            
            logger.info("Email settings updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting email settings: {e}")
            return False
    
    def add_price_alert(self, user_id, symbol, target_price, direction='above', notify_email=None, alert_id=None):
        """
        Add a price alert for a cryptocurrency.
        
        Args:
            user_id: User identifier (email or username)
            symbol: Cryptocurrency symbol
            target_price: Target price for alert
            direction: 'above' or 'below'
            notify_email: Email to send notification (optional)
            alert_id: Custom alert ID (optional)
            
        Returns:
            Alert ID if successful, None otherwise
        """
        try:
            # Validate inputs
            if direction not in ['above', 'below']:
                logger.error(f"Invalid direction: {direction}")
                return None
            
            # Generate alert ID if not provided
            if alert_id is None:
                alert_id = f"price_{symbol.lower()}_{direction}_{int(time.time())}"
            
            # Create alert
            alert = {
                'id': alert_id,
                'user_id': user_id,
                'type': 'price',
                'symbol': symbol,
                'target_price': float(target_price),
                'direction': direction,
                'notify_email': notify_email,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'triggered': False,
                'triggered_at': None
            }
            
            # Add to user alerts
            self.user_alerts[alert_id] = alert
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Added price alert {alert_id} for {symbol}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error adding price alert: {e}")
            return None
    
    def add_percent_change_alert(self, user_id, symbol, percent_change, time_period='24h', notify_email=None, alert_id=None):
        """
        Add a percent change alert for a cryptocurrency.
        
        Args:
            user_id: User identifier (email or username)
            symbol: Cryptocurrency symbol
            percent_change: Target percent change for alert (e.g., 10 for 10%)
            time_period: Time period for percent change ('1h', '24h', '7d')
            notify_email: Email to send notification (optional)
            alert_id: Custom alert ID (optional)
            
        Returns:
            Alert ID if successful, None otherwise
        """
        try:
            # Validate inputs
            if time_period not in ['1h', '24h', '7d']:
                logger.error(f"Invalid time period: {time_period}")
                return None
            
            # Generate alert ID if not provided
            if alert_id is None:
                alert_id = f"pct_{symbol.lower()}_{time_period}_{int(time.time())}"
            
            # Create alert
            alert = {
                'id': alert_id,
                'user_id': user_id,
                'type': 'percent_change',
                'symbol': symbol,
                'percent_change': float(percent_change),
                'time_period': time_period,
                'notify_email': notify_email,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'triggered': False,
                'triggered_at': None
            }
            
            # Add to user alerts
            self.user_alerts[alert_id] = alert
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Added percent change alert {alert_id} for {symbol}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error adding percent change alert: {e}")
            return None
    
    def add_technical_indicator_alert(self, user_id, symbol, indicator, condition, value, notify_email=None, alert_id=None):
        """
        Add a technical indicator alert for a cryptocurrency.
        
        Args:
            user_id: User identifier (email or username)
            symbol: Cryptocurrency symbol
            indicator: Technical indicator ('rsi', 'macd', 'ema_cross', etc.)
            condition: Condition for alert ('above', 'below', 'crosses_above', 'crosses_below')
            value: Target value for alert
            notify_email: Email to send notification (optional)
            alert_id: Custom alert ID (optional)
            
        Returns:
            Alert ID if successful, None otherwise
        """
        try:
            # Validate inputs
            valid_indicators = ['rsi', 'macd', 'ema_cross', 'sma_cross', 'bollinger']
            valid_conditions = ['above', 'below', 'crosses_above', 'crosses_below']
            
            if indicator not in valid_indicators:
                logger.error(f"Invalid indicator: {indicator}")
                return None
            
            if condition not in valid_conditions:
                logger.error(f"Invalid condition: {condition}")
                return None
            
            # Generate alert ID if not provided
            if alert_id is None:
                alert_id = f"tech_{symbol.lower()}_{indicator}_{condition}_{int(time.time())}"
            
            # Create alert
            alert = {
                'id': alert_id,
                'user_id': user_id,
                'type': 'technical',
                'symbol': symbol,
                'indicator': indicator,
                'condition': condition,
                'value': float(value) if value is not None else None,
                'notify_email': notify_email,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'triggered': False,
                'triggered_at': None
            }
            
            # Add to user alerts
            self.user_alerts[alert_id] = alert
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Added technical indicator alert {alert_id} for {symbol}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error adding technical indicator alert: {e}")
            return None
    
    def add_score_alert(self, user_id, symbol, score_type, target_score, direction='above', notify_email=None, alert_id=None):
        """
        Add a score alert for a cryptocurrency.
        
        Args:
            user_id: User identifier (email or username)
            symbol: Cryptocurrency symbol
            score_type: Type of score ('technical', 'fundamental', 'sentiment', 'total')
            target_score: Target score for alert (0-100)
            direction: 'above' or 'below'
            notify_email: Email to send notification (optional)
            alert_id: Custom alert ID (optional)
            
        Returns:
            Alert ID if successful, None otherwise
        """
        try:
            # Validate inputs
            valid_score_types = ['technical', 'fundamental', 'sentiment', 'total']
            
            if score_type not in valid_score_types:
                logger.error(f"Invalid score type: {score_type}")
                return None
            
            if direction not in ['above', 'below']:
                logger.error(f"Invalid direction: {direction}")
                return None
            
            # Generate alert ID if not provided
            if alert_id is None:
                alert_id = f"score_{symbol.lower()}_{score_type}_{direction}_{int(time.time())}"
            
            # Create alert
            alert = {
                'id': alert_id,
                'user_id': user_id,
                'type': 'score',
                'symbol': symbol,
                'score_type': score_type,
                'target_score': float(target_score),
                'direction': direction,
                'notify_email': notify_email,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'triggered': False,
                'triggered_at': None
            }
            
            # Add to user alerts
            self.user_alerts[alert_id] = alert
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Added score alert {alert_id} for {symbol}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error adding score alert: {e}")
            return None
    
    def add_ml_prediction_alert(self, user_id, symbol, prediction_type, threshold, direction='above', notify_email=None, alert_id=None):
        """
        Add a machine learning prediction alert for a cryptocurrency.
        
        Args:
            user_id: User identifier (email or username)
            symbol: Cryptocurrency symbol
            prediction_type: Type of prediction ('price_change', 'trend', 'confidence')
            threshold: Threshold value for alert
            direction: 'above' or 'below'
            notify_email: Email to send notification (optional)
            alert_id: Custom alert ID (optional)
            
        Returns:
            Alert ID if successful, None otherwise
        """
        try:
            # Validate inputs
            valid_prediction_types = ['price_change', 'trend', 'confidence']
            
            if prediction_type not in valid_prediction_types:
                logger.error(f"Invalid prediction type: {prediction_type}")
                return None
            
            if direction not in ['above', 'below']:
                logger.error(f"Invalid direction: {direction}")
                return None
            
            # Generate alert ID if not provided
            if alert_id is None:
                alert_id = f"ml_{symbol.lower()}_{prediction_type}_{direction}_{int(time.time())}"
            
            # Create alert
            alert = {
                'id': alert_id,
                'user_id': user_id,
                'type': 'ml_prediction',
                'symbol': symbol,
                'prediction_type': prediction_type,
                'threshold': float(threshold) if prediction_type != 'trend' else threshold,
                'direction': direction,
                'notify_email': notify_email,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'triggered': False,
                'triggered_at': None
            }
            
            # Add to user alerts
            self.user_alerts[alert_id] = alert
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Added ML prediction alert {alert_id} for {symbol}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error adding ML prediction alert: {e}")
            return None
    
    def remove_alert(self, alert_id):
        """
        Remove an alert.
        
        Args:
            alert_id: Alert ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if alert_id in self.user_alerts:
                del self.user_alerts[alert_id]
                
                # Save configuration
                self._save_config()
                
                logger.info(f"Removed alert {alert_id}")
                return True
            else:
                logger.warning(f"Alert {alert_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing alert: {e}")
            return False
    
    def get_user_alerts(self, user_id):
        """
        Get all alerts for a user.
        
        Args:
            user_id: User identifier (email or username)
            
        Returns:
            List of alerts for the user
        """
        try:
            user_alerts = []
            
            for alert_id, alert in self.user_alerts.items():
                if alert['user_id'] == user_id:
                    user_alerts.append(alert)
            
            return user_alerts
            
        except Exception as e:
            logger.error(f"Error getting user alerts: {e}")
            return []
    
    def check_price_alert(self, alert):
        """
        Check if a price alert should be triggered.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is triggered, False otherwise
        """
        try:
            symbol = alert['symbol']
            
            # Get current price from rankings
            if self.rankings is None:
                logger.warning("Rankings data not available")
                return False
            
            crypto_data = self.rankings[self.rankings['symbol'] == symbol]
            if crypto_data.empty:
                logger.warning(f"No data found for {symbol}")
                return False
            
            current_price = crypto_data.iloc[0]['price_usd']
            target_price = alert['target_price']
            direction = alert['direction']
            
            # Check if alert should be triggered
            if direction == 'above' and current_price > target_price:
                logger.info(f"Price alert triggered: {symbol} above {target_price}")
                return True
            elif direction == 'below' and current_price < target_price:
                logger.info(f"Price alert triggered: {symbol} below {target_price}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking price alert: {e}")
            return False
    
    def check_percent_change_alert(self, alert):
        """
        Check if a percent change alert should be triggered.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is triggered, False otherwise
        """
        try:
            symbol = alert['symbol']
            
            # Get crypto data from rankings
            if self.rankings is None:
                logger.warning("Rankings data not available")
                return False
            
            crypto_data = self.rankings[self.rankings['symbol'] == symbol]
            if crypto_data.empty:
                logger.warning(f"No data found for {symbol}")
                return False
            
            # Get percent change based on time period
            time_period = alert['time_period']
            percent_change = 0
            
            if time_period == '1h':
                percent_change = crypto_data.iloc[0].get('percent_change_1h', 0)
            elif time_period == '24h':
                percent_change = crypto_data.iloc[0].get('percent_change_24h', 0)
            elif time_period == '7d':
                percent_change = crypto_data.iloc[0].get('percent_change_7d', 0)
            
            target_change = alert['percent_change']
            
            # Check if alert should be triggered
            if abs(percent_change) >= abs(target_change):
                # Check if direction matches
                if (percent_change > 0 and target_change > 0) or (percent_change < 0 and target_change < 0):
                    logger.info(f"Percent change alert triggered: {symbol} {percent_change}% in {time_period}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking percent change alert: {e}")
            return False
    
    def check_technical_indicator_alert(self, alert):
        """
        Check if a technical indicator alert should be triggered.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is triggered, False otherwise
        """
        try:
            symbol = alert['symbol']
            indicator = alert['indicator']
            condition = alert['condition']
            value = alert['value']
            
            # Get technical data
            technical_file = f"{self.analysis_dir}/technical/{symbol.lower()}_technical.json"
            
            if not os.path.exists(technical_file):
                logger.warning(f"Technical data not found for {symbol}")
                return False
            
            with open(technical_file, 'r') as f:
                technical_data = json.load(f)
            
            # Check indicator based on type
            if indicator == 'rsi':
                # RSI indicator
                current_value = technical_data.get('rsi_14', 0)
                
                if condition == 'above' and current_value > value:
                    logger.info(f"Technical alert triggered: {symbol} RSI above {value}")
                    return True
                elif condition == 'below' and current_value < value:
                    logger.info(f"Technical alert triggered: {symbol} RSI below {value}")
                    return True
                
            elif indicator == 'macd':
                # MACD indicator
                current_value = technical_data.get('macd', 0)
                signal_value = technical_data.get('macd_signal', 0)
                
                if condition == 'above' and current_value > value:
                    logger.info(f"Technical alert triggered: {symbol} MACD above {value}")
                    return True
                elif condition == 'below' and current_value < value:
                    logger.info(f"Technical alert triggered: {symbol} MACD below {value}")
                    return True
                elif condition == 'crosses_above' and current_value > signal_value:
                    logger.info(f"Technical alert triggered: {symbol} MACD crosses above signal")
                    return True
                elif condition == 'crosses_below' and current_value < signal_value:
                    logger.info(f"Technical alert triggered: {symbol} MACD crosses below signal")
                    return True
                
            elif indicator == 'ema_cross':
                # EMA cross
                ema_12 = technical_data.get('ema_12', 0)
                ema_26 = technical_data.get('ema_26', 0)
                
                if condition == 'crosses_above' and ema_12 > ema_26:
                    logger.info(f"Technical alert triggered: {symbol} EMA-12 crosses above EMA-26")
                    return True
                elif condition == 'crosses_below' and ema_12 < ema_26:
                    logger.info(f"Technical alert triggered: {symbol} EMA-12 crosses below EMA-26")
                    return True
                
            elif indicator == 'sma_cross':
                # SMA cross
                ma_7 = technical_data.get('ma_7', 0)
                ma_30 = technical_data.get('ma_30', 0)
                
                if condition == 'crosses_above' and ma_7 > ma_30:
                    logger.info(f"Technical alert triggered: {symbol} MA-7 crosses above MA-30")
                    return True
                elif condition == 'crosses_below' and ma_7 < ma_30:
                    logger.info(f"Technical alert triggered: {symbol} MA-7 crosses below MA-30")
                    return True
                
            elif indicator == 'bollinger':
                # Bollinger Bands
                price = technical_data.get('price', 0)
                bb_upper = technical_data.get('bb_upper', 0)
                bb_lower = technical_data.get('bb_lower', 0)
                
                if condition == 'above' and price > bb_upper:
                    logger.info(f"Technical alert triggered: {symbol} price above upper Bollinger Band")
                    return True
                elif condition == 'below' and price < bb_lower:
                    logger.info(f"Technical alert triggered: {symbol} price below lower Bollinger Band")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking technical indicator alert: {e}")
            return False
    
    def check_score_alert(self, alert):
        """
        Check if a score alert should be triggered.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is triggered, False otherwise
        """
        try:
            symbol = alert['symbol']
            score_type = alert['score_type']
            target_score = alert['target_score']
            direction = alert['direction']
            
            # Get score from rankings
            if self.rankings is None:
                logger.warning("Rankings data not available")
                return False
            
            crypto_data = self.rankings[self.rankings['symbol'] == symbol]
            if crypto_data.empty:
                logger.warning(f"No data found for {symbol}")
                return False
            
            # Get score based on type
            score_column = f"{score_type}_score"
            if score_column not in crypto_data.columns:
                logger.warning(f"Score column {score_column} not found")
                return False
            
            current_score = crypto_data.iloc[0][score_column]
            
            # Check if alert should be triggered
            if direction == 'above' and current_score > target_score:
                logger.info(f"Score alert triggered: {symbol} {score_type} score above {target_score}")
                return True
            elif direction == 'below' and current_score < target_score:
                logger.info(f"Score alert triggered: {symbol} {score_type} score below {target_score}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking score alert: {e}")
            return False
    
    def check_ml_prediction_alert(self, alert):
        """
        Check if a machine learning prediction alert should be triggered.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is triggered, False otherwise
        """
        try:
            symbol = alert['symbol']
            prediction_type = alert['prediction_type']
            threshold = alert['threshold']
            direction = alert['direction']
            
            # Get ML prediction data
            prediction_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_trend_prediction.json"
            
            if not os.path.exists(prediction_file):
                logger.warning(f"ML prediction data not found for {symbol}")
                return False
            
            with open(prediction_file, 'r') as f:
                prediction_data = json.load(f)
            
            # Check prediction based on type
            if prediction_type == 'price_change':
                current_value = prediction_data.get('price_change_percent', 0)
                
                if direction == 'above' and current_value > threshold:
                    logger.info(f"ML alert triggered: {symbol} predicted price change above {threshold}%")
                    return True
                elif direction == 'below' and current_value < threshold:
                    logger.info(f"ML alert triggered: {symbol} predicted price change below {threshold}%")
                    return True
                
            elif prediction_type == 'trend':
                current_trend = prediction_data.get('trend', 'Neutral')
                
                if current_trend == threshold:
                    logger.info(f"ML alert triggered: {symbol} predicted trend is {threshold}")
                    return True
                
            elif prediction_type == 'confidence':
                current_value = prediction_data.get('confidence', 0)
                
                if direction == 'above' and current_value > threshold:
                    logger.info(f"ML alert triggered: {symbol} prediction confidence above {threshold}%")
                    return True
                elif direction == 'below' and current_value < threshold:
                    logger.info(f"ML alert triggered: {symbol} prediction confidence below {threshold}%")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking ML prediction alert: {e}")
            return False
    
    def check_all_alerts(self):
        """
        Check all alerts and trigger notifications for those that match conditions.
        
        Returns:
            List of triggered alerts
        """
        try:
            # Reset triggered alerts
            self.triggered_alerts = []
            
            # Check each alert
            for alert_id, alert in self.user_alerts.items():
                # Skip already triggered alerts
                if alert.get('triggered', False):
                    continue
                
                # Check different alert types
                triggered = False
                
                if alert['type'] == 'price':
                    triggered = self.check_price_alert(alert)
                    
                elif alert['type'] == 'percent_change':
                    triggered = self.check_percent_change_alert(alert)
                    
                elif alert['type'] == 'technical':
                    triggered = self.check_technical_indicator_alert(alert)
                    
                elif alert['type'] == 'score':
                    triggered = self.check_score_alert(alert)
                    
                elif alert['type'] == 'ml_prediction':
                    triggered = self.check_ml_prediction_alert(alert)
                
                # If alert is triggered, update status and notify
                if triggered:
                    # Update alert status
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add to triggered alerts
                    self.triggered_alerts.append(alert)
                    
                    # Send notification if email is provided
                    if alert.get('notify_email'):
                        self.send_alert_notification(alert)
            
            # Save updated alert statuses
            self._save_config()
            
            return self.triggered_alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
    
    def create_alert_visualization(self, alert):
        """
        Create a visualization for an alert.
        
        Args:
            alert: Alert to visualize
            
        Returns:
            Path to visualization file
        """
        try:
            symbol = alert['symbol']
            alert_type = alert['type']
            
            # Get historical data
            if symbol not in self.historical_data:
                logger.warning(f"Historical data not found for {symbol}")
                return None
            
            df = self.historical_data[symbol].copy()
            df = df.sort_values('timestamp')
            
            # Create visualization based on alert type
            plt.figure(figsize=(12, 6))
            
            if alert_type == 'price':
                # Price alert visualization
                target_price = alert['target_price']
                direction = alert['direction']
                
                # Plot price history
                plt.plot(df['timestamp'], df['price'], label='Price', color='blue')
                
                # Plot target price line
                plt.axhline(y=target_price, color='red', linestyle='--', 
                           label=f"Target: {target_price} ({direction})")
                
                # Add labels
                plt.title(f"{symbol} Price Alert")
                plt.xlabel("Date")
                plt.ylabel("Price (USD)")
                
            elif alert_type == 'percent_change':
                # Percent change visualization
                time_period = alert['time_period']
                target_change = alert['percent_change']
                
                # Calculate percent change
                if time_period == '1h':
                    window = 1
                elif time_period == '24h':
                    window = 24
                else:  # 7d
                    window = 168  # 24*7
                
                # Simplify by using last N data points based on window
                tail_size = min(len(df), window * 2)
                recent_df = df.tail(tail_size).copy()
                
                # Calculate percent change from oldest to newest in window
                start_price = recent_df['price'].iloc[0]
                end_price = recent_df['price'].iloc[-1]
                actual_change = (end_price - start_price) / start_price * 100
                
                # Plot price history
                plt.plot(recent_df['timestamp'], recent_df['price'], label='Price', color='blue')
                
                # Annotate percent change
                plt.annotate(
                    f"{actual_change:.2f}% change",
                    xy=(recent_df['timestamp'].iloc[-1], recent_df['price'].iloc[-1]),
                    xytext=(10, -30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red')
                )
                
                # Add labels
                plt.title(f"{symbol} {time_period} Percent Change Alert ({target_change}%)")
                plt.xlabel("Date")
                plt.ylabel("Price (USD)")
                
            elif alert_type == 'technical':
                # Technical indicator visualization
                indicator = alert['indicator']
                condition = alert['condition']
                value = alert.get('value')
                
                # Plot price history
                plt.plot(df['timestamp'], df['price'], label='Price', color='blue')
                
                # Add technical indicator based on type
                if indicator == 'rsi':
                    # Calculate RSI
                    def calculate_rsi(prices, period=14):
                        # Calculate price differences
                        delta = prices.diff()
                        
                        # Separate gains and losses
                        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                        
                        # Calculate RS
                        rs = gain / loss
                        
                        # Calculate RSI
                        rsi = 100 - (100 / (1 + rs))
                        return rsi
                    
                    # Add subplot for RSI
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    
                    # Calculate RSI
                    df['rsi'] = calculate_rsi(df['price'])
                    
                    # Plot RSI
                    ax2.plot(df['timestamp'], df['rsi'], label='RSI', color='green')
                    
                    # Plot threshold line
                    if value is not None:
                        ax2.axhline(y=value, color='red', linestyle='--', 
                                  label=f"Threshold: {value}")
                    
                    # Add RSI labels
                    ax2.set_ylabel('RSI')
                    ax2.legend(loc='upper right')
                    
                    # Add title
                    plt.title(f"{symbol} RSI Alert")
                    
                elif indicator == 'macd':
                    # Calculate MACD
                    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
                    df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
                    df['macd'] = df['ema_12'] - df['ema_26']
                    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    
                    # Add subplot for MACD
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    
                    # Plot MACD and signal
                    ax2.plot(df['timestamp'], df['macd'], label='MACD', color='green')
                    ax2.plot(df['timestamp'], df['signal'], label='Signal', color='red')
                    
                    # Plot threshold line if condition is above/below
                    if condition in ['above', 'below'] and value is not None:
                        ax2.axhline(y=value, color='purple', linestyle='--', 
                                  label=f"Threshold: {value}")
                    
                    # Add MACD labels
                    ax2.set_ylabel('MACD')
                    ax2.legend(loc='upper right')
                    
                    # Add title
                    plt.title(f"{symbol} MACD Alert")
                    
                elif indicator in ['ema_cross', 'sma_cross']:
                    # Calculate moving averages
                    if indicator == 'ema_cross':
                        df['ma1'] = df['price'].ewm(span=12, adjust=False).mean()
                        df['ma2'] = df['price'].ewm(span=26, adjust=False).mean()
                        ma1_label = 'EMA-12'
                        ma2_label = 'EMA-26'
                    else:
                        df['ma1'] = df['price'].rolling(window=7).mean()
                        df['ma2'] = df['price'].rolling(window=30).mean()
                        ma1_label = 'SMA-7'
                        ma2_label = 'SMA-30'
                    
                    # Plot moving averages
                    plt.plot(df['timestamp'], df['ma1'], label=ma1_label, color='green')
                    plt.plot(df['timestamp'], df['ma2'], label=ma2_label, color='red')
                    
                    # Add title
                    plt.title(f"{symbol} Moving Average Crossover Alert")
                    
                elif indicator == 'bollinger':
                    # Calculate Bollinger Bands
                    df['ma20'] = df['price'].rolling(window=20).mean()
                    df['std20'] = df['price'].rolling(window=20).std()
                    df['upper'] = df['ma20'] + 2 * df['std20']
                    df['lower'] = df['ma20'] - 2 * df['std20']
                    
                    # Plot Bollinger Bands
                    plt.plot(df['timestamp'], df['upper'], label='Upper Band', color='red')
                    plt.plot(df['timestamp'], df['ma20'], label='Middle Band', color='green')
                    plt.plot(df['timestamp'], df['lower'], label='Lower Band', color='red')
                    
                    # Add title
                    plt.title(f"{symbol} Bollinger Bands Alert")
                
            elif alert_type == 'score':
                # Score alert visualization
                score_type = alert['score_type']
                target_score = alert['target_score']
                direction = alert['direction']
                
                # Get score from rankings
                if self.rankings is not None:
                    crypto_data = self.rankings[self.rankings['symbol'] == symbol]
                    if not crypto_data.empty:
                        score_column = f"{score_type}_score"
                        if score_column in crypto_data.columns:
                            current_score = crypto_data.iloc[0][score_column]
                            
                            # Create bar chart for score
                            plt.figure(figsize=(8, 6))
                            plt.bar(['Current Score'], [current_score], color='blue')
                            
                            # Add target line
                            plt.axhline(y=target_score, color='red', linestyle='--', 
                                       label=f"Target: {target_score} ({direction})")
                            
                            # Add labels
                            plt.title(f"{symbol} {score_type.capitalize()} Score Alert")
                            plt.ylabel('Score (0-100)')
                            plt.ylim(0, 100)
                
            elif alert_type == 'ml_prediction':
                # ML prediction visualization
                prediction_type = alert['prediction_type']
                threshold = alert['threshold']
                direction = alert.get('direction')
                
                # Get prediction data
                prediction_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_trend_prediction.json"
                
                if os.path.exists(prediction_file):
                    with open(prediction_file, 'r') as f:
                        prediction_data = json.load(f)
                    
                    # Create visualization based on prediction type
                    if prediction_type == 'price_change':
                        current_price = prediction_data.get('current_price', 0)
                        predicted_price = prediction_data.get('predicted_price', 0)
                        price_change = prediction_data.get('price_change_percent', 0)
                        
                        # Create bar chart for current and predicted price
                        plt.bar(['Current', 'Predicted'], [current_price, predicted_price], color=['blue', 'green'])
                        
                        # Add percent change annotation
                        plt.annotate(
                            f"{price_change:.2f}%",
                            xy=(1, predicted_price),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center'
                        )
                        
                        # Add threshold line
                        if direction == 'above':
                            target_price = current_price * (1 + threshold / 100)
                            plt.axhline(y=target_price, color='red', linestyle='--', 
                                       label=f"Target: +{threshold}%")
                        else:
                            target_price = current_price * (1 - threshold / 100)
                            plt.axhline(y=target_price, color='red', linestyle='--', 
                                       label=f"Target: -{threshold}%")
                        
                        # Add labels
                        plt.title(f"{symbol} Price Change Prediction Alert")
                        plt.ylabel('Price (USD)')
                        
                    elif prediction_type == 'trend':
                        # Create text visualization for trend
                        current_trend = prediction_data.get('trend', 'Neutral')
                        target_trend = threshold
                        
                        plt.text(0.5, 0.5, f"Predicted Trend: {current_trend}\nTarget Trend: {target_trend}", 
                                ha='center', va='center', fontsize=16)
                        plt.axis('off')
                        
                        # Add title
                        plt.title(f"{symbol} Trend Prediction Alert")
                        
                    elif prediction_type == 'confidence':
                        confidence = prediction_data.get('confidence', 0)
                        
                        # Create gauge chart for confidence
                        plt.pie([confidence, 100-confidence], colors=['green', 'lightgray'], startangle=90, 
                               counterclock=False, labels=['', ''], wedgeprops=dict(width=0.3))
                        plt.text(0, 0, f"{confidence:.1f}%", ha='center', va='center', fontsize=24)
                        
                        # Add threshold annotation
                        plt.annotate(
                            f"Threshold: {threshold}%",
                            xy=(0, -0.5),
                            xytext=(0, -70),
                            textcoords='offset points',
                            ha='center'
                        )
                        
                        # Add title
                        plt.title(f"{symbol} Prediction Confidence Alert")
            
            # Add legend
            plt.legend()
            
            # Format dates
            plt.gcf().autofmt_xdate()
            
            # Save visualization
            vis_path = f"alerts/visualizations/{alert['id']}.png"
            plt.savefig(vis_path, dpi=300)
            plt.close()
            
            return vis_path
            
        except Exception as e:
            logger.error(f"Error creating alert visualization: {e}")
            return None
    
    def send_alert_notification(self, alert):
        """
        Send email notification for a triggered alert.
        
        Args:
            alert: Triggered alert
            
        Returns:
            True if notification is sent successfully, False otherwise
        """
        try:
            # Check if email settings and notification email are provided
            if not self.email_settings or not alert.get('notify_email'):
                logger.warning("Email settings or notification email not provided")
                return False
            
            # Create visualization
            vis_path = self.create_alert_visualization(alert)
            
            # Create email message
            symbol = alert['symbol']
            alert_type = alert['type']
            
            # Get symbol name from rankings
            symbol_name = symbol
            if self.rankings is not None:
                crypto_data = self.rankings[self.rankings['symbol'] == symbol]
                if not crypto_data.empty:
                    symbol_name = crypto_data.iloc[0].get('name', symbol)
            
            # Create email subject
            subject = f"Crypto Alert: {symbol_name} ({symbol}) {alert_type.replace('_', ' ').title()} Alert"
            
            # Create email body based on alert type
            if alert_type == 'price':
                target_price = alert['target_price']
                direction = alert['direction']
                body = f"Your price alert for {symbol_name} ({symbol}) has been triggered.\n\n"
                body += f"The price is now {direction} {target_price} USD."
                
            elif alert_type == 'percent_change':
                time_period = alert['time_period']
                target_change = alert['percent_change']
                body = f"Your percent change alert for {symbol_name} ({symbol}) has been triggered.\n\n"
                body += f"The price has changed by at least {target_change}% in the last {time_period}."
                
            elif alert_type == 'technical':
                indicator = alert['indicator']
                condition = alert['condition']
                value = alert.get('value')
                body = f"Your technical indicator alert for {symbol_name} ({symbol}) has been triggered.\n\n"
                
                if indicator == 'rsi':
                    body += f"The RSI is now {condition} {value}."
                elif indicator == 'macd':
                    if condition in ['above', 'below']:
                        body += f"The MACD is now {condition} {value}."
                    else:
                        body += f"The MACD has crossed {condition.replace('crosses_', '')} the signal line."
                elif indicator == 'ema_cross':
                    body += f"The EMA-12 has crossed {condition.replace('crosses_', '')} the EMA-26."
                elif indicator == 'sma_cross':
                    body += f"The SMA-7 has crossed {condition.replace('crosses_', '')} the SMA-30."
                elif indicator == 'bollinger':
                    if condition == 'above':
                        body += f"The price is now above the upper Bollinger Band."
                    else:
                        body += f"The price is now below the lower Bollinger Band."
                
            elif alert_type == 'score':
                score_type = alert['score_type']
                target_score = alert['target_score']
                direction = alert['direction']
                body = f"Your score alert for {symbol_name} ({symbol}) has been triggered.\n\n"
                body += f"The {score_type} score is now {direction} {target_score}."
                
            elif alert_type == 'ml_prediction':
                prediction_type = alert['prediction_type']
                threshold = alert['threshold']
                direction = alert.get('direction')
                body = f"Your machine learning prediction alert for {symbol_name} ({symbol}) has been triggered.\n\n"
                
                if prediction_type == 'price_change':
                    body += f"The predicted price change is now {direction} {threshold}%."
                elif prediction_type == 'trend':
                    body += f"The predicted trend is now {threshold}."
                elif prediction_type == 'confidence':
                    body += f"The prediction confidence is now {direction} {threshold}%."
            
            # Add timestamp
            body += f"\n\nAlert triggered at: {alert['triggered_at']}"
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.email_settings['sender_email']
            msg['To'] = alert['notify_email']
            msg['Subject'] = subject
            
            # Attach text
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach visualization if available
            if vis_path and os.path.exists(vis_path):
                with open(vis_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=f"{alert['id']}.png")
                    msg.attach(img)
            
            # Send email
            server = smtplib.SMTP(self.email_settings['smtp_server'], self.email_settings['smtp_port'])
            server.starttls()
            server.login(self.email_settings['username'], self.email_settings['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert notification sent to {alert['notify_email']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            return False


# Run alert system if script is executed directly
if __name__ == "__main__":
    alert_system = CryptoAlertSystem()
    
    # Example: Add a price alert
    alert_system.add_price_alert(
        user_id="user@example.com",
        symbol="BTC",
        target_price=60000,
        direction="above",
        notify_email="user@example.com"
    )
    
    # Check alerts
    triggered_alerts = alert_system.check_all_alerts()
    
    if triggered_alerts:
        print(f"Triggered {len(triggered_alerts)} alerts:")
        for alert in triggered_alerts:
            print(f"  {alert['id']}: {alert['symbol']} {alert['type']} alert")
    else:
        print("No alerts triggered")
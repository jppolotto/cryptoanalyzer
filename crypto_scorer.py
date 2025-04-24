"""
Scoring system for ranking cryptocurrencies based on technical, fundamental,
sentiment and relative performance metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema, find_peaks
from sklearn.linear_model import LinearRegression
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_scorer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("crypto_scorer")

def calculate_rsi(prices, period=14):
    """Calcula o RSI (Relative Strength Index)"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else float('inf')
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up/down if down != 0 else float('inf')
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcula o MACD (Moving Average Convergence Divergence)"""
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd.values, signal_line.values, histogram.values

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calcula as Bandas de Bollinger"""
    rolling_mean = pd.Series(prices).rolling(window=period).mean()
    rolling_std = pd.Series(prices).rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band.values, rolling_mean.values, lower_band.values

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calcula o Estocástico"""
    lowest_low = pd.Series(low).rolling(window=k_period).min()
    highest_high = pd.Series(high).rolling(window=k_period).max()
    k = 100 * ((pd.Series(close) - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k.values, d.values

def calculate_adx(high, low, close, period=14):
    """Calcula o ADX (Average Directional Index)"""
    # True Range
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Plus Directional Movement
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff()
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)

    # Minus Directional Movement
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx.values, plus_di.values, minus_di.values

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

class CryptoScorer:
    """Class to score and rank cryptocurrencies based on multiple factors."""
    
    def __init__(self, data_dir='data'):
        """
        Initialize the CryptoScorer.
        
        Args:
            data_dir: Directory containing cryptocurrency data
        """
        self.data_dir = data_dir
        self.top_cryptos = None
        self.historical_data = {}
        self.global_metrics = None
        self.fear_greed = None
        self.scores = None
        self.market_trends = None
        
        # Create output directories
        os.makedirs('analysis', exist_ok=True)
        os.makedirs('analysis/technical', exist_ok=True)
        os.makedirs('analysis/fundamental', exist_ok=True)
        os.makedirs('analysis/sentiment', exist_ok=True)
        os.makedirs('analysis/patterns', exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Calculate market trends
        self._calculate_market_trends()

    def _load_data(self):
        """Load cryptocurrency data from files."""
        try:
            # Load top cryptocurrencies
            if os.path.exists(f"{self.data_dir}/top_cryptos.csv"):
                self.top_cryptos = pd.read_csv(f"{self.data_dir}/top_cryptos.csv")
                logger.info(f"Loaded data for {len(self.top_cryptos)} cryptocurrencies")
            else:
                logger.warning("Top cryptocurrencies data not found")
                
            # Load global metrics
            if os.path.exists(f"{self.data_dir}/global_metrics.csv"):
                self.global_metrics = pd.read_csv(f"{self.data_dir}/global_metrics.csv").iloc[0].to_dict()
                logger.info("Loaded global market metrics")
            else:
                logger.warning("Global metrics data not found")
                
            # Load Fear & Greed Index
            if os.path.exists(f"{self.data_dir}/fear_greed_index.csv"):
                self.fear_greed = pd.read_csv(f"{self.data_dir}/fear_greed_index.csv").iloc[0].to_dict()
                logger.info("Loaded Fear & Greed Index")
            else:
                logger.warning("Fear & Greed Index data not found")
                
            # Load historical data for each cryptocurrency
            if self.top_cryptos is not None:
                for _, row in self.top_cryptos.iterrows():
                    symbol = row['symbol']
                    daily_file = f"{self.data_dir}/{symbol.lower()}_historical_daily.csv"
                    
                    if os.path.exists(daily_file):
                        try:
                            df = pd.read_csv(daily_file)
                            # Convert timestamp to datetime
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            # Sort by timestamp
                            df = df.sort_values('timestamp')
                            # Store data
                            self.historical_data[symbol] = {'daily': df}
                        except Exception as e:
                            logger.error(f"Error loading historical data for {symbol}: {e}")
                            continue
                
                logger.info(f"Loaded historical data for {len(self.historical_data)} cryptocurrencies")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error(traceback.format_exc())

    def _calculate_market_trends(self):
        """Calculate overall market trends and metrics."""
        try:
            if self.top_cryptos is None or self.top_cryptos.empty:
                logger.warning("No cryptocurrency data available for market trends")
                return
            
            # Calculate market caps by category
            total_market_cap = self.top_cryptos['market_cap_usd'].sum()
            
            # Get top 5, top 10, top 20 market caps
            top5_market_cap = self.top_cryptos.head(5)['market_cap_usd'].sum()
            top10_market_cap = self.top_cryptos.head(10)['market_cap_usd'].sum()
            top20_market_cap = self.top_cryptos.head(20)['market_cap_usd'].sum()
            
            # Calculate percentages
            top5_dominance = (top5_market_cap / total_market_cap) * 100
            top10_dominance = (top10_market_cap / total_market_cap) * 100
            top20_dominance = (top20_market_cap / total_market_cap) * 100
            
            # Calculate average changes
            avg_change_24h = self.top_cryptos['percent_change_24h'].mean()
            avg_change_7d = self.top_cryptos['percent_change_7d'].mean()
            
            # Get BTC dominance from global metrics
            btc_dominance = self.global_metrics.get('btc_dominance', 0) if self.global_metrics else 0
            
            # Determine market trend
            if avg_change_24h > 3 and avg_change_7d > 7:
                market_trend = "Strong Bullish"
            elif avg_change_24h > 1 and avg_change_7d > 3:
                market_trend = "Bullish"
            elif avg_change_24h < -3 and avg_change_7d < -7:
                market_trend = "Strong Bearish"
            elif avg_change_24h < -1 and avg_change_7d < -3:
                market_trend = "Bearish"
            else:
                market_trend = "Neutral"
            
            # Calculate market sentiment
            if self.fear_greed:
                fear_greed_value = self.fear_greed.get('value', 50)
                if fear_greed_value <= 25:
                    market_sentiment = "Extreme Fear"
                elif fear_greed_value <= 40:
                    market_sentiment = "Fear"
                elif fear_greed_value <= 60:
                    market_sentiment = "Neutral"
                elif fear_greed_value <= 75:
                    market_sentiment = "Greed"
                else:
                    market_sentiment = "Extreme Greed"
            else:
                market_sentiment = "Unknown"
            
            # Store market trends
            self.market_trends = {
                'total_market_cap': total_market_cap,
                'top5_market_cap': top5_market_cap,
                'top10_market_cap': top10_market_cap,
                'top20_market_cap': top20_market_cap,
                'top5_dominance': top5_dominance,
                'top10_dominance': top10_dominance,
                'top20_dominance': top20_dominance,
                'btc_dominance': btc_dominance,
                'avg_change_24h': avg_change_24h,
                'avg_change_7d': avg_change_7d,
                'market_trend': market_trend,
                'market_sentiment': market_sentiment
            }
            
            logger.info(f"Calculated market trends: {market_trend}, {market_sentiment}")
            
        except Exception as e:
            logger.error(f"Error calculating market trends: {e}")
            logger.error(traceback.format_exc())

# ... rest of the existing code ... 
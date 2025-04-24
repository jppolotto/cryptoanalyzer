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

// ... rest of the existing code ... 
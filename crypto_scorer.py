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
import talib
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
            self.market_trends = {}
    
    def _preprocess_data(self, df):
        """Preprocess and clean historical data."""
        try:
            # Check for NaN values
            if df.isnull().values.any():
                # Fill NaN values using forward fill
                df = df.fillna(method='ffill')
                # If still have NaNs (e.g., at the beginning), use backward fill
                df = df.fillna(method='bfill')
            
            # Check for duplicate timestamps
            if df.duplicated(subset=['timestamp']).any():
                # Keep only the last entry for each timestamp
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Ensure data is sorted by timestamp
            df = df.sort_values('timestamp')
            
            # Fill missing dates (if any) with interpolated values
            if len(df) >= 2:
                date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
                if len(date_range) > len(df):
                    # Create a complete dataframe with all dates
                    full_df = pd.DataFrame({'timestamp': date_range})
                    # Merge with original data
                    df = pd.merge(full_df, df, on='timestamp', how='left')
                    # Interpolate missing values
                    df = df.interpolate(method='time')
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for historical data.
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure numeric types for calculations
            numeric_cols = ['price', 'volume_24h', 'market_cap']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 1. Moving Averages
            df['ma_7'] = df['price'].rolling(window=7).mean()
            df['ma_30'] = df['price'].rolling(window=30).mean()
            df['ma_90'] = df['price'].rolling(window=90).mean()
            df['ma_200'] = df['price'].rolling(window=200).mean()
            
            # Calcular média móvel de 7 dias para o volume
            if 'volume_24h' in df.columns:
                df['volume_ma_7'] = df['volume_24h'].rolling(window=7).mean()
            
            # 2. Exponential Moving Averages
            df['ema_9'] = df['price'].ewm(span=9, adjust=False).mean()
            df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
            df['ema_50'] = df['price'].ewm(span=50, adjust=False).mean()
            
            # 3. Relative Strength Index (RSI)
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                
                # Avoid division by zero
                loss = np.where(loss == 0, 0.000001, loss)
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            df['rsi_14'] = calculate_rsi(df['price'])
            
            # 4. MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 5. Bollinger Bands
            df['bb_middle'] = df['price'].rolling(window=20).mean()
            df['bb_std'] = df['price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 6. Price Rate of Change
            df['roc_7'] = df['price'].pct_change(periods=7) * 100
            df['roc_30'] = df['price'].pct_change(periods=30) * 100
            
            # 7. On-Balance Volume (OBV)
            df['daily_return'] = df['price'].pct_change()
            df['obv'] = 0
            
            # Use numpy for better performance
            price_change = df['daily_return'].values
            volume = df['volume_24h'].values
            obv = np.zeros_like(price_change)
            
            for i in range(1, len(price_change)):
                if price_change[i] > 0:
                    obv[i] = obv[i-1] + volume[i]
                elif price_change[i] < 0:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            df['obv'] = obv
            
            # 8. Ichimoku Cloud
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['price'].rolling(window=9).max()
            low_9 = df['price'].rolling(window=9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['price'].rolling(window=26).max()
            low_26 = df['price'].rolling(window=26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 shifted forward 26 periods
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted forward 26 periods
            high_52 = df['price'].rolling(window=52).max()
            low_52 = df['price'].rolling(window=52).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Current closing price time-shifted backwards 26 periods
            df['chikou_span'] = df['price'].shift(-26)
            
            # 9. Average Directional Index (ADX)
            # Calculate true range
            df['prev_close'] = df['price'].shift(1)
            df['high'] = df['price'] * 1.0005  # Create proxy high price (0.05% above close)
            df['low'] = df['price'] * 0.9995   # Create proxy low price (0.05% below close)
            
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr_14'] = df['tr'].rolling(window=14).mean()
            
            # Directional movement
            df['up_move'] = df['high'] - df['high'].shift(1)
            df['down_move'] = df['low'].shift(1) - df['low']
            
            # Positive and negative directional movement
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # Normalize with ATR
            df['plus_di_14'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr_14'])
            df['minus_di_14'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr_14'])
            
            # Calculate DX
            df['dx'] = 100 * abs(df['plus_di_14'] - df['minus_di_14']) / (df['plus_di_14'] + df['minus_di_14'])
            df['dx'] = df['dx'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate ADX
            df['adx'] = df['dx'].rolling(window=14).mean()
            
            # 10. Stochastic Oscillator
            # Use rolling windows to calculate the highest high and lowest low
            high_14 = df['high'].rolling(window=14).max()
            low_14 = df['low'].rolling(window=14).min()
            
            # Calculate %K
            df['stoch_k'] = 100 * ((df['price'] - low_14) / (high_14 - low_14))
            # Replace infinite values with 0
            df['stoch_k'] = df['stoch_k'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate %D
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # 11. Money Flow Index (MFI)
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['price']) / 3
            
            # Calculate raw money flow
            df['money_flow'] = df['typical_price'] * df['volume_24h']
            
            # Determine positive and negative money flow
            df['pmf'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
            df['nmf'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)
            
            # Calculate money flow ratio and index
            df['pmf_14'] = df['pmf'].rolling(window=14).sum()
            df['nmf_14'] = df['nmf'].rolling(window=14).sum()
            
            # Avoid division by zero
            df['mfr'] = np.where(df['nmf_14'] == 0, 100, df['pmf_14'] / df['nmf_14'])
            df['mfi'] = 100 - (100 / (1 + df['mfr']))
            df['mfi'] = df['mfi'].replace([np.inf, -np.inf], np.nan).fillna(50)
            
            # 12. Hull Moving Average (HMA)
            # Calculate weighted moving average
            def calculate_wma(data, period):
                weights = np.arange(1, period + 1)
                return data.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
            
            # Calculate HMA
            wma_half_period = calculate_wma(df['price'], int(10 / 2))
            wma_full_period = calculate_wma(df['price'], 10)
            df['hma_10'] = calculate_wma(2 * wma_half_period - wma_full_period, int(np.sqrt(10)))
            
            # 13. Keltner Channels
            df['keltner_middle'] = df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
            df['keltner_upper'] = df['keltner_middle'] + 2 * df['atr_14']
            df['keltner_lower'] = df['keltner_middle'] - 2 * df['atr_14']
            
            # 14. Parabolic SAR
            df['psar'] = 0.0
            
            # Simple implementation for parabolic SAR
            acceleration_factor = 0.02
            max_acceleration = 0.2
            uptrend = True
            
            # Initial values
            sar = df['price'].iloc[0]
            ep = df['price'].iloc[0]  # Extreme point
            af = acceleration_factor  # Acceleration factor
            
            # Calculate PSAR
            for i in range(1, len(df)):
                if uptrend:
                    # Rising trend
                    if df['price'].iloc[i] > ep:
                        ep = df['price'].iloc[i]
                        af = min(af + acceleration_factor, max_acceleration)
                    
                    sar = sar + af * (ep - sar)
                    
                    # Ensure PSAR is below the low of the prior two bars
                    sar = min(sar, df['low'].iloc[max(0, i-2):i].min())
                    
                    # Check for trend reversal
                    if df['price'].iloc[i] < sar:
                        uptrend = False
                        sar = ep
                        ep = df['price'].iloc[i]
                        af = acceleration_factor
                else:
                    # Falling trend
                    if df['price'].iloc[i] < ep:
                        ep = df['price'].iloc[i]
                        af = min(af + acceleration_factor, max_acceleration)
                    
                    sar = sar - af * (sar - ep)
                    
                    # Ensure PSAR is above the high of the prior two bars
                    sar = max(sar, df['high'].iloc[max(0, i-2):i].max())
                    
                    # Check for trend reversal
                    if df['price'].iloc[i] > sar:
                        uptrend = True
                        sar = ep
                        ep = df['price'].iloc[i]
                        af = acceleration_factor
                
                df.at[df.index[i], 'psar'] = sar
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def calculate_technical_score(self, symbol):
        """
        Calculate technical score for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Technical score (0-100)
        """
        if symbol not in self.historical_data:
            logger.warning(f"No historical data found for {symbol}")
            return 0
        
        try:
            # Get daily historical data
            df = self.historical_data[symbol]['daily'].copy()
            
            if len(df) < 30:
                logger.warning(f"Insufficient historical data for {symbol}")
                return 0
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Get latest values for scoring
            latest = df.iloc[-1]
            
            # Initialize score components
            score_components = {}
            
            # 1. Trend Score (0-20 points)
            trend_score = 0
            
            # Price above moving averages
            if latest['price'] > latest['ma_7']:
                trend_score += 3
            if latest['price'] > latest['ma_30']:
                trend_score += 5
            if latest['price'] > latest['ma_90']:
                trend_score += 6
            if latest['price'] > latest['ma_200']:
                trend_score += 6
                
            score_components['trend_score'] = trend_score
            
            # 2. Momentum Score (0-20 points)
            momentum_score = 0
            
            # RSI
            rsi = latest['rsi_14']
            if 40 <= rsi <= 60:  # Neutral
                momentum_score += 8
            elif 60 < rsi <= 70:  # Bullish but not overbought
                momentum_score += 12
            elif 30 <= rsi < 40:  # Bearish but not oversold
                momentum_score += 5
            elif rsi > 70:  # Overbought
                momentum_score += 2
            elif rsi < 30:  # Oversold (potential reversal)
                momentum_score += 10
                
            # MACD
            if latest['macd'] > latest['macd_signal']:
                momentum_score += 4
            if latest['macd'] > 0:
                momentum_score += 4
            
            # Cap momentum score at 20
            momentum_score = min(momentum_score, 20)
            score_components['momentum_score'] = momentum_score
            
            # 3. Volatility Score (0-10 points)
            volatility_score = 0
            
            # Bollinger Band width
            bb_width_sorted = sorted(df['bb_width'].dropna().tolist())
            bb_width_percentiles = {
                'p25': np.percentile(bb_width_sorted, 25),
                'p50': np.percentile(bb_width_sorted, 50),
                'p75': np.percentile(bb_width_sorted, 75)
            }
            
            if latest['bb_width'] < bb_width_percentiles['p25']:  # Low volatility
                volatility_score += 10
            elif latest['bb_width'] < bb_width_percentiles['p50']:  # Medium-low volatility
                volatility_score += 7
            elif latest['bb_width'] < bb_width_percentiles['p75']:  # Medium-high volatility
                volatility_score += 4
            else:  # High volatility
                volatility_score += 1
                
            score_components['volatility_score'] = volatility_score
            
            # 4. Support/Resistance Score (0-10 points)
            support_resistance_score = 0
            
            # Price position within Bollinger Bands
            bb_position = (latest['price'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            
            if 0.4 <= bb_position <= 0.6:  # Middle of the band
                support_resistance_score += 7
            elif 0.1 <= bb_position < 0.4:  # Near support but not oversold
                support_resistance_score += 8
            elif 0.6 < bb_position <= 0.9:  # Near resistance but not overbought
                support_resistance_score += 6
            elif bb_position < 0.1:  # At/below support (potential bounce)
                support_resistance_score += 9
            elif bb_position > 0.9:  # At/above resistance (potential drop)
                support_resistance_score += 3
                
            # Recent bounce from support or rejection from resistance
            if len(df) >= 3:
                if (df['price'].iloc[-3] < df['bb_lower'].iloc[-3] and 
                    df['price'].iloc[-1] > df['bb_lower'].iloc[-1]):
                    support_resistance_score += 1  # Recent bounce from support
                elif (df['price'].iloc[-3] > df['bb_upper'].iloc[-3] and 
                      df['price'].iloc[-1] < df['bb_upper'].iloc[-1]):
                    support_resistance_score -= 2  # Recent rejection from resistance
            
            # Cap support/resistance score
            support_resistance_score = max(0, min(support_resistance_score, 10))
            score_components['support_resistance_score'] = support_resistance_score
            

            
            # 5. Volume Score (0-10 points)
            volume_score = 0

            # Volume trend
            if 'volume_ma_7' in df.columns and not df['volume_ma_7'].isna().all():
                if latest['volume_24h'] > df['volume_ma_7'].iloc[-1]:
                    volume_score += 5
                if latest['volume_24h'] > df['volume_24h'].rolling(window=30).mean().iloc[-1]:
                    volume_score += 5
            else:
                # Atribuir uma pontuação padrão se os dados não estiverem disponíveis
                volume_score += 5
                
            score_components['volume_score'] = volume_score
            
            # 6. Ichimoku Score (0-10 points)
            ichimoku_score = 0
            
            # Check if Ichimoku components are available
            if (pd.notnull(latest['tenkan_sen']) and pd.notnull(latest['kijun_sen']) and
                pd.notnull(latest['senkou_span_a']) and pd.notnull(latest['senkou_span_b'])):
                
                # Determine the Kumo (cloud)
                kumo_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
                kumo_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
                
                # Check price position relative to Kumo
                if latest['price'] > kumo_top:  # Price above cloud (bullish)
                    ichimoku_score += 5
                elif latest['price'] < kumo_bottom:  # Price below cloud (bearish)
                    ichimoku_score += 0
                else:  # Price inside cloud (neutral)
                    ichimoku_score += 2
                
                # Check Tenkan-sen vs Kijun-sen
                if latest['tenkan_sen'] > latest['kijun_sen']:  # Tenkan above Kijun (bullish)
                    ichimoku_score += 5
                elif latest['tenkan_sen'] < latest['kijun_sen']:  # Tenkan below Kijun (bearish)
                    ichimoku_score += 0
                else:  # Tenkan equals Kijun (neutral)
                    ichimoku_score += 2
            else:
                # Default value if Ichimoku data is not available
                ichimoku_score = 5
            
            score_components['ichimoku_score'] = ichimoku_score
            
            # 7. OBV Score (0-5 points)
            obv_score = 0
            
            if pd.notnull(latest['obv']):
                # Calculate OBV trend using 20-day moving average
                df['obv_ma_20'] = df['obv'].rolling(window=20).mean()
                
                # OBV trend
                if latest['obv'] > df['obv_ma_20'].iloc[-1]:  # OBV rising (accumulation)
                    obv_score += 5
                else:  # OBV falling (distribution)
                    obv_score += 0
            else:
                # Default value if OBV is not available
                obv_score = 2
            
            score_components['obv_score'] = obv_score
            
            # 8. ADX Score (0-5 points)
            adx_score = 0
            
            if pd.notnull(latest['adx']):
                # ADX strength
                if latest['adx'] > 30:  # Strong trend
                    adx_score += 5
                elif latest['adx'] > 20:  # Moderate trend
                    adx_score += 3
                elif latest['adx'] > 15:  # Weak trend
                    adx_score += 1
                else:  # No trend
                    adx_score += 0
                    
                # Check if it's a bullish or bearish trend
                if latest['plus_di_14'] > latest['minus_di_14'] and adx_score > 0:
                    # It's a bullish trend, keep the score
                    pass
                elif latest['plus_di_14'] < latest['minus_di_14'] and adx_score > 0:
                    # It's a bearish trend, reduce the score
                    adx_score = max(0, adx_score - 2)
            else:
                # Default value if ADX is not available
                adx_score = 2
            
            score_components['adx_score'] = adx_score
            
            # 9. Parabolic SAR Score (0-5 points)
            psar_score = 0
            
            if pd.notnull(latest['psar']):
                if latest['price'] > latest['psar']:  # Price above PSAR (bullish)
                    psar_score += 5
                else:  # Price below PSAR (bearish)
                    psar_score += 0
            else:
                # Default value if PSAR is not available
                psar_score = 2
            
            score_components['psar_score'] = psar_score
            
            # 10. Moving Average Cross Score (0-5 points)
            ma_cross_score = 0
            
            # Check for recent golden cross (7 and 30 day moving averages)
            if len(df) >= 10:
                # Was there a recent cross?
                was_below = df['ma_7'].iloc[-10] < df['ma_30'].iloc[-10]
                is_above = latest['ma_7'] > latest['ma_30']
                
                if was_below and is_above:  # Recent golden cross (bullish)
                    ma_cross_score += 5
                elif not was_below and not is_above:  # Recent death cross (bearish)
                    ma_cross_score += 0
                elif is_above:  # Already in a golden cross state
                    ma_cross_score += 3
                else:  # Already in a death cross state
                    ma_cross_score += 1
            else:
                # Default if not enough data
                ma_cross_score = 2
            
            score_components['ma_cross_score'] = ma_cross_score
            
            # Combine all scores
            technical_score = (
                trend_score + 
                momentum_score + 
                volatility_score + 
                support_resistance_score + 
                volume_score +
                ichimoku_score + 
                obv_score + 
                adx_score + 
                psar_score + 
                ma_cross_score
            )
            
            # Ensure score is between 0 and 100
            technical_score = max(0, min(100, technical_score))
            
            # Save technical indicators for later use
            technical_data = {
                'symbol': symbol,
                'price': float(latest['price']),
                'ma_7': float(latest['ma_7']),
                'ma_30': float(latest['ma_30']),
                'ma_90': float(latest['ma_90']),
                'ma_200': float(latest['ma_200']),
                'rsi_14': float(latest['rsi_14']),
                'macd': float(latest['macd']),
                'macd_signal': float(latest['macd_signal']),
                'bb_upper': float(latest['bb_upper']),
                'bb_middle': float(latest['bb_middle']),
                'bb_lower': float(latest['bb_lower']),
                'bb_width': float(latest['bb_width']),
                'roc_7': float(latest['roc_7']),
                'roc_30': float(latest['roc_30']),
                'obv': float(latest['obv']),
                'tenkan_sen': float(latest['tenkan_sen']) if pd.notnull(latest['tenkan_sen']) else None,
                'kijun_sen': float(latest['kijun_sen']) if pd.notnull(latest['kijun_sen']) else None,
                'adx': float(latest['adx']) if pd.notnull(latest['adx']) else None,
                'stoch_k': float(latest['stoch_k']) if pd.notnull(latest['stoch_k']) else None,
                'stoch_d': float(latest['stoch_d']) if pd.notnull(latest['stoch_d']) else None,
                'mfi': float(latest['mfi']) if pd.notnull(latest['mfi']) else None,
                'psar': float(latest['psar']) if pd.notnull(latest['psar']) else None,
                # Add all score components
                'technical_score': technical_score,
                'score_components': score_components
            }
            
            # Save technical data
            os.makedirs('analysis/technical', exist_ok=True)
            with open(f"analysis/technical/{symbol.lower()}_technical.json", 'w') as f:
                json.dump(technical_data, f, indent=4, default=json_serializable)
            
            logger.info(f"Calculated technical score for {symbol}: {technical_score}")
            return technical_score
            
        except Exception as e:
            logger.error(f"Error calculating technical score for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def calculate_fundamental_score(self, symbol):
        """
        Calculate fundamental score for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Fundamental score (0-100)
        """
        try:
            # Get cryptocurrency data
            crypto_data = self.top_cryptos[self.top_cryptos['symbol'] == symbol]
            
            if crypto_data.empty:
                logger.warning(f"No data found for {symbol}")
                return 0
            
            crypto_data = crypto_data.iloc[0]
            
            # Initialize score components
            score_components = {}
            
            # 1. Market Cap Score (0-25 points)
            market_cap_score = 0
            market_cap = crypto_data['market_cap_usd']
            
            # Log scale for market cap (higher market cap = higher stability)
            if market_cap > 100e9:  # > $100B
                market_cap_score = 25
            elif market_cap > 50e9:  # > $50B
                market_cap_score = 22
            elif market_cap > 10e9:  # > $10B
                market_cap_score = 18
            elif market_cap > 5e9:  # > $5B
                market_cap_score = 15
            elif market_cap > 1e9:  # > $1B
                market_cap_score = 10
            elif market_cap > 500e6:  # > $500M
                market_cap_score = 7
            elif market_cap > 100e6:  # > $100M
                market_cap_score = 4
            else:
                market_cap_score = 2
                
            score_components['market_cap_score'] = market_cap_score
            
            # 2. Volume/Market Cap Ratio Score (0-20 points)
            volume_mcap_score = 0
            volume = crypto_data['volume_24h_usd']
            
            # Calculate volume/market cap ratio (higher ratio = higher liquidity)
            volume_mcap_ratio = volume / market_cap if market_cap > 0 else 0
            
            if volume_mcap_ratio > 0.30:  # Extremely high liquidity
                volume_mcap_score = 20
            elif volume_mcap_ratio > 0.20:  # Very high liquidity
                volume_mcap_score = 18
            elif volume_mcap_ratio > 0.15:  # High liquidity
                volume_mcap_score = 15
            elif volume_mcap_ratio > 0.10:  # Good liquidity
                volume_mcap_score = 12
            elif volume_mcap_ratio > 0.05:  # Moderate liquidity
                volume_mcap_score = 8
            elif volume_mcap_ratio > 0.02:  # Low liquidity
                volume_mcap_score = 5
            else:  # Very low liquidity
                volume_mcap_score = 2
                
            score_components['volume_mcap_score'] = volume_mcap_score
            
            # 3. Market Rank Score (0-15 points)
            rank_score = 0
            rank = crypto_data['cmc_rank']
            
            if rank <= 3:  # Top 3
                rank_score = 15
            elif rank <= 5:  # Top 5
                rank_score = 14
            elif rank <= 10:  # Top 10
                rank_score = 12
            elif rank <= 20:  # Top 20
                rank_score = 10
            elif rank <= 50:  # Top 50
                rank_score = 7
            elif rank <= 100:  # Top 100
                rank_score = 5
            elif rank <= 200:  # Top 200
                rank_score = 3
            else:  # Below top 200
                rank_score = 1
                
            score_components['rank_score'] = rank_score
            
            # 4. Price Stability Score (0-15 points)
            stability_score = 0
            
            # Use historical volatility as a measure of stability
            if symbol in self.historical_data:
                df = self.historical_data[symbol]['daily'].copy()
                
                if len(df) >= 30:
                    # Calculate daily returns
                    df['daily_return'] = df['price'].pct_change()
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility_30d = df['daily_return'].tail(30).std() * np.sqrt(365)  # Annualized
                    
                    # Lower volatility = higher stability score
                    if volatility_30d < 0.5:  # < 50% annual volatility
                        stability_score = 15
                    elif volatility_30d < 0.75:  # < 75% annual volatility
                        stability_score = 12
                    elif volatility_30d < 1.0:  # < 100% annual volatility
                        stability_score = 9
                    elif volatility_30d < 1.5:  # < 150% annual volatility
                        stability_score = 6
                    elif volatility_30d < 2.0:  # < 200% annual volatility
                        stability_score = 3
                    else:  # >= 200% annual volatility
                        stability_score = 1
            else:
                # Default if no historical data
                stability_score = 5
            
            score_components['stability_score'] = stability_score
            
            # 5. Growth Score (0-15 points)
            growth_score = 0
            
            # Use 30-day price change as growth indicator
            price_change_30d = crypto_data.get('percent_change_30d', 0)
            
            if price_change_30d > 100:  # > 100% growth (exceptional)
                growth_score = 15
            elif price_change_30d > 50:  # > 50% growth (excellent)
                growth_score = 13
            elif price_change_30d > 25:  # > 25% growth (very good)
                growth_score = 11
            elif price_change_30d > 10:  # > 10% growth (good)
                growth_score = 9
            elif price_change_30d > 5:  # > 5% growth (moderate)
                growth_score = 7
            elif price_change_30d > 0:  # > 0% growth (slight)
                growth_score = 5
            elif price_change_30d > -10:  # > -10% decline (slight decline)
                growth_score = 3
            elif price_change_30d > -25:  # > -25% decline (moderate decline)
                growth_score = 2
            else:  # >= -25% decline (significant decline)
                growth_score = 1
                
            score_components['growth_score'] = growth_score
            
            # 6. Market Position Score (0-10 points)
            market_position_score = 0
            
            # Compare market cap to total market
            total_market_cap = self.top_cryptos['market_cap_usd'].sum()
            market_share = (market_cap / total_market_cap) * 100 if total_market_cap > 0 else 0
            
            if market_share > 20:  # Dominant (like BTC)
                market_position_score = 10
            elif market_share > 10:  # Major (like ETH)
                market_position_score = 9
            elif market_share > 5:  # Very significant
                market_position_score = 8
            elif market_share > 2:  # Significant
                market_position_score = 7
            elif market_share > 1:  # Notable
                market_position_score = 6
            elif market_share > 0.5:  # Moderate
                market_position_score = 5
            elif market_share > 0.2:  # Small
                market_position_score = 4
            elif market_share > 0.1:  # Very small
                market_position_score = 3
            elif market_share > 0.05:  # Minimal
                market_position_score = 2
            else:  # Negligible
                market_position_score = 1
                
            score_components['market_position_score'] = market_position_score
            
            # Combine all scores
            fundamental_score = (
                market_cap_score + 
                volume_mcap_score + 
                rank_score + 
                stability_score + 
                growth_score + 
                market_position_score
            )
            
            # Ensure score is between 0 and 100
            fundamental_score = max(0, min(100, fundamental_score))
            
            # Save fundamental data
            fundamental_data = {
                'symbol': symbol,
                'market_cap_usd': float(market_cap),
                'volume_24h_usd': float(volume),
                'volume_mcap_ratio': float(volume_mcap_ratio),
                'market_share': float(market_share),
                'cmc_rank': int(rank),
                'price_change_30d': float(price_change_30d),
                'fundamental_score': fundamental_score,
                'score_components': score_components
            }
            
            # Save fundamental data
            os.makedirs('analysis/fundamental', exist_ok=True)
            with open(f"analysis/fundamental/{symbol.lower()}_fundamental.json", 'w') as f:
                json.dump(fundamental_data, f, indent=4, default=json_serializable)
            
            logger.info(f"Calculated fundamental score for {symbol}: {fundamental_score}")
            return fundamental_score
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def calculate_sentiment_score(self, symbol):
        """
        Calculate sentiment score for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Sentiment score (0-100)
        """
        try:
            # Get cryptocurrency data
            crypto_data = self.top_cryptos[self.top_cryptos['symbol'] == symbol]
            
            if crypto_data.empty:
                logger.warning(f"No data found for {symbol}")
                return 0
            
            crypto_data = crypto_data.iloc[0]
            
            # Initialize score components
            score_components = {}
            
            # 1. Price Momentum Score (0-35 points)
            momentum_score = 0
            
            # Use recent price changes as sentiment indicators
            price_change_1h = crypto_data.get('percent_change_1h', 0)
            price_change_24h = crypto_data.get('percent_change_24h', 0)
            price_change_7d = crypto_data.get('percent_change_7d', 0)
            
            # 1-hour momentum (0-10 points)
            if price_change_1h > 5:  # Strong positive
                momentum_score += 10
            elif price_change_1h > 3:  # Positive
                momentum_score += 8
            elif price_change_1h > 1:  # Slightly positive
                momentum_score += 6
            elif price_change_1h > 0:  # Minimal positive
                momentum_score += 5
            elif price_change_1h > -1:  # Minimal negative
                momentum_score += 4
            elif price_change_1h > -3:  # Slightly negative
                momentum_score += 3
            elif price_change_1h > -5:  # Negative
                momentum_score += 2
            else:  # Strong negative
                momentum_score += 1
                
            # 24-hour momentum (0-12 points)
            if price_change_24h > 15:  # Exceptional positive
                momentum_score += 12
            elif price_change_24h > 10:  # Strong positive
                momentum_score += 10
            elif price_change_24h > 5:  # Positive
                momentum_score += 8
            elif price_change_24h > 2:  # Slightly positive
                momentum_score += 6
            elif price_change_24h > 0:  # Minimal positive
                momentum_score += 5
            elif price_change_24h > -2:  # Minimal negative
                momentum_score += 4
            elif price_change_24h > -5:  # Slightly negative
                momentum_score += 3
            elif price_change_24h > -10:  # Negative
                momentum_score += 2
            else:  # Strong negative
                momentum_score += 1
                
            # 7-day momentum (0-13 points)
            if price_change_7d > 50:  # Exceptional positive
                momentum_score += 13
            elif price_change_7d > 30:  # Very strong positive
                momentum_score += 11
            elif price_change_7d > 20:  # Strong positive
                momentum_score += 10
            elif price_change_7d > 10:  # Positive
                momentum_score += 8
            elif price_change_7d > 5:  # Slightly positive
                momentum_score += 7
            elif price_change_7d > 0:  # Minimal positive
                momentum_score += 6
            elif price_change_7d > -5:  # Minimal negative
                momentum_score += 5
            elif price_change_7d > -10:  # Slightly negative
                momentum_score += 4
            elif price_change_7d > -20:  # Negative
                momentum_score += 3
            elif price_change_7d > -30:  # Strong negative
                momentum_score += 2
            else:  # Very strong negative
                momentum_score += 1
                
            score_components['momentum_score'] = momentum_score
            
            # 2. Market Sentiment Score (0-30 points)
            market_sentiment_score = 0
            
            # Use Fear & Greed Index as market sentiment indicator
            if self.fear_greed is not None:
                fear_greed_value = self.fear_greed.get('value', 50)
                
                # Contrarian approach: extreme fear = buying opportunity
                if fear_greed_value <= 20:  # Extreme Fear
                    market_sentiment_score += 25
                elif fear_greed_value <= 40:  # Fear
                    market_sentiment_score += 20
                elif fear_greed_value <= 60:  # Neutral
                    market_sentiment_score += 15
                elif fear_greed_value <= 80:  # Greed
                    market_sentiment_score += 10
                else:  # Extreme Greed
                    market_sentiment_score += 5
                    
                # Add bonus for Bitcoin and top coins during fear
                if symbol in ['BTC', 'ETH'] and fear_greed_value <= 40:
                    market_sentiment_score += 5
            else:
                # Default to neutral if no Fear & Greed data
                market_sentiment_score += 15
                
            score_components['market_sentiment_score'] = market_sentiment_score
            
            # 3. Relative Performance Score (0-25 points)
            relative_score = 0
            
            # Compare to market average performance
            if self.top_cryptos is not None:
                avg_change_24h = self.top_cryptos['percent_change_24h'].mean()
                avg_change_7d = self.top_cryptos['percent_change_7d'].mean()
                
                # 24-hour relative performance (0-10 points)
                rel_perf_24h = price_change_24h - avg_change_24h
                
                if rel_perf_24h > 15:  # Exceptional outperformance
                    relative_score += 10
                elif rel_perf_24h > 10:  # Strong outperformance
                    relative_score += 8
                elif rel_perf_24h > 5:  # Good outperformance
                    relative_score += 6
                elif rel_perf_24h > 2:  # Slight outperformance
                    relative_score += 5
                elif rel_perf_24h > -2:  # In line with market
                    relative_score += 4
                elif rel_perf_24h > -5:  # Slight underperformance
                    relative_score += 3
                elif rel_perf_24h > -10:  # Underperformance
                    relative_score += 2
                else:  # Significant underperformance
                    relative_score += 1
                    
                # 7-day relative performance (0-15 points)
                rel_perf_7d = price_change_7d - avg_change_7d
                
                if rel_perf_7d > 30:  # Exceptional outperformance
                    relative_score += 15
                elif rel_perf_7d > 20:  # Very strong outperformance
                    relative_score += 13
                elif rel_perf_7d > 15:  # Strong outperformance
                    relative_score += 11
                elif rel_perf_7d > 10:  # Good outperformance
                    relative_score += 9
                elif rel_perf_7d > 5:  # Moderate outperformance
                    relative_score += 7
                elif rel_perf_7d > 0:  # Slight outperformance
                    relative_score += 5
                elif rel_perf_7d > -5:  # In line with market
                    relative_score += 4
                elif rel_perf_7d > -10:  # Slight underperformance
                    relative_score += 3
                elif rel_perf_7d > -15:  # Moderate underperformance
                    relative_score += 2
                else:  # Significant underperformance
                    relative_score += 1
            else:
                # Default to neutral if no market data
                relative_score += 12
                
            score_components['relative_score'] = relative_score
            
            # 4. Market Trend Alignment Score (0-10 points)
            alignment_score = 0
            
            # Check if the cryptocurrency is aligned with the market trend
            if self.market_trends:
                market_trend = self.market_trends.get('market_trend', 'Neutral')
                
                # For bullish market, upward movement is good
                if market_trend in ['Bullish', 'Strong Bullish']:
                    if price_change_7d > 0:  # Moving up in bullish market
                        alignment_score += 5
                        if price_change_7d > avg_change_7d:  # Outperforming in bullish market
                            alignment_score += 5
                        else:  # Underperforming in bullish market
                            alignment_score += 2
                    else:  # Moving down in bullish market (negative)
                        alignment_score += 0
                
                # For bearish market, being more resilient is good
                elif market_trend in ['Bearish', 'Strong Bearish']:
                    if price_change_7d > 0:  # Moving up in bearish market (excellent)
                        alignment_score += 10
                    elif price_change_7d > avg_change_7d:  # Better than market in bearish market
                        alignment_score += 7
                    else:  # Worse than market in bearish market
                        alignment_score += 3
                
                # For neutral market, relative performance matters
                else:
                    if price_change_7d > avg_change_7d + 5:  # Significantly outperforming
                        alignment_score += 10
                    elif price_change_7d > avg_change_7d:  # Outperforming
                        alignment_score += 7
                    elif price_change_7d > avg_change_7d - 5:  # Slightly underperforming
                        alignment_score += 5
                    else:  # Significantly underperforming
                        alignment_score += 2
            else:
                # Default if no market trend data
                alignment_score = 5
                
            score_components['alignment_score'] = alignment_score
            
            # Combine all scores
            sentiment_score = momentum_score + market_sentiment_score + relative_score + alignment_score
            
            # Ensure score is between 0 and 100
            sentiment_score = max(0, min(100, sentiment_score))
            
            # Save sentiment data
            sentiment_data = {
                'symbol': symbol,
                'price_change_1h': float(price_change_1h),
                'price_change_24h': float(price_change_24h),
                'price_change_7d': float(price_change_7d),
                'market_avg_change_24h': float(avg_change_24h) if 'avg_change_24h' in locals() else None,
                'market_avg_change_7d': float(avg_change_7d) if 'avg_change_7d' in locals() else None,
                'relative_performance_24h': float(rel_perf_24h) if 'rel_perf_24h' in locals() else None,
                'relative_performance_7d': float(rel_perf_7d) if 'rel_perf_7d' in locals() else None,
                'fear_greed_value': self.fear_greed.get('value', 50) if self.fear_greed else 50,
                'fear_greed_classification': self.fear_greed.get('value_classification', 'Neutral') if self.fear_greed else 'Neutral',
                'market_trend': self.market_trends.get('market_trend', 'Neutral') if self.market_trends else 'Neutral',
                'sentiment_score': sentiment_score,
                'score_components': score_components
            }
            
            # Save sentiment data
            os.makedirs('analysis/sentiment', exist_ok=True)
            with open(f"analysis/sentiment/{symbol.lower()}_sentiment.json", 'w') as f:
                json.dump(sentiment_data, f, indent=4, default=json_serializable)
            
            logger.info(f"Calculated sentiment score for {symbol}: {sentiment_score}")
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def calculate_dynamic_weights(self, symbol=None):
        """
        Calculate dynamic weights based on market conditions and specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (optional)
            
        Returns:
            Dictionary of weight factors for each component
        """
        try:
            # Default weights
            weights = {
                'technical': 0.40,
                'fundamental': 0.35,
                'sentiment': 0.25
            }
            
            # Adjust based on market sentiment (Fear & Greed Index)
            if self.fear_greed:
                fear_greed_value = self.fear_greed.get('value', 50)
                
                # In extreme fear or greed, technical and sentiment matter more
                if fear_greed_value <= 25 or fear_greed_value >= 75:
                    weights = {
                        'technical': 0.45,  # Increase technical weight
                        'fundamental': 0.30,  # Decrease fundamental weight
                        'sentiment': 0.25    # Keep sentiment weight
                    }
                # In neutral market, fundamentals matter more
                elif 40 <= fear_greed_value <= 60:
                    weights = {
                        'technical': 0.35,   # Decrease technical weight
                        'fundamental': 0.45,  # Increase fundamental weight
                        'sentiment': 0.20     # Decrease sentiment weight
                    }
            
            # Adjust based on market trend
            if self.market_trends:
                market_trend = self.market_trends.get('market_trend', 'Neutral')
                
                # In strongly trending markets, technical matters more
                if market_trend in ['Strong Bullish', 'Strong Bearish']:
                    weights['technical'] = min(0.50, weights['technical'] + 0.05)
                    weights['fundamental'] = max(0.25, weights['fundamental'] - 0.05)
                    # Keep sentiment weight the same
            
            # Adjust for specific cryptocurrency if provided
            if symbol:
                # For top cryptocurrencies (BTC, ETH), fundamentals matter more
                if symbol in ['BTC', 'ETH']:
                    weights['technical'] = weights['technical'] - 0.05
                    weights['fundamental'] = weights['fundamental'] + 0.05
                
                # For very small cryptocurrencies, technicals and sentiment matter more
                elif symbol in self.top_cryptos['symbol'].values:
                    crypto_data = self.top_cryptos[self.top_cryptos['symbol'] == symbol].iloc[0]
                    rank = crypto_data['cmc_rank']
                    
                    if rank > 100:  # For smaller cryptocurrencies
                        weights['technical'] = min(0.50, weights['technical'] + 0.05)
                        weights['fundamental'] = max(0.25, weights['fundamental'] - 0.05)
            
            # Ensure weights sum to 1
            total = sum(weights.values())
            if total != 1:
                # Normalize weights
                weights = {k: v/total for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {e}")
            logger.error(traceback.format_exc())
            # Return default weights on error
            return {
                'technical': 0.40,
                'fundamental': 0.35,
                'sentiment': 0.25
            }
    
    def calculate_total_score(self, symbol):
        """
        Calculate total score for a cryptocurrency with dynamic weights.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Total score (0-100)
        """
        try:
            # Calculate component scores
            technical_score = self.calculate_technical_score(symbol)
            fundamental_score = self.calculate_fundamental_score(symbol)
            sentiment_score = self.calculate_sentiment_score(symbol)
            
            # Calculate dynamic weights for this specific cryptocurrency
            weights = self.calculate_dynamic_weights(symbol)
            
# Weighted average of component scores with dynamic weights
            total_score = (
                weights['technical'] * technical_score +
                weights['fundamental'] * fundamental_score +
                weights['sentiment'] * sentiment_score
            )
            
            # Round to 1 decimal place
            total_score = round(total_score, 1)
            
            logger.info(f"Calculated total score for {symbol}: {total_score} (T:{technical_score}, F:{fundamental_score}, S:{sentiment_score})")
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating total score for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def detect_divergences(self, symbol, lookback_period=30):
        """
        Detect divergences between price and indicators like RSI, MACD, etc.
        
        Args:
            symbol: Cryptocurrency symbol
            lookback_period: Number of days to look back for divergences
            
        Returns:
            Dictionary with detected divergences
        """
        if symbol not in self.historical_data:
            return {}
        
        try:
            df = self.historical_data[symbol]['daily'].copy()
            if len(df) < lookback_period:
                return {}
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Calculate technical indicators if not already calculated
            if 'rsi_14' not in df.columns:
                df = self.calculate_technical_indicators(df)
            
            # Get data for analysis
            df = df.tail(lookback_period)
            
            # Find price peaks and troughs
            # Use find_peaks for better peak detection
            price_series = df['price'].values
            price_highs_idx, _ = find_peaks(price_series, distance=3)
            price_lows_idx, _ = find_peaks(-price_series, distance=3)
            
            # If not enough peaks found, use argrelextrema as backup
            if len(price_highs_idx) < 2 or len(price_lows_idx) < 2:
                price_highs_idx = argrelextrema(price_series, np.greater, order=3)[0]
                price_lows_idx = argrelextrema(price_series, np.less, order=3)[0]
            
            # Get RSI for divergence analysis
            rsi_series = df['rsi_14'].values
            rsi_highs_idx, _ = find_peaks(rsi_series, distance=3)
            rsi_lows_idx, _ = find_peaks(-rsi_series, distance=3)
            
            # Get MACD for divergence analysis
            if 'macd' in df.columns:
                macd_series = df['macd'].values
                macd_highs_idx, _ = find_peaks(macd_series, distance=3)
                macd_lows_idx, _ = find_peaks(-macd_series, distance=3)
            else:
                macd_highs_idx = []
                macd_lows_idx = []
            
            # Initialize divergence results
            divergences = {
                'bearish_divergence': False,
                'bullish_divergence': False,
                'bearish_divergence_strength': 0,  # 0-10 scale
                'bullish_divergence_strength': 0,  # 0-10 scale
                'rsi_divergences': [],
                'macd_divergences': []
            }
            
            # Check for RSI Bearish Divergence (price making higher highs, RSI making lower highs)
            if len(price_highs_idx) >= 2 and len(rsi_highs_idx) >= 2:
                # Check last two price highs
                last_price_high = price_series[price_highs_idx[-1]]
                prev_price_high = price_series[price_highs_idx[-2]]
                
                # Find corresponding RSI highs (closest in time)
                rsi_high_candidates = [i for i in rsi_highs_idx if abs(i - price_highs_idx[-1]) <= 2]
                prev_rsi_high_candidates = [i for i in rsi_highs_idx if abs(i - price_highs_idx[-2]) <= 2]
                
                if rsi_high_candidates and prev_rsi_high_candidates:
                    last_rsi_high = rsi_series[rsi_high_candidates[0]]
                    prev_rsi_high = rsi_series[prev_rsi_high_candidates[0]]
                    
                    # Check for bearish divergence
                    if last_price_high > prev_price_high and last_rsi_high < prev_rsi_high:
                        divergences['bearish_divergence'] = True
                        # Calculate strength based on the difference
                        price_change_pct = (last_price_high / prev_price_high - 1) * 100
                        rsi_change_pct = (prev_rsi_high / last_rsi_high - 1) * 100
                        strength = min(10, int((price_change_pct + rsi_change_pct) / 2))
                        divergences['bearish_divergence_strength'] = max(1, strength)
                        divergences['rsi_divergences'].append('bearish')
            
            # Check for RSI Bullish Divergence (price making lower lows, RSI making higher lows)
            if len(price_lows_idx) >= 2 and len(rsi_lows_idx) >= 2:
                # Check last two price lows
                last_price_low = price_series[price_lows_idx[-1]]
                prev_price_low = price_series[price_lows_idx[-2]]
                
                # Find corresponding RSI lows (closest in time)
                rsi_low_candidates = [i for i in rsi_lows_idx if abs(i - price_lows_idx[-1]) <= 2]
                prev_rsi_low_candidates = [i for i in rsi_lows_idx if abs(i - price_lows_idx[-2]) <= 2]
                
                if rsi_low_candidates and prev_rsi_low_candidates:
                    last_rsi_low = rsi_series[rsi_low_candidates[0]]
                    prev_rsi_low = rsi_series[prev_rsi_low_candidates[0]]
                    
                    # Check for bullish divergence
                    if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
                        divergences['bullish_divergence'] = True
                        # Calculate strength based on the difference
                        price_change_pct = (prev_price_low / last_price_low - 1) * 100
                        rsi_change_pct = (last_rsi_low / prev_rsi_low - 1) * 100
                        strength = min(10, int((price_change_pct + rsi_change_pct) / 2))
                        divergences['bullish_divergence_strength'] = max(1, strength)
                        divergences['rsi_divergences'].append('bullish')
            
            # Check for MACD Divergence if MACD data is available
            if 'macd' in df.columns and len(macd_highs_idx) >= 2 and len(macd_lows_idx) >= 2:
                # Similar logic for MACD divergence detection
                # MACD Bearish Divergence
                if len(price_highs_idx) >= 2 and len(macd_highs_idx) >= 2:
                    last_price_high = price_series[price_highs_idx[-1]]
                    prev_price_high = price_series[price_highs_idx[-2]]
                    
                    macd_high_candidates = [i for i in macd_highs_idx if abs(i - price_highs_idx[-1]) <= 2]
                    prev_macd_high_candidates = [i for i in macd_highs_idx if abs(i - price_highs_idx[-2]) <= 2]
                    
                    if macd_high_candidates and prev_macd_high_candidates:
                        last_macd_high = macd_series[macd_high_candidates[0]]
                        prev_macd_high = macd_series[prev_macd_high_candidates[0]]
                        
                        if last_price_high > prev_price_high and last_macd_high < prev_macd_high:
                            divergences['bearish_divergence'] = True
                            divergences['macd_divergences'].append('bearish')
                            # Increase strength if both RSI and MACD show divergence
                            if 'bearish' in divergences['rsi_divergences']:
                                divergences['bearish_divergence_strength'] = min(10, divergences['bearish_divergence_strength'] + 2)
                
                # MACD Bullish Divergence
                if len(price_lows_idx) >= 2 and len(macd_lows_idx) >= 2:
                    last_price_low = price_series[price_lows_idx[-1]]
                    prev_price_low = price_series[price_lows_idx[-2]]
                    
                    macd_low_candidates = [i for i in macd_lows_idx if abs(i - price_lows_idx[-1]) <= 2]
                    prev_macd_low_candidates = [i for i in macd_lows_idx if abs(i - price_lows_idx[-2]) <= 2]
                    
                    if macd_low_candidates and prev_macd_low_candidates:
                        last_macd_low = macd_series[macd_low_candidates[0]]
                        prev_macd_low = macd_series[prev_macd_low_candidates[0]]
                        
                        if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
                            divergences['bullish_divergence'] = True
                            divergences['macd_divergences'].append('bullish')
                            # Increase strength if both RSI and MACD show divergence
                            if 'bullish' in divergences['rsi_divergences']:
                                divergences['bullish_divergence_strength'] = min(10, divergences['bullish_divergence_strength'] + 2)
            
            # Save divergence data in patterns directory
            os.makedirs('analysis/patterns', exist_ok=True)
            with open(f"analysis/patterns/{symbol.lower()}_divergences.json", 'w') as f:
                json.dump(divergences, f, indent=4, default=json_serializable)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting divergences for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {
                'bearish_divergence': False,
                'bullish_divergence': False,
                'bearish_divergence_strength': 0,
                'bullish_divergence_strength': 0,
                'rsi_divergences': [],
                'macd_divergences': []
            }
    
    def detect_chart_patterns(self, symbol, lookback_period=60):
        """
        Detect common chart patterns like head and shoulders, triangles, etc.
        
        Args:
            symbol: Cryptocurrency symbol
            lookback_period: Number of days to look back for patterns
            
        Returns:
            Dictionary with detected patterns
        """
        if symbol not in self.historical_data:
            return {}
        
        try:
            df = self.historical_data[symbol]['daily'].copy()
            if len(df) < lookback_period:
                return {}
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Get data for analysis
            df = df.tail(lookback_period)
            
            # Get price series
            price_series = df['price'].values
            
            # Initialize patterns dictionary
            patterns = {
                'head_and_shoulders': False,
                'inverse_head_and_shoulders': False,
                'double_top': False,
                'double_bottom': False,
                'ascending_triangle': False,
                'descending_triangle': False,
                'symmetric_triangle': False,
                'bullish_flag': False,
                'bearish_flag': False,
                'pattern_strength': 0  # 0-10 scale
            }
            
            # 1. Detect Head and Shoulders pattern
            # Find peaks (local maxima)
            peaks, _ = find_peaks(price_series, distance=5)
            
            if len(peaks) >= 3:
                # Sort peaks by height
                peak_heights = [(i, price_series[i]) for i in peaks]
                peak_heights.sort(key=lambda x: x[1], reverse=True)
                
                # Get highest peak (potential head)
                head_idx = peak_heights[0][0]
                
                # Get potential left and right shoulders
                left_candidates = [i for i in peaks if i < head_idx]
                right_candidates = [i for i in peaks if i > head_idx]
                
                if left_candidates and right_candidates:
                    # Select closest peaks as shoulders
                    left_shoulder = max(left_candidates)
                    right_shoulder = min(right_candidates)
                    
                    # Check if left and right shoulders are at similar heights
                    left_height = price_series[left_shoulder]
                    right_height = price_series[right_shoulder]
                    head_height = price_series[head_idx]
                    
                    height_diff_pct = abs(left_height - right_height) / ((left_height + right_height) / 2)
                    
                    # Head should be significantly higher than shoulders
                    head_prominence = min(
                        (head_height / left_height - 1) * 100,
                        (head_height / right_height - 1) * 100
                    )
                    
                    if height_diff_pct < 0.15 and head_prominence > 5:
                        patterns['head_and_shoulders'] = True
                        # Calculate pattern strength
                        neckline = min(left_height, right_height)
                        breakdown_potential = (price_series[-1] - neckline) / neckline * 100
                        if breakdown_potential < 0:  # Price already below neckline
                            strength = min(10, int(abs(breakdown_potential)))
                        else:  # Price still above neckline
                            strength = min(5, max(1, 5 - int(breakdown_potential)))
                        patterns['pattern_strength'] = max(patterns['pattern_strength'], strength)
            
            # 2. Detect Inverse Head and Shoulders pattern
            # Find troughs (local minima)
            troughs, _ = find_peaks(-price_series, distance=5)
            
            if len(troughs) >= 3:
                # Sort troughs by depth (lowest first)
                trough_depths = [(i, price_series[i]) for i in troughs]
                trough_depths.sort(key=lambda x: x[1])
                
                # Get lowest trough (potential head)
                inv_head_idx = trough_depths[0][0]
                
                # Get potential left and right shoulders
                inv_left_candidates = [i for i in troughs if i < inv_head_idx]
                inv_right_candidates = [i for i in troughs if i > inv_head_idx]
                
                if inv_left_candidates and inv_right_candidates:
                    # Select closest troughs as shoulders
                    inv_left_shoulder = max(inv_left_candidates)
                    inv_right_shoulder = min(inv_right_candidates)
                    
                    # Check if left and right shoulders are at similar depths
                    inv_left_depth = price_series[inv_left_shoulder]
                    inv_right_depth = price_series[inv_right_shoulder]
                    inv_head_depth = price_series[inv_head_idx]
                    
                    inv_depth_diff_pct = abs(inv_left_depth - inv_right_depth) / ((inv_left_depth + inv_right_depth) / 2)
                    
                    # Head should be significantly lower than shoulders
                    inv_head_prominence = min(
                        (inv_left_depth / inv_head_depth - 1) * 100,
                        (inv_right_depth / inv_head_depth - 1) * 100
                    )
                    
                    if inv_depth_diff_pct < 0.15 and inv_head_prominence > 5:
                        patterns['inverse_head_and_shoulders'] = True
                        # Calculate pattern strength
                        neckline = max(inv_left_depth, inv_right_depth)
                        breakout_potential = (neckline - price_series[-1]) / neckline * 100
                        if breakout_potential < 0:  # Price already above neckline
                            strength = min(10, int(abs(breakout_potential)))
                        else:  # Price still below neckline
                            strength = min(5, max(1, 5 - int(breakout_potential)))
                        patterns['pattern_strength'] = max(patterns['pattern_strength'], strength)
            
            # 3. Detect Double Top pattern
            if len(peaks) >= 2:
                # Get two highest peaks
                peak_heights = [(i, price_series[i]) for i in peaks]
                peak_heights.sort(key=lambda x: x[1], reverse=True)
                
                if len(peak_heights) >= 2:
                    top1_idx, top1_height = peak_heights[0]
                    top2_idx, top2_height = peak_heights[1]
                    
                    # Ensure proper order (left to right)
                    if top1_idx > top2_idx:
                        top1_idx, top2_idx = top2_idx, top1_idx
                        top1_height, top2_height = top2_height, top1_height
                    
                    # Check if peaks are at similar heights
                    height_diff_pct = abs(top1_height - top2_height) / ((top1_height + top2_height) / 2)
                    
                    # Find trough between peaks
                    between_idxs = [i for i in range(top1_idx + 1, top2_idx)]
                    if between_idxs:
                        trough_idx = min(between_idxs, key=lambda i: price_series[i])
                        trough_height = price_series[trough_idx]
                        
                        # Calculate pattern parameters
                        height_diff = top1_height - trough_height
                        time_diff = top2_idx - top1_idx
                        
                        if height_diff_pct < 0.1 and height_diff > 0 and time_diff > 5:
                            patterns['double_top'] = True
                            # Calculate pattern strength
                            neckline = trough_height
                            breakdown_potential = (price_series[-1] - neckline) / neckline * 100
                            if breakdown_potential < 0:  # Price already below neckline
                                strength = min(10, int(abs(breakdown_potential)))
                            else:  # Price still above neckline
                                strength = min(5, max(1, 5 - int(breakdown_potential)))
                            patterns['pattern_strength'] = max(patterns['pattern_strength'], strength)
            
            # 4. Detect Double Bottom pattern
            if len(troughs) >= 2:
                # Get two lowest troughs
                trough_depths = [(i, price_series[i]) for i in troughs]
                trough_depths.sort(key=lambda x: x[1])
                
                if len(trough_depths) >= 2:
                    bottom1_idx, bottom1_depth = trough_depths[0]
                    bottom2_idx, bottom2_depth = trough_depths[1]
                    
                    # Ensure proper order (left to right)
                    if bottom1_idx > bottom2_idx:
                        bottom1_idx, bottom2_idx = bottom2_idx, bottom1_idx
                        bottom1_depth, bottom2_depth = bottom2_depth, bottom1_depth
                    
                    # Check if troughs are at similar depths
                    depth_diff_pct = abs(bottom1_depth - bottom2_depth) / ((bottom1_depth + bottom2_depth) / 2)
                    
                    # Find peak between troughs
                    between_idxs = [i for i in range(bottom1_idx + 1, bottom2_idx)]
                    if between_idxs:
                        peak_idx = max(between_idxs, key=lambda i: price_series[i])
                        peak_height = price_series[peak_idx]
                        
                        # Calculate pattern parameters
                        height_diff = peak_height - bottom1_depth
                        time_diff = bottom2_idx - bottom1_idx
                        
                        if depth_diff_pct < 0.1 and height_diff > 0 and time_diff > 5:
                            patterns['double_bottom'] = True
                            # Calculate pattern strength
                            neckline = peak_height
                            breakout_potential = (neckline - price_series[-1]) / neckline * 100
                            if breakout_potential < 0:  # Price already above neckline
                                strength = min(10, int(abs(breakout_potential)))
                            else:  # Price still below neckline
                                strength = min(5, max(1, 5 - int(breakout_potential)))
                            patterns['pattern_strength'] = max(patterns['pattern_strength'], strength)
            
            # 5. Detect Triangle patterns
            if len(peaks) >= 3 and len(troughs) >= 3:
                # Get last few highs and lows for trendline analysis
                recent_highs = [(i, price_series[i]) for i in peaks if i >= len(price_series) - 30]
                recent_lows = [(i, price_series[i]) for i in troughs if i >= len(price_series) - 30]
                
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    # Calculate slope of high trendline
                    high_x = [i for i, _ in recent_highs]
                    high_y = [p for _, p in recent_highs]
                    high_slope, high_intercept = np.polyfit(high_x, high_y, 1)
                    
                    # Calculate slope of low trendline
                    low_x = [i for i, _ in recent_lows]
                    low_y = [p for _, p in recent_lows]
                    low_slope, low_intercept = np.polyfit(low_x, low_y, 1)
                    
                    # Determine triangle type
                    if abs(high_slope) < 0.01 and low_slope > 0.01:
                        patterns['ascending_triangle'] = True
                        patterns['pattern_strength'] = max(patterns['pattern_strength'], 7)
                    elif high_slope < -0.01 and abs(low_slope) < 0.01:
                        patterns['descending_triangle'] = True
                        patterns['pattern_strength'] = max(patterns['pattern_strength'], 7)
                    elif high_slope < -0.01 and low_slope > 0.01:
                        patterns['symmetric_triangle'] = True
                        patterns['pattern_strength'] = max(patterns['pattern_strength'], 6)
            
            # 6. Detect Flag patterns
            # Requires a strong preceding move (pole)
            if len(price_series) >= 20:
                # Check for bullish flag (consolidation after strong upward move)
                pole_start_idx = max(0, len(price_series) - 20)
                pole_end_idx = max(0, len(price_series) - 10)
                
                pole_price_change = (price_series[pole_end_idx] / price_series[pole_start_idx] - 1) * 100
                recent_volatility = np.std(price_series[-10:]) / np.mean(price_series[-10:]) * 100
                
                if pole_price_change > 20 and recent_volatility < 5:
                    patterns['bullish_flag'] = True
                    patterns['pattern_strength'] = max(patterns['pattern_strength'], int(min(10, pole_price_change / 5)))
                
                # Check for bearish flag (consolidation after strong downward move)
                if pole_price_change < -20 and recent_volatility < 5:
                    patterns['bearish_flag'] = True
                    patterns['pattern_strength'] = max(patterns['pattern_strength'], int(min(10, abs(pole_price_change) / 5)))
            
            # Save pattern data
            os.makedirs('analysis/patterns', exist_ok=True)
            with open(f"analysis/patterns/{symbol.lower()}_patterns.json", 'w') as f:
                json.dump(patterns, f, indent=4, default=json_serializable)
            
            # Log detected patterns
            detected_patterns = [p for p, detected in patterns.items() if detected and p != 'pattern_strength']
            if detected_patterns:
                logger.info(f"Detected patterns for {symbol}: {', '.join(detected_patterns)}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {
                'head_and_shoulders': False,
                'inverse_head_and_shoulders': False,
                'double_top': False,
                'double_bottom': False,
                'ascending_triangle': False,
                'descending_triangle': False,
                'symmetric_triangle': False,
                'bullish_flag': False,
                'bearish_flag': False,
                'pattern_strength': 0
            }
    
    def get_risk_profile(self, symbol):
        """
        Determine risk profile for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Risk profile (Conservative, Moderate, Aggressive, Speculative)
        """
        try:
            # Get cryptocurrency data
            crypto_data = self.top_cryptos[self.top_cryptos['symbol'] == symbol]
            
            if crypto_data.empty:
                return "Unknown"
            
            crypto_data = crypto_data.iloc[0]
            
            # Get market cap and rank
            market_cap = crypto_data['market_cap_usd']
            rank = crypto_data['cmc_rank']
            
            # Get volatility if available
            volatility = None
            if symbol in self.historical_data:
                df = self.historical_data[symbol]['daily'].copy()
                if len(df) >= 30:
                    df['daily_return'] = df['price'].pct_change()
                    volatility = df['daily_return'].tail(30).std() * np.sqrt(365)  # Annualized
            
            # Get trading volume ratio
            volume = crypto_data['volume_24h_usd']
            volume_to_mcap = volume / market_cap if market_cap > 0 else 0
            
            # Determine risk profile based on multiple factors
            if (rank <= 5 and market_cap > 50e9 and 
                (volatility is None or volatility < 0.75)):
                return "Conservative"
            elif (rank <= 20 and market_cap > 5e9 and 
                  (volatility is None or volatility < 1.2)):
                return "Moderate"
            elif (rank <= 100 and market_cap > 500e6 and 
                  (volatility is None or volatility < 2.0)):
                return "Aggressive"
            else:
                return "Speculative"
                
        except Exception as e:
            logger.error(f"Error determining risk profile for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return "Unknown"
    
    def get_market_opportunity(self, symbol):
        """
        Evaluate market opportunity for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Opportunity rating (Strong, Good, Moderate, Limited, Poor)
        """
        try:
            # Get cryptocurrency scores
            technical_score = self.calculate_technical_score(symbol)
            fundamental_score = self.calculate_fundamental_score(symbol)
            sentiment_score = self.calculate_sentiment_score(symbol)
            total_score = self.calculate_total_score(symbol)
            
            # Get patterns and divergences
            patterns = self.detect_chart_patterns(symbol)
            divergences = self.detect_divergences(symbol)
            
            # Calculate opportunity score (0-100)
            opportunity_score = total_score
            
            # Adjust for patterns
            pattern_strength = patterns.get('pattern_strength', 0)
            if patterns.get('bullish_flag', False) or patterns.get('inverse_head_and_shoulders', False) or patterns.get('double_bottom', False):
                opportunity_score += pattern_strength * 0.5
            elif patterns.get('symmetric_triangle', False) or patterns.get('ascending_triangle', False):
                opportunity_score += pattern_strength * 0.3
            elif patterns.get('bearish_flag', False) or patterns.get('head_and_shoulders', False) or patterns.get('double_top', False) or patterns.get('descending_triangle', False):
                opportunity_score -= pattern_strength * 0.5
            
            # Adjust for divergences
            bullish_strength = divergences.get('bullish_divergence_strength', 0)
            bearish_strength = divergences.get('bearish_divergence_strength', 0)
            opportunity_score += bullish_strength * 0.7
            opportunity_score -= bearish_strength * 0.7
            
            # Ensure score is within 0-100 range
            opportunity_score = max(0, min(100, opportunity_score))
            
            # Map score to opportunity rating
            if opportunity_score >= 80:
                rating = "Strong"
            elif opportunity_score >= 65:
                rating = "Good"
            elif opportunity_score >= 50:
                rating = "Moderate"
            elif opportunity_score >= 35:
                rating = "Limited"
            else:
                rating = "Poor"
            
            return rating
            
        except Exception as e:
            logger.error(f"Error evaluating market opportunity for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return "Unknown"
    
    def get_investment_recommendation(self, symbol, total_score=None):
        """
        Generate investment recommendation based on score.
        
        Args:
            symbol: Cryptocurrency symbol
            total_score: Total score (0-100), calculated if not provided
            
        Returns:
            Recommendation (Strong Buy, Buy, Hold, Sell, Strong Sell)
        """
        try:
            if total_score is None:
                total_score = self.calculate_total_score(symbol)
            
            # Get market opportunity
            opportunity = self.get_market_opportunity(symbol)
            
            # Get patterns and divergences
            patterns = self.detect_chart_patterns(symbol)
            divergences = self.detect_divergences(symbol)
            
            # Base recommendation on total score
            base_recommendation = ""
            if total_score >= 80:
                base_recommendation = "Strong Buy"
            elif total_score >= 65:
                base_recommendation = "Buy"
            elif total_score >= 45:
                base_recommendation = "Hold"
            elif total_score >= 30:
                base_recommendation = "Sell"
            else:
                base_recommendation = "Strong Sell"
            
            # Adjust for strong patterns and divergences
            final_recommendation = base_recommendation
            
            # Bullish patterns can upgrade a Hold to Buy
            if (base_recommendation == "Hold" and 
                (patterns.get('bullish_flag', False) or 
                 patterns.get('inverse_head_and_shoulders', False) or 
                 patterns.get('double_bottom', False) or 
                 divergences.get('bullish_divergence', False)) and
                patterns.get('pattern_strength', 0) >= 7):
                final_recommendation = "Buy"
            
            # Bearish patterns can downgrade a Hold to Sell
            elif (base_recommendation == "Hold" and 
                  (patterns.get('bearish_flag', False) or 
                   patterns.get('head_and_shoulders', False) or 
                   patterns.get('double_top', False) or 
                   divergences.get('bearish_divergence', False)) and
                  patterns.get('pattern_strength', 0) >= 7):
                final_recommendation = "Sell"
            
            # Extremely strong bullish patterns can upgrade a Buy to Strong Buy
            elif (base_recommendation == "Buy" and 
                 ((patterns.get('bullish_flag', False) or 
                   patterns.get('inverse_head_and_shoulders', False) or 
                   patterns.get('double_bottom', False)) and
                  patterns.get('pattern_strength', 0) >= 9)):
                final_recommendation = "Strong Buy"
            
            # Extremely strong bearish patterns can downgrade a Sell to Strong Sell
            elif (base_recommendation == "Sell" and 
                 ((patterns.get('bearish_flag', False) or 
                   patterns.get('head_and_shoulders', False) or 
                   patterns.get('double_top', False)) and
                  patterns.get('pattern_strength', 0) >= 9)):
                final_recommendation = "Strong Sell"
            
            return final_recommendation
            
        except Exception as e:
            logger.error(f"Error generating investment recommendation for {symbol}: {e}")
            logger.error(traceback.format_exc())
            # Default to Hold on error
            return "Hold"
    
    def score_all_cryptocurrencies(self):
        """
        Score and rank all cryptocurrencies.
        
        Returns:
            DataFrame with cryptocurrency scores and rankings
        """
        if self.top_cryptos is None or self.top_cryptos.empty:
            logger.warning("No cryptocurrency data available")
            return pd.DataFrame()
        
        logger.info("Scoring cryptocurrencies...")
        
        # Create results list
        results = []
        
        # Score each cryptocurrency
        for _, row in self.top_cryptos.iterrows():
            symbol = row['symbol']
            name = row['name']
            
            logger.info(f"Scoring {name} ({symbol})...")
            
            try:
                # Calculate scores
                technical_score = self.calculate_technical_score(symbol)
                fundamental_score = self.calculate_fundamental_score(symbol)
                sentiment_score = self.calculate_sentiment_score(symbol)
                total_score = self.calculate_total_score(symbol)
                
                # Detect patterns and divergences
                patterns = self.detect_chart_patterns(symbol)
                divergences = self.detect_divergences(symbol)
                
                # Get risk profile, opportunity, and recommendation
                risk_profile = self.get_risk_profile(symbol)
                market_opportunity = self.get_market_opportunity(symbol)
                recommendation = self.get_investment_recommendation(symbol, total_score)
                
                # Get dynamic weights
                weights = self.calculate_dynamic_weights(symbol)
                
                # Add to results
                result = {
                    'symbol': symbol,
                    'name': name,
                    'price_usd': row['price_usd'],
                    'market_cap_usd': row['market_cap_usd'],
                    'volume_24h_usd': row['volume_24h_usd'],
                    'percent_change_24h': row.get('percent_change_24h', 0),
                    'percent_change_7d': row.get('percent_change_7d', 0),
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'sentiment_score': sentiment_score,
                    'total_score': total_score,
                    'technical_weight': weights['technical'],
                    'fundamental_weight': weights['fundamental'],
                    'sentiment_weight': weights['sentiment'],
                    'risk_profile': risk_profile,
                    'market_opportunity': market_opportunity,
                    'recommendation': recommendation,
                    'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Add pattern detection results
                for pattern_name, detected in patterns.items():
                    if pattern_name != 'pattern_strength':
                        result[f'has_{pattern_name}'] = detected
                result['pattern_strength'] = patterns.get('pattern_strength', 0)
                
                # Add divergence detection results
                result['has_bullish_divergence'] = divergences.get('bullish_divergence', False)
                result['has_bearish_divergence'] = divergences.get('bearish_divergence', False)
                result['bullish_divergence_strength'] = divergences.get('bullish_divergence_strength', 0)
                result['bearish_divergence_strength'] = divergences.get('bearish_divergence_strength', 0)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            logger.warning("No cryptocurrencies scored successfully")
            return df
        
        # Sort by total score (descending)
        df = df.sort_values('total_score', ascending=False).reset_index(drop=True)
        
        # Add rank
        df['rank'] = df.index + 1
        
        # Save results
        df.to_csv('analysis/crypto_rankings.csv', index=False)
        
        # Save as JSON for the dashboard
        with open('analysis/crypto_rankings.json', 'w') as f:
            json.dump(df.to_dict(orient='records'), f, indent=4, default=json_serializable)
        
        logger.info(f"Cryptocurrency scoring completed! Scored {len(df)} cryptocurrencies.")
        
        # Store scores and rankings (make them point to the same object)
        self.scores = df
        self.rankings = df  # Adicionada para compatibilidade
        
        return df
    
    def generate_summary_visualizations(self):
        """Generate summary visualizations of cryptocurrency rankings."""
        if self.scores is None or self.scores.empty:
            logger.warning("No scoring data available")
            return
        
        logger.info("Generating summary visualizations...")
        
        # Create visualizations directory
        os.makedirs('analysis/visualizations', exist_ok=True)
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Top 10 Cryptocurrencies by Total Score
            plt.figure(figsize=(12, 8))
            top10 = self.scores.head(10)
            
            # Create horizontal bar chart
            bars = plt.barh(top10['symbol'], top10['total_score'], color='skyblue')
            
            # Add score labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                         f"{width:.1f}", ha='left', va='center')
            
            # Add recommendation colors
            for i, rec in enumerate(top10['recommendation']):
                color = 'green' if rec in ['Strong Buy', 'Buy'] else 'orange' if rec == 'Hold' else 'red'
                plt.text(5, i, f" {rec}", color=color, va='center', fontweight='bold')
            
            plt.xlabel('Total Score (0-100)')
            plt.title('Top 10 Cryptocurrencies by Total Score')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/top10_by_score.png', dpi=300)
            plt.close()
            
            # 2. Score Components Comparison
            plt.figure(figsize=(14, 8))
            
            # Get top 10 cryptocurrencies
            top10 = self.scores.head(10)
            
            # Set width of bars
            barWidth = 0.25
            
            # Set positions of bars on X axis
            r1 = np.arange(len(top10))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            
            # Create bars
            plt.bar(r1, top10['technical_score'], width=barWidth, label='Technical', color='skyblue')
            plt.bar(r2, top10['fundamental_score'], width=barWidth, label='Fundamental', color='lightgreen')
            plt.bar(r3, top10['sentiment_score'], width=barWidth, label='Sentiment', color='salmon')
            
            # Add labels and title
            plt.xlabel('Cryptocurrency')
            plt.ylabel('Score (0-100)')
            plt.title('Score Components for Top 10 Cryptocurrencies')
            plt.xticks([r + barWidth for r in range(len(top10))], top10['symbol'])
            plt.legend()
            plt.tight_layout()
            plt.savefig('analysis/visualizations/score_components.png', dpi=300)
            plt.close()
            
            # 3. Risk Profile Distribution
            plt.figure(figsize=(10, 6))
            
            # Count cryptocurrencies by risk profile
            risk_counts = self.scores['risk_profile'].value_counts()
            
            # Create pie chart
            colors = ['green', 'orange', 'red', 'purple', 'gray']
            plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
                    startangle=90, colors=colors)
            plt.axis('equal')
            plt.title('Cryptocurrency Distribution by Risk Profile')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/risk_profile_distribution.png', dpi=300)
            plt.close()
            
            # 4. Recommendation Distribution
            plt.figure(figsize=(10, 6))
            
            # Count cryptocurrencies by recommendation
            rec_counts = self.scores['recommendation'].value_counts()
            
            # Create pie chart
            colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
            plt.pie(rec_counts, labels=rec_counts.index, autopct='%1.1f%%', 
                    startangle=90, colors=colors)
            plt.axis('equal')
            plt.title('Cryptocurrency Distribution by Recommendation')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/recommendation_distribution.png', dpi=300)
            plt.close()
            
            # 5. Market Opportunity Distribution
            plt.figure(figsize=(10, 6))
            
            # Count cryptocurrencies by market opportunity
            opp_counts = self.scores['market_opportunity'].value_counts()
            
            # Create pie chart
            colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
            plt.pie(opp_counts, labels=opp_counts.index, autopct='%1.1f%%', 
                    startangle=90, colors=colors)
            plt.axis('equal')
            plt.title('Cryptocurrency Distribution by Market Opportunity')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/market_opportunity_distribution.png', dpi=300)
            plt.close()
            
            # 6. Score vs. Market Cap
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            scatter = plt.scatter(
                self.scores['market_cap_usd'], 
                self.scores['total_score'],
                c=self.scores['total_score'],
                cmap='viridis',
                alpha=0.7,
                s=100
            )
            
            # Add labels for each point
            for i, symbol in enumerate(self.scores['symbol']):
                plt.annotate(
                    symbol,
                    (self.scores['market_cap_usd'].iloc[i], self.scores['total_score'].iloc[i]),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
            
            # Add colorbar
            plt.colorbar(scatter, label='Total Score')
            
            # Set log scale for market cap
            plt.xscale('log')
            
            # Add labels and title
            plt.xlabel('Market Cap (USD, log scale)')
            plt.ylabel('Total Score (0-100)')
            plt.title('Cryptocurrency Score vs. Market Cap')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/score_vs_market_cap.png', dpi=300)
            plt.close()
            
            # 7. Correlation Heatmap
            plt.figure(figsize=(12, 10))
            
            # Select numeric columns
            numeric_cols = [
                'price_usd', 'market_cap_usd', 'volume_24h_usd',
                'percent_change_24h', 'percent_change_7d',
                'technical_score', 'fundamental_score', 'sentiment_score', 'total_score'
            ]
            
            # Calculate correlation matrix
            corr_matrix = self.scores[numeric_cols].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title('Correlation Matrix of Cryptocurrency Metrics')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/correlation_heatmap.png', dpi=300)
            plt.close()
            
            # 8. Pattern Distribution
            plt.figure(figsize=(14, 8))
            
            # Get pattern columns
            pattern_cols = [col for col in self.scores.columns if col.startswith('has_') and col != 'has_bullish_divergence' and col != 'has_bearish_divergence']
            
            # Count patterns
            pattern_counts = {}
            for col in pattern_cols:
                pattern_name = col.replace('has_', '').replace('_', ' ').title()
                pattern_counts[pattern_name] = self.scores[col].sum()
            
            # Create bar chart
            plt.bar(pattern_counts.keys(), pattern_counts.values(), color='skyblue')
            plt.ylabel('Number of Cryptocurrencies')
            plt.title('Chart Patterns Detected Across Cryptocurrencies')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/detected_patterns.png', dpi=300)
            plt.close()
            
            # 9. Divergence Distribution
            plt.figure(figsize=(10, 6))
            
            # Count divergences
            divergence_counts = {
                'Bullish Divergence': self.scores['has_bullish_divergence'].sum(),
                'Bearish Divergence': self.scores['has_bearish_divergence'].sum(),
                'No Divergence': len(self.scores) - self.scores['has_bullish_divergence'].sum() - self.scores['has_bearish_divergence'].sum()
            }
            
            # Create pie chart
            colors = ['green', 'red', 'gray']
            plt.pie(divergence_counts.values(), labels=divergence_counts.keys(), autopct='%1.1f%%', 
                    startangle=90, colors=colors)
            plt.axis('equal')
            plt.title('Divergence Detection Distribution')
            plt.tight_layout()
            plt.savefig('analysis/visualizations/divergence_distribution.png', dpi=300)
            plt.close()
            
            logger.info("Summary visualizations generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating summary visualizations: {e}")
            logger.error(traceback.format_exc())
    
    def generate_detailed_analysis(self, symbol):
        """
        Generate detailed analysis for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with detailed analysis
        """
        if self.scores is None or self.scores.empty:
            logger.warning("No scoring data available")
            return {}
        
        try:
            # Get cryptocurrency data
            crypto_data = self.scores[self.scores['symbol'] == symbol]
            
            if crypto_data.empty:
                logger.warning(f"No data found for {symbol}")
                return {}
            
            crypto_data = crypto_data.iloc[0].to_dict()
            
            # Load technical data
            technical_file = f"analysis/technical/{symbol.lower()}_technical.json"
            if os.path.exists(technical_file):
                with open(technical_file, 'r') as f:
                    technical_data = json.load(f)
            else:
                technical_data = {}
                
            # Load fundamental data
            fundamental_file = f"analysis/fundamental/{symbol.lower()}_fundamental.json"
            if os.path.exists(fundamental_file):
                with open(fundamental_file, 'r') as f:
                    fundamental_data = json.load(f)
            else:
                fundamental_data = {}
                
            # Load sentiment data
            sentiment_file = f"analysis/sentiment/{symbol.lower()}_sentiment.json"
            if os.path.exists(sentiment_file):
                with open(sentiment_file, 'r') as f:
                    sentiment_data = json.load(f)
            else:
                sentiment_data = {}
                
            # Load pattern data
            pattern_file = f"analysis/patterns/{symbol.lower()}_patterns.json"
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    pattern_data = json.load(f)
            else:
                pattern_data = {}
                
            # Load divergence data
            divergence_file = f"analysis/patterns/{symbol.lower()}_divergences.json"
            if os.path.exists(divergence_file):
                with open(divergence_file, 'r') as f:
                    divergence_data = json.load(f)
            else:
                divergence_data = {}
                
            # Get historical data
            historical_data = None
            if symbol in self.historical_data:
                historical_data = self.historical_data[symbol]['daily'].to_dict(orient='records')
            
            # Generate investment thesis
            investment_thesis = self._generate_investment_thesis(symbol, crypto_data)
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(symbol, crypto_data)
            
            # Combine all data
            detailed_analysis = {
                'symbol': symbol,
                'name': crypto_data.get('name', ''),
                'price_usd': crypto_data.get('price_usd', 0),
                'market_cap_usd': crypto_data.get('market_cap_usd', 0),
                'volume_24h_usd': crypto_data.get('volume_24h_usd', 0),
                'percent_change_24h': crypto_data.get('percent_change_24h', 0),
                'percent_change_7d': crypto_data.get('percent_change_7d', 0),
                'technical_score': crypto_data.get('technical_score', 0),
                'fundamental_score': crypto_data.get('fundamental_score', 0),
                'sentiment_score': crypto_data.get('sentiment_score', 0),
                'total_score': crypto_data.get('total_score', 0),
                'technical_weight': crypto_data.get('technical_weight', 0),
                'fundamental_weight': crypto_data.get('fundamental_weight', 0),
                'sentiment_weight': crypto_data.get('sentiment_weight', 0),
                'risk_profile': crypto_data.get('risk_profile', 'Unknown'),
                'market_opportunity': crypto_data.get('market_opportunity', 'Unknown'),
                'recommendation': crypto_data.get('recommendation', 'Unknown'),
                'rank': crypto_data.get('rank', 0),
                'technical_data': technical_data,
                'fundamental_data': fundamental_data,
                'sentiment_data': sentiment_data,
                'pattern_data': pattern_data,
                'divergence_data': divergence_data,
                'historical_data': historical_data,
                'investment_thesis': investment_thesis,
                'risk_assessment': risk_assessment,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save detailed analysis
            os.makedirs('analysis/detailed', exist_ok=True)
            with open(f"analysis/detailed/{symbol.lower()}_analysis.json", 'w') as f:
                json.dump(detailed_analysis, f, indent=4, default=json_serializable)
            
            logger.info(f"Generated detailed analysis for {symbol}")
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"Error generating detailed analysis for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {}
    def generate_all_analyses(self):
        """
        Generate detailed analyses for all cryptocurrencies.
        
        Returns:
            Success status
        """
        if self.scores is None or self.scores.empty:
            logger.warning("No cryptocurrency data available")
            return False
        
        logger.info("Generating detailed analyses for all cryptocurrencies...")
        
        success_count = 0
        for _, row in self.scores.iterrows():
            symbol = row['symbol']
            try:
                self.generate_detailed_analysis(symbol)
                success_count += 1
            except Exception as e:
                logger.error(f"Error generating detailed analysis for {symbol}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Generated detailed analyses for {success_count}/{len(self.scores)} cryptocurrencies")
        return success_count > 0
    
    def _generate_investment_thesis(self, symbol, crypto_data):
        """Generate investment thesis based on analysis."""
        try:
            # Get recommendations, scores, etc.
            recommendation = crypto_data.get('recommendation', 'Unknown')
            opportunity = crypto_data.get('market_opportunity', 'Unknown')
            technical_score = crypto_data.get('technical_score', 0)
            fundamental_score = crypto_data.get('fundamental_score', 0)
            sentiment_score = crypto_data.get('sentiment_score', 0)
            total_score = crypto_data.get('total_score', 0)
            
            # Initialize thesis based on recommendation
            if recommendation in ['Strong Buy', 'Buy']:
                thesis = f"{symbol} presents a {opportunity.lower()} investment opportunity based on its overall score of {total_score:.1f}/100. "
                
                # Add technical analysis
                if technical_score >= 70:
                    thesis += f"Technically, {symbol} shows very strong indicators with a score of {technical_score:.1f}/100, suggesting positive price momentum. "
                elif technical_score >= 50:
                    thesis += f"Technical indicators are favorable with a score of {technical_score:.1f}/100, showing balanced price action. "
                else:
                    thesis += f"Despite some technical weaknesses (score: {technical_score:.1f}/100), other factors support a positive outlook. "
                
                # Add fundamental analysis
                if fundamental_score >= 70:
                    thesis += f"Fundamentally strong ({fundamental_score:.1f}/100), this cryptocurrency demonstrates solid market presence and stability. "
                elif fundamental_score >= 50:
                    thesis += f"Fundamentals are reasonably solid ({fundamental_score:.1f}/100), suggesting adequate market position. "
                else:
                    thesis += f"While fundamentals are less robust ({fundamental_score:.1f}/100), they may improve with market development. "
                
                # Add sentiment analysis
                if sentiment_score >= 70:
                    thesis += f"Market sentiment is highly positive ({sentiment_score:.1f}/100), indicating strong investor confidence. "
                elif sentiment_score >= 50:
                    thesis += f"Sentiment indicators are favorable ({sentiment_score:.1f}/100), showing positive market perception. "
                else:
                    thesis += f"Despite mixed sentiment ({sentiment_score:.1f}/100), other factors suggest potential upside. "
                
            elif recommendation == 'Hold':
                thesis = f"{symbol} represents a {opportunity.lower()} opportunity with a balanced risk-reward profile (score: {total_score:.1f}/100). "
                
                # Add technical analysis
                if technical_score >= 60:
                    thesis += f"Technical indicators are moderately positive ({technical_score:.1f}/100), suggesting potential for price appreciation. "
                elif technical_score >= 40:
                    thesis += f"Technical factors are balanced ({technical_score:.1f}/100), showing mixed price signals. "
                else:
                    thesis += f"Technical indicators show weakness ({technical_score:.1f}/100), indicating potential price pressure. "
                
                # Add fundamental analysis
                if fundamental_score >= 60:
                    thesis += f"Fundamentals remain relatively strong ({fundamental_score:.1f}/100), supporting a stable market position. "
                elif fundamental_score >= 40:
                    thesis += f"Fundamental factors are average ({fundamental_score:.1f}/100), suggesting a moderate market foundation. "
                else:
                    thesis += f"Fundamental weaknesses ({fundamental_score:.1f}/100) suggest caution in the medium term. "
                
                # Add sentiment analysis
                if sentiment_score >= 60:
                    thesis += f"Current market sentiment is positive ({sentiment_score:.1f}/100), potentially supporting price stability. "
                elif sentiment_score >= 40:
                    thesis += f"Sentiment indicators are neutral ({sentiment_score:.1f}/100), showing balanced market perception. "
                else:
                    thesis += f"Negative sentiment trends ({sentiment_score:.1f}/100) suggest monitoring closely for changes. "
                
            else:  # Sell or Strong Sell
                thesis = f"{symbol} presents a {opportunity.lower()} opportunity with significant risks (score: {total_score:.1f}/100). "
                
                # Add technical analysis
                if technical_score >= 40:
                    thesis += f"Despite some positive technical signals ({technical_score:.1f}/100), overall market factors suggest caution. "
                else:
                    thesis += f"Technical indicators are weak ({technical_score:.1f}/100), showing negative price momentum. "
                
                # Add fundamental analysis
                if fundamental_score >= 40:
                    thesis += f"While some fundamental metrics remain adequate ({fundamental_score:.1f}/100), they may be outweighed by other concerns. "
                else:
                    thesis += f"Fundamental weaknesses ({fundamental_score:.1f}/100) indicate potential challenges ahead. "
                
                # Add sentiment analysis
                if sentiment_score >= 40:
                    thesis += f"Despite relatively neutral sentiment ({sentiment_score:.1f}/100), other factors suggest downside risk. "
                else:
                    thesis += f"Negative market sentiment ({sentiment_score:.1f}/100) reflects poor investor confidence. "
            
            # Add pattern analysis if available
            pattern_file = f"analysis/patterns/{symbol.lower()}_patterns.json"
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    pattern_data = json.load(f)
                
                # Add bullish patterns
                bullish_patterns = []
                if pattern_data.get('inverse_head_and_shoulders', False):
                    bullish_patterns.append("inverse head and shoulders")
                if pattern_data.get('double_bottom', False):
                    bullish_patterns.append("double bottom")
                if pattern_data.get('bullish_flag', False):
                    bullish_patterns.append("bullish flag")
                if pattern_data.get('ascending_triangle', False):
                    bullish_patterns.append("ascending triangle")
                
                if bullish_patterns:
                    if len(bullish_patterns) == 1:
                        thesis += f"Chart analysis reveals a {bullish_patterns[0]} pattern, supporting a bullish outlook. "
                    else:
                        thesis += f"Multiple bullish patterns ({', '.join(bullish_patterns)}) have been identified, strengthening the positive case. "
                
                # Add bearish patterns
                bearish_patterns = []
                if pattern_data.get('head_and_shoulders', False):
                    bearish_patterns.append("head and shoulders")
                if pattern_data.get('double_top', False):
                    bearish_patterns.append("double top")
                if pattern_data.get('bearish_flag', False):
                    bearish_patterns.append("bearish flag")
                if pattern_data.get('descending_triangle', False):
                    bearish_patterns.append("descending triangle")
                
                if bearish_patterns:
                    if len(bearish_patterns) == 1:
                        thesis += f"Chart analysis reveals a {bearish_patterns[0]} pattern, suggesting potential downside risk. "
                    else:
                        thesis += f"Multiple bearish patterns ({', '.join(bearish_patterns)}) have been identified, indicating caution is warranted. "
            
            # Add divergence analysis if available
            divergence_file = f"analysis/patterns/{symbol.lower()}_divergences.json"
            if os.path.exists(divergence_file):
                with open(divergence_file, 'r') as f:
                    divergence_data = json.load(f)
                
                if divergence_data.get('bullish_divergence', False):
                    thesis += f"A bullish divergence has been detected, potentially signaling a reversal to the upside. "
                elif divergence_data.get('bearish_divergence', False):
                    thesis += f"A bearish divergence has been detected, potentially signaling a reversal to the downside. "
            
            # Add market trend context
            if self.market_trends:
                market_trend = self.market_trends.get('market_trend', 'Neutral')
                fear_greed = self.fear_greed.get('value_classification', 'Neutral') if self.fear_greed else 'Neutral'
                
                thesis += f"This analysis is set against a broader market that is currently in a {market_trend.lower()} trend with sentiment showing {fear_greed.lower()}. "
            
            # Add final recommendation
            if recommendation == 'Strong Buy':
                thesis += f"Based on comprehensive analysis, {symbol} is rated as a Strong Buy with above-average potential for significant price appreciation."
            elif recommendation == 'Buy':
                thesis += f"Overall, {symbol} merits a Buy rating, suggesting positive but measured price appreciation potential."
            elif recommendation == 'Hold':
                thesis += f"The current recommendation for {symbol} is Hold, suggesting investors maintain existing positions while monitoring for changes in market conditions."
            elif recommendation == 'Sell':
                thesis += f"The analysis supports a Sell rating for {symbol}, indicating potential for price depreciation and suggesting reducing exposure."
            else:  # Strong Sell
                thesis += f"The comprehensive assessment results in a Strong Sell rating for {symbol}, indicating significant downside risk that warrants immediate consideration."
            
            return thesis
            
        except Exception as e:
            logger.error(f"Error generating investment thesis for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return f"Analysis of {symbol} indicates a {crypto_data.get('recommendation', 'Hold')} recommendation based on available data and market conditions."
    
    def _generate_risk_assessment(self, symbol, crypto_data):
        """Generate risk assessment based on analysis."""
        try:
            # Get risk profile and other data
            risk_profile = crypto_data.get('risk_profile', 'Unknown')
            market_cap = crypto_data.get('market_cap_usd', 0)
            volume = crypto_data.get('volume_24h_usd', 0)
            rank = crypto_data.get('rank', 0)
            
            # Calculate liquidity ratio
            liquidity_ratio = volume / market_cap if market_cap > 0 else 0
            
            # Get technical data for volatility
            volatility = None
            if symbol in self.historical_data:
                df = self.historical_data[symbol]['daily'].copy()
                if len(df) >= 30:
                    df['daily_return'] = df['price'].pct_change()
                    volatility = df['daily_return'].tail(30).std() * np.sqrt(365)  # Annualized
            
            # Initialize risk assessment
            assessment = f"{symbol} is classified as a {risk_profile.lower()} risk investment. "
            
# Add market position context
            if risk_profile == 'Conservative':
                assessment += f"As a large-cap cryptocurrency (market cap: ${market_cap/1e9:.2f}B) ranked #{rank} globally, it presents relatively lower volatility compared to smaller cryptocurrencies. "
                
                if volatility is not None:
                    assessment += f"Historical volatility ({volatility*100:.1f}% annualized) is moderate for the crypto market. "
                
                assessment += f"With a daily trading volume of ${volume/1e6:.2f}M and liquidity ratio of {liquidity_ratio:.3f}, {symbol} offers sufficient liquidity for most investors. "
                
                assessment += "Primary risks include regulatory changes, market-wide corrections, and competition from other established cryptocurrencies."
                
            elif risk_profile == 'Moderate':
                assessment += f"With a market cap of ${market_cap/1e9:.2f}B and global rank of #{rank}, it balances growth potential with reasonable stability. "
                
                if volatility is not None:
                    assessment += f"Annualized volatility of {volatility*100:.1f}% indicates moderate price fluctuations. "
                
                assessment += f"Daily trading volume of ${volume/1e6:.2f}M and liquidity ratio of {liquidity_ratio:.3f} suggest adequate liquidity for most trading activities. "
                
                assessment += "Investors should be aware of risks including increased competition, potential regulatory impacts, and higher sensitivity to market sentiment."
                
            elif risk_profile == 'Aggressive':
                assessment += f"With a market cap of ${market_cap/1e9:.2f}B and rank #{rank}, it offers higher growth potential with corresponding volatility. "
                
                if volatility is not None:
                    assessment += f"Annualized volatility of {volatility*100:.1f}% indicates significant price swings are common. "
                
                assessment += f"Daily trading volume of ${volume/1e6:.2f}M and liquidity ratio of {liquidity_ratio:.3f} suggest {liquidity_ratio > 0.1 and 'good' or 'moderate'} liquidity. "
                
                assessment += "Key risks include high competition, technological challenges, potential regulatory scrutiny, and pronounced market cycle effects."
                
            else:  # Speculative
                assessment += f"With a market cap of ${market_cap/1e9:.2f}B and rank #{rank}, it represents a speculative investment with potential for both high returns and significant losses. "
                
                if volatility is not None:
                    assessment += f"High annualized volatility of {volatility*100:.1f}% indicates substantial price fluctuations. "
                
                assessment += f"Daily trading volume of ${volume/1e6:.2f}M and liquidity ratio of {liquidity_ratio:.3f} suggest {liquidity_ratio > 0.15 and 'adequate' or 'potentially limited'} liquidity. "
                
                assessment += "Major risk factors include extreme volatility, limited history, potential liquidity challenges, regulatory uncertainty, and elevated market manipulation risk."
            
            # Add specific technical risks if available
            technical_file = f"analysis/technical/{symbol.lower()}_technical.json"
            if os.path.exists(technical_file):
                with open(technical_file, 'r') as f:
                    technical_data = json.load(f)
                
                # Check for overbought/oversold conditions
                rsi = technical_data.get('rsi_14', 50)
                if rsi > 70:
                    assessment += f" The RSI of {rsi:.1f} indicates overbought conditions, increasing the risk of a short-term correction."
                elif rsi < 30:
                    assessment += f" The RSI of {rsi:.1f} indicates oversold conditions, which might present a buying opportunity but also suggests recent significant selling pressure."
            
            # Add market context
            if self.market_trends:
                market_trend = self.market_trends.get('market_trend', 'Neutral')
                btc_dominance = self.market_trends.get('btc_dominance', 50)
                
                if market_trend in ['Strong Bullish', 'Bullish']:
                    assessment += f" The current bullish market environment potentially reduces short-term downside risk, though caution is warranted for any sudden sentiment changes."
                elif market_trend in ['Strong Bearish', 'Bearish']:
                    assessment += f" The current bearish market environment may amplify downside risks in the near term."
                
                if symbol != 'BTC' and btc_dominance > 60:
                    assessment += f" High Bitcoin dominance ({btc_dominance:.1f}%) may limit altcoin performance in the near term."
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error generating risk assessment for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return f"{symbol} is classified as a {risk_profile.lower()} risk investment based on market position and volatility metrics."

    def generate_report(self, top_n=10):
        """
        Generate a comprehensive report for the top N cryptocurrencies.
        
        Args:
            top_n: Number of top cryptocurrencies to include in the report
            
        Returns:
            Report text
        """
        if self.scores is None or self.scores.empty:
            logger.warning("No scoring data available")
            return "No cryptocurrency data available for report generation."
        
        try:
            # Get top N cryptocurrencies
            top_cryptos = self.scores.head(top_n)
            
            # Initialize report
            report = "# Cryptocurrency Market Analysis Report\n\n"
            report += f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            
            # Add market overview
            report += "## Market Overview\n\n"
            
            if self.market_trends:
                total_market_cap = self.market_trends.get('total_market_cap', 0)
                btc_dominance = self.market_trends.get('btc_dominance', 0)
                market_trend = self.market_trends.get('market_trend', 'Neutral')
                avg_change_24h = self.market_trends.get('avg_change_24h', 0)
                avg_change_7d = self.market_trends.get('avg_change_7d', 0)
                
                report += f"**Total Market Cap**: ${total_market_cap/1e9:.2f}B\n"
                report += f"**Bitcoin Dominance**: {btc_dominance:.2f}%\n"
                report += f"**Market Trend**: {market_trend}\n"
                report += f"**24h Average Change**: {avg_change_24h:.2f}%\n"
                report += f"**7d Average Change**: {avg_change_7d:.2f}%\n"
            
            if self.fear_greed:
                fear_greed_value = self.fear_greed.get('value', 50)
                fear_greed_class = self.fear_greed.get('value_classification', 'Neutral')
                
                report += f"**Fear & Greed Index**: {fear_greed_value} ({fear_greed_class})\n"
            
            report += "\n"
            
            # Add top cryptocurrencies
            report += "## Top Performing Cryptocurrencies\n\n"
            
            # Create table header
            report += "| Rank | Symbol | Name | Price (USD) | 24h Change | Total Score | Recommendation |\n"
            report += "|------|--------|------|------------|------------|-------------|----------------|\n"
            
            # Add rows for each cryptocurrency
            for _, crypto in top_cryptos.iterrows():
                report += f"| {crypto['rank']} | {crypto['symbol']} | {crypto['name']} | "
                report += f"${crypto['price_usd']:.2f} | {crypto['percent_change_24h']:.2f}% | "
                report += f"{crypto['total_score']:.1f} | {crypto['recommendation']} |\n"
            
            report += "\n"
            
            # Add detailed analysis for top 3
            report += "## Detailed Analysis\n\n"
            
            for i in range(min(3, len(top_cryptos))):
                crypto = top_cryptos.iloc[i]
                symbol = crypto['symbol']
                
                report += f"### {crypto['name']} ({symbol})\n\n"
                
                # Generate detailed analysis if not already available
                analysis_file = f"analysis/detailed/{symbol.lower()}_analysis.json"
                if not os.path.exists(analysis_file):
                    self.generate_detailed_analysis(symbol)
                
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as f:
                        analysis = json.load(f)
                    
                    # Add investment thesis
                    report += "#### Investment Thesis\n\n"
                    report += f"{analysis.get('investment_thesis', 'No investment thesis available.')}\n\n"
                    
                    # Add risk assessment
                    report += "#### Risk Assessment\n\n"
                    report += f"{analysis.get('risk_assessment', 'No risk assessment available.')}\n\n"
                    
                    # Add technical highlights
                    report += "#### Technical Highlights\n\n"
                    
                    technical_data = analysis.get('technical_data', {})
                    score_components = technical_data.get('score_components', {})
                    
                    report += f"- **Technical Score**: {crypto['technical_score']:.1f}/100\n"
                    report += f"- **Current Price**: ${technical_data.get('price', crypto['price_usd']):.4f}\n"
                    report += f"- **RSI (14)**: {technical_data.get('rsi_14', 'N/A')}\n"
                    
                    # Add key moving averages
                    report += "- **Key Moving Averages**:\n"
                    report += f"  - MA (7): ${technical_data.get('ma_7', 'N/A')}\n"
                    report += f"  - MA (30): ${technical_data.get('ma_30', 'N/A')}\n"
                    report += f"  - MA (90): ${technical_data.get('ma_90', 'N/A')}\n"
                    
                    # Add pattern analysis
                    pattern_data = analysis.get('pattern_data', {})
                    detected_patterns = [p for p, detected in pattern_data.items() if detected and p != 'pattern_strength']
                    
                    if detected_patterns:
                        report += "- **Detected Patterns**: " + ", ".join([p.replace('_', ' ').title() for p in detected_patterns]) + "\n"
                    
                    # Add divergence analysis
                    divergence_data = analysis.get('divergence_data', {})
                    if divergence_data.get('bullish_divergence', False):
                        report += f"- **Bullish Divergence Detected** (Strength: {divergence_data.get('bullish_divergence_strength', 0)}/10)\n"
                    elif divergence_data.get('bearish_divergence', False):
                        report += f"- **Bearish Divergence Detected** (Strength: {divergence_data.get('bearish_divergence_strength', 0)}/10)\n"
                    
                    report += "\n"
                    
                    # Add fundamental highlights
                    report += "#### Fundamental Highlights\n\n"
                    
                    fundamental_data = analysis.get('fundamental_data', {})
                    
                    report += f"- **Fundamental Score**: {crypto['fundamental_score']:.1f}/100\n"
                    report += f"- **Market Cap**: ${fundamental_data.get('market_cap_usd', crypto['market_cap_usd'])/1e9:.2f}B\n"
                    report += f"- **24h Volume**: ${fundamental_data.get('volume_24h_usd', crypto['volume_24h_usd'])/1e6:.2f}M\n"
                    report += f"- **Volume/MCap Ratio**: {fundamental_data.get('volume_mcap_ratio', 0):.4f}\n"
                    report += f"- **Market Share**: {fundamental_data.get('market_share', 0):.2f}%\n"
                    
                    report += "\n"
                    
                    # Add sentiment highlights
                    report += "#### Sentiment Highlights\n\n"
                    
                    sentiment_data = analysis.get('sentiment_data', {})
                    
                    report += f"- **Sentiment Score**: {crypto['sentiment_score']:.1f}/100\n"
                    report += f"- **Price Change (24h)**: {sentiment_data.get('price_change_24h', crypto['percent_change_24h']):.2f}%\n"
                    report += f"- **Price Change (7d)**: {sentiment_data.get('price_change_7d', crypto['percent_change_7d']):.2f}%\n"
                    
                    if 'relative_performance_24h' in sentiment_data:
                        report += f"- **Relative to Market (24h)**: {sentiment_data['relative_performance_24h']:.2f}%\n"
                    
                    if 'relative_performance_7d' in sentiment_data:
                        report += f"- **Relative to Market (7d)**: {sentiment_data['relative_performance_7d']:.2f}%\n"
                    
                    report += "\n"
                else:
                    report += "Detailed analysis not available.\n\n"
            
            # Add market insights
            report += "## Market Insights\n\n"
            
            # Distribution of recommendations
            rec_counts = self.scores['recommendation'].value_counts()
            
            report += "### Recommendation Distribution\n\n"
            for rec, count in rec_counts.items():
                report += f"- **{rec}**: {count} cryptocurrencies ({count/len(self.scores)*100:.1f}%)\n"
            
            report += "\n"
            
            # Distribution of risk profiles
            risk_counts = self.scores['risk_profile'].value_counts()
            
            report += "### Risk Profile Distribution\n\n"
            for risk, count in risk_counts.items():
                report += f"- **{risk}**: {count} cryptocurrencies ({count/len(self.scores)*100:.1f}%)\n"
            
            report += "\n"
            
            # Add disclaimer
            report += "## Disclaimer\n\n"
            report += "This report is generated for informational purposes only and does not constitute financial advice. "
            report += "Cryptocurrency investments are subject to high market risk. Past performance is not indicative of future results. "
            report += "Always conduct your own research and consider your financial situation before making investment decisions."
            
            # Save report
            with open('analysis/market_report.md', 'w') as f:
                f.write(report)
            
            logger.info("Generated comprehensive market report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.error(traceback.format_exc())
            return "Error generating report. See logs for details."
    
    def run_analysis(self, top_n=10):
        """
        Run complete analysis pipeline.
        
        Args:
            top_n: Number of top cryptocurrencies to include in detailed report
            
        Returns:
            Success status
        """
        try:
            logger.info("Starting cryptocurrency analysis pipeline...")
            
            # Score and rank all cryptocurrencies
            self.score_all_cryptocurrencies()
            
            if self.scores is None or self.scores.empty:
                logger.error("Failed to score cryptocurrencies")
                return False
            
            # Generate summary visualizations
            self.generate_summary_visualizations()
            
            # Generate detailed analysis for top N cryptocurrencies
            logger.info(f"Generating detailed analysis for top {top_n} cryptocurrencies...")
            for i, (_, crypto) in enumerate(self.scores.head(top_n).iterrows()):
                symbol = crypto['symbol']
                logger.info(f"Analyzing {symbol} ({i+1}/{top_n})...")
                self.generate_detailed_analysis(symbol)
            
            # Generate comprehensive report
            logger.info("Generating comprehensive report...")
            self.generate_report(top_n)
            
            logger.info("Analysis pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            logger.error(traceback.format_exc())
            return False

    def score_all(self):
        """Alias para score_all_cryptocurrencies para compatibilidade com dashboard."""
        return self.score_all_cryptocurrencies()

    def generate_predictions(self, symbol):
        """Gera previsões de preço para uma criptomoeda."""
        logger.info(f"Generating predictions for {symbol}...")
        
        try:
            if symbol not in self.historical_data:
                logger.warning(f"No historical data for {symbol}")
                return False
                
            df = self.historical_data[symbol]['daily'].copy()
            if len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}")
                return False
                
            # Usar os últimos 30 dias para extrapolar tendência
            df = df.sort_values('timestamp').tail(30).reset_index(drop=True)
            df['days'] = range(len(df))
            
            # Regressão linear simples
            X = df['days'].values.reshape(-1, 1)
            y = df['price'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Prever próximos 30 dias
            future_days = np.array(range(len(df), len(df) + 30)).reshape(-1, 1)
            predicted_prices = model.predict(future_days)
            
            # Garantir que não haja preços negativos
            predicted_prices = np.maximum(predicted_prices, 0)
            
            # Atualizar a análise detalhada com previsões
            crypto_data = self.scores[self.scores['symbol'] == symbol]
            if not crypto_data.empty:
                idx = crypto_data.index[0]
                self.scores.at[idx, 'predicted_price_7d'] = predicted_prices[6]  # 7º dia
                self.scores.at[idx, 'predicted_price_30d'] = predicted_prices[-1]  # 30º dia
                self.scores.at[idx, 'predicted_prices'] = predicted_prices.tolist()
                
                # Calcular mudanças percentuais
                current_price = df['price'].iloc[-1]
                self.scores.at[idx, 'predicted_change_7d'] = ((predicted_prices[6] / current_price) - 1) * 100
                self.scores.at[idx, 'predicted_change_30d'] = ((predicted_prices[-1] / current_price) - 1) * 100
                
                # Confiança da previsão (baseada no R² do modelo)
                confidence = max(0, min(100, model.score(X, y) * 100))
                self.scores.at[idx, 'prediction_confidence'] = confidence
                
                # Salvar rankings atualizados
                self.scores.to_csv('analysis/crypto_rankings.csv', index=False)
                with open('analysis/crypto_rankings.json', 'w') as f:
                    json.dump(self.scores.to_dict(orient='records'), f, indent=4, default=json_serializable)
                
                logger.info(f"Generated predictions for {symbol}")
                return True
            else:
                logger.warning(f"{symbol} not found in scores")
                return False
                
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    def analyze_patterns(self, symbol):
        """
        Analyze patterns for a specific cryptocurrency.
        This is an alias that combines pattern and divergence detection.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with combined pattern analysis
        """
        try:
            # Detect chart patterns
            patterns = self.detect_chart_patterns(symbol)
            
            # Detect divergences
            divergences = self.detect_divergences(symbol)
            
            # Combine results
            analysis = {
                'symbol': symbol,
                'patterns': patterns,
                'divergences': divergences,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save combined analysis
            os.makedirs('analysis/patterns', exist_ok=True)
            with open(f"analysis/patterns/{symbol.lower()}_combined_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=4, default=json_serializable)
            
            logger.info(f"Completed pattern analysis for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing patterns for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {
                'symbol': symbol,
                'patterns': {},
                'divergences': {},
                'error': str(e)
            }


# Run scoring if script is executed directly
if __name__ == "__main__":
    scorer = CryptoScorer()
    rankings = scorer.score_all_cryptocurrencies()
    scorer.generate_summary_visualizations()
    scorer.generate_all_analyses()
    
    # Print top 5 cryptocurrencies
    if rankings is not None and not rankings.empty:
        print("\nTop 5 Cryptocurrencies:")
        for _, row in rankings.head(5).iterrows():
            print(f"{row['rank']}. {row['name']} ({row['symbol']}): Score {row['total_score']}, {row['recommendation']}")
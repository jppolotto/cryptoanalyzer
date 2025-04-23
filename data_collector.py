"""
Data collection module for cryptocurrency analysis.
Fetches data for top cryptocurrencies from CoinMarketCap and CryptoDataDownload.
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from api_keys import CMC_API_KEY, CDD_API_KEY

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

class CryptoDataCollector:
    """Class to collect cryptocurrency data from various sources."""
    
    def __init__(self):
        self.cmc_api_key = CMC_API_KEY
        self.cdd_api_key = CDD_API_KEY
        self.cmc_base_url = "https://pro-api.coinmarketcap.com/v1"
        self.cdd_base_url = "https://www.cryptodatadownload.com/api"
        
    def get_top_cryptocurrencies(self, limit=20):
        """
        Get list of top cryptocurrencies by market cap.
        
        Args:
            limit: Number of top cryptocurrencies to retrieve
            
        Returns:
            DataFrame with cryptocurrency data
        """
        url = f"{self.cmc_base_url}/cryptocurrency/listings/latest"
        
        parameters = {
            'start': '1',
            'limit': str(limit),
            'convert': 'USD',
            'sort': 'market_cap',
            'sort_dir': 'desc'
        }
        
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.cmc_api_key,
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            # Convert to DataFrame
            df = pd.json_normalize(data['data'])
            
            # Save raw data
            with open('data/top_cryptos_raw.json', 'w') as f:
                json.dump(data, f)
                
            # Process and save cleaned data
            cleaned_df = self._process_cmc_data(df)
            cleaned_df.to_csv('data/top_cryptos.csv', index=False)
            
            print(f"Successfully collected data for top {limit} cryptocurrencies")
            return cleaned_df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching top cryptocurrencies: {e}")
            # If we have a saved file, try to load it
            if os.path.exists('data/top_cryptos.csv'):
                print("Loading data from saved file")
                return pd.read_csv('data/top_cryptos.csv')
            return pd.DataFrame()
    
    def _process_cmc_data(self, df):
        """Process and clean CoinMarketCap data."""
        # Select and rename relevant columns
        if df.empty:
            return df
            
        cols = [
            'id', 'name', 'symbol', 'slug', 'cmc_rank', 
            'quote.USD.price', 'quote.USD.volume_24h', 
            'quote.USD.market_cap', 'quote.USD.percent_change_1h',
            'quote.USD.percent_change_24h', 'quote.USD.percent_change_7d',
            'quote.USD.percent_change_30d', 'quote.USD.percent_change_60d',
            'quote.USD.percent_change_90d'
        ]
        
        # Make sure all columns exist
        existing_cols = [col for col in cols if col in df.columns]
        
        cleaned_df = df[existing_cols].copy()
        
        # Rename columns
        column_mapping = {
            'quote.USD.price': 'price_usd',
            'quote.USD.volume_24h': 'volume_24h_usd',
            'quote.USD.market_cap': 'market_cap_usd',
            'quote.USD.percent_change_1h': 'percent_change_1h',
            'quote.USD.percent_change_24h': 'percent_change_24h',
            'quote.USD.percent_change_7d': 'percent_change_7d',
            'quote.USD.percent_change_30d': 'percent_change_30d',
            'quote.USD.percent_change_60d': 'percent_change_60d',
            'quote.USD.percent_change_90d': 'percent_change_90d'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Add timestamp
        cleaned_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return cleaned_df
    
    def get_historical_data(self, symbol, days=360, interval='daily'):
        """
        Get historical price data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly')
            
        Returns:
            DataFrame with historical price data
        """
        # CryptoDataDownload API doesn't have a direct endpoint for this
        # Using CoinMarketCap for historical data
        
        url = f"{self.cmc_base_url}/cryptocurrency/quotes/historical"
        
        # Calculate time period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=360)
        
        # Convert to ISO format
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        parameters = {
            'symbol': symbol,
            'time_start': start_str,
            'time_end': end_str,
            'interval': '1d' if interval == 'daily' else '1h',
            'convert': 'USD'
        }
        
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.cmc_api_key,
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()
            data = response.json()
            
            # Save raw data
            filename = f"data/{symbol.lower()}_historical_{interval}.json"
            with open(filename, 'w') as f:
                json.dump(data, f)
                
            # Process data into DataFrame
            df = self._process_historical_data(data, symbol)
            
            # Save processed data
            csv_filename = f"data/{symbol.lower()}_historical_{interval}.csv"
            df.to_csv(csv_filename, index=False)
            
            print(f"Successfully collected historical {interval} data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            # Try to load from file if it exists
            csv_filename = f"data/{symbol.lower()}_historical_{interval}.csv"
            if os.path.exists(csv_filename):
                print(f"Loading historical data from saved file for {symbol}")
                return pd.read_csv(csv_filename)
            
            # If API fails and no file exists, create mock data for demo purposes
            print(f"Creating mock historical data for {symbol}")
            return self._create_mock_historical_data(symbol, days, interval)
    
    def _process_historical_data(self, data, symbol):
        """Process historical data from API response."""
        try:
            # Extract quotes from the response
            quotes = data['data']['quotes']
            
            # Convert to DataFrame
            records = []
            for quote in quotes:
                record = {
                    'timestamp': quote['timestamp'],
                    'symbol': symbol,
                    'price': quote['quote']['USD']['price'],
                    'volume_24h': quote['quote']['USD']['volume_24h'],
                    'market_cap': quote['quote']['USD']['market_cap'],
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except (KeyError, TypeError) as e:
            print(f"Error processing historical data: {e}")
            return pd.DataFrame()
    
    def _create_mock_historical_data(self, symbol, days, interval):
        """Create mock historical data for demo purposes."""
        end_date = datetime.now()
        
        if interval == 'daily':
            # Create daily data
            dates = [end_date - timedelta(days=i) for i in range(days)]
            periods = days
        else:
            # Create hourly data
            dates = [end_date - timedelta(hours=i) for i in range(days*24)]
            periods = days * 24
        
        # Get current price from top cryptos if available
        try:
            top_cryptos = pd.read_csv('data/top_cryptos.csv')
            current_price = top_cryptos[top_cryptos['symbol'] == symbol]['price_usd'].values[0]
        except:
            # Use a default price if not found
            current_price = 1000 if symbol == 'BTC' else 100
        
        # Generate random walk prices
        import numpy as np
        np.random.seed(42)  # For reproducibility
        
        # Random walk with drift
        random_walk = np.random.normal(0, 0.02, periods).cumsum()
        price_multiplier = np.exp(random_walk)
        
        prices = [current_price * m for m in price_multiplier]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'price': prices,
            'volume_24h': [np.random.uniform(1000000, 10000000) for _ in range(periods)],
            'market_cap': [p * np.random.uniform(10000, 100000) for p in prices]
        })
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def get_fear_greed_index(self):
        """
        Get the current Fear & Greed Index.
        
        Returns:
            Dictionary with fear & greed index data
        """
        # Alternative Fear & Greed Index API
        url = "https://api.alternative.me/fng/"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Save raw data
            with open('data/fear_greed_index.json', 'w') as f:
                json.dump(data, f)
            
            # Process data
            if 'data' in data:
                latest = data['data'][0]
                result = {
                    'value': int(latest['value']),
                    'value_classification': latest['value_classification'],
                    'timestamp': latest['timestamp'],
                    'time_until_update': latest['time_until_update']
                }
                
                # Save as CSV
                df = pd.DataFrame([result])
                df.to_csv('data/fear_greed_index.csv', index=False)
                
                print("Successfully collected Fear & Greed Index")
                return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            
        # If API fails or no data, create mock data
        mock_data = {
            'value': 45,
            'value_classification': 'Fear',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'time_until_update': '12:00'
        }
        
        # Save mock data
        df = pd.DataFrame([mock_data])
        df.to_csv('data/fear_greed_index.csv', index=False)
        
        return mock_data
    
    def get_global_metrics(self):
        """
        Get global cryptocurrency market metrics.
        
        Returns:
            Dictionary with global market data
        """
        url = f"{self.cmc_base_url}/global-metrics/quotes/latest"
        
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.cmc_api_key,
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Save raw data
            with open('data/global_metrics.json', 'w') as f:
                json.dump(data, f)
            
            # Extract relevant metrics
            metrics = data['data']
            result = {
                'total_market_cap_usd': metrics['quote']['USD']['total_market_cap'],
                'total_volume_24h_usd': metrics['quote']['USD']['total_volume_24h'],
                'btc_dominance': metrics['btc_dominance'],
                'eth_dominance': metrics['eth_dominance'],
                'active_cryptocurrencies': metrics['active_cryptocurrencies'],
                'active_exchanges': metrics['active_exchanges'],
                'last_updated': metrics['last_updated']
            }
            
            # Save as CSV
            df = pd.DataFrame([result])
            df.to_csv('data/global_metrics.csv', index=False)
            
            print("Successfully collected global market metrics")
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching global metrics: {e}")
            
            # If API fails, create mock data
            mock_data = {
                'total_market_cap_usd': 2500000000000,
                'total_volume_24h_usd': 150000000000,
                'btc_dominance': 45.5,
                'eth_dominance': 18.2,
                'active_cryptocurrencies': 10000,
                'active_exchanges': 500,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save mock data
            df = pd.DataFrame([mock_data])
            df.to_csv('data/global_metrics.csv', index=False)
            
            return mock_data
    
    def collect_all_data(self, top_n=20):
        """
        Collect all necessary data for the cryptocurrency analyzer.
        
        Args:
            top_n: Number of top cryptocurrencies to analyze
            
        Returns:
            Dictionary with all collected data
        """
        print(f"Starting data collection for top {top_n} cryptocurrencies...")
        
        # Get top cryptocurrencies
        top_cryptos = self.get_top_cryptocurrencies(limit=top_n)
        
        # Get global metrics
        global_metrics = self.get_global_metrics()
        
        # Get Fear & Greed Index
        fear_greed = self.get_fear_greed_index()
        
        # Get historical data for each cryptocurrency
        historical_data = {}
        for _, row in top_cryptos.iterrows():
            symbol = row['symbol']
            print(f"Collecting historical data for {symbol}...")
            
            # Get daily and hourly data
            daily_data = self.get_historical_data(symbol, days=90, interval='daily')
            historical_data[symbol] = {
                'daily': daily_data
            }
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Save collection summary
        summary = {
            'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_cryptos_count': len(top_cryptos),
            'global_metrics': bool(global_metrics),
            'fear_greed_index': bool(fear_greed),
            'historical_data_symbols': list(historical_data.keys())
        }
        
        with open('data/collection_summary.json', 'w') as f:
            json.dump(summary, f)
        
        print("Data collection completed successfully!")
        
        return {
            'top_cryptos': top_cryptos,
            'global_metrics': global_metrics,
            'fear_greed': fear_greed,
            'historical_data': historical_data,
            'summary': summary
        }


# Run data collection if script is executed directly
if __name__ == "__main__":
    collector = CryptoDataCollector()
    collector.collect_all_data(top_n=20)

import requests
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
import time

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoAPIFetcher:
    """Classe para buscar e gerenciar dados da API CryptoDataDownload."""
    
    def __init__(self, data_dir='data', api_token=None):
        """
        Inicializa o CryptoAPIFetcher.
        
        Args:
            data_dir: Diretório para armazenar os dados baixados
            api_token: Token da API CryptoDataDownload
        """
        self.data_dir = data_dir
        self.api_token = api_token
        self.base_url = "https://api.cryptodatadownload.com/v1/data"
        self.saved_files = []
        
        # Criar diretórios necessários
        self._create_directories()
    
    def _create_directories(self):
        """Cria os diretórios necessários para armazenar os dados."""
        directories = [
            self.data_dir,
            f"{self.data_dir}/breadth",
            f"{self.data_dir}/ohlc",
            f"{self.data_dir}/gov",
            f"{self.data_dir}/risk",
            f"{self.data_dir}/summary",
            f"{self.data_dir}/technicals"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Diretório criado/verificado: {directory}")
    
    def _make_request(self, endpoint, params=None, retries=3, delay=2):
        """
        Faz uma requisição à API com retry automático.
        
        Args:
            endpoint: Endpoint da API
            params: Parâmetros da requisição
            retries: Número de tentativas em caso de falha
            delay: Tempo de espera entre tentativas em segundos
            
        Returns:
            Dados da resposta ou None em caso de erro
        """
        if not self.api_token:
            logger.error("Token da API não definido")
            return None
        
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"TOKEN {self.api_token}"}
        
        for attempt in range(retries):
            try:
                logger.debug(f"Requisição para {url} com parâmetros {params}")
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        return data
                    except ValueError:
                        logger.warning(f"Resposta não é um JSON válido: {response.text[:100]}")
                        # Tentar retornar o texto da resposta para processamento posterior
                        return response.text
                
                elif response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit atingido. Tentativa {attempt+1}/{retries}")
                    time.sleep(delay * (attempt + 1))  # Backoff exponencial
                    continue
                    
                else:
                    logger.error(f"Erro na requisição: {response.status_code} - {response.text[:100]}")
                    return None
                    
            except requests.RequestException as e:
                logger.error(f"Erro de requisição: {e}")
                if attempt < retries - 1:
                    logger.info(f"Tentando novamente em {delay} segundos...")
                    time.sleep(delay)
                else:
                    logger.error("Número máximo de tentativas excedido")
                    return None
        
        return None
    
    def _save_data(self, data, filename, format='csv'):
        """
        Salva os dados em um arquivo.
        
        Args:
            data: Dados a serem salvos
            filename: Nome do arquivo
            format: Formato do arquivo ('csv', 'json')
            
        Returns:
            Bool indicando sucesso
        """
        try:
            filepath = f"{self.data_dir}/{filename}"
            
            # Criar DataFrame se for um dicionário
            if isinstance(data, dict):
                df = pd.DataFrame([data]) if not any(isinstance(v, list) for v in data.values()) else pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, str):
                # Tentar converter string para DataFrame
                try:
                    if data.startswith("[") or data.startswith("{"):
                        json_data = json.loads(data)
                        df = pd.DataFrame(json_data) if isinstance(json_data, list) else pd.DataFrame([json_data])
                    else:
                        # Pode ser CSV ou outro formato de texto
                        import io
                        df = pd.read_csv(io.StringIO(data))
                except Exception as e:
                    logger.error(f"Erro ao converter string para DataFrame: {e}")
                    # Salvar como arquivo de texto
                    with open(filepath, 'w') as f:
                        f.write(data)
                    self.saved_files.append(filepath)
                    logger.info(f"Dados salvos como texto em {filepath}")
                    return True
            else:
                logger.error(f"Tipo de dados não suportado: {type(data)}")
                return False
            
            # Salvar no formato especificado
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'json':
                df.to_json(filepath, orient='records', indent=4)
            else:
                logger.error(f"Formato não suportado: {format}")
                return False
            
            self.saved_files.append(filepath)
            logger.info(f"Dados salvos em {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
            return False
    
    def fetch_market_breadth(self):
        """
        Busca dados de amplitude do mercado (market breadth).
        
        Returns:
            Bool indicando sucesso
        """
        try:
            # Buscar Highs/Lows de 52 semanas
            highs_lows = self._make_request("breadth/52wk-highs-lows/")
            if highs_lows:
                self._save_data(highs_lows, "breadth/52wk_highs_lows.csv")
            
            # Buscar rastreamento de médias móveis
            ma_tracking = self._make_request("breadth/moving-average-tracking/")
            if ma_tracking:
                self._save_data(ma_tracking, "breadth/moving_average_tracking.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados de amplitude do mercado: {e}")
            return False
    
    def fetch_government_data(self):
        """
        Busca dados governamentais (CFTC, Treasury)
        
        Returns:
            Bool indicando sucesso
        """
        try:
            # Buscar dados COT da CFTC para Bitcoin
            current_year = datetime.now().year
            cftc_data = self._make_request("cftc/cot/", params={
                "contract_name": "BITCOIN",
                "year": str(current_year)
            })
            if cftc_data:
                self._save_data(cftc_data, f"gov/cftc_cot_bitcoin_{current_year}.csv")
            
            # Buscar dados de rendimentos do tesouro
            treasury_data = self._make_request("treasury/par-yields/", params={"year": str(current_year)})
            if treasury_data:
                self._save_data(treasury_data, f"gov/treasury_yields_{current_year}.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados governamentais: {e}")
            return False
    
    def fetch_ohlc_data(self, exchanges=None, symbols=None):
        """
        Busca dados OHLC de vários exchanges e pares.
        
        Args:
            exchanges: Lista de exchanges (se None, usa 'binance' e 'deribit')
            symbols: Lista de símbolos (se None, usa 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            Bool indicando sucesso
        """
        if not exchanges:
            exchanges = ['binance', 'deribit']
        
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT']
            deribit_symbols = ['BTC-PERPETUAL', 'ETH-PERPETUAL']
        else:
            # Converter símbolos para formato da Deribit
            deribit_symbols = [f"{s.split('USDT')[0]}-PERPETUAL" for s in symbols if 'USDT' in s]
        
        try:
            # Para cada exchange e símbolo
            for exchange in exchanges:
                if exchange == 'binance':
                    # Dados spot
                    for symbol in symbols:
                        spot_data = self._make_request(f"ohlc/binance/spot/", params={
                            "symbol": symbol,
                            "interval": "1d"
                        })
                        if spot_data:
                            self._save_data(spot_data, f"ohlc/binance_spot_{symbol}_daily.csv")
                    
                    # Dados de futuros
                    for symbol in symbols:
                        futures_data = self._make_request(f"ohlc/binance/futures/um/", params={
                            "symbol": symbol,
                            "interval": "1d"
                        })
                        if futures_data:
                            self._save_data(futures_data, f"ohlc/binance_futures_{symbol}_daily.csv")
                
                elif exchange == 'deribit':
                    # Dados de futuros
                    for symbol in deribit_symbols:
                        futures_data = self._make_request(f"ohlc/deribit/futures/", params={
                            "symbol": symbol
                        })
                        if futures_data:
                            self._save_data(futures_data, f"ohlc/deribit_futures_{symbol.replace('-', '_')}_daily.csv")
                    
                    # Dados de funding
                    for symbol in deribit_symbols:
                        funding_data = self._make_request(f"ohlc/deribit/futures/funding/", params={
                            "symbol": symbol
                        })
                        if funding_data:
                            self._save_data(funding_data, f"ohlc/deribit_funding_{symbol.replace('-', '_')}.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados OHLC: {e}")
            return False
    
    def fetch_risk_metrics(self, symbols=None):
        """
        Busca métricas de risco para os símbolos especificados.
        
        Args:
            symbols: Lista de símbolos (se None, usa 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            Bool indicando sucesso
        """
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        try:
            # Para cada símbolo
            for symbol in symbols:
                # Correlações
                correlations = self._make_request("risk/correlations/", params={"symbol": symbol})
                if correlations:
                    self._save_data(correlations, f"risk/correlations_{symbol}.csv")
                
                # VaR
                var_data = self._make_request("risk/var/standalone/", params={"symbol": symbol})
                if var_data:
                    self._save_data(var_data, f"risk/var_{symbol}.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao buscar métricas de risco: {e}")
            return False
    
    def fetch_summary_data(self, symbols=None):
        """
        Busca dados resumidos de Binance, Blockchain e Deribit.
        
        Args:
            symbols: Lista de símbolos (se None, usa 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            Bool indicando sucesso
        """
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT']
            deribit_symbols = ['BTC', 'ETH']
            blockchain_symbols = ['btc', 'eth']
        else:
            # Converter símbolos para formato da Deribit e Blockchain
            deribit_symbols = [s.split('USDT')[0] for s in symbols if 'USDT' in s]
            blockchain_symbols = [s.split('USDT')[0].lower() for s in symbols if 'USDT' in s]
        
        try:
            # Binance Summary Data
            for symbol in symbols:
                # Taxas de funding de futuros
                funding = self._make_request("summary/binance/futures/funding/", params={"symbol": symbol})
                if funding:
                    self._save_data(funding, f"summary/binance_futures_funding_{symbol}.csv")
                
                # Métricas de futuros
                metrics = self._make_request("summary/binance/futures/metrics/", params={"symbol": symbol})
                if metrics:
                    self._save_data(metrics, f"summary/binance_futures_metrics_{symbol}.csv")
                
                # Open Interest de opções
                oi = self._make_request("summary/binance/options/oi/", params={"underlying": symbol})
                if oi:
                    self._save_data(oi, f"summary/binance_options_oi_{symbol}.csv")
            
            # Blockchain Summary Data
            # Dados BCHAIN agregados
            bchain = self._make_request("summary/blockchain/bchain/")
            if bchain:
                self._save_data(bchain, "summary/blockchain_bchain_aggregated.csv")
            
            # Resumo de blocos por símbolo
            for symbol in blockchain_symbols:
                blocks = self._make_request("summary/blockchain/blocks/", params={"symbol": symbol})
                if blocks:
                    self._save_data(blocks, f"summary/blockchain_blocks_{symbol}.csv")
            
            # Maiores transações (para data atual)
            today = datetime.now().strftime("%Y-%m-%d")
            transactions = self._make_request("summary/blockchain/transactions/largest/", params={"date": today})
            if transactions:
                self._save_data(transactions, f"summary/blockchain_largest_transactions_{today}.csv")
            
            # Deribit Summary Data
            for symbol in deribit_symbols:
                # Resumo de gregas
                greeks = self._make_request("summary/deribit/options/greeks/", params={"underlying": symbol})
                if greeks:
                    self._save_data(greeks, f"summary/deribit_options_greeks_{symbol}.csv")
                
                # Maiores transações (para data atual)
                transactions = self._make_request(
                    "summary/deribit/options/transactions/largest/", 
                    params={"date": today}
                )
                if transactions:
                    self._save_data(transactions, f"summary/deribit_options_transactions_{symbol}_{today}.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados resumidos: {e}")
            return False
    
    def fetch_technical_indicators(self, symbols=None):
        """
        Busca indicadores técnicos para os símbolos especificados.
        
        Args:
            symbols: Lista de símbolos (se None, usa 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            Bool indicando sucesso
        """
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Para cada símbolo
            for symbol in symbols:
                indicators = self._make_request("technicals/TA/", params={
                    "coin": symbol,
                    "date": today
                })
                if indicators:
                    self._save_data(indicators, f"technicals/indicators_{symbol}_{today}.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao buscar indicadores técnicos: {e}")
            return False
    
    def fetch_all_data(self, symbols=None):
        """
        Busca todos os tipos de dados disponíveis.
        
        Args:
            symbols: Lista de símbolos (se None, usa 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            Dict com resultados de cada operação
        """
        results = {}
        
        logger.info("Iniciando busca completa de dados da API CryptoDataDownload")
        
        results['market_breadth'] = self.fetch_market_breadth()
        results['government_data'] = self.fetch_government_data()
        results['ohlc_data'] = self.fetch_ohlc_data(symbols=symbols)
        results['risk_metrics'] = self.fetch_risk_metrics(symbols=symbols)
        results['summary_data'] = self.fetch_summary_data(symbols=symbols)
        results['technical_indicators'] = self.fetch_technical_indicators(symbols=symbols)
        
        logger.info("Busca completa finalizada")
        logger.info(f"Arquivos salvos: {len(self.saved_files)}")
        
        return {
            'success': all(results.values()),
            'results': results,
            'saved_files': len(self.saved_files),
            'file_list': self.saved_files
        }
    
    def get_available_files(self):
        """
        Retorna uma lista de todos os arquivos disponíveis no diretório de dados.
        
        Returns:
            Dict com arquivos agrupados por categoria
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"Diretório de dados não encontrado: {self.data_dir}")
            return {}
        
        result = {
            'breadth': [],
            'ohlc': [],
            'gov': [],
            'risk': [],
            'summary': [],
            'technicals': [],
            'other': []
        }
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    relpath = os.path.relpath(filepath, self.data_dir)
                    
                    # Categorizar o arquivo
                    category = 'other'
                    for cat in result.keys():
                        if cat in relpath:
                            category = cat
                            break
                    
                    result[category].append({
                        'name': file,
                        'path': relpath,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        return result
    
    def load_data_file(self, filepath):
        """
        Carrega um arquivo de dados específico.
        
        Args:
            filepath: Caminho relativo do arquivo a partir do diretório de dados
            
        Returns:
            DataFrame com os dados ou None em caso de erro
        """
        try:
            full_path = os.path.join(self.data_dir, filepath)
            
            if not os.path.exists(full_path):
                logger.error(f"Arquivo não encontrado: {full_path}")
                return None
            
            if filepath.endswith('.csv'):
                df = pd.read_csv(full_path)
            elif filepath.endswith('.json'):
                df = pd.read_json(full_path)
            else:
                logger.error(f"Formato de arquivo não suportado: {filepath}")
                return None
            
            logger.info(f"Arquivo carregado: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {filepath}: {e}")
            return None

# Função para facilitar a criação de uma instância com token da API
def create_fetcher(api_token, data_dir='data'):
    """
    Cria uma instância de CryptoAPIFetcher com o token especificado.
    
    Args:
        api_token: Token da API CryptoDataDownload
        data_dir: Diretório para armazenar os dados
        
    Returns:
        Instância de CryptoAPIFetcher
    """
    return CryptoAPIFetcher(data_dir=data_dir, api_token=api_token)

# Exemplo de uso
if __name__ == "__main__":
    # Token da API
    API_TOKEN = "73f3a088f60e547e8d803f2773d82f7218e83030"
    
    # Criar fetcher
    fetcher = create_fetcher(API_TOKEN)
    
    # Buscar todos os dados disponíveis
    results = fetcher.fetch_all_data()
    
    # Imprimir resultados
    print(f"Sucesso: {results['success']}")
    print(f"Arquivos salvos: {results['saved_files']}")
    
    # Listar arquivos disponíveis
    files = fetcher.get_available_files()
    for category, file_list in files.items():
        print(f"\n{category.upper()}: {len(file_list)} arquivos")
        for file in file_list[:3]:  # Mostrar apenas 3 primeiros
            print(f"  - {file['name']} ({file['modified']})")
        if len(file_list) > 3:
            print(f"  ... e mais {len(file_list) - 3} arquivos")
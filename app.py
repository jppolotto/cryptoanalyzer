import streamlit as st
from crypto_api_fetcher import CryptoAPIFetcher # Supondo que esta classe existe
from crypto_data_analysis import CryptoDataAnalysis # Atualizado para usar a classe correta

def main():
    st.title("Análise Avançada de Mercado Cripto")
    
    # Inicialização
    # Certifique-se de que CryptoAPIFetcher está definido corretamente em seu ambiente
    try:
        fetcher = CryptoAPIFetcher()
    except NameError:
        st.error("Erro: A classe CryptoAPIFetcher não foi encontrada. Verifique as importações.")
        # Mock fetcher para permitir que a UI carregue parcialmente
        class MockFetcher:
            def load_data_file(self, filename):
                st.warning(f"MockFetcher: Tentativa de carregar {filename}")
                return None
        fetcher = MockFetcher()
        
    analyzer = CryptoDataAnalysis(fetcher)
    
    # Sidebar para configurações
    st.sidebar.header("Configurações")
    symbol = st.sidebar.selectbox(
        "Selecione o par de trading",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"],
        key='symbol_select' # Adiciona uma chave única
    )
    
    # Seleção de análises - Nomes atualizados para clareza
    st.sidebar.header("Análises Disponíveis")
    analysis_options = {
        # Básicas
        "Amplitude de Mercado (Básica)": "basic_breadth",
        "Mercado Spot (OHLC)": "basic_ohlc",
        "Sumário (Funding, Vol/OI)": "basic_summary",
        "Dados Governamentais (COT, Yields)": "basic_gov",
        "Métricas de Risco (VaR, Correl)": "basic_risk",
        # Avançadas
        "Amplitude de Mercado (Avançada)": "advanced_breadth",
        "Estrutura a Termo de Futuros": "advanced_futures_term",
        "Superfície de Volatilidade (Opções)": "advanced_vol_surface",
        "Detecção de Regime de Mercado": "advanced_regime",
        "Correlações Cross-Asset": "advanced_cross_asset"
    }
    
    selected_analysis_names = st.sidebar.multiselect(
        "Selecione as análises desejadas",
        list(analysis_options.keys()),
        default=list(analysis_options.keys()), # Seleciona tudo por padrão
        key='analysis_multiselect' # Adiciona uma chave única
    )
    
    # Mapeamento de nomes selecionados para métodos e argumentos
    analysis_map = {
        "Amplitude de Mercado (Básica)": analyzer.analyze_market_breadth,
        "Mercado Spot (OHLC)": analyzer.analyze_ohlc_data,
        "Sumário (Funding, Vol/OI)": analyzer.analyze_summary_data,
        "Dados Governamentais (COT, Yields)": analyzer.analyze_government_data,
        "Métricas de Risco (VaR, Correl)": analyzer.analyze_risk_metrics,
        "Amplitude de Mercado (Avançada)": analyzer.advanced_analyzer.enhance_market_breadth_analysis,
        "Estrutura a Termo de Futuros": analyzer.advanced_analyzer.analyze_futures_term_structure,
        "Superfície de Volatilidade (Opções)": analyzer.advanced_analyzer.analyze_volatility_surface,
        "Detecção de Regime de Mercado": analyzer.advanced_analyzer.detect_market_regime,
        "Correlações Cross-Asset": analyzer.advanced_analyzer.analyze_cross_asset_correlations
    }

    args_map = {
        "Amplitude de Mercado (Básica)": [],
        "Mercado Spot (OHLC)": [symbol],
        "Sumário (Funding, Vol/OI)": [symbol],
        "Dados Governamentais (COT, Yields)": [],
        "Métricas de Risco (VaR, Correl)": [symbol],
        "Amplitude de Mercado (Avançada)": [],
        "Estrutura a Termo de Futuros": [symbol],
        "Superfície de Volatilidade (Opções)": [symbol],
        "Detecção de Regime de Mercado": [symbol],
        "Correlações Cross-Asset": [symbol]
    }

    # Execução das análises selecionadas - Lógica Corrigida
    if st.sidebar.button("Executar Análises", key='run_button'):
        if not selected_analysis_names:
            st.warning("Nenhuma análise selecionada.")
        else:
            st.info(f"Executando {len(selected_analysis_names)} análise(s) selecionada(s) para {symbol}...")
            with st.spinner("Processando..."):
                executed_count = 0
                error_count = 0
                # Adiciona um container para os resultados para melhor organização
                results_container = st.container()

                for analysis_name in selected_analysis_names:
                    method_to_call = analysis_map.get(analysis_name)
                    args_to_pass = args_map.get(analysis_name, [])
                    
                    if method_to_call:
                        results_container.markdown(f"<hr><h4> Executando: {analysis_name} </h4>", unsafe_allow_html=True)
                        try:
                            # Chama o método específico
                            method_to_call(*args_to_pass) 
                            executed_count += 1
                        except Exception as e:
                            error_count += 1
                            results_container.error(f"Erro ao executar '{analysis_name}': {str(e)}")
                            import traceback
                            # Mostra traceback dentro de um expander para não poluir a UI
                            with results_container.expander("Ver detalhes do erro"): 
                                st.text(traceback.format_exc())
                    else:
                         results_container.warning(f"Método não encontrado para a análise: '{analysis_name}'")
            
            if executed_count > 0 and error_count == 0:
                st.success(f"Todas as {executed_count} análises selecionadas foram concluídas com sucesso!")
            elif executed_count > 0 and error_count > 0:
                 st.warning(f"{executed_count} análise(s) concluída(s), mas {error_count} falharam.")
            elif error_count > 0:
                 st.error(f"Todas as {error_count} análises selecionadas falharam.")

if __name__ == "__main__":
    # Adiciona checagem básica para garantir que as classes necessárias estão disponíveis
    missing_classes = []
    try:
        from crypto_api_fetcher import CryptoAPIFetcher
    except ImportError:
        missing_classes.append("CryptoAPIFetcher")
    try:
        from crypto_data_analysis import CryptoDataAnalysis, CryptoDataAnalyzer, AdvancedCryptoAnalyzer
    except ImportError:
        missing_classes.append("CryptoDataAnalysis/CryptoDataAnalyzer/AdvancedCryptoAnalyzer")

    if missing_classes:
        st.error(f"Erro Crítico: As seguintes classes não puderam ser importadas: {', '.join(missing_classes)}. Verifique os arquivos e nomes.")
    else:
        main() 
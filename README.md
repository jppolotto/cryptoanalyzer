# Crypto Analyzer

A comprehensive cryptocurrency analysis tool that collects real-time data, scores and ranks cryptocurrencies, and provides an interactive dashboard for investment decision-making.

## Features

- **Real-time Data Collection**: Fetches current data for the top 20 cryptocurrencies using CoinMarketCap and CryptoDataDownload APIs
- **Comprehensive Scoring System**: Evaluates cryptocurrencies based on technical, fundamental, and sentiment factors
- **Risk Profiling**: Categorizes cryptocurrencies as Conservative, Moderate, or Aggressive based on multiple factors
- **Investment Recommendations**: Provides clear Buy/Hold/Sell recommendations with detailed justifications
- **Interactive Dashboard**: User-friendly interface with multiple views and comparison tools
- **Visualization Tools**: Charts and graphs for price history, score components, correlations, and more

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Unzip the package to your desired location
2. Open a terminal/command prompt and navigate to the extracted directory
3. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn plotly streamlit requests
```

## Usage

### Running the Full Analysis

To run the complete analysis pipeline (data collection, scoring, and dashboard preparation):

```bash
python run_analyzer.py
```

This will:
1. Collect data for the top 20 cryptocurrencies
2. Calculate scores and rankings
3. Generate visualizations
4. Prepare the dashboard

### Launching the Dashboard

After running the analysis, start the interactive dashboard with:

```bash
streamlit run dashboard.py
```

This will open the dashboard in your default web browser.

### Dashboard Navigation

The dashboard includes four main sections:

1. **Market Overview**: Global market metrics, Fear & Greed Index, and top cryptocurrencies
2. **Cryptocurrency Rankings**: Sortable and filterable table of all analyzed cryptocurrencies
3. **Detailed Analysis**: In-depth analysis of a selected cryptocurrency
4. **Compare Cryptocurrencies**: Side-by-side comparison of two cryptocurrencies

## API Keys

The application uses the following APIs:
- CoinMarketCap API
- CryptoDataDownload API

Your API keys are stored in `api_keys.py`. Please ensure this file is kept secure and not shared publicly.

## Customization

### Adding More Cryptocurrencies

To analyze more than the top 20 cryptocurrencies, modify the `MAX_CRYPTOS` variable in `data_collector.py`.

### Adjusting Scoring Weights

To change how technical, fundamental, and sentiment factors are weighted, modify the weights in the `calculate_total_score` method in `crypto_scorer.py`.

## Troubleshooting

- If you encounter API rate limit issues, try reducing the number of cryptocurrencies analyzed or implementing a delay between requests
- For dashboard display issues, ensure you're using a recent version of your web browser
- Check the log files (`crypto_analyzer.log` and `package_app.log`) for detailed error information

## Disclaimer

This tool is for informational purposes only and does not constitute investment advice. Cryptocurrency investments are subject to high market risk. Always conduct your own research before making investment decisions.

## License

This project is licensed for personal use only and is not to be redistributed without permission.

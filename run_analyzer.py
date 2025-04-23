#!/usr/bin/env python3
"""
Main application script for Crypto Analyzer.
This script runs the complete cryptocurrency analysis pipeline:
1. Collects data for top cryptocurrencies
2. Scores and ranks cryptocurrencies
3. Launches the interactive dashboard
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("crypto_analyzer")

def main():
    """Run the complete cryptocurrency analysis pipeline."""
    start_time = time.time()
    logger.info("Starting Crypto Analyzer application")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('analysis/technical', exist_ok=True)
    os.makedirs('analysis/fundamental', exist_ok=True)
    os.makedirs('analysis/sentiment', exist_ok=True)
    os.makedirs('analysis/detailed', exist_ok=True)
    os.makedirs('analysis/visualizations', exist_ok=True)
    
    try:
        # Step 1: Collect data
        logger.info("Step 1: Collecting cryptocurrency data")
        from data_collector import CryptoDataCollector
        
        collector = CryptoDataCollector()
        collector.collect_all_data()
        
        # Step 2: Score and rank cryptocurrencies
        logger.info("Step 2: Scoring and ranking cryptocurrencies")
        from crypto_scorer import CryptoScorer
        
        scorer = CryptoScorer()
        rankings = scorer.score_all_cryptocurrencies()
        scorer.generate_summary_visualizations()
        scorer.generate_all_analyses()
        
        # Step 3: Launch interactive dashboard
        logger.info("Step 3: Launching interactive dashboard")
        logger.info("Total processing time: {:.2f} seconds".format(time.time() - start_time))
        logger.info("Dashboard is ready to use. Run 'streamlit run dashboard.py' to start the dashboard.")
        
        print("\n" + "="*80)
        print("Crypto Analyzer completed successfully!")
        print("To launch the interactive dashboard, run:")
        print("streamlit run dashboard.py")
        print("="*80 + "\n")
        
        # Automatically launch the dashboard if requested
        if len(sys.argv) > 1 and sys.argv[1] == "--launch":
            import subprocess
            subprocess.run(["streamlit", "run", "dashboard.py"])
        
    except Exception as e:
        logger.error(f"Error running Crypto Analyzer: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Check crypto_analyzer.log for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

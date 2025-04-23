"""
Integration module for the cryptocurrency analyzer system.
Connects all components and provides a unified interface.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import logging
import time
import schedule
from threading import Thread

# Import project modules
from data_collector import CryptoDataCollector
from crypto_scorer import CryptoScorer
from ml_analyzer import CryptoMLAnalyzer
from alert_system import CryptoAlertSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_analyzer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("crypto_integration")

class CryptoAnalyzerIntegration:
    """
    Integration class for the cryptocurrency analyzer system.
    Connects all components and provides a unified interface.
    """
    
    def __init__(self, data_dir='data', analysis_dir='analysis'):
        """
        Initialize the integration module.
        
        Args:
            data_dir: Directory containing cryptocurrency data
            analysis_dir: Directory containing analysis results
        """
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        
        # Initialize components
        self.collector = CryptoDataCollector()
        self.scorer = CryptoScorer(data_dir)
        self.ml_analyzer = CryptoMLAnalyzer(data_dir, analysis_dir)
        self.alert_system = CryptoAlertSystem(data_dir, analysis_dir)
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Status
        self.scheduler_running = False
    
    def run_full_pipeline(self, top_n=20):
        """
        Run the complete cryptocurrency analysis pipeline.
        
        Args:
            top_n: Number of top cryptocurrencies to analyze
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Start time
            start_time = time.time()
            logger.info("Starting full cryptocurrency analysis pipeline...")
            
            # Step 1: Collect data
            logger.info("Step 1: Collecting cryptocurrency data...")
            collected_data = self.collector.collect_all_data(top_n=top_n)
            
            if not collected_data:
                logger.error("Data collection failed")
                return False
            
            logger.info(f"Collected data for {len(collected_data.get('top_cryptos', []))} cryptocurrencies")
            
            # Step 2: Score and rank cryptocurrencies
            logger.info("Step 2: Scoring and ranking cryptocurrencies...")
            rankings = self.scorer.score_all_cryptocurrencies()
            
            if rankings is None or rankings.empty:
                logger.error("Scoring and ranking failed")
                return False
            
            logger.info(f"Scored and ranked {len(rankings)} cryptocurrencies")
            
            # Step 3: Generate visualizations
            logger.info("Step 3: Generating visualizations...")
            self.scorer.generate_summary_visualizations()
            
            # Step 4: Generate detailed analyses
            logger.info("Step 4: Generating detailed analyses...")
            self.scorer.generate_all_analyses()
            
            # Step 5: Run ML analysis for top cryptocurrencies
            logger.info("Step 5: Running machine learning analysis...")
            ml_predictions = self.ml_analyzer.analyze_all_cryptocurrencies(top_n=min(top_n, 10), days=30)
            
            if ml_predictions is None or ml_predictions.empty:
                logger.warning("ML analysis failed or produced no results")
            else:
                logger.info(f"Generated ML predictions for {len(ml_predictions)} cryptocurrencies")
            
            # Step 6: Check alerts
            logger.info("Step 6: Checking alerts...")
            triggered_alerts = self.alert_system.check_all_alerts()
            
            if triggered_alerts:
                logger.info(f"Triggered {len(triggered_alerts)} alerts")
            else:
                logger.info("No alerts triggered")
            
            # End time
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Cryptocurrency analysis pipeline completed in {duration:.2f} seconds")
            
            # Generate execution summary
            summary = {
                'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': duration,
                'cryptocurrencies_analyzed': len(rankings),
                'ml_predictions_generated': len(ml_predictions) if ml_predictions is not None else 0,
                'alerts_triggered': len(triggered_alerts),
                'top_crypto': rankings.iloc[0]['symbol'] if len(rankings) > 0 else None,
                'top_score': rankings.iloc[0]['total_score'] if len(rankings) > 0 else None
            }
            
            # Save summary
            with open(f"{self.analysis_dir}/execution_summary.json", 'w') as f:
                json.dump(summary, f, indent=4)
            
            return True
            
        except Exception as e:
            logger.error(f"Error running full pipeline: {e}", exc_info=True)
            return False
    
    def update_data_only(self, top_n=20):
        """
        Update cryptocurrency data without running the full pipeline.
        
        Args:
            top_n: Number of top cryptocurrencies to collect
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Updating cryptocurrency data...")
            
            # Collect data
            collected_data = self.collector.collect_all_data(top_n=top_n)
            
            if not collected_data:
                logger.error("Data update failed")
                return False
            
            logger.info(f"Updated data for {len(collected_data.get('top_cryptos', []))} cryptocurrencies")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data: {e}", exc_info=True)
            return False
    
    def check_alerts_only(self):
        """
        Check alerts without running the full pipeline.
        
        Returns:
            List of triggered alerts
        """
        try:
            logger.info("Checking alerts...")
            
            # Check alerts
            triggered_alerts = self.alert_system.check_all_alerts()
            
            if triggered_alerts:
                logger.info(f"Triggered {len(triggered_alerts)} alerts")
            else:
                logger.info("No alerts triggered")
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}", exc_info=True)
            return []
    
    def launch_dashboard(self):
        """
        Launch the interactive dashboard.
        
        Returns:
            Subprocess of the dashboard
        """
        try:
            logger.info("Launching interactive dashboard...")
            
            # Launch dashboard
            dashboard_process = subprocess.Popen(["streamlit", "run", "dashboard.py"])
            
            logger.info("Dashboard launched successfully")
            return dashboard_process
            
        except Exception as e:
            logger.error(f"Error launching dashboard: {e}", exc_info=True)
            return None
    
    def start_scheduler(self):
        """
        Start scheduled tasks.
        
        Returns:
            True if scheduler started successfully, False otherwise
        """
        if self.scheduler_running:
            logger.warning("Scheduler is already running")
            return False
        
        try:
            logger.info("Starting scheduler...")
            
            # Schedule data update every 1 hour
            schedule.every(1).hours.do(self.update_data_only)
            
            # Schedule full pipeline every 24 hours
            schedule.every(24).hours.do(self.run_full_pipeline)
            
            # Schedule alert check every 5 minutes
            schedule.every(5).minutes.do(self.check_alerts_only)
            
            # Start scheduler thread
            self.scheduler_thread = Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            self.scheduler_running = True
            
            logger.info("Scheduler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}", exc_info=True)
            return False
    
    def stop_scheduler(self):
        """
        Stop scheduled tasks.
        
        Returns:
            True if scheduler stopped successfully, False otherwise
        """
        if not self.scheduler_running:
            logger.warning("Scheduler is not running")
            return False
        
        try:
            logger.info("Stopping scheduler...")
            
            # Clear all scheduled jobs
            schedule.clear()
            
            # Set flag to stop thread
            self.scheduler_running = False
            
            logger.info("Scheduler stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}", exc_info=True)
            return False
    
    def _run_scheduler(self):
        """Run scheduler in a loop."""
        logger.info("Scheduler thread started")
        
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(1)
        
        logger.info("Scheduler thread stopped")
    
    def get_system_status(self):
        """
        Get the status of the cryptocurrency analyzer system.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Get general system status
            status = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'scheduler_running': self.scheduler_running,
                'data_available': os.path.exists(f"{self.data_dir}/top_cryptos.csv"),
                'analysis_available': os.path.exists(f"{self.analysis_dir}/crypto_rankings.csv"),
                'ml_predictions_available': os.path.exists(f"{self.analysis_dir}/ml/predictions/all_cryptocurrencies_predictions.csv"),
                'alert_system_configured': len(self.alert_system.email_settings) > 0,
                'active_alerts': len(self.alert_system.user_alerts)
            }
            
            # Get last execution summary if available
            summary_file = f"{self.analysis_dir}/execution_summary.json"
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    execution_summary = json.load(f)
                
                status['last_execution'] = execution_summary.get('execution_time')
                status['last_duration'] = execution_summary.get('duration_seconds')
                status['cryptocurrencies_analyzed'] = execution_summary.get('cryptocurrencies_analyzed')
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }


def main():
    """Main function to run the cryptocurrency analyzer integration."""
    print("\n" + "="*80)
    print("Cryptocurrency Analyzer Integration")
    print("="*80 + "\n")
    
    # Create integration
    integration = CryptoAnalyzerIntegration()
    
    # Menu loop
    while True:
        print("\nSelect an option:")
        print("1. Run full analysis pipeline")
        print("2. Update data only")
        print("3. Check alerts only")
        print("4. Launch dashboard")
        print("5. Start scheduler")
        print("6. Stop scheduler")
        print("7. Get system status")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-7): ")
        
        if choice == '1':
            print("\nRunning full analysis pipeline...")
            top_n = int(input("Number of top cryptocurrencies to analyze: ") or "20")
            success = integration.run_full_pipeline(top_n=top_n)
            if success:
                print("Analysis pipeline completed successfully!")
            else:
                print("Analysis pipeline failed. Check logs for details.")
        
        elif choice == '2':
            print("\nUpdating cryptocurrency data...")
            top_n = int(input("Number of top cryptocurrencies to collect: ") or "20")
            success = integration.update_data_only(top_n=top_n)
            if success:
                print("Data updated successfully!")
            else:
                print("Data update failed. Check logs for details.")
        
        elif choice == '3':
            print("\nChecking alerts...")
            triggered_alerts = integration.check_alerts_only()
            if triggered_alerts:
                print(f"Triggered {len(triggered_alerts)} alerts:")
                for alert in triggered_alerts:
                    print(f"  {alert['id']}: {alert['symbol']} {alert['type']} alert")
            else:
                print("No alerts triggered")
        
        elif choice == '4':
            print("\nLaunching dashboard...")
            dashboard_process = integration.launch_dashboard()
            if dashboard_process:
                print("Dashboard launched successfully!")
                print("Press Enter to continue (dashboard will keep running)...")
                input()
            else:
                print("Dashboard launch failed. Check logs for details.")
        
        elif choice == '5':
            print("\nStarting scheduler...")
            success = integration.start_scheduler()
            if success:
                print("Scheduler started successfully!")
            else:
                print("Scheduler start failed. Check logs for details.")
        
        elif choice == '6':
            print("\nStopping scheduler...")
            success = integration.stop_scheduler()
            if success:
                print("Scheduler stopped successfully!")
            else:
                print("Scheduler stop failed. Check logs for details.")
        
        elif choice == '7':
            print("\nGetting system status...")
            status = integration.get_system_status()
            print("\nSystem Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        elif choice == '0':
            print("\nExiting...")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
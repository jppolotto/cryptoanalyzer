"""
Machine Learning Analyzer for cryptocurrency price prediction.
Implements various ML models to identify trends and patterns.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
import traceback  # Para logar erros com mais detalhes

class CryptoMLAnalyzer:
    """Class for machine learning analysis and prediction of cryptocurrency prices."""
    
    def __init__(self, data_dir='data', analysis_dir='analysis'):
        """
        Initialize the ML analyzer.
        
        Args:
            data_dir: Directory containing cryptocurrency data
            analysis_dir: Directory for saving analysis results
        """
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        self.historical_data = {}
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
        # Create directories
        os.makedirs(f"{analysis_dir}/ml", exist_ok=True)
        os.makedirs(f"{analysis_dir}/ml/models", exist_ok=True)
        os.makedirs(f"{analysis_dir}/ml/predictions", exist_ok=True)
        os.makedirs(f"{analysis_dir}/ml/visualizations", exist_ok=True)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load cryptocurrency historical data from files."""
        try:
            # Load top cryptocurrencies
            top_cryptos_file = f"{self.data_dir}/top_cryptos.csv"
            if os.path.exists(top_cryptos_file):
                top_cryptos = pd.read_csv(top_cryptos_file)
                print(f"Loaded data for {len(top_cryptos)} cryptocurrencies")
                
                # Load historical data for each cryptocurrency
                for _, row in top_cryptos.iterrows():
                    symbol = row['symbol']
                    daily_file = f"{self.data_dir}/{symbol.lower()}_historical_daily.csv"
                    
                    if os.path.exists(daily_file):
                        self.historical_data[symbol] = pd.read_csv(daily_file)
                        # Convert timestamp to datetime
                        self.historical_data[symbol]['timestamp'] = pd.to_datetime(
                            self.historical_data[symbol]['timestamp']
                        )
                
                print(f"Loaded historical data for {len(self.historical_data)} cryptocurrencies")
            else:
                print("Top cryptocurrencies data not found")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            print(traceback.format_exc())
    
    def prepare_data(self, symbol, window_size=10, prediction_days=1):
        """
        Prepare data for machine learning model.
        
        Args:
            symbol: Cryptocurrency symbol
            window_size: Number of previous days to use as features
            prediction_days: Number of days to predict
            
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        if symbol not in self.historical_data:
            print(f"No historical data found for {symbol}")
            return None, None, None, None, None
        
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            if len(df) < window_size + prediction_days:
                print(f"Insufficient historical data for {symbol}")
                return None, None, None, None, None
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Create features and target
            data = df[['price', 'volume_24h', 'market_cap']].values
            
            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Store scaler for later use
            self.scalers[symbol] = scaler
            
            # Create feature set (sliding window)
            X, y = [], []
            
            for i in range(window_size, len(scaled_data) - prediction_days + 1):
                # Features: window_size days of price, volume, market_cap
                X.append(scaled_data[i - window_size:i])
                
                # Target: next prediction_days day of price (changed to support single day prediction)
                y.append(scaled_data[i, 0])  # Only predict next day price
            
            X, y = np.array(X), np.array(y)
            
            # Split data into training and testing sets (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            return X_train, X_test, y_train, y_test, scaler
            
        except Exception as e:
            print(f"Error preparing data for {symbol}: {e}")
            print(traceback.format_exc())
            return None, None, None, None, None
    
    def train_linear_regression(self, symbol, window_size=10, prediction_days=1):
        """
        Train a linear regression model for price prediction.
        
        Args:
            symbol: Cryptocurrency symbol
            window_size: Number of previous days to use as features
            prediction_days: Number of days to predict
            
        Returns:
            Trained model, evaluation metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_data(
            symbol, window_size, prediction_days
        )
        
        if X_train is None:
            return None, {}
        
        try:
            # Reshape input for linear regression
            n_samples_train, n_steps, n_features = X_train.shape
            n_samples_test, _, _ = X_test.shape
            
            X_train_reshaped = X_train.reshape(n_samples_train, n_steps * n_features)
            X_test_reshaped = X_test.reshape(n_samples_test, n_steps * n_features)
            
            # Create and train model
            model = LinearRegression()
            model.fit(X_train_reshaped, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_reshaped)
            
            # Evaluate model
            metrics = self._evaluate_model(y_test, y_pred, scaler)
            
            # Store model
            self.models[f"{symbol}_linear"] = model
            
            # Save model
            joblib.dump(model, f"{self.analysis_dir}/ml/models/{symbol.lower()}_linear.joblib")
            
            return model, metrics
            
        except Exception as e:
            print(f"Error training linear regression model for {symbol}: {e}")
            print(traceback.format_exc())
            return None, {}
    
    def train_random_forest(self, symbol, window_size=10, prediction_days=1):
        """
        Train a random forest regression model for price prediction.
        
        Args:
            symbol: Cryptocurrency symbol
            window_size: Number of previous days to use as features
            prediction_days: Number of days to predict
            
        Returns:
            Trained model, evaluation metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_data(
            symbol, window_size, prediction_days
        )
        
        if X_train is None:
            return None, {}
        
        try:
            # Reshape input for random forest
            n_samples_train, n_steps, n_features = X_train.shape
            n_samples_test, _, _ = X_test.shape
            
            X_train_reshaped = X_train.reshape(n_samples_train, n_steps * n_features)
            X_test_reshaped = X_test.reshape(n_samples_test, n_steps * n_features)
            
            # Create and train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_reshaped, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_reshaped)
            
            # Evaluate model
            metrics = self._evaluate_model(y_test, y_pred, scaler)
            
            # Store model
            self.models[f"{symbol}_rf"] = model
            
            # Save model
            joblib.dump(model, f"{self.analysis_dir}/ml/models/{symbol.lower()}_rf.joblib")
            
            return model, metrics
            
        except Exception as e:
            print(f"Error training random forest model for {symbol}: {e}")
            print(traceback.format_exc())
            return None, {}
    
    def train_gradient_boosting(self, symbol, window_size=10, prediction_days=1):
        """
        Train a gradient boosting regression model for price prediction.
        
        Args:
            symbol: Cryptocurrency symbol
            window_size: Number of previous days to use as features
            prediction_days: Number of days to predict
            
        Returns:
            Trained model, evaluation metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_data(
            symbol, window_size, prediction_days
        )
        
        if X_train is None:
            return None, {}
        
        try:
            # Reshape input for gradient boosting
            n_samples_train, n_steps, n_features = X_train.shape
            n_samples_test, _, _ = X_test.shape
            
            X_train_reshaped = X_train.reshape(n_samples_train, n_steps * n_features)
            X_test_reshaped = X_test.reshape(n_samples_test, n_steps * n_features)
            
            # Create and train model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_reshaped, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_reshaped)
            
            # Evaluate model
            metrics = self._evaluate_model(y_test, y_pred, scaler)
            
            # Store model
            self.models[f"{symbol}_gb"] = model
            
            # Save model
            joblib.dump(model, f"{self.analysis_dir}/ml/models/{symbol.lower()}_gb.joblib")
            
            return model, metrics
            
        except Exception as e:
            print(f"Error training gradient boosting model for {symbol}: {e}")
            print(traceback.format_exc())
            return None, {}
    
    def _evaluate_model(self, y_true, y_pred, scaler):
        """
        Evaluate model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            scaler: Scaler used to normalize data
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Calculate metrics for normalized data
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate Mean Absolute Percentage Error (MAPE)
            # Prevent division by zero
            mask = y_true != 0
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = 0
            
            # Prepare dummy data for inverse transform to calculate metrics in actual scale
            dummy_data = np.zeros((len(y_true), 3))
            
            # First column is price, rest are dummy values for volume and market_cap
            dummy_true = dummy_data.copy()
            dummy_true[:, 0] = y_true
            
            dummy_pred = dummy_data.copy()
            dummy_pred[:, 0] = y_pred
            
            # Inverse transform
            y_true_actual = scaler.inverse_transform(dummy_true)[:, 0]
            y_pred_actual = scaler.inverse_transform(dummy_pred)[:, 0]
            
            # Calculate metrics for actual data
            mse_actual = mean_squared_error(y_true_actual, y_pred_actual)
            rmse_actual = np.sqrt(mse_actual)
            mae_actual = mean_absolute_error(y_true_actual, y_pred_actual)
            
            # Prevent division by zero for MAPE calculation
            mask_actual = y_true_actual != 0
            if np.sum(mask_actual) > 0:
                mape_actual = np.mean(np.abs((y_true_actual[mask_actual] - y_pred_actual[mask_actual]) / y_true_actual[mask_actual])) * 100
            else:
                mape_actual = 0
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'mse_actual': mse_actual,
                'rmse_actual': rmse_actual,
                'mae_actual': mae_actual,
                'mape_actual': mape_actual
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            print(traceback.format_exc())
            return {
                'mse': 0,
                'rmse': 0,
                'mae': 0,
                'r2': 0,
                'mape': 0,
                'mse_actual': 0,
                'rmse_actual': 0,
                'mae_actual': 0,
                'mape_actual': 0
            }
    
    def predict_future_prices(self, symbol, days=30, model_type='rf'):
        """
        Predict future prices for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to predict into the future
            model_type: Type of model to use (linear, rf, gb)
            
        Returns:
            DataFrame with predicted prices
        """
        if symbol not in self.historical_data:
            print(f"No historical data found for {symbol}")
            return None
        
        # Check if model exists
        model_key = f"{symbol}_{model_type}"
        model_file = f"{self.analysis_dir}/ml/models/{symbol.lower()}_{model_type}.joblib"
        
        if model_key not in self.models and os.path.exists(model_file):
            try:
                self.models[model_key] = joblib.load(model_file)
            except Exception as e:
                print(f"Error loading model: {e}")
                print(traceback.format_exc())
        
        if model_key not in self.models:
            print(f"Model not found for {symbol}. Training new model...")
            
            if model_type == 'linear':
                self.train_linear_regression(symbol)
            elif model_type == 'rf':
                self.train_random_forest(symbol)
            elif model_type == 'gb':
                self.train_gradient_boosting(symbol)
            else:
                print(f"Invalid model type: {model_type}")
                return None
        
        try:
            # Check again if model was successfully trained
            if model_key not in self.models:
                print(f"Failed to train model for {symbol}")
                return None
                
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Get the last window_size days of data
            window_size = 10  # Same as used in training
            data = df[['price', 'volume_24h', 'market_cap']].values[-window_size:]
            
            # Normalize data
            if symbol not in self.scalers:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df[['price', 'volume_24h', 'market_cap']].values)
                self.scalers[symbol] = scaler
            else:
                scaler = self.scalers[symbol]
            
            scaled_data = scaler.transform(data)
            
            # Reshape data for prediction (flattened)
            X = scaled_data.reshape(1, -1)
            
            # Get model
            model = self.models[model_key]
            
            # Make predictions
            predicted_prices = []
            last_window = scaled_data.copy()
            
            for i in range(days):
                try:
                    # Predict next day (using flattened input)
                    next_day_scaled = float(model.predict(X)[0])  # Convert to float scalar
                    
                    # Create dummy data for inverse transform
                    dummy_pred = np.zeros((1, 3))
                    dummy_pred[0, 0] = next_day_scaled
                    
                    # Convert to actual price
                    next_day_price = float(scaler.inverse_transform(dummy_pred)[0, 0])  # Convert to float scalar
                    
                    # Add prediction to list
                    current_date = df['timestamp'].iloc[-1] + timedelta(days=i + 1)
                    predicted_prices.append({
                        'timestamp': current_date,
                        'predicted_price': next_day_price
                    })
                    
                    # Update input window for next prediction
                    # Shift window and add new prediction at the end
                    new_day_data = np.zeros((1, 3))
                    new_day_data[0, 0] = next_day_scaled
                    # Use last values for volume and market cap (we don't predict these)
                    new_day_data[0, 1:] = last_window[-1, 1:]
                    
                    # Update last_window by removing the first row and adding the new prediction
                    last_window = np.vstack([last_window[1:], new_day_data])
                    
                    # Reshape for next prediction
                    X = last_window.reshape(1, -1)
                    
                except Exception as e:
                    print(f"Error during day {i} prediction for {symbol}: {e}")
                    print(traceback.format_exc())
                    # Continue with next day if possible
                    continue
            
            # Check if we have any predictions
            if not predicted_prices:
                print(f"Failed to generate any predictions for {symbol}")
                return None
                
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame(predicted_prices)
            
            # Store predictions
            self.predictions[f"{symbol}_{model_type}"] = predictions_df
            
            # Save predictions
            predictions_df.to_csv(
                f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_{model_type}_predictions.csv",
                index=False
            )
            
            return predictions_df
            
        except Exception as e:
            print(f"Error predicting future prices for {symbol}: {e}")
            print(traceback.format_exc())
            return None
    
    def visualize_predictions(self, symbol, model_type='rf', days=30):
        """
        Visualize price predictions for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: Type of model to use (linear, rf, gb)
            days: Number of days to predict into the future
            
        Returns:
            Path to saved visualization
        """
        # Check if predictions exist
        pred_key = f"{symbol}_{model_type}"
        pred_file = f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_{model_type}_predictions.csv"
        
        if pred_key not in self.predictions and os.path.exists(pred_file):
            try:
                self.predictions[pred_key] = pd.read_csv(pred_file)
                self.predictions[pred_key]['timestamp'] = pd.to_datetime(
                    self.predictions[pred_key]['timestamp']
                )
            except Exception as e:
                print(f"Error loading predictions: {e}")
                print(traceback.format_exc())
        
        if pred_key not in self.predictions:
            print(f"Predictions not found for {symbol}. Generating predictions...")
            self.predict_future_prices(symbol, days, model_type)
        
        if pred_key not in self.predictions:
            print(f"Failed to generate predictions for {symbol}")
            return None
        
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Get predictions
            predictions_df = self.predictions[pred_key]
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Plot historical prices
            plt.plot(
                df['timestamp'].values,
                df['price'].values,
                label='Historical Price',
                color='blue'
            )
            
            # Plot predicted prices
            plt.plot(
                predictions_df['timestamp'].values,
                predictions_df['predicted_price'].values,
                label=f'Predicted Price ({model_type.upper()} Model)',
                color='red',
                linestyle='--'
            )
            
            # Add confidence interval (simple approximation)
            if len(predictions_df) > 0:
                # Calculate standard deviation of historical prices for last 30 days
                last_n = min(30, len(df))
                hist_std = df['price'].tail(last_n).std()
                
                # Create confidence interval (mean +/- 2*std)
                plt.fill_between(
                    predictions_df['timestamp'].values,
                    predictions_df['predicted_price'].values - 2 * hist_std,
                    predictions_df['predicted_price'].values + 2 * hist_std,
                    color='red',
                    alpha=0.2,
                    label='95% Confidence Interval'
                )
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title(f'{symbol} Price Prediction - {model_type.upper()} Model')
            plt.legend()
            plt.grid(True)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Set y-axis to start from 0 if lowest price is close to 0
            if df['price'].min() < df['price'].max() * 0.1:
                plt.ylim(bottom=0)
            
            # Save visualization
            vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_{model_type}_prediction.png"
            plt.savefig(vis_path, dpi=300)
            plt.close()
            
            return vis_path
            
        except Exception as e:
            print(f"Error visualizing predictions for {symbol}: {e}")
            print(traceback.format_exc())
            return None
    
    def compare_model_performances(self, symbol):
        """
        Compare performance of different models for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with model performance metrics
        """
        try:
            # Train models if they don't exist
            print(f"Training Linear Regression model for {symbol}...")
            _, linear_metrics = self.train_linear_regression(symbol)
            
            print(f"Training Random Forest model for {symbol}...")
            _, rf_metrics = self.train_random_forest(symbol)
            
            print(f"Training Gradient Boosting model for {symbol}...")
            _, gb_metrics = self.train_gradient_boosting(symbol)
            
            # Create comparison DataFrame with default values
            default_metrics = {
                'rmse_actual': 0,
                'mae_actual': 0,
                'mape_actual': 0,
                'r2': 0
            }
            
            # Use metrics if available, otherwise use defaults
            comparison_data = {
                'Linear Regression': linear_metrics if linear_metrics else default_metrics,
                'Random Forest': rf_metrics if rf_metrics else default_metrics,
                'Gradient Boosting': gb_metrics if gb_metrics else default_metrics
            }
            
            comparison = pd.DataFrame(comparison_data).T
            
            # Select key metrics (ensuring they exist)
            selected_metrics = ['rmse_actual', 'mae_actual', 'mape_actual', 'r2']
            for metric in selected_metrics:
                if metric not in comparison.columns:
                    comparison[metric] = 0
                    
            comparison = comparison[selected_metrics]
            
            # Rename columns for better readability
            comparison.columns = ['RMSE', 'MAE', 'MAPE (%)', 'R²']
            
            # Save comparison
            comparison.to_csv(
                f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_model_comparison.csv"
            )
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Bar chart for RMSE, MAE, MAPE
            ax1 = plt.subplot(121)
            comparison[['RMSE', 'MAE', 'MAPE (%)']].plot(kind='bar', ax=ax1)
            plt.title('Error Metrics by Model Type')
            plt.ylabel('Error Value')
            plt.grid(True, axis='y')
            
            # Bar chart for R²
            ax2 = plt.subplot(122)
            comparison[['R²']].plot(kind='bar', ax=ax2, color='green')
            plt.title('R² by Model Type')
            plt.ylabel('R² Value')
            plt.grid(True, axis='y')
            
            plt.suptitle(f'Model Performance Comparison for {symbol}')
            plt.tight_layout()
            
            # Save visualization
            vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_model_comparison.png"
            plt.savefig(vis_path, dpi=300)
            plt.close()
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing model performances for {symbol}: {e}")
            print(traceback.format_exc())
            return None
    
    def predict_price_trends(self, symbol, days=30):
        """
        Predict price trends for a cryptocurrency using ensemble of models.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to predict into the future
            
        Returns:
            Dictionary with trend prediction and confidence
        """
        try:
            print(f"Generating predictions for {symbol}...")
            
            # Generate predictions from all models (with error handling)
            linear_pred = None
            rf_pred = None
            gb_pred = None
            
            try:
                linear_pred = self.predict_future_prices(symbol, days, 'linear')
            except Exception as e:
                print(f"Error with linear model: {e}")
                
            try:
                rf_pred = self.predict_future_prices(symbol, days, 'rf')
            except Exception as e:
                print(f"Error with random forest model: {e}")
                
            try:
                gb_pred = self.predict_future_prices(symbol, days, 'gb')
            except Exception as e:
                print(f"Error with gradient boosting model: {e}")
            
            # Check if we have any successful predictions
            successful_models = []
            if linear_pred is not None:
                successful_models.append(('linear', linear_pred))
            if rf_pred is not None:
                successful_models.append(('rf', rf_pred))
            if gb_pred is not None:
                successful_models.append(('gb', gb_pred))
                
            if not successful_models:
                print(f"Failed to generate any predictions for {symbol}")
                return None
                
            # Use the random forest model as default if available
            best_model_type = 'rf' if rf_pred is not None else successful_models[0][0]
            best_model = 'Random Forest' if best_model_type == 'rf' else 'Linear Regression' if best_model_type == 'linear' else 'Gradient Boosting'
            best_pred = rf_pred if rf_pred is not None else successful_models[0][1]
            
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Get current price
            current_price = df['price'].iloc[-1]
            
            # Get predicted prices
            if len(best_pred) > 0:
                final_predicted_price = best_pred['predicted_price'].iloc[-1]
                
                # Calculate trend
                price_change = (final_predicted_price - current_price) / current_price * 100
                
                # Determine trend direction
                if price_change > 5:
                    trend = "Bullish"
                    strength = "Strong" if price_change > 15 else "Moderate"
                elif price_change > 1:
                    trend = "Bullish"
                    strength = "Weak"
                elif price_change < -5:
                    trend = "Bearish"
                    strength = "Strong" if price_change < -15 else "Moderate"
                elif price_change < -1:
                    trend = "Bearish"
                    strength = "Weak"
                else:
                    trend = "Neutral"
                    strength = "Neutral"
                
                # Calculate confidence based on available models
                agreement_count = 0
                for model_type, pred_df in successful_models:
                    if len(pred_df) > 0:
                        model_final_price = pred_df['predicted_price'].iloc[-1]
                        model_change = (model_final_price - current_price) / current_price * 100
                        
                        if trend == "Bullish" and model_change > 0:
                            agreement_count += 1
                        elif trend == "Bearish" and model_change < 0:
                            agreement_count += 1
                        elif trend == "Neutral" and abs(model_change) <= 1:
                            agreement_count += 1
                
                # Calculate confidence
                confidence = (agreement_count / len(successful_models)) * 100
                
                # Create trend prediction summary
                trend_prediction = {
                    'symbol': symbol,
                    'current_price': float(current_price),  # Ensure it's a regular float
                    'predicted_price': float(final_predicted_price),
                    'prediction_days': days,
                    'price_change_percent': float(price_change),
                    'trend': trend,
                    'trend_strength': strength,
                    'confidence': float(confidence),
                    'best_model': best_model,
                    'model_agreement': agreement_count,
                    'available_models': len(successful_models),
                    'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save trend prediction
                with open(f"{self.analysis_dir}/ml/predictions/{symbol.lower()}_trend_prediction.json", 'w') as f:
                    json.dump(trend_prediction, f, indent=4)
                
                # Create visualization comparing all models
                plt.figure(figsize=(14, 8))
                
                # Plot historical prices
                plt.plot(
                    df['timestamp'].values,
                    df['price'].values,
                    label='Historical Price',
                    color='blue'
                )
                
                # Plot predicted prices from available models
                for model_type, pred_df in successful_models:
                    if model_type == 'linear':
                        plt.plot(
                            pred_df['timestamp'].values,
                            pred_df['predicted_price'].values,
                            label='Linear Regression',
                            color='green',
                            linestyle='--'
                        )
                    elif model_type == 'rf':
                        plt.plot(
                            pred_df['timestamp'].values,
                            pred_df['predicted_price'].values,
                            label='Random Forest',
                            color='red',
                            linestyle='--'
                        )
                    elif model_type == 'gb':
                        plt.plot(
                            pred_df['timestamp'].values,
                            pred_df['predicted_price'].values,
                            label='Gradient Boosting',
                            color='purple',
                            linestyle='--'
                        )
                
                # Add labels and title
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                plt.title(f'{symbol} Price Predictions - Model Comparison')
                plt.legend()
                plt.grid(True)
                
                # Format x-axis dates
                plt.gcf().autofmt_xdate()
                
                # Save visualization
                vis_path = f"{self.analysis_dir}/ml/visualizations/{symbol.lower()}_all_models_comparison.png"
                plt.savefig(vis_path, dpi=300)
                plt.close()
                
                return trend_prediction
                
            else:
                print(f"No predictions available for {symbol}")
                return None
            
        except Exception as e:
            print(f"Error predicting price trends for {symbol}: {e}")
            print(traceback.format_exc())
            return None
    
    def analyze_all_cryptocurrencies(self, top_n=10, days=30):
        """
        Analyze all cryptocurrencies with ML models.
        
        Args:
            top_n: Number of top cryptocurrencies to analyze
            days: Number of days to predict into the future
            
        Returns:
            DataFrame with trend predictions for all cryptocurrencies
        """
        try:
            # Load top cryptocurrencies
            top_cryptos_file = f"{self.data_dir}/top_cryptos.csv"
            if os.path.exists(top_cryptos_file):
                top_cryptos = pd.read_csv(top_cryptos_file)
                
                # Limit to top N
                top_cryptos = top_cryptos.head(top_n)
                
                # Create results list
                results = []
                
                # Analyze each cryptocurrency
                for _, row in top_cryptos.iterrows():
                    symbol = row['symbol']
                    name = row['name']
                    
                    print(f"Analyzing {name} ({symbol})...")
                    
                    try:
                        # Predict trends
                        trend_prediction = self.predict_price_trends(symbol, days)
                        
                        if trend_prediction:
                            trend_prediction['name'] = name
                            results.append(trend_prediction)
                    except Exception as e:
                        print(f"Error analyzing {symbol}: {e}")
                        print(traceback.format_exc())
                        continue
                
                # Convert to DataFrame
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Save results
                    results_df.to_csv(
                        f"{self.analysis_dir}/ml/predictions/all_cryptocurrencies_predictions.csv",
                        index=False
                    )
                    
                    return results_df
                else:
                    print("No results generated")
                    return pd.DataFrame()  # Return empty DataFrame instead of None
            else:
                print("Top cryptocurrencies data not found")
                return pd.DataFrame()  # Return empty DataFrame instead of None
                
        except Exception as e:
            print(f"Error analyzing all cryptocurrencies: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()  # Return empty DataFrame instead of None


# Run analysis if script is executed directly
if __name__ == "__main__":
    analyzer = CryptoMLAnalyzer()
    
    # Analyze top 10 cryptocurrencies
    predictions = analyzer.analyze_all_cryptocurrencies(top_n=10, days=30)
    
    # Print trend predictions
    if predictions is not None and not predictions.empty:
        print("\nTrend Predictions:")
        for _, row in predictions.iterrows():
            print(f"{row['name']} ({row['symbol']}): {row['trend']} trend with {row['confidence']:.1f}% confidence. "
                  f"Predicted price change: {row['price_change_percent']:.2f}%")
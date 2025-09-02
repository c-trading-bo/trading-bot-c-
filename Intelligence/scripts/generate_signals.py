#!/usr/bin/env python3
"""
Signal Generator for Intelligence Pipeline
Generates trading signals from ML models and feature data
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        self.features_dir = "Intelligence/data/features"
        self.models_dir = "Intelligence/models"
        self.signals_dir = "Intelligence/data/signals"
        self.news_dir = "Intelligence/data/raw/news"
        
        # Create directories
        for dir_path in [self.models_dir, self.signals_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_news_analysis(self):
        """Load the latest news analysis"""
        try:
            news_file = os.path.join(self.news_dir, "latest_analysis.json")
            if not os.path.exists(news_file):
                logger.warning("No news analysis found, using defaults")
                return {
                    'sentiment': 'neutral',
                    'intensity': 25.0,
                    'confidence': 50.0,
                    'volatility_score': 2.0,
                    'fomc_detected': False,
                    'cpi_detected': False,
                    'earnings_season': False,
                    'fed_speak': False
                }
            
            with open(news_file, 'r') as f:
                news_data = json.load(f)
            
            logger.info(f"Loaded news analysis: {news_data['sentiment']} sentiment, "
                       f"{news_data['intensity']:.1f} intensity, {news_data['article_count']} articles")
            return news_data
            
        except Exception as e:
            logger.warning(f"Error loading news analysis: {e}, using defaults")
            return {
                'sentiment': 'neutral',
                'intensity': 25.0,
                'confidence': 50.0,
                'volatility_score': 2.0,
                'fomc_detected': False,
                'cpi_detected': False,
                'earnings_season': False,
                'fed_speak': False
            }
    
    def load_features(self):
        """Load the latest feature data"""
        try:
            features_file = os.path.join(self.features_dir, "latest_features.csv")
            if not os.path.exists(features_file):
                logger.warning("No feature data found")
                return None
            
            df = pd.read_csv(features_file)
            logger.info(f"Loaded features: {len(df)} observations with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        if df is None or df.empty:
            return None, None, None
        
        try:
            # Select relevant features for prediction
            feature_columns = [
                'regime_confidence', 'vix_level', 'trend_strength',
                'spx_change_pct', 'spx_volatility_5d', 'spx_volatility_20d',
                'news_intensity', 'news_sentiment_numeric', 'news_confidence',
                'news_article_count', 'is_cpi_day', 'is_fomc_day'
            ]
            
            # Keep only columns that exist
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) < 3:
                logger.warning(f"Insufficient features available: {available_features}")
                return None, None, None
            
            # Get the feature matrix
            X = df[available_features].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            # Convert boolean columns to numeric
            bool_columns = X.select_dtypes(include=['bool']).columns
            X[bool_columns] = X[bool_columns].astype(int)
            
            # Get the latest observation for prediction
            latest_features = X.iloc[-1:].copy()
            
            logger.info(f"Prepared features: {len(available_features)} features for {len(X)} observations")
            return X, latest_features, available_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None
    
    def create_simple_model(self, X, feature_names):
        """Create a simple model when no training data is available"""
        try:
            # Simple heuristic-based model
            latest = X.iloc[-1]
            
            # Market regime analysis
            vix_level = latest.get('vix_level', 20)
            spx_change = latest.get('spx_change_pct', 0)
            news_sentiment = latest.get('news_sentiment_numeric', 0)
            trend_strength = latest.get('trend_strength', 0)
            
            # Simple regime classification
            if vix_level > 25:
                regime = "Volatile"
                confidence = min(0.8, (vix_level - 20) / 20)
            elif vix_level < 15 and abs(spx_change) < 1:
                regime = "Ranging"
                confidence = 0.6
            else:
                regime = "Trending"
                confidence = 0.7
            
            # Bias determination
            bias_score = news_sentiment * 0.3 + (spx_change / 2) * 0.7
            
            if bias_score > 0.5:
                primary_bias = "Long"
            elif bias_score < -0.5:
                primary_bias = "Short"
            else:
                primary_bias = "Neutral"
            
            # Generate simple trade setups
            setups = []
            
            if regime == "Trending" and abs(spx_change) > 0.5:
                direction = "Long" if spx_change > 0 else "Short"
                setups.append({
                    "timeWindow": "Opening30Min",
                    "direction": direction,
                    "confidenceScore": confidence * 0.8,
                    "suggestedRiskMultiple": 1.2,
                    "rationale": f"Trending market with {spx_change:+.1f}% move"
                })
            
            if vix_level > 20 and news_sentiment != 0:
                direction = "Long" if news_sentiment > 0 else "Short"
                setups.append({
                    "timeWindow": "Afternoon",
                    "direction": direction,
                    "confidenceScore": confidence * 0.6,
                    "suggestedRiskMultiple": 0.8,
                    "rationale": f"Volatility play with {['negative', 'neutral', 'positive'][int(news_sentiment)+1]} news"
                })
            
            return {
                'regime': regime,
                'confidence': confidence,
                'primary_bias': primary_bias,
                'news_intensity': latest.get('news_intensity', 0),
                'setups': setups
            }
            
        except Exception as e:
            logger.error(f"Error creating simple model: {e}")
            return self.default_signals()
    
    def load_or_train_model(self, X, feature_names):
        """Load existing model or create simple heuristic model"""
        try:
            # Try to load existing model
            model_file = os.path.join(self.models_dir, "regime_model.joblib")
            scaler_file = os.path.join(self.models_dir, "feature_scaler.joblib")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                try:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    logger.info("Loaded existing ML model")
                    return model, scaler
                except Exception as e:
                    logger.warning(f"Failed to load existing model: {e}")
            
            # Create and train a simple model if we have enough data
            if len(X) >= 10:
                # Create synthetic labels based on future volatility (placeholder)
                # In a real implementation, you'd have historical trade outcomes
                returns = X['spx_change_pct'].shift(-1)  # Next day return
                
                # Simple labeling: positive return = 1, negative = 0
                y = (returns > 0).astype(int).fillna(0)
                
                # Remove the last row (no future return)
                X_train = X.iloc[:-1]
                y_train = y.iloc[:-1]
                
                if len(X_train) >= 5:
                    # Train simple model
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train)
                    
                    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
                    model.fit(X_scaled, y_train)
                    
                    # Save model
                    joblib.dump(model, model_file)
                    joblib.dump(scaler, scaler_file)
                    
                    logger.info(f"Trained new model with {len(X_train)} samples")
                    return model, scaler
            
            # Fallback to heuristic model
            logger.info("Using heuristic-based predictions")
            return None, None
            
        except Exception as e:
            logger.error(f"Error with model: {e}")
            return None, None
    
    def generate_predictions(self, model, scaler, latest_features, X, feature_names):
        """Generate predictions from the model"""
        try:
            if model is not None and scaler is not None:
                # ML-based predictions
                X_scaled = scaler.transform(latest_features)
                prediction_proba = model.predict_proba(X_scaled)[0]
                
                # Get regime from recent data
                latest = latest_features.iloc[0]
                vix_level = latest.get('vix_level', 20)
                
                if vix_level > 25:
                    regime = "Volatile"
                elif vix_level < 15:
                    regime = "Ranging"
                else:
                    regime = "Trending"
                
                confidence = max(prediction_proba)
                primary_bias = "Long" if prediction_proba[1] > 0.5 else "Short"
                
                return {
                    'regime': regime,
                    'confidence': float(confidence),
                    'primary_bias': primary_bias,
                    'model_prediction': float(prediction_proba[1]),
                    'news_intensity': float(latest.get('news_intensity', 0))
                }
            else:
                # Heuristic-based predictions
                return self.create_simple_model(latest_features, feature_names)
                
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return self.default_signals()
    
    def default_signals(self):
        """Generate default signals when everything else fails"""
        return {
            'regime': 'Ranging',
            'confidence': 0.3,
            'primary_bias': 'Neutral',
            'news_intensity': 0,
            'setups': []
        }
    
    def create_market_context(self, predictions, df):
        """Create the MarketContext object for the C# bot"""
        try:
            today = datetime.now()
            
            # Load news analysis
            news_data = self.load_news_analysis()
            
            # Get calendar information - prioritize news detection over data features
            is_cpi_day = news_data.get('cpi_detected', False)
            is_fomc_day = news_data.get('fomc_detected', False)
            
            # Fallback to data features if news doesn't detect events
            if df is not None and not df.empty:
                latest_row = df.iloc[-1]
                if not is_cpi_day:
                    is_cpi_day = bool(latest_row.get('is_cpi_day', False))
                if not is_fomc_day:
                    is_fomc_day = bool(latest_row.get('is_fomc_day', False))
            
            # Adjust regime based on news sentiment and volatility
            regime = predictions['regime']
            news_sentiment = news_data.get('sentiment', 'neutral')
            volatility_score = news_data.get('volatility_score', 2.0)
            
            # Override regime if news indicates high volatility or strong directional bias
            if volatility_score >= 7 or is_fomc_day or is_cpi_day:
                regime = "Volatile"
            elif news_sentiment == 'bullish' and news_data.get('confidence', 50) > 70:
                regime = "Trending"
            elif news_sentiment == 'bearish' and news_data.get('confidence', 50) > 70:
                regime = "Trending"
            
            # Adjust confidence based on news confidence
            base_confidence = predictions['confidence']
            news_confidence = news_data.get('confidence', 50) / 100.0
            
            # Blend model confidence with news confidence
            blended_confidence = (base_confidence * 0.6 + news_confidence * 0.4)
            
            # Adjust primary bias based on news sentiment
            primary_bias = predictions['primary_bias']
            if news_sentiment == 'bullish' and news_confidence > 0.7:
                primary_bias = "Long"
            elif news_sentiment == 'bearish' and news_confidence > 0.7:
                primary_bias = "Short"
            
            # Create trade setups
            setups = predictions.get('setups', [])
            
            # If no setups, create default ones based on regime and news
            if not setups:
                confidence = blended_confidence
                bias = primary_bias
                
                # Adjust risk multipliers based on volatility and events
                base_risk = 1.0
                if is_fomc_day or is_cpi_day:
                    base_risk = 0.5  # Reduce risk on major events
                elif volatility_score >= 6:
                    base_risk = 0.7  # Reduce risk in high volatility
                elif news_confidence > 0.8:
                    base_risk = 1.3  # Increase risk with high confidence
                
                if regime == "Trending" and bias != "Neutral":
                    setups.append({
                        "timeWindow": "Opening30Min",
                        "direction": bias,
                        "confidenceScore": confidence * 0.8,
                        "suggestedRiskMultiple": base_risk,
                        "rationale": f"Trending market with {bias.lower()} bias from news sentiment"
                    })
                
                if confidence > 0.6:
                    setups.append({
                        "timeWindow": "Afternoon",
                        "direction": bias if bias != "Neutral" else "Long",
                        "confidenceScore": confidence * 0.7,
                        "suggestedRiskMultiple": base_risk * 0.8,
                        "rationale": f"High confidence {regime.lower()} regime with {news_sentiment} sentiment"
                    })
                
                # Add volatility-based setups
                if regime == "Volatile" and confidence > 0.5:
                    setups.append({
                        "timeWindow": "EventWindow",
                        "direction": bias if bias != "Neutral" else "Long",
                        "confidenceScore": confidence * 0.6,
                        "suggestedRiskMultiple": 0.4,  # Small size for volatility
                        "rationale": f"Volatility trading on {regime.lower()} regime"
                    })
            
            market_context = {
                "date": today.strftime("%Y-%m-%d"),
                "regime": regime,
                "newsIntensity": float(news_data.get('intensity', 25.0)),
                "isCpiDay": is_cpi_day,
                "isFomcDay": is_fomc_day,
                "modelConfidence": float(blended_confidence),
                "primaryBias": primary_bias,
                "setups": setups,
                "newsDetails": {
                    "sentiment": news_sentiment,
                    "volatilityScore": volatility_score,
                    "articleCount": news_data.get('article_count', 0),
                    "newsConfidence": news_data.get('confidence', 50),
                    "earningsSeason": news_data.get('earnings_season', False),
                    "fedSpeak": news_data.get('fed_speak', False)
                }
            }
            
            return market_context
            
        except Exception as e:
            logger.error(f"Error creating market context: {e}")
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "regime": "Ranging",
                "newsIntensity": 0.0,
                "isCpiDay": False,
                "isFomcDay": False,
                "modelConfidence": 0.3,
                "primaryBias": "Neutral",
                "setups": []
            }
    
    def save_signals(self, market_context):
        """Save signals to files for bot consumption"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Add generation metadata
            market_context['generatedAt'] = datetime.now().isoformat()
            market_context['version'] = "1.0"
            
            # Save timestamped signal file
            signal_file = os.path.join(self.signals_dir, f"signals_{timestamp_str}.json")
            with open(signal_file, 'w') as f:
                json.dump(market_context, f, indent=2)
            
            # Save latest signal file for bot consumption
            latest_file = os.path.join(self.signals_dir, "latest.json")
            with open(latest_file, 'w') as f:
                json.dump(market_context, f, indent=2)
            
            logger.info(f"Saved signals: regime={market_context['regime']}, confidence={market_context['modelConfidence']:.2f}, setups={len(market_context['setups'])}")
            return signal_file
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting signal generation...")
    
    generator = SignalGenerator()
    
    # Load features
    df = generator.load_features()
    
    # Prepare features for ML
    X, latest_features, feature_names = generator.prepare_features(df)
    
    if X is None:
        logger.warning("No features available, generating default signals")
        predictions = generator.default_signals()
    else:
        # Load or train model
        model, scaler = generator.load_or_train_model(X, feature_names)
        
        # Generate predictions
        predictions = generator.generate_predictions(model, scaler, latest_features, X, feature_names)
    
    # Create market context
    market_context = generator.create_market_context(predictions, df)
    
    # Save signals
    signal_file = generator.save_signals(market_context)
    
    if signal_file:
        logger.info("Signal generation completed successfully")
    else:
        logger.error("Signal generation failed")

if __name__ == "__main__":
    main()
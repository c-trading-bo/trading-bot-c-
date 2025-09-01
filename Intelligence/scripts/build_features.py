#!/usr/bin/env python3
"""
Feature Builder for Intelligence Pipeline
Combines news, market data, and other sources into ML-ready features
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self):
        self.raw_data_dir = "Intelligence/data/raw"
        self.features_dir = "Intelligence/data/features"
        os.makedirs(self.features_dir, exist_ok=True)
    
    def load_news_data(self, days_back=7):
        """Load and aggregate news data from the last N days"""
        try:
            news_dir = os.path.join(self.raw_data_dir, "news")
            if not os.path.exists(news_dir):
                return pd.DataFrame()
            
            # Find news analysis files from last N days
            cutoff_date = datetime.now() - timedelta(days=days_back)
            analysis_files = glob.glob(os.path.join(news_dir, "analysis_*.json"))
            
            news_features = []
            
            for file_path in analysis_files:
                try:
                    # Extract date from filename
                    filename = os.path.basename(file_path)
                    date_str = filename.split('_')[1].split('.')[0]  # analysis_YYYYMMDD_HHMMSS.json
                    file_date = datetime.strptime(date_str[:8], "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        continue
                    
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract features from news analysis
                    feature_row = {
                        'date': file_date.strftime('%Y-%m-%d'),
                        'timestamp': data.get('timestamp', ''),
                        'news_intensity': data.get('intensity', 0),
                        'news_sentiment': data.get('sentiment', 'neutral'),
                        'news_confidence': data.get('confidence', 0),
                        'news_article_count': data.get('article_count', 0),
                        'news_avg_tone': data.get('avg_tone', 0)
                    }
                    
                    # Convert sentiment to numeric
                    sentiment_map = {'bearish': -1, 'neutral': 0, 'bullish': 1}
                    feature_row['news_sentiment_numeric'] = sentiment_map.get(feature_row['news_sentiment'], 0)
                    
                    news_features.append(feature_row)
                    
                except Exception as e:
                    logger.warning(f"Failed to process news file {file_path}: {e}")
                    continue
            
            if news_features:
                df = pd.DataFrame(news_features)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            return pd.DataFrame()
    
    def load_market_data(self, days_back=30):
        """Load and process market data from the last N days"""
        try:
            indices_dir = os.path.join(self.raw_data_dir, "indices")
            if not os.path.exists(indices_dir):
                return pd.DataFrame()
            
            # Find market data files from last N days
            cutoff_date = datetime.now() - timedelta(days=days_back)
            data_files = glob.glob(os.path.join(indices_dir, "market_data_*.json"))
            
            market_features = []
            
            for file_path in data_files:
                try:
                    # Extract date from filename
                    filename = os.path.basename(file_path)
                    date_str = filename.split('_')[2].split('.')[0]  # market_data_YYYYMMDD_HHMMSS.json
                    file_date = datetime.strptime(date_str[:8], "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        continue
                    
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract market features
                    market_data = data.get('market_data', {})
                    regime_analysis = data.get('regime_analysis', {})
                    calendar_info = data.get('economic_calendar', {})
                    
                    feature_row = {
                        'date': file_date.strftime('%Y-%m-%d'),
                        'timestamp': data.get('timestamp', ''),
                        
                        # Regime features
                        'regime': regime_analysis.get('regime', 'Unknown'),
                        'regime_confidence': regime_analysis.get('confidence', 0),
                        'vix_level': regime_analysis.get('vix_level', 0),
                        'trend_strength': regime_analysis.get('trend_strength', 0),
                        
                        # Calendar features
                        'is_cpi_day': calendar_info.get('is_cpi_day', False),
                        'is_fomc_day': calendar_info.get('is_fomc_day', False),
                    }
                    
                    # Add individual symbol features
                    for symbol, symbol_data in market_data.items():
                        prefix = f"{symbol.lower()}_"
                        feature_row.update({
                            f"{prefix}close": symbol_data.get('close', 0),
                            f"{prefix}change_pct": symbol_data.get('change_pct', 0),
                            f"{prefix}volatility_5d": symbol_data.get('volatility_5d', 0),
                            f"{prefix}volatility_20d": symbol_data.get('volatility_20d', 0),
                        })
                    
                    # Convert regime to numeric
                    regime_map = {'Ranging': 0, 'Trending': 1, 'Volatile': 2, 'Unknown': -1}
                    feature_row['regime_numeric'] = regime_map.get(feature_row['regime'], -1)
                    
                    market_features.append(feature_row)
                    
                except Exception as e:
                    logger.warning(f"Failed to process market file {file_path}: {e}")
                    continue
            
            if market_features:
                df = pd.DataFrame(market_features)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def create_technical_features(self, market_df):
        """Create technical analysis features from market data"""
        if market_df.empty or 'spx_close' not in market_df.columns:
            return market_df
        
        try:
            # Sort by date for proper time series calculation
            market_df = market_df.sort_values('date').copy()
            
            # Simple moving averages
            if len(market_df) >= 5:
                market_df['spx_sma_5'] = market_df['spx_close'].rolling(window=5, min_periods=1).mean()
                market_df['spx_above_sma_5'] = (market_df['spx_close'] > market_df['spx_sma_5']).astype(int)
            
            if len(market_df) >= 10:
                market_df['spx_sma_10'] = market_df['spx_close'].rolling(window=10, min_periods=1).mean()
                market_df['spx_above_sma_10'] = (market_df['spx_close'] > market_df['spx_sma_10']).astype(int)
            
            # Momentum features
            if len(market_df) >= 2:
                market_df['spx_momentum_1d'] = market_df['spx_change_pct']
            
            if len(market_df) >= 5:
                market_df['spx_momentum_5d'] = market_df['spx_close'].pct_change(periods=5) * 100
            
            # VIX features
            if 'vix_level' in market_df.columns:
                market_df['vix_regime'] = pd.cut(market_df['vix_level'], 
                                               bins=[0, 15, 25, 100], 
                                               labels=['low', 'medium', 'high'])
                market_df['vix_regime_numeric'] = market_df['vix_regime'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(-1)
            
            # Volatility spread
            if 'spx_volatility_5d' in market_df.columns and 'spx_volatility_20d' in market_df.columns:
                market_df['vol_spread'] = market_df['spx_volatility_5d'] - market_df['spx_volatility_20d']
            
            logger.info(f"Created technical features for {len(market_df)} market observations")
            return market_df
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return market_df
    
    def merge_features(self, news_df, market_df):
        """Merge news and market features by date"""
        try:
            if news_df.empty and market_df.empty:
                return pd.DataFrame()
            
            if news_df.empty:
                return market_df
            
            if market_df.empty:
                return news_df
            
            # Aggregate news by date (multiple news updates per day)
            news_daily = news_df.groupby('date').agg({
                'news_intensity': 'mean',
                'news_sentiment_numeric': 'mean',
                'news_confidence': 'mean',
                'news_article_count': 'sum',
                'news_avg_tone': 'mean'
            }).reset_index()
            
            # Market data should already be daily
            market_daily = market_df.drop_duplicates(subset=['date'], keep='last')
            
            # Merge on date
            combined = pd.merge(market_daily, news_daily, on='date', how='outer')
            
            # Fill missing news data with neutral values
            news_columns = ['news_intensity', 'news_sentiment_numeric', 'news_confidence', 'news_article_count', 'news_avg_tone']
            for col in news_columns:
                if col in combined.columns:
                    combined[col] = combined[col].fillna(0)
            
            # Sort by date
            combined = combined.sort_values('date')
            
            logger.info(f"Merged features: {len(combined)} days with {len(combined.columns)} features")
            return combined
            
        except Exception as e:
            logger.error(f"Error merging features: {e}")
            return pd.DataFrame()
    
    def save_features(self, features_df):
        """Save feature matrix to files"""
        if features_df.empty:
            logger.warning("No features to save")
            return None
        
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save timestamped CSV
            csv_file = os.path.join(self.features_dir, f"features_{timestamp_str}.csv")
            features_df.to_csv(csv_file, index=False)
            
            # Save latest CSV for model training
            latest_csv = os.path.join(self.features_dir, "latest_features.csv")
            features_df.to_csv(latest_csv, index=False)
            
            # Save feature metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'feature_count': len(features_df.columns),
                'observation_count': len(features_df),
                'date_range': {
                    'start': features_df['date'].min() if 'date' in features_df.columns else None,
                    'end': features_df['date'].max() if 'date' in features_df.columns else None
                },
                'columns': list(features_df.columns)
            }
            
            metadata_file = os.path.join(self.features_dir, f"metadata_{timestamp_str}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            latest_metadata = os.path.join(self.features_dir, "latest_metadata.json")
            with open(latest_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved features: {len(features_df)} observations with {len(features_df.columns)} features")
            return csv_file
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting feature building...")
    
    builder = FeatureBuilder()
    
    # Load raw data
    logger.info("Loading news data...")
    news_df = builder.load_news_data(days_back=30)
    
    logger.info("Loading market data...")
    market_df = builder.load_market_data(days_back=30)
    
    # Create technical features
    logger.info("Creating technical features...")
    market_df = builder.create_technical_features(market_df)
    
    # Merge all features
    logger.info("Merging features...")
    features_df = builder.merge_features(news_df, market_df)
    
    # Save feature matrix
    if not features_df.empty:
        builder.save_features(features_df)
        logger.info("Feature building completed successfully")
    else:
        logger.warning("No features generated - check data availability")

if __name__ == "__main__":
    main()
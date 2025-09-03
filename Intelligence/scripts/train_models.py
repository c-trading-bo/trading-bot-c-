#!/usr/bin/env python3
"""
Model Trainer for Intelligence Pipeline
Trains ML models using historical features and trade outcomes
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Suppress specific warnings that are not critical for trading operations
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.features_dir = "Intelligence/data/features"
        self.trades_dir = "Intelligence/data/trades"
        self.models_dir = "Intelligence/models"
        
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_training_data(self):
        """Load features and trade outcomes for training"""
        try:
            # Load features
            features_file = os.path.join(self.features_dir, "latest_features.csv")
            if not os.path.exists(features_file):
                logger.warning("No feature data found")
                return None, None
            
            features_df = pd.read_csv(features_file)
            
            # Load trade results if available
            trades_file = os.path.join(self.trades_dir, "results.jsonl")
            if os.path.exists(trades_file):
                trades_data = []
                with open(trades_file, 'r') as f:
                    for line in f:
                        try:
                            trades_data.append(json.loads(line.strip()))
                        except:
                            continue
                
                if trades_data:
                    trades_df = pd.DataFrame(trades_data)
                    trades_df['date'] = pd.to_datetime(trades_df['Timestamp']).dt.date.astype(str)
                    logger.info(f"Loaded {len(trades_df)} trade results")
                else:
                    trades_df = None
            else:
                trades_df = None
                logger.info("No trade results found, using synthetic labels")
            
            return features_df, trades_df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None, None
    
    def create_labels(self, features_df, trades_df):
        """Create training labels from trade outcomes or synthetic data"""
        try:
            if trades_df is not None and len(trades_df) > 10:
                # Use actual trade outcomes
                daily_pnl = trades_df.groupby('date')['PnL'].sum().reset_index()
                daily_pnl['profitable'] = (daily_pnl['PnL'] > 0).astype(int)
                
                # Merge with features
                merged = pd.merge(features_df, daily_pnl[['date', 'profitable']], on='date', how='left')
                merged['profitable'] = merged['profitable'].fillna(0)  # Assume no trade = no profit
                
                logger.info(f"Created labels from {len(daily_pnl)} trading days")
                return merged['profitable'].values
            
            else:
                # Create synthetic labels based on market patterns
                logger.info("Creating synthetic labels from market patterns")
                
                # Use next-day return as proxy
                if 'spx_change_pct' in features_df.columns:
                    next_day_returns = features_df['spx_change_pct'].shift(-1)
                    labels = (next_day_returns > 0).astype(int)
                    
                    # Add some noise based on other factors
                    if 'news_sentiment_numeric' in features_df.columns:
                        news_boost = features_df['news_sentiment_numeric'] * 0.1
                        labels = labels + (news_boost > 0.5).astype(int)
                        labels = np.clip(labels, 0, 1)
                    
                    # Remove last row (no future return)
                    return labels.iloc[:-1].values
                else:
                    # Fallback: random labels with slight bias
                    np.random.seed(42)
                    return np.random.choice([0, 1], size=len(features_df), p=[0.45, 0.55])
                    
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            np.random.seed(42)
            return np.random.choice([0, 1], size=len(features_df), p=[0.45, 0.55])
    
    def prepare_features(self, features_df):
        """Prepare features for training"""
        try:
            # Select numerical features for training
            feature_columns = [
                'regime_confidence', 'vix_level', 'trend_strength',
                'spx_change_pct', 'spx_volatility_5d', 'spx_volatility_20d',
                'news_intensity', 'news_sentiment_numeric', 'news_confidence',
                'news_article_count', 'is_cpi_day', 'is_fomc_day'
            ]
            
            # Add technical features if available
            tech_features = [col for col in features_df.columns if 'sma' in col or 'momentum' in col or 'vol_spread' in col]
            feature_columns.extend(tech_features)
            
            # Keep only available columns
            available_features = [col for col in feature_columns if col in features_df.columns]
            
            if len(available_features) < 3:
                logger.error(f"Insufficient features: {available_features}")
                return None, None
            
            X = features_df[available_features].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            # Convert boolean to numeric
            bool_columns = X.select_dtypes(include=['bool']).columns
            X[bool_columns] = X[bool_columns].astype(int)
            
            logger.info(f"Prepared {len(available_features)} features for training")
            return X.values, available_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None
    
    def train_models(self, X, y, feature_names):
        """Train multiple models and select the best one"""
        if X is None or len(X) < 10:
            logger.warning("Insufficient data for training")
            return None, None, None
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define models to try
            models = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
                'gradient_boost': GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
            }
            
            best_model = None
            best_score = 0
            best_name = None
            
            results = {}
            
            for name, model in models.items():
                try:
                    # Train model
                    if name == 'logistic':
                        model.fit(X_train_scaled, y_train)
                        predictions = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    # Evaluate
                    accuracy = accuracy_score(y_test, predictions)
                    
                    # Cross-validation score
                    if name == 'logistic':
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2))
                    
                    cv_mean = cv_scores.mean()
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'cv_score': cv_mean,
                        'cv_std': cv_scores.std()
                    }
                    
                    logger.info(f"{name}: accuracy={accuracy:.3f}, cv_score={cv_mean:.3f}±{cv_scores.std():.3f}")
                    
                    # Select best model based on CV score
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_model = model
                        best_name = name
                
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
                    continue
            
            if best_model is not None:
                logger.info(f"Best model: {best_name} with CV score {best_score:.3f}")
                
                # Feature importance (if available)
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    logger.info("Top 5 most important features:")
                    for _, row in importance_df.head().iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.3f}")
                
                return best_model, scaler, results
            
            else:
                logger.error("No models trained successfully")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return None, None, None
    
    def save_models(self, model, scaler, feature_names, results):
        """Save trained models and metadata"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model and scaler
            model_file = os.path.join(self.models_dir, "regime_model.joblib")
            scaler_file = os.path.join(self.models_dir, "feature_scaler.joblib")
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            
            # Save versioned copies
            versioned_model = os.path.join(self.models_dir, f"regime_model_{timestamp_str}.joblib")
            versioned_scaler = os.path.join(self.models_dir, f"feature_scaler_{timestamp_str}.joblib")
            
            joblib.dump(model, versioned_model)
            joblib.dump(scaler, versioned_scaler)
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'feature_count': len(feature_names),
                'features': feature_names,
                'training_results': results,
                'model_file': model_file,
                'scaler_file': scaler_file
            }
            
            metadata_file = os.path.join(self.models_dir, "model_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            versioned_metadata = os.path.join(self.models_dir, f"model_metadata_{timestamp_str}.json")
            with open(versioned_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved models and metadata to {self.models_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False

def main():
    """Main execution function"""
    logger.info("Starting model training...")
    
    trainer = ModelTrainer()
    
    # Load training data
    features_df, trades_df = trainer.load_training_data()
    
    if features_df is None:
        logger.error("No features available for training")
        return
    
    # Create labels
    y = trainer.create_labels(features_df, trades_df)
    
    # Prepare features
    X, feature_names = trainer.prepare_features(features_df)
    
    if X is None:
        logger.error("Failed to prepare features")
        return
    
    # Ensure we have the same number of labels as features
    if len(y) > len(X):
        y = y[:len(X)]
    elif len(y) < len(X):
        X = X[:len(y)]
    
    # Train models
    model, scaler, results = trainer.train_models(X, y, feature_names)
    
    if model is not None:
        # Save models
        if trainer.save_models(model, scaler, feature_names, results):
            logger.info("Model training completed successfully")
        else:
            logger.error("Failed to save models")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()
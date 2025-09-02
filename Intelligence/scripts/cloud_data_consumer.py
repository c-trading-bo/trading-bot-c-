#!/usr/bin/env python3
"""
Cloud Data Consumer for Local Bot Integration
Consumes data collected by GitHub Actions cloud workflows for local trading bot use
"""

import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudDataConsumer:
    """Consumes cloud-collected data for local bot usage"""
    
    def __init__(self, intelligence_dir="Intelligence"):
        self.intelligence_dir = Path(intelligence_dir)
        self.data_dir = self.intelligence_dir / "data"
        
        # Ensure data directories exist
        self.options_dir = self.data_dir / "options" / "flow"
        self.macro_dir = self.data_dir / "macro" 
        self.news_dir = self.data_dir / "news"
        
        for dir_path in [self.options_dir, self.macro_dir, self.news_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_latest_options_data(self):
        """Get latest enhanced options and gamma data"""
        try:
            latest_file = self.options_dir / "latest_enhanced_options.json"
            
            if not latest_file.exists():
                logger.warning("No latest options data found")
                return None
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics for trading decisions
            market_summary = data.get('market_summary', {})
            
            trading_signals = {
                'timestamp': data.get('timestamp'),
                'market_put_call_ratio': market_summary.get('market_put_call_ratio', 0),
                'total_call_volume': market_summary.get('total_call_volume', 0),
                'total_put_volume': market_summary.get('total_put_volume', 0),
                'symbols': {}
            }
            
            # Extract key data for SPY/QQQ (main trading instruments)
            for symbol in ['SPY', 'QQQ']:
                if symbol in data.get('symbols', {}):
                    symbol_data = data['symbols'][symbol]
                    trading_signals['symbols'][symbol] = {
                        'current_price': symbol_data.get('current_price'),
                        'price_change_pct': symbol_data.get('price_change_pct'),
                        'put_call_ratio': symbol_data.get('summary', {}).get('avg_put_call_ratio', 0),
                        'net_gamma_exposure': symbol_data.get('summary', {}).get('total_net_gamma_exposure', 0),
                        'gamma_flip_level': symbol_data.get('summary', {}).get('gamma_flip_level', 0)
                    }
            
            logger.info(f"✅ Latest options data loaded - P/C Ratio: {trading_signals['market_put_call_ratio']:.3f}")
            return trading_signals
            
        except Exception as e:
            logger.error(f"❌ Error loading options data: {e}")
            return None
    
    def get_latest_macro_data(self):
        """Get latest macro economic data"""
        try:
            latest_file = self.macro_dir / "latest_macro_data.json"
            
            if not latest_file.exists():
                logger.warning("No latest macro data found")
                return None
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics for trading decisions
            trading_signals = {
                'timestamp': data.get('timestamp'),
                'treasury_yields': {},
                'dollar_strength': {},
                'volatility': {},
                'risk_sentiment': {}
            }
            
            # Treasury yield curve signals
            yields = data.get('treasury_yields', {})
            if '10_year' in yields and '2_year' in yields:
                ten_year = yields.get('10_year', {}).get('current', 0)
                # Approximate 2-year from 3-month if 2-year not available
                two_year = yields.get('2_year', {}).get('current', yields.get('3_month', {}).get('current', 0))
                
                trading_signals['treasury_yields'] = {
                    '10_year': ten_year,
                    '2_year': two_year,
                    'yield_curve_spread': ten_year - two_year,
                    'curve_inverted': ten_year < two_year
                }
            
            # Dollar strength
            currencies = data.get('currencies', {})
            if 'dxy' in currencies:
                dxy_data = currencies['dxy']
                trading_signals['dollar_strength'] = {
                    'dxy_level': dxy_data.get('current', 0),
                    'dxy_change_pct': dxy_data.get('change_pct', 0),
                    'dollar_strengthening': dxy_data.get('change_pct', 0) > 0
                }
            
            # Volatility regime
            vol_indices = data.get('volatility_indices', {})
            if 'vix' in vol_indices:
                vix_data = vol_indices['vix']
                vix_level = vix_data.get('current', 20)
                
                vol_regime = "Low"
                if vix_level > 30:
                    vol_regime = "High"
                elif vix_level > 20:
                    vol_regime = "Medium"
                
                trading_signals['volatility'] = {
                    'vix_level': vix_level,
                    'vix_change': vix_data.get('change', 0),
                    'volatility_regime': vol_regime,
                    'fear_spike': vix_level > 25 and vix_data.get('change', 0) > 2
                }
            
            # Risk sentiment
            sentiment = data.get('sentiment_indicators', {})
            if 'fear_greed_index' in sentiment:
                fg_data = sentiment['fear_greed_index']
                trading_signals['risk_sentiment'] = {
                    'fear_greed_score': fg_data.get('score', 50),
                    'sentiment_label': fg_data.get('label', 'Neutral'),
                    'extreme_fear': fg_data.get('score', 50) < 25,
                    'extreme_greed': fg_data.get('score', 50) > 75
                }
            
            logger.info(f"✅ Latest macro data loaded - VIX: {trading_signals.get('volatility', {}).get('vix_level', 'N/A')}")
            return trading_signals
            
        except Exception as e:
            logger.error(f"❌ Error loading macro data: {e}")
            return None
    
    def get_latest_news_sentiment(self):
        """Get latest news and sentiment analysis"""
        try:
            latest_file = self.news_dir / "latest_news_sentiment.json"
            
            if not latest_file.exists():
                logger.warning("No latest news sentiment data found")
                return None
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Extract key sentiment signals
            trading_signals = {
                'timestamp': data.get('timestamp'),
                'sentiment': {},
                'trending_topics': {},
                'news_impact': {}
            }
            
            # Overall market sentiment
            sentiment = data.get('sentiment_analysis', {})
            trading_signals['sentiment'] = {
                'polarity': sentiment.get('overall_polarity', 0),
                'label': sentiment.get('sentiment_label', 'Neutral'),
                'confidence': sentiment.get('confidence', 0.5),
                'bullish': sentiment.get('overall_polarity', 0) > 0.1,
                'bearish': sentiment.get('overall_polarity', 0) < -0.1
            }
            
            # Trending keywords analysis
            keywords = data.get('trending_topics', {}).get('keywords', {})
            
            # Categorize keywords by impact
            high_impact_words = ['fed', 'federal reserve', 'fomc', 'powell', 'inflation', 'cpi', 'recession']
            medium_impact_words = ['earnings', 'revenue', 'unemployment', 'jobs', 'gdp']
            
            high_impact_mentions = sum(keywords.get(word, 0) for word in high_impact_words)
            medium_impact_mentions = sum(keywords.get(word, 0) for word in medium_impact_words)
            
            trading_signals['trending_topics'] = {
                'top_keywords': list(keywords.keys())[:5],
                'high_impact_mentions': high_impact_mentions,
                'medium_impact_mentions': medium_impact_mentions,
                'fed_focus': keywords.get('fed', 0) + keywords.get('federal reserve', 0) + keywords.get('fomc', 0)
            }
            
            # News impact assessment
            total_headlines = data.get('news_summary', {}).get('total_headlines', 0)
            trading_signals['news_impact'] = {
                'headlines_count': total_headlines,
                'high_impact_ratio': high_impact_mentions / max(total_headlines, 1),
                'significant_news': high_impact_mentions > 3 or total_headlines > 20
            }
            
            logger.info(f"✅ Latest news sentiment loaded - Sentiment: {trading_signals['sentiment']['label']}")
            return trading_signals
            
        except Exception as e:
            logger.error(f"❌ Error loading news sentiment: {e}")
            return None
    
    def get_comprehensive_market_signals(self):
        """Get all cloud data combined into comprehensive trading signals"""
        try:
            options_data = self.get_latest_options_data()
            macro_data = self.get_latest_macro_data()
            news_data = self.get_latest_news_sentiment()
            
            # Combine into comprehensive signals
            signals = {
                'timestamp': datetime.now().isoformat(),
                'data_availability': {
                    'options': options_data is not None,
                    'macro': macro_data is not None,
                    'news': news_data is not None
                },
                'risk_regime': 'UNKNOWN',
                'market_bias': 'NEUTRAL',
                'volatility_regime': 'MEDIUM',
                'position_sizing_factor': 1.0,
                'entry_confidence': 0.5
            }
            
            confidence_factors = []
            
            # Risk regime assessment
            risk_factors = []
            
            if macro_data:
                # VIX regime
                vix_level = macro_data.get('volatility', {}).get('vix_level', 20)
                if vix_level > 25:
                    risk_factors.append('HIGH_VOL')
                elif vix_level < 15:
                    risk_factors.append('LOW_VOL')
                
                # Yield curve
                if macro_data.get('treasury_yields', {}).get('curve_inverted', False):
                    risk_factors.append('INVERTED_CURVE')
                
                # Dollar strength
                if macro_data.get('dollar_strength', {}).get('dxy_change_pct', 0) > 1:
                    risk_factors.append('STRONG_DOLLAR')
            
            if options_data:
                # Put/Call ratio
                pcr = options_data.get('market_put_call_ratio', 1)
                if pcr > 1.2:
                    risk_factors.append('HIGH_PUT_DEMAND')
                elif pcr < 0.8:
                    risk_factors.append('HIGH_CALL_DEMAND')
            
            if news_data:
                # Sentiment extremes
                if news_data.get('sentiment', {}).get('bearish', False):
                    risk_factors.append('NEGATIVE_SENTIMENT')
                elif news_data.get('sentiment', {}).get('bullish', False):
                    risk_factors.append('POSITIVE_SENTIMENT')
            
            # Determine overall risk regime
            high_risk_count = len([f for f in risk_factors if f in ['HIGH_VOL', 'INVERTED_CURVE', 'HIGH_PUT_DEMAND', 'NEGATIVE_SENTIMENT']])
            
            if high_risk_count >= 2:
                signals['risk_regime'] = 'HIGH_RISK'
                signals['position_sizing_factor'] = 0.5
            elif high_risk_count == 0:
                signals['risk_regime'] = 'LOW_RISK'
                signals['position_sizing_factor'] = 1.2
            else:
                signals['risk_regime'] = 'MEDIUM_RISK'
                signals['position_sizing_factor'] = 1.0
            
            # Market bias
            bullish_factors = len([f for f in risk_factors if f in ['LOW_VOL', 'HIGH_CALL_DEMAND', 'POSITIVE_SENTIMENT']])
            bearish_factors = len([f for f in risk_factors if f in ['HIGH_VOL', 'HIGH_PUT_DEMAND', 'NEGATIVE_SENTIMENT', 'INVERTED_CURVE']])
            
            if bullish_factors > bearish_factors:
                signals['market_bias'] = 'BULLISH'
            elif bearish_factors > bullish_factors:
                signals['market_bias'] = 'BEARISH'
            else:
                signals['market_bias'] = 'NEUTRAL'
            
            # Entry confidence based on data quality and agreement
            base_confidence = 0.3
            if signals['data_availability']['options']:
                base_confidence += 0.2
            if signals['data_availability']['macro']:
                base_confidence += 0.3
            if signals['data_availability']['news']:
                base_confidence += 0.2
            
            signals['entry_confidence'] = min(1.0, base_confidence)
            
            # Add raw data for detailed analysis
            signals['raw_data'] = {
                'options': options_data,
                'macro': macro_data,  
                'news': news_data
            }
            
            signals['risk_factors'] = risk_factors
            
            logger.info(f"✅ Comprehensive signals generated - Regime: {signals['risk_regime']}, Bias: {signals['market_bias']}")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive signals: {e}")
            return None
    
    def is_data_fresh(self, max_age_minutes=30):
        """Check if cloud data is fresh enough for trading decisions"""
        try:
            files_to_check = [
                self.options_dir / "latest_enhanced_options.json",
                self.macro_dir / "latest_macro_data.json",
                self.news_dir / "latest_news_sentiment.json"
            ]
            
            current_time = datetime.now()
            
            for file_path in files_to_check:
                if not file_path.exists():
                    logger.warning(f"Missing data file: {file_path}")
                    continue
                
                # Check file modification time
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                age_minutes = (current_time - file_time).total_seconds() / 60
                
                if age_minutes > max_age_minutes:
                    logger.warning(f"Stale data: {file_path} is {age_minutes:.1f} minutes old")
                    return False
            
            logger.info(f"✅ All cloud data is fresh (< {max_age_minutes} minutes old)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking data freshness: {e}")
            return False

def main():
    """Demo usage of CloudDataConsumer"""
    consumer = CloudDataConsumer()
    
    print("🚀 Cloud Data Consumer - Local Bot Integration")
    print("=" * 50)
    
    # Check data freshness
    is_fresh = consumer.is_data_fresh(max_age_minutes=60)
    print(f"📊 Data Freshness: {'✅ Fresh' if is_fresh else '⚠️ Stale'}")
    
    # Get comprehensive signals
    signals = consumer.get_comprehensive_market_signals()
    
    if signals:
        print(f"\n🎯 TRADING SIGNALS SUMMARY")
        print(f"Risk Regime: {signals['risk_regime']}")
        print(f"Market Bias: {signals['market_bias']}")
        print(f"Position Sizing Factor: {signals['position_sizing_factor']:.2f}")
        print(f"Entry Confidence: {signals['entry_confidence']:.2f}")
        print(f"Risk Factors: {', '.join(signals['risk_factors'])}")
        
        # Data availability
        avail = signals['data_availability']
        print(f"\n📊 Data Sources: Options={avail['options']}, Macro={avail['macro']}, News={avail['news']}")
        
        # Save combined signals for C# bot consumption
        output_file = consumer.data_dir / "combined_trading_signals.json"
        with open(output_file, 'w') as f:
            json.dump(signals, f, indent=2)
        
        print(f"💾 Combined signals saved to: {output_file}")
        
    else:
        print("❌ Failed to generate comprehensive trading signals")

if __name__ == "__main__":
    main()
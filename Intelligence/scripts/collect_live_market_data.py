#!/usr/bin/env python3
"""
Enhanced Live Market Data Collection
Real-time market intelligence with news integration and advanced analytics
"""

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from dataclasses import dataclass
import websockets
import ssl

@dataclass
class MarketSnapshot:
    """Real-time market data snapshot"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    volatility: float
    momentum_1m: float
    momentum_5m: float
    rsi: float
    volume_profile: Dict[str, float]
    news_sentiment: float
    
class EnhancedMarketDataCollector:
    """
    Advanced market data collection with real-time processing
    Integrates price data, volume analysis, sentiment, and microstructure
    """
    
    def __init__(self, data_dir: str = "Intelligence/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "market"
        self.features_dir = self.data_dir / "features"
        self.signals_dir = self.data_dir / "signals"
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.features_dir, self.signals_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger = self._setup_logger()
        
        # Data storage
        self.price_history = {}
        self.volume_history = {}
        self.sentiment_cache = {}
        self.live_data = {}
        
        # Market hours
        self.market_open = 9.5  # 9:30 AM
        self.market_close = 16.0  # 4:00 PM
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("EnhancedMarketData")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (EST)"""
        now = datetime.now()
        weekday = now.weekday()  # Monday = 0, Sunday = 6
        
        # Skip weekends
        if weekday >= 5:
            return False
            
        current_hour = now.hour + now.minute / 60.0
        return self.market_open <= current_hour <= self.market_close
    
    async def collect_symbol_data(self, symbol: str) -> Optional[MarketSnapshot]:
        """Collect comprehensive data for a single symbol"""
        try:
            # Get current market data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                self.logger.warning(f"No history data for {symbol}")
                return None
                
            current_data = hist.iloc[-1]
            
            # Calculate technical indicators
            prices = hist['Close'].values
            volumes = hist['Volume'].values
            
            # Momentum calculations
            momentum_1m = self._calculate_momentum(prices, 1)
            momentum_5m = self._calculate_momentum(prices, 5)
            
            # RSI calculation
            rsi = self._calculate_rsi(prices, 14)
            
            # Volatility (rolling standard deviation)
            volatility = np.std(prices[-20:]) if len(prices) >= 20 else 0.0
            
            # Volume profile
            volume_profile = self._calculate_volume_profile(hist)
            
            # Get sentiment data
            sentiment = await self._get_news_sentiment(symbol)
            
            # Bid/Ask spread (simulated for historical data)
            spread = current_data['High'] - current_data['Low']
            bid = current_data['Close'] - spread / 2
            ask = current_data['Close'] + spread / 2
            
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                price=float(current_data['Close']),
                volume=int(current_data['Volume']),
                bid=float(bid),
                ask=float(ask),
                spread=float(spread),
                volatility=float(volatility),
                momentum_1m=float(momentum_1m),
                momentum_5m=float(momentum_5m),
                rsi=float(rsi),
                volume_profile=volume_profile,
                news_sentiment=sentiment
            )
            
            self.logger.debug(f"Collected data for {symbol}: ${snapshot.price:.2f}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            return None
    
    def _calculate_momentum(self, prices: np.ndarray, periods: int) -> float:
        """Calculate price momentum over specified periods"""
        if len(prices) < periods + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-(periods + 1)]
        
        return (current_price - past_price) / past_price * 100
    
    def _calculate_rsi(self, prices: np.ndarray, periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < periods + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_volume_profile(self, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-weighted price levels"""
        try:
            # Create price buckets
            price_min = hist['Low'].min()
            price_max = hist['High'].max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return {"poc": float(hist['Close'].iloc[-1])}
            
            # Simple volume profile calculation
            total_volume = hist['Volume'].sum()
            vwap = (hist['Close'] * hist['Volume']).sum() / total_volume if total_volume > 0 else hist['Close'].mean()
            
            # Point of Control (highest volume price level)
            poc = vwap  # Simplified
            
            return {
                "poc": float(poc),
                "vwap": float(vwap),
                "volume_ratio": float(hist['Volume'].iloc[-1] / hist['Volume'].mean()) if hist['Volume'].mean() > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {"poc": 0.0, "vwap": 0.0, "volume_ratio": 1.0}
    
    async def _get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment for symbol (cached)"""
        # Check cache first
        if symbol in self.sentiment_cache:
            cache_time, sentiment = self.sentiment_cache[symbol]
            if datetime.now() - cache_time < timedelta(minutes=15):
                return sentiment
        
        try:
            # Read from news analysis if available
            news_file = self.data_dir / "news" / "latest.json"
            if news_file.exists():
                with open(news_file, 'r') as f:
                    news_data = json.load(f)
                    
                # Look for symbol-related sentiment
                sentiment = 0.0
                count = 0
                
                for article in news_data.get('articles', []):
                    if symbol.replace('=F', '').upper() in article.get('title', '').upper():
                        article_sentiment = article.get('sentiment', 0.0)
                        sentiment += article_sentiment
                        count += 1
                
                final_sentiment = sentiment / count if count > 0 else 0.0
                
                # Cache result
                self.sentiment_cache[symbol] = (datetime.now(), final_sentiment)
                return final_sentiment
            
        except Exception as e:
            self.logger.debug(f"Error getting sentiment for {symbol}: {e}")
        
        return 0.0  # Neutral sentiment as fallback
    
    async def collect_multi_symbol_data(self, symbols: List[str]) -> Dict[str, MarketSnapshot]:
        """Collect data for multiple symbols concurrently"""
        tasks = [self.collect_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, MarketSnapshot):
                data[symbol] = result
            elif isinstance(result, Exception):
                self.logger.error(f"Error collecting {symbol}: {result}")
        
        return data
    
    async def generate_market_features(self, snapshots: Dict[str, MarketSnapshot]) -> Dict:
        """Generate ML-ready features from market snapshots"""
        try:
            features = {
                "timestamp": datetime.now().isoformat(),
                "market_regime": self._detect_market_regime(snapshots),
                "cross_asset_correlation": self._calculate_correlations(snapshots),
                "volatility_regime": self._classify_volatility(snapshots),
                "volume_regime": self._classify_volume(snapshots),
                "sentiment_aggregate": self._aggregate_sentiment(snapshots),
                "features": {}
            }
            
            # Individual symbol features
            for symbol, snapshot in snapshots.items():
                symbol_features = {
                    "price": snapshot.price,
                    "volume_ratio": snapshot.volume_profile.get("volume_ratio", 1.0),
                    "spread_bps": (snapshot.spread / snapshot.price) * 10000,
                    "momentum_1m": snapshot.momentum_1m,
                    "momentum_5m": snapshot.momentum_5m,
                    "rsi": snapshot.rsi,
                    "volatility": snapshot.volatility,
                    "sentiment": snapshot.news_sentiment,
                    "relative_strength": self._calculate_relative_strength(symbol, snapshots)
                }
                features["features"][symbol] = symbol_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}
    
    def _detect_market_regime(self, snapshots: Dict[str, MarketSnapshot]) -> str:
        """Detect current market regime"""
        try:
            # Use ES (S&P 500) as primary indicator
            es_data = snapshots.get("ES=F")
            if not es_data:
                return "unknown"
            
            # Simple regime classification
            if es_data.momentum_5m > 0.1 and es_data.volatility < 0.02:
                return "trending_up"
            elif es_data.momentum_5m < -0.1 and es_data.volatility < 0.02:
                return "trending_down"
            elif es_data.volatility > 0.03:
                return "high_volatility"
            else:
                return "ranging"
                
        except Exception:
            return "unknown"
    
    def _calculate_correlations(self, snapshots: Dict[str, MarketSnapshot]) -> Dict[str, float]:
        """Calculate cross-asset correlations"""
        try:
            correlations = {}
            
            # Get momentum values
            momentums = {symbol: data.momentum_5m for symbol, data in snapshots.items()}
            
            # Calculate simple correlation proxy (same direction = positive correlation)
            if "ES=F" in momentums and "NQ=F" in momentums:
                es_mom = momentums["ES=F"]
                nq_mom = momentums["NQ=F"]
                
                # Simple correlation indicator
                if (es_mom > 0 and nq_mom > 0) or (es_mom < 0 and nq_mom < 0):
                    correlations["ES_NQ"] = 0.8
                else:
                    correlations["ES_NQ"] = -0.2
            
            return correlations
            
        except Exception:
            return {}
    
    def _classify_volatility(self, snapshots: Dict[str, MarketSnapshot]) -> str:
        """Classify current volatility regime"""
        try:
            # Average volatility across symbols
            volatilities = [data.volatility for data in snapshots.values()]
            avg_vol = np.mean(volatilities) if volatilities else 0.0
            
            if avg_vol > 0.04:
                return "high"
            elif avg_vol > 0.02:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def _classify_volume(self, snapshots: Dict[str, MarketSnapshot]) -> str:
        """Classify current volume regime"""
        try:
            # Average volume ratio across symbols
            volume_ratios = [data.volume_profile.get("volume_ratio", 1.0) for data in snapshots.values()]
            avg_vol_ratio = np.mean(volume_ratios) if volume_ratios else 1.0
            
            if avg_vol_ratio > 1.5:
                return "high"
            elif avg_vol_ratio > 0.8:
                return "normal"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def _aggregate_sentiment(self, snapshots: Dict[str, MarketSnapshot]) -> float:
        """Aggregate sentiment across all symbols"""
        try:
            sentiments = [data.news_sentiment for data in snapshots.values()]
            return float(np.mean(sentiments)) if sentiments else 0.0
        except Exception:
            return 0.0
    
    def _calculate_relative_strength(self, symbol: str, snapshots: Dict[str, MarketSnapshot]) -> float:
        """Calculate relative strength vs market"""
        try:
            symbol_momentum = snapshots[symbol].momentum_5m
            
            # Use ES as benchmark
            benchmark_momentum = snapshots.get("ES=F", snapshots[symbol]).momentum_5m
            
            return symbol_momentum - benchmark_momentum
            
        except Exception:
            return 0.0
    
    async def save_market_data(self, snapshots: Dict[str, MarketSnapshot], features: Dict):
        """Save collected market data and features"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save raw snapshots
            raw_data = {
                "timestamp": datetime.now().isoformat(),
                "snapshots": {
                    symbol: {
                        "price": snapshot.price,
                        "volume": snapshot.volume,
                        "bid": snapshot.bid,
                        "ask": snapshot.ask,
                        "spread": snapshot.spread,
                        "volatility": snapshot.volatility,
                        "momentum_1m": snapshot.momentum_1m,
                        "momentum_5m": snapshot.momentum_5m,
                        "rsi": snapshot.rsi,
                        "volume_profile": snapshot.volume_profile,
                        "news_sentiment": snapshot.news_sentiment
                    }
                    for symbol, snapshot in snapshots.items()
                }
            }
            
            raw_file = self.raw_dir / f"market_data_{timestamp}.json"
            with open(raw_file, 'w') as f:
                json.dump(raw_data, f, indent=2)
            
            # Save processed features
            features_file = self.features_dir / f"market_features_{timestamp}.json"
            with open(features_file, 'w') as f:
                json.dump(features, f, indent=2)
            
            # Update latest files
            latest_raw = self.raw_dir / "latest.json"
            with open(latest_raw, 'w') as f:
                json.dump(raw_data, f, indent=2)
                
            latest_features = self.features_dir / "latest.json"
            with open(latest_features, 'w') as f:
                json.dump(features, f, indent=2)
            
            self.logger.info(f"Saved market data for {len(snapshots)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
    
    async def run_collection_cycle(self, symbols: List[str] = ["ES=F", "NQ=F"]):
        """Run one complete data collection cycle"""
        try:
            self.logger.info(f"Starting collection cycle for {symbols}")
            
            # Collect raw data
            snapshots = await self.collect_multi_symbol_data(symbols)
            
            if not snapshots:
                self.logger.warning("No snapshots collected")
                return
            
            # Generate features
            features = await self.generate_market_features(snapshots)
            
            # Save data
            await self.save_market_data(snapshots, features)
            
            self.logger.info(f"Collection cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in collection cycle: {e}")
    
    async def start_live_collection(self, symbols: List[str] = ["ES=F", "NQ=F"], 
                                   interval: int = 30):
        """Start continuous live data collection"""
        self.logger.info(f"Starting live market data collection for {symbols}")
        self.logger.info(f"Collection interval: {interval} seconds")
        
        while True:
            try:
                # Only collect during market hours
                if self.is_market_hours():
                    await self.run_collection_cycle(symbols)
                else:
                    self.logger.debug("Outside market hours, skipping collection")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                self.logger.info("Live collection stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in live collection: {e}")
                await asyncio.sleep(60)  # Wait before retrying


async def main():
    """Main function for standalone execution"""
    collector = EnhancedMarketDataCollector()
    
    # Run one collection cycle
    await collector.run_collection_cycle(["ES=F", "NQ=F"])
    
    print("Enhanced market data collection completed")


if __name__ == "__main__":
    asyncio.run(main())
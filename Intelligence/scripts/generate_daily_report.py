#!/usr/bin/env python3
"""
Daily Report Generator for Intelligence Pipeline
Generates comprehensive daily analysis reports
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyReportGenerator:
    def __init__(self):
        self.data_dir = "Intelligence/data"
        self.reports_dir = "Intelligence/reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def load_latest_data(self):
        """Load the latest intelligence data"""
        data = {}
        
        try:
            # Load latest signals
            signals_file = os.path.join(self.data_dir, "signals", "latest.json")
            if os.path.exists(signals_file):
                with open(signals_file, 'r') as f:
                    data['signals'] = json.load(f)
            
            # Load latest news analysis
            news_file = os.path.join(self.data_dir, "raw", "news", "latest_analysis.json")
            if os.path.exists(news_file):
                with open(news_file, 'r') as f:
                    data['news'] = json.load(f)
            
            # Load latest market data
            market_file = os.path.join(self.data_dir, "raw", "indices", "latest_market_data.json")
            if os.path.exists(market_file):
                with open(market_file, 'r') as f:
                    data['market'] = json.load(f)
            
            # Load features if available
            features_file = os.path.join(self.data_dir, "features", "latest_metadata.json")
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    data['features_meta'] = json.load(f)
            
            logger.info(f"Loaded data: {list(data.keys())}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}
    
    def generate_executive_summary(self, data):
        """Generate executive summary of current market intelligence"""
        try:
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S UTC'),
                'status': 'Active'
            }
            
            # Signals summary
            if 'signals' in data:
                signals = data['signals']
                summary.update({
                    'regime': signals.get('regime', 'Unknown'),
                    'regime_confidence': signals.get('modelConfidence', 0),
                    'primary_bias': signals.get('primaryBias', 'Neutral'),
                    'trade_setups': len(signals.get('setups', [])),
                    'special_events': {
                        'cpi_day': signals.get('isCpiDay', False),
                        'fomc_day': signals.get('isFomcDay', False)
                    }
                })
            
            # News summary
            if 'news' in data:
                news = data['news']
                summary.update({
                    'news_intensity': news.get('intensity', 0),
                    'news_sentiment': news.get('sentiment', 'neutral'),
                    'news_confidence': news.get('confidence', 0),
                    'news_articles': news.get('article_count', 0)
                })
            
            # Market summary
            if 'market' in data:
                market = data['market']
                regime_analysis = market.get('regime_analysis', {})
                summary.update({
                    'vix_level': regime_analysis.get('vix_level', 0),
                    'market_regime': regime_analysis.get('regime', 'Unknown'),
                    'market_confidence': regime_analysis.get('confidence', 0)
                })
                
                # SPX data if available
                market_data = market.get('market_data', {})
                if 'SPX' in market_data:
                    spx = market_data['SPX']
                    summary.update({
                        'spx_close': spx.get('close', 0),
                        'spx_change': spx.get('change_pct', 0)
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {'date': datetime.now().strftime('%Y-%m-%d'), 'status': 'Error'}
    
    def generate_trade_analysis(self, data):
        """Generate detailed trade analysis and recommendations"""
        try:
            analysis = {
                'recommendation': 'Neutral',
                'confidence': 0.0,
                'risk_level': 'Medium',
                'position_sizing': 'Standard',
                'time_preferences': [],
                'warnings': [],
                'opportunities': []
            }
            
            if 'signals' not in data:
                analysis['warnings'].append('No signal data available')
                return analysis
            
            signals = data['signals']
            regime = signals.get('regime', 'Unknown')
            confidence = signals.get('modelConfidence', 0)
            bias = signals.get('primaryBias', 'Neutral')
            setups = signals.get('setups', [])
            
            # Overall recommendation
            analysis['recommendation'] = bias
            analysis['confidence'] = confidence
            
            # Risk assessment
            vix_level = 0
            if 'market' in data:
                vix_level = data['market'].get('regime_analysis', {}).get('vix_level', 0)
            
            if vix_level > 25 or regime == 'Volatile':
                analysis['risk_level'] = 'High'
                analysis['position_sizing'] = 'Reduced'
                analysis['warnings'].append(f'High volatility environment (VIX: {vix_level:.1f})')
            elif confidence < 0.4:
                analysis['risk_level'] = 'Medium'
                analysis['position_sizing'] = 'Reduced'
                analysis['warnings'].append(f'Low model confidence ({confidence:.1f})')
            else:
                analysis['risk_level'] = 'Medium'
                analysis['position_sizing'] = 'Standard'
            
            # Special events
            if signals.get('isCpiDay', False):
                analysis['warnings'].append('CPI release day - expect volatility')
                analysis['risk_level'] = 'High'
            
            if signals.get('isFomcDay', False):
                analysis['warnings'].append('FOMC meeting day - expect volatility')
                analysis['risk_level'] = 'High'
            
            # Setup analysis
            for setup in setups:
                time_window = setup.get('timeWindow', 'Unknown')
                direction = setup.get('direction', 'Unknown')
                setup_confidence = setup.get('confidenceScore', 0)
                rationale = setup.get('rationale', 'No rationale provided')
                
                if setup_confidence > 0.6:
                    analysis['opportunities'].append({
                        'time': time_window,
                        'direction': direction,
                        'confidence': setup_confidence,
                        'reason': rationale
                    })
                
                analysis['time_preferences'].append(time_window)
            
            # News factor
            if 'news' in data:
                news = data['news']
                news_intensity = news.get('intensity', 0)
                
                if news_intensity > 70:
                    analysis['warnings'].append(f'High news intensity ({news_intensity:.0f}/100)')
                elif news_intensity > 0:
                    analysis['opportunities'].append({
                        'type': 'news',
                        'intensity': news_intensity,
                        'sentiment': news.get('sentiment', 'neutral')
                    })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating trade analysis: {e}")
            return {'recommendation': 'Neutral', 'confidence': 0.0}
    
    def generate_market_overview(self, data):
        """Generate comprehensive market overview"""
        try:
            overview = {
                'indices': {},
                'volatility': {},
                'sentiment': {},
                'technical': {}
            }
            
            # Market data
            if 'market' in data:
                market_data = data['market'].get('market_data', {})
                
                for symbol, symbol_data in market_data.items():
                    overview['indices'][symbol] = {
                        'price': symbol_data.get('close', 0),
                        'change': symbol_data.get('change_pct', 0),
                        'volume': symbol_data.get('volume', 0)
                    }
                    
                    # Add volatility info
                    if symbol == 'VIX':
                        overview['volatility']['current'] = symbol_data.get('close', 0)
                        overview['volatility']['change'] = symbol_data.get('change_pct', 0)
                    
                    # Add technical indicators if available
                    if 'volatility_20d' in symbol_data:
                        overview['technical'][f'{symbol}_vol_20d'] = symbol_data['volatility_20d']
            
            # News sentiment
            if 'news' in data:
                news = data['news']
                overview['sentiment'] = {
                    'direction': news.get('sentiment', 'neutral'),
                    'strength': news.get('intensity', 0),
                    'confidence': news.get('confidence', 0),
                    'sources': news.get('article_count', 0)
                }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error generating market overview: {e}")
            return {}
    
    def format_html_report(self, summary, analysis, overview):
        """Generate HTML formatted daily report"""
        try:
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Daily Intelligence Report - {summary['date']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                    .opportunity {{ background-color: #d4edda; border-color: #c3e6cb; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Daily Intelligence Report</h1>
                    <p><strong>Date:</strong> {summary['date']} at {summary['time']}</p>
                    <p><strong>Status:</strong> {summary['status']}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="metric"><strong>Market Regime:</strong> {summary.get('regime', 'Unknown')}</div>
                    <div class="metric"><strong>Primary Bias:</strong> {summary.get('primary_bias', 'Neutral')}</div>
                    <div class="metric"><strong>Model Confidence:</strong> {summary.get('regime_confidence', 0):.1%}</div>
                    <div class="metric"><strong>Trade Setups:</strong> {summary.get('trade_setups', 0)}</div>
                </div>
                
                <div class="section">
                    <h2>Trade Analysis</h2>
                    <p><strong>Recommendation:</strong> {analysis['recommendation']}</p>
                    <p><strong>Confidence:</strong> {analysis['confidence']:.1%}</p>
                    <p><strong>Risk Level:</strong> {analysis['risk_level']}</p>
                    <p><strong>Position Sizing:</strong> {analysis['position_sizing']}</p>
                    
                    {'<div class="section warning"><h3>Warnings</h3><ul>' + ''.join(f'<li>{w}</li>' for w in analysis['warnings']) + '</ul></div>' if analysis['warnings'] else ''}
                    
                    {'<div class="section opportunity"><h3>Opportunities</h3><ul>' + ''.join(f'<li>{o.get("reason", str(o))}</li>' for o in analysis['opportunities']) + '</ul></div>' if analysis['opportunities'] else ''}
                </div>
                
                <div class="section">
                    <h2>Market Overview</h2>
                    <table>
                        <tr><th>Index</th><th>Price</th><th>Change</th></tr>
            """
            
            # Add market data to table
            for symbol, data in overview.get('indices', {}).items():
                change_color = 'green' if data['change'] >= 0 else 'red'
                html_template += f"""
                        <tr>
                            <td>{symbol}</td>
                            <td>{data['price']:.2f}</td>
                            <td style="color: {change_color};">{data['change']:+.2f}%</td>
                        </tr>
                """
            
            html_template += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>News Sentiment</h2>
            """
            
            sentiment_data = overview.get('sentiment', {})
            if sentiment_data:
                html_template += f"""
                    <p><strong>Direction:</strong> {sentiment_data.get('direction', 'neutral').title()}</p>
                    <p><strong>Intensity:</strong> {sentiment_data.get('strength', 0):.0f}/100</p>
                    <p><strong>Confidence:</strong> {sentiment_data.get('confidence', 0):.1f}%</p>
                    <p><strong>Articles Analyzed:</strong> {sentiment_data.get('sources', 0)}</p>
                """
            else:
                html_template += "<p>No news sentiment data available</p>"
            
            html_template += """
                </div>
                
                <div class="section">
                    <small>
                        Generated by Intelligence Pipeline at {timestamp}<br>
                        This report is for informational purposes only. Always apply proper risk management.
                    </small>
                </div>
            </body>
            </html>
            """.format(timestamp=datetime.now().isoformat())
            
            return html_template
            
        except Exception as e:
            logger.error(f"Error formatting HTML report: {e}")
            return f"<html><body><h1>Report Generation Error</h1><p>{e}</p></body></html>"
    
    def save_reports(self, summary, analysis, overview):
        """Save reports in multiple formats"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_str = datetime.now().strftime("%Y-%m-%d")
            
            # Save JSON summary
            json_report = {
                'summary': summary,
                'analysis': analysis,
                'overview': overview,
                'generated_at': datetime.now().isoformat()
            }
            
            json_file = os.path.join(self.reports_dir, f"daily_report_{timestamp_str}.json")
            with open(json_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            # Save latest JSON
            latest_json = os.path.join(self.reports_dir, "latest_report.json")
            with open(latest_json, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            # Generate and save HTML report
            html_content = self.format_html_report(summary, analysis, overview)
            html_file = os.path.join(self.reports_dir, f"daily_report_{date_str}.html")
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            # Save latest HTML
            latest_html = os.path.join(self.reports_dir, "latest_report.html")
            with open(latest_html, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Saved reports: JSON={json_file}, HTML={html_file}")
            return json_file, html_file
            
        except Exception as e:
            logger.error(f"Error saving reports: {e}")
            return None, None

def main():
    """Main execution function"""
    logger.info("Starting daily report generation...")
    
    generator = DailyReportGenerator()
    
    # Load latest data
    data = generator.load_latest_data()
    
    if not data:
        logger.warning("No data available for report generation")
        return
    
    # Generate report sections
    summary = generator.generate_executive_summary(data)
    analysis = generator.generate_trade_analysis(data)
    overview = generator.generate_market_overview(data)
    
    # Save reports
    json_file, html_file = generator.save_reports(summary, analysis, overview)
    
    if json_file and html_file:
        logger.info("Daily report generation completed successfully")
        logger.info(f"Reports available at: {generator.reports_dir}")
    else:
        logger.error("Failed to generate reports")

if __name__ == "__main__":
    main()
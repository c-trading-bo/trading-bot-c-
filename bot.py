#!/usr/bin/env python3
"""
Trading Bot Launcher Script

Wrapper script that launches the C# UnifiedOrchestrator trading bot with proper
environment setup and logging. This script is designed to be run from GitHub Actions
or other CI/CD environments.

Usage:
    python bot.py [--dry-run] [--log-file PATH]
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging to both console and file."""
    logger = logging.getLogger("bot_launcher")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def validate_environment(logger: logging.Logger) -> bool:
    """Validate required environment variables and dependencies."""
    logger.info("=== Environment Validation ===")
    
    # Check required environment variables
    required_vars = ['TOPSTEPX_API_KEY', 'TOPSTEPX_USERNAME']
    optional_vars = ['TOPSTEPX_ACCOUNT_ID', 'PYTHON_EXECUTABLE']
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: [SET]")
        else:
            logger.error(f"✗ {var}: [MISSING]")
            missing_vars.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: {value}")
        else:
            logger.info(f"- {var}: [NOT SET]")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check if project-x-py is installed
    logger.info("\n=== SDK Validation ===")
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import project_x_py; print(project_x_py.__version__)'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✓ project-x-py SDK installed: v{version}")
        else:
            logger.error(f"✗ project-x-py SDK validation failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to validate project-x-py SDK: {e}")
        return False
    
    # Check dotnet availability
    logger.info("\n=== .NET Runtime Validation ===")
    try:
        result = subprocess.run(
            ['dotnet', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✓ .NET Runtime available: v{version}")
        else:
            logger.error("✗ .NET Runtime not found")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to validate .NET Runtime: {e}")
        return False
    
    logger.info("\n✓ All validations passed\n")
    return True


def run_trading_bot(logger: logging.Logger, dry_run: bool = False) -> int:
    """Launch the C# trading bot using dotnet run."""
    logger.info("=== Starting Trading Bot ===")
    
    # Navigate to UnifiedOrchestrator directory
    bot_path = Path(__file__).parent / 'src' / 'UnifiedOrchestrator'
    
    if not bot_path.exists():
        logger.error(f"Bot directory not found: {bot_path}")
        return 1
    
    logger.info(f"Working directory: {bot_path}")
    
    # Build the command
    cmd = ['dotnet', 'run', '--project', str(bot_path)]
    
    if dry_run:
        cmd.append('--dry-run')
        logger.info("Running in DRY-RUN mode")
    
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("Starting bot...\n")
    logger.info("=" * 60)
    
    try:
        # Run the bot and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output to logger
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        logger.info("=" * 60)
        logger.info(f"\nBot exited with code: {return_code}")
        
        return return_code
        
    except KeyboardInterrupt:
        logger.warning("\nReceived interrupt signal, stopping bot...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.error("Bot did not stop gracefully, killing process...")
            process.kill()
        return 130
        
    except Exception as e:
        logger.error(f"Failed to run trading bot: {e}", exc_info=True)
        return 1


def main():
    """Main entry point for bot launcher."""
    parser = argparse.ArgumentParser(
        description='Trading Bot Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot.py                           # Run bot normally
  python bot.py --dry-run                 # Run in dry-run mode
  python bot.py --log-file logs/bot.log   # Log to specific file
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run bot in dry-run mode (no live trading)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=f'logs/bot-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        help='Path to log file (default: logs/bot-TIMESTAMP.log)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 60)
    logger.info("  Trading Bot Launcher")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info("")
    
    # Validate environment
    if not validate_environment(logger):
        logger.error("\n❌ Environment validation failed")
        return 1
    
    # Run the bot
    exit_code = run_trading_bot(logger, dry_run=args.dry_run)
    
    logger.info(f"\nCompleted at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

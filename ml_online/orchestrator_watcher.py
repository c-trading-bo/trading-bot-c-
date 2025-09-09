#!/usr/bin/env python3
"""
Python Orchestrator Watcher for Model Hot-Reload
Implements requirement 4: Enable Model Hot-Reload - Python orchestrator watcher
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Set
import yaml
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class PythonOrchestratorWatcher:
    """Python-side orchestrator watcher for model updates"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.last_checked: Dict[str, float] = {}
        self.monitoring_paths = {
            'registry': 'data/registry',
            'sac': 'data/rl/sac',
            'local_models': 'data/local_models'
        }
        self.running = False
        
    async def start_watching(self):
        """Start the orchestrator watcher"""
        self.running = True
        logger.info(f"[ORCHESTRATOR_WATCHER] Starting model hot-reload watcher (interval: {self.check_interval}s)")
        
        while self.running:
            try:
                await self.check_all_paths()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"[ORCHESTRATOR_WATCHER] Error in watch loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
                
    async def stop_watching(self):
        """Stop the orchestrator watcher"""
        self.running = False
        logger.info("[ORCHESTRATOR_WATCHER] Stopped model hot-reload watcher")
        
    async def check_all_paths(self):
        """Check all monitored paths for updates"""
        for path_type, path in self.monitoring_paths.items():
            await self.check_path_updates(path_type, path)
            
    async def check_path_updates(self, path_type: str, path: str):
        """Check a specific path for model updates"""
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                return
                
            if path_type == 'registry':
                await self.check_registry_updates(path)
            elif path_type == 'sac':
                await self.check_sac_updates(path)
            elif path_type == 'local_models':
                await self.check_local_models_updates(path)
                
        except Exception as e:
            logger.error(f"[ORCHESTRATOR_WATCHER] Error checking {path_type} at {path}: {e}")
            
    async def check_registry_updates(self, registry_path: str):
        """Check registry for *_latest.yaml updates"""
        yaml_files = list(Path(registry_path).glob("*_latest.yaml"))
        
        for yaml_file in yaml_files:
            try:
                mod_time = yaml_file.stat().st_mtime
                file_key = f"registry_{yaml_file.name}"
                
                if file_key not in self.last_checked or mod_time > self.last_checked[file_key]:
                    logger.info(f"[ORCHESTRATOR_WATCHER] Registry update detected: {yaml_file.name}")
                    
                    # Parse YAML metadata
                    with open(yaml_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                        
                    await self.handle_registry_update(yaml_file.name, metadata)
                    self.last_checked[file_key] = mod_time
                    
            except Exception as e:
                logger.error(f"[ORCHESTRATOR_WATCHER] Error processing registry file {yaml_file}: {e}")
                
    async def check_sac_updates(self, sac_path: str):
        """Check SAC directory for model updates"""
        sac_files = []
        for ext in ['*.zip', '*.pkl', '*.pt']:
            sac_files.extend(list(Path(sac_path).glob(ext)))
            
        for sac_file in sac_files:
            try:
                mod_time = sac_file.stat().st_mtime
                file_key = f"sac_{sac_file.name}"
                
                if file_key not in self.last_checked or mod_time > self.last_checked[file_key]:
                    logger.info(f"[ORCHESTRATOR_WATCHER] SAC model update detected: {sac_file.name}")
                    
                    await self.handle_sac_update(sac_file.name, str(sac_file))
                    self.last_checked[file_key] = mod_time
                    
            except Exception as e:
                logger.error(f"[ORCHESTRATOR_WATCHER] Error processing SAC file {sac_file}: {e}")
                
    async def check_local_models_updates(self, models_path: str):
        """Check local models directory for updates"""
        model_files = list(Path(models_path).glob("*.onnx")) + list(Path(models_path).glob("*.pkl"))
        
        for model_file in model_files:
            try:
                mod_time = model_file.stat().st_mtime
                file_key = f"local_{model_file.name}"
                
                if file_key not in self.last_checked or mod_time > self.last_checked[file_key]:
                    logger.info(f"[ORCHESTRATOR_WATCHER] Local model update detected: {model_file.name}")
                    
                    await self.handle_local_model_update(model_file.name, str(model_file))
                    self.last_checked[file_key] = mod_time
                    
            except Exception as e:
                logger.error(f"[ORCHESTRATOR_WATCHER] Error processing local model {model_file}: {e}")
                
    async def handle_registry_update(self, filename: str, metadata: Dict):
        """Handle registry metadata update"""
        try:
            model_type = filename.replace('_latest.yaml', '')
            version = metadata.get('version', 'unknown')
            model_path = metadata.get('model_path', '')
            
            logger.info(f"[ORCHESTRATOR_WATCHER] Registry model promoted: {model_type} v{version} at {model_path}")
            
            # TODO: Trigger model reload in appropriate system components
            # This could notify the C# side via IPC or shared state
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR_WATCHER] Error handling registry update {filename}: {e}")
            
    async def handle_sac_update(self, filename: str, filepath: str):
        """Handle SAC model update"""
        try:
            logger.info(f"[ORCHESTRATOR_WATCHER] Reloading SAC model: {filename}")
            
            # TODO: Reload SAC model in the appropriate agent
            # This would update the SAC policy for position sizing
            
            # Placeholder for SAC model loading
            if filename.endswith('.zip'):
                # Stable-Baselines3 model
                logger.info(f"[ORCHESTRATOR_WATCHER] Loading SB3 SAC model from {filepath}")
            elif filename.endswith('.pkl'):
                # Pickled model
                logger.info(f"[ORCHESTRATOR_WATCHER] Loading pickled SAC model from {filepath}")
            elif filename.endswith('.pt'):
                # PyTorch model
                logger.info(f"[ORCHESTRATOR_WATCHER] Loading PyTorch SAC model from {filepath}")
                
        except Exception as e:
            logger.error(f"[ORCHESTRATOR_WATCHER] Error handling SAC update {filename}: {e}")
            
    async def handle_local_model_update(self, filename: str, filepath: str):
        """Handle local model update"""
        try:
            logger.info(f"[ORCHESTRATOR_WATCHER] Reloading local model: {filename}")
            
            # TODO: Trigger local model reload
            # This would update models used by the online learning system
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR_WATCHER] Error handling local model update {filename}: {e}")

# Global watcher instance
orchestrator_watcher = PythonOrchestratorWatcher()

async def start_orchestrator_watcher():
    """Start the orchestrator watcher"""
    await orchestrator_watcher.start_watching()

async def stop_orchestrator_watcher():
    """Stop the orchestrator watcher"""
    await orchestrator_watcher.stop_watching()

if __name__ == "__main__":
    # Run the watcher standalone
    import sys
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(start_orchestrator_watcher())
    except KeyboardInterrupt:
        logger.info("[ORCHESTRATOR_WATCHER] Shutdown requested")
        sys.exit(0)
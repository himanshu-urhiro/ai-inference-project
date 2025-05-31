"""
Main entry point for the PyTorch Video Inference API

This script initializes and runs the video inference system,
setting up all components and starting the REST API server.
"""

import os
import sys
import argparse
import logging

from src.config.config_manager import ConfigManager
from src.models.model_manager import ModelManager
from src.video.video_processor import VideoProcessor
from src.inference.inference_engine import InferenceEngine
from src.api.api import run_api_server
from src.utils.performance import setup_logging, optimize_torch_for_edge, ResourceMonitor

def main():
    """
    Main entry point for the application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch Video Inference API")
    parser.add_argument("--config", type=str, default="./config/config.json",
                        help="Path to configuration file")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for log files")
    parser.add_argument("--host", type=str, default=None,
                        help="Host address to bind API server (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to run API server (overrides config)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_dir, log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PyTorch Video Inference API")
    
    # Load configuration
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    logger.info(f"Loading configuration from {config_path}")
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # Override config with command line arguments if provided
    if args.host:
        config.setdefault("api", {})["host"] = args.host
    if args.port:
        config.setdefault("api", {})["port"] = args.port
    
    # Optimize PyTorch for edge deployment
    optimize_torch_for_edge()
    
    # Start resource monitoring
    resource_monitor = ResourceMonitor(interval=config.get("monitoring", {}).get("interval", 5.0))
    resource_monitor.start()
    
    # Start API server
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    
    try:
        logger.info(f"Starting API server on {host}:{port}")
        run_api_server(config_path, host, port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running API server: {str(e)}")
    finally:
        resource_monitor.stop()
        logger.info("PyTorch Video Inference API stopped")

if __name__ == "__main__":
    main()

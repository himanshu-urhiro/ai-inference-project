"""
Utility Functions Module for PyTorch Video Inference

This module provides common utility functions used across the project,
including logging, performance monitoring, and edge-specific optimizations.
"""

import os
import time
import logging
import threading
import psutil
import json
from typing import Dict, List, Any, Optional, Callable, Union
import torch

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"inference_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")

class ResourceMonitor:
    """
    Monitor system resources for edge deployment.
    
    Tracks CPU, memory, and GPU usage to ensure efficient operation
    on resource-constrained edge devices.
    """
    
    def __init__(self, interval: float = 5.0):
        """
        Initialize the resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        self.stats = {
            "cpu": [],
            "memory": [],
            "gpu": [],
            "temperature": []
        }
        self.callbacks = []
        
    def start(self) -> None:
        """
        Start resource monitoring.
        """
        if self.running:
            logger.warning("Resource monitor already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def _monitor_resources(self) -> None:
        """
        Internal method to continuously monitor system resources.
        """
        while self.running:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Get GPU stats if available
                gpu_percent = 0
                gpu_temp = 0
                
                if torch.cuda.is_available():
                    # This is a simplified approach - in a real implementation,
                    # you would use libraries like pynvml for detailed GPU stats
                    gpu_percent = 0  # Placeholder
                    gpu_temp = 0     # Placeholder
                
                # Record stats
                timestamp = time.time()
                current_stats = {
                    "timestamp": timestamp,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "gpu_percent": gpu_percent,
                    "temperature": gpu_temp
                }
                
                # Keep last 100 measurements
                self.stats["cpu"].append((timestamp, cpu_percent))
                if len(self.stats["cpu"]) > 100:
                    self.stats["cpu"].pop(0)
                    
                self.stats["memory"].append((timestamp, memory_percent))
                if len(self.stats["memory"]) > 100:
                    self.stats["memory"].pop(0)
                    
                self.stats["gpu"].append((timestamp, gpu_percent))
                if len(self.stats["gpu"]) > 100:
                    self.stats["gpu"].pop(0)
                    
                self.stats["temperature"].append((timestamp, gpu_temp))
                if len(self.stats["temperature"]) > 100:
                    self.stats["temperature"].pop(0)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(current_stats)
                    except Exception as e:
                        logger.error(f"Error in resource monitor callback: {str(e)}")
                
                # Check for resource limits
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                    
                if memory_percent > 90:
                    logger.warning(f"High memory usage: {memory_percent}%")
                    
                if gpu_percent > 90:
                    logger.warning(f"High GPU usage: {gpu_percent}%")
                    
                # Sleep until next interval
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(self.interval)
                
        logger.info("Resource monitoring stopped")
    
    def stop(self) -> None:
        """
        Stop resource monitoring.
        """
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
    def get_stats(self) -> Dict[str, List]:
        """
        Get current resource statistics.
        
        Returns:
            Dict[str, List]: Dictionary of resource statistics
        """
        return self.stats
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function for resource updates.
        
        Args:
            callback: Function to call with resource stats
        """
        self.callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Unregister a callback function.
        
        Args:
            callback: Function to remove
            
        Returns:
            bool: True if callback was removed, False otherwise
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            return True
        return False

class PerformanceTracker:
    """
    Track performance metrics for inference operations.
    
    Records and analyzes timing information for different stages
    of the inference pipeline.
    """
    
    def __init__(self):
        """
        Initialize the performance tracker.
        """
        self.metrics = {}
        
    def start_timer(self, name: str) -> None:
        """
        Start a timer for a specific operation.
        
        Args:
            name: Name of the operation to time
        """
        if name not in self.metrics:
            self.metrics[name] = {
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "start_time": None
            }
            
        self.metrics[name]["start_time"] = time.time()
        
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record the elapsed time.
        
        Args:
            name: Name of the operation
            
        Returns:
            float: Elapsed time in seconds
        """
        if name not in self.metrics or self.metrics[name]["start_time"] is None:
            logger.warning(f"Timer '{name}' not started")
            return 0
            
        elapsed = time.time() - self.metrics[name]["start_time"]
        self.metrics[name]["count"] += 1
        self.metrics[name]["total_time"] += elapsed
        self.metrics[name]["min_time"] = min(self.metrics[name]["min_time"], elapsed)
        self.metrics[name]["max_time"] = max(self.metrics[name]["max_time"], elapsed)
        self.metrics[name]["start_time"] = None
        
        return elapsed
        
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current performance metrics.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of performance metrics
        """
        result = {}
        
        for name, data in self.metrics.items():
            if data["count"] > 0:
                avg_time = data["total_time"] / data["count"]
                result[name] = {
                    "count": data["count"],
                    "total_time": data["total_time"],
                    "avg_time": avg_time,
                    "min_time": data["min_time"],
                    "max_time": data["max_time"]
                }
                
        return result
        
    def reset(self) -> None:
        """
        Reset all performance metrics.
        """
        self.metrics = {}
        
    def export_metrics(self, file_path: str) -> bool:
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to save metrics
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.get_metrics(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return False

def optimize_torch_for_edge() -> None:
    """
    Configure PyTorch for optimal performance on edge devices.
    """
    # Set thread count for CPU operations
    if hasattr(torch, 'set_num_threads'):
        # Use half of available CPU cores
        num_threads = max(1, psutil.cpu_count(logical=False) // 2)
        torch.set_num_threads(num_threads)
        logger.info(f"PyTorch CPU threads set to {num_threads}")
    
    # Disable gradient calculation for inference
    torch.set_grad_enabled(False)
    logger.info("PyTorch gradients disabled for inference")
    
    # Set default tensor type
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        logger.info("PyTorch default tensor type set to CUDA")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        logger.info("PyTorch default tensor type set to CPU")

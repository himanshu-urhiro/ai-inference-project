"""
Inference Engine Module for PyTorch Video Inference

This module handles the core inference logic, coordinating between
model management and video processing for efficient edge deployment.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

from ..models.model_manager import ModelManager
from ..video.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Manages the inference process for video streams on edge devices.
    
    Coordinates between model management and video processing to perform
    efficient inference on video frames.
    """
    
    def __init__(self, model_manager: ModelManager, config: Dict[str, Any]):
        """
        Initialize the inference engine.
        
        Args:
            model_manager: ModelManager instance for model access
            config: Configuration dictionary with inference parameters
        """
        self.model_manager = model_manager
        self.config = config
        self.current_model_name = config.get("default_model", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 1)
        self.inference_stats = {
            "total_frames": 0,
            "inference_time": 0.0,
            "avg_fps": 0.0
        }
        
    def set_model(self, model_name: str) -> bool:
        """
        Set the current model for inference.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            bool: True if model was set successfully, False otherwise
        """
        model = self.model_manager.get_model(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return False
            
        self.current_model_name = model_name
        logger.info(f"Set current model to {model_name}")
        return True
        
    def process_frame(self, frame_data: Tuple[int, np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Process a single frame with the current model.
        
        Args:
            frame_data: Tuple of (frame_id, preprocessed_frame, original_frame)
            
        Returns:
            Dict[str, Any]: Results of inference with metadata
        """
        frame_id, preprocessed_frame, original_frame = frame_data
        
        if self.current_model_name is None:
            logger.error("No model selected for inference")
            return {"error": "No model selected", "frame_id": frame_id}
            
        model = self.model_manager.get_model(self.current_model_name)
        if model is None:
            logger.error(f"Model {self.current_model_name} not available")
            return {"error": "Model not available", "frame_id": frame_id}
            
        try:
            # Convert numpy array to torch tensor
            input_tensor = torch.from_numpy(preprocessed_frame).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                
            inference_time = time.time() - start_time
            
            # Update statistics
            self.inference_stats["total_frames"] += 1
            self.inference_stats["inference_time"] += inference_time
            self.inference_stats["avg_fps"] = self.inference_stats["total_frames"] / self.inference_stats["inference_time"]
            
            # Process outputs (this would depend on the specific model)
            # For demonstration, we'll assume classification output
            if isinstance(output, torch.Tensor):
                # Move to CPU and convert to numpy
                output_np = output.cpu().numpy()
                
                # For classification, get top predictions
                if len(output_np.shape) == 2:  # [batch_size, num_classes]
                    top_indices = np.argsort(output_np[0])[-5:][::-1]  # Top 5 indices
                    top_scores = output_np[0][top_indices]
                    
                    results = {
                        "frame_id": frame_id,
                        "model": self.current_model_name,
                        "inference_time_ms": inference_time * 1000,
                        "top_indices": top_indices.tolist(),
                        "top_scores": top_scores.tolist(),
                        "timestamp": time.time()
                    }
                else:
                    # For other types of outputs (detection, segmentation, etc.)
                    results = {
                        "frame_id": frame_id,
                        "model": self.current_model_name,
                        "inference_time_ms": inference_time * 1000,
                        "raw_output_shape": output_np.shape,
                        "timestamp": time.time()
                    }
            else:
                # Handle non-tensor outputs
                results = {
                    "frame_id": frame_id,
                    "model": self.current_model_name,
                    "inference_time_ms": inference_time * 1000,
                    "output_type": str(type(output)),
                    "timestamp": time.time()
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return {
                "error": str(e),
                "frame_id": frame_id,
                "model": self.current_model_name
            }
    
    def process_batch(self, frames: List[Tuple[int, np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Process a batch of frames with the current model.
        
        Args:
            frames: List of frame data tuples
            
        Returns:
            List[Dict[str, Any]]: Results for each frame
        """
        if not frames:
            return []
            
        if self.current_model_name is None:
            logger.error("No model selected for batch inference")
            return [{"error": "No model selected", "frame_id": f[0]} for f in frames]
            
        model = self.model_manager.get_model(self.current_model_name)
        if model is None:
            logger.error(f"Model {self.current_model_name} not available")
            return [{"error": "Model not available", "frame_id": f[0]} for f in frames]
            
        try:
            # Extract frame IDs and preprocessed frames
            frame_ids = [f[0] for f in frames]
            preprocessed_frames = [f[1] for f in frames]
            
            # Concatenate preprocessed frames into a batch
            batch = np.concatenate(preprocessed_frames, axis=0)
            
            # Convert to torch tensor
            input_tensor = torch.from_numpy(batch).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                batch_output = model(input_tensor)
                
            inference_time = time.time() - start_time
            
            # Update statistics
            self.inference_stats["total_frames"] += len(frames)
            self.inference_stats["inference_time"] += inference_time
            self.inference_stats["avg_fps"] = self.inference_stats["total_frames"] / self.inference_stats["inference_time"]
            
            # Process outputs for each frame in the batch
            results = []
            
            if isinstance(batch_output, torch.Tensor):
                batch_output_np = batch_output.cpu().numpy()
                
                for i, frame_id in enumerate(frame_ids):
                    if i < len(batch_output_np):
                        # For classification output
                        if len(batch_output_np.shape) == 2:
                            top_indices = np.argsort(batch_output_np[i])[-5:][::-1]
                            top_scores = batch_output_np[i][top_indices]
                            
                            results.append({
                                "frame_id": frame_id,
                                "model": self.current_model_name,
                                "inference_time_ms": (inference_time * 1000) / len(frames),
                                "top_indices": top_indices.tolist(),
                                "top_scores": top_scores.tolist(),
                                "timestamp": time.time()
                            })
                        else:
                            # For other output types
                            results.append({
                                "frame_id": frame_id,
                                "model": self.current_model_name,
                                "inference_time_ms": (inference_time * 1000) / len(frames),
                                "raw_output_shape": batch_output_np[i].shape,
                                "timestamp": time.time()
                            })
            else:
                # Handle non-tensor outputs
                for frame_id in frame_ids:
                    results.append({
                        "frame_id": frame_id,
                        "model": self.current_model_name,
                        "inference_time_ms": (inference_time * 1000) / len(frames),
                        "output_type": str(type(batch_output)),
                        "timestamp": time.time()
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Batch inference error: {str(e)}")
            return [{"error": str(e), "frame_id": f[0], "model": self.current_model_name} for f in frames]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current inference statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of inference statistics
        """
        return {
            "total_frames": self.inference_stats["total_frames"],
            "total_inference_time_s": self.inference_stats["inference_time"],
            "average_fps": self.inference_stats["avg_fps"],
            "current_model": self.current_model_name,
            "device": str(self.device),
            "batch_size": self.batch_size
        }

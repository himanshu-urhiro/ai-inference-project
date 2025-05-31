"""
Video Processing Module for PyTorch Video Inference

This module handles video frame capture, preprocessing, and management
for efficient inference on edge devices.
"""

import cv2
import numpy as np
import logging
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Union, Generator

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Processes video streams for inference on edge devices.
    
    Handles frame capture, preprocessing, and buffering for efficient inference.
    """
    
    def __init__(self, config: Dict[str, any]):
        """
        Initialize the video processor.
        
        Args:
            config: Configuration dictionary with video processing parameters
        """
        self.config = config
        self.frame_queue = queue.Queue(maxsize=config.get("queue_size", 30))
        self.processed_frames = {}
        self.running = False
        self.capture_thread = None
        self.source = None
        
    def start_capture(self, source: Union[int, str]) -> bool:
        """
        Start capturing frames from a video source.
        
        Args:
            source: Video source (camera index or file path)
            
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        if self.running:
            logger.warning("Video capture already running")
            return False
            
        try:
            self.source = source
            self.running = True
            self.capture_thread = threading.Thread(
                target=self._capture_frames,
                daemon=True
            )
            self.capture_thread.start()
            logger.info(f"Started video capture from source: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video capture: {str(e)}")
            self.running = False
            return False
    
    def _capture_frames(self) -> None:
        """
        Internal method to continuously capture frames from the video source.
        """
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            self.running = False
            return
            
        frame_count = 0
        drop_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("End of video stream reached")
                    break
                    
                frame_count += 1
                
                # Skip frames if necessary for performance on edge devices
                if frame_count % self.config.get("frame_skip", 1) != 0:
                    continue
                
                # Process frame for inference
                processed_frame = self.preprocess_frame(frame)
                
                # Try to add to queue, drop if full (important for real-time processing)
                try:
                    self.frame_queue.put((frame_count, processed_frame, frame), block=False)
                except queue.Full:
                    drop_count += 1
                    if drop_count % 100 == 0:
                        logger.warning(f"Dropped {drop_count} frames due to full queue")
                        
        except Exception as e:
            logger.error(f"Error in frame capture: {str(e)}")
        finally:
            cap.release()
            self.running = False
            logger.info(f"Video capture stopped. Processed {frame_count} frames, dropped {drop_count}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame for inference.
        
        Args:
            frame: Raw video frame
            
        Returns:
            np.ndarray: Preprocessed frame ready for model inference
        """
        # Resize frame to target dimensions
        target_size = self.config.get("target_size", (224, 224))
        resized = cv2.resize(frame, target_size)
        
        # Convert to RGB if needed (OpenCV uses BGR)
        if self.config.get("convert_to_rgb", True):
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        if self.config.get("normalize", True):
            resized = resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization if specified
            if self.config.get("imagenet_norm", False):
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                resized = (resized - mean) / std
        
        # Add batch dimension if needed
        if self.config.get("add_batch_dim", True):
            resized = np.expand_dims(resized, axis=0)
            
        # Transpose for PyTorch (N,C,H,W)
        if self.config.get("to_pytorch_format", True):
            resized = np.transpose(resized, (0, 3, 1, 2))
            
        return resized
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Get the next preprocessed frame from the queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Optional[Tuple[int, np.ndarray, np.ndarray]]: 
                Tuple of (frame_id, preprocessed_frame, original_frame) or None if queue is empty
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_capture(self) -> None:
        """
        Stop the video capture process.
        """
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        logger.info("Video capture stopped")
        
    def get_frame_generator(self) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Get a generator that yields frames as they become available.
        
        Yields:
            Tuple[int, np.ndarray, np.ndarray]: Tuple of (frame_id, preprocessed_frame, original_frame)
        """
        while self.running or not self.frame_queue.empty():
            frame_data = self.get_frame(timeout=0.1)
            if frame_data:
                yield frame_data
            else:
                # No frame available, yield control back to prevent blocking
                time.sleep(0.01)

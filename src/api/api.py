"""
REST API Module for PyTorch Video Inference

This module provides a FastAPI-based REST API for video inference
on edge devices, allowing remote access to inference capabilities.
"""

import os
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import threading
import json
from pydantic import BaseModel

from ..models.model_manager import ModelManager
from ..video.video_processor import VideoProcessor
from ..inference.inference_engine import InferenceEngine
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses
class ModelRequest(BaseModel):
    model_name: str
    
class InferenceResponse(BaseModel):
    request_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
class StatusResponse(BaseModel):
    status: str
    active_model: Optional[str] = None
    stats: Dict[str, Any]
    
# Global variables for API state
active_requests = {}
video_sources = {}

def create_api(config_path: str) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="Video Inference API",
        description="REST API for PyTorch-based video inference on edge devices",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    model_manager = ModelManager(config_path)
    video_processor = VideoProcessor(config.get("video", {}))
    inference_engine = InferenceEngine(model_manager, config.get("inference", {}))
    
    # Load default model if specified
    default_model = config.get("default_model", {})
    if default_model.get("name") and default_model.get("path"):
        model_manager.load_model(default_model["name"], default_model["path"])
        inference_engine.set_model(default_model["name"])
    
    @app.get("/")
    async def root():
        """API root endpoint"""
        return {
            "name": "Video Inference API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get current API and inference engine status"""
        stats = inference_engine.get_stats()
        return {
            "status": "running",
            "active_model": inference_engine.current_model_name,
            "stats": stats
        }
    
    @app.post("/models/load")
    async def load_model(model_request: ModelRequest):
        """
        Load a model for inference.
        
        Args:
            model_request: Request containing model name to load
        """
        model_name = model_request.model_name
        model_path = os.path.join(
            config.get("model_dir", "./models"),
            f"{model_name}.pt"
        )
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        success = model_manager.load_model(model_name, model_path)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {model_name}")
        
        inference_engine.set_model(model_name)
        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
    
    @app.post("/video/start/{source_id}")
    async def start_video_source(source_id: str, source_path: str = Query(...)):
        """
        Start a video source for inference.
        
        Args:
            source_id: Identifier for the video source
            source_path: Path or URL to video source (file, camera index, RTSP URL)
        """
        if source_id in video_sources:
            return {"status": "warning", "message": f"Video source {source_id} already running"}
        
        # Convert camera index from string to integer if needed
        if source_path.isdigit():
            source_path = int(source_path)
        
        success = video_processor.start_capture(source_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start video capture")
        
        video_sources[source_id] = {
            "source_path": source_path,
            "start_time": time.time()
        }
        
        return {
            "status": "success", 
            "message": f"Video source {source_id} started",
            "source_info": video_sources[source_id]
        }
    
    @app.post("/video/stop/{source_id}")
    async def stop_video_source(source_id: str):
        """
        Stop a video source.
        
        Args:
            source_id: Identifier for the video source to stop
        """
        if source_id not in video_sources:
            raise HTTPException(status_code=404, detail=f"Video source {source_id} not found")
        
        video_processor.stop_capture()
        source_info = video_sources.pop(source_id)
        
        return {
            "status": "success",
            "message": f"Video source {source_id} stopped",
            "source_info": source_info
        }
    
    @app.post("/inference/frame", response_model=InferenceResponse)
    async def process_single_frame(file: UploadFile = File(...)):
        """
        Process a single frame image for inference.
        
        Args:
            file: Uploaded image file
        """
        if not inference_engine.current_model_name:
            raise HTTPException(status_code=400, detail="No model selected for inference")
        
        try:
            # Read image file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Process frame
            processed_frame = video_processor.preprocess_frame(frame)
            frame_id = int(time.time() * 1000)  # Use timestamp as frame ID
            
            # Run inference
            results = inference_engine.process_frame((frame_id, processed_frame, frame))
            
            request_id = str(uuid.uuid4())
            return {
                "request_id": request_id,
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/inference/stream/{source_id}")
    async def stream_inference(source_id: str, max_fps: int = Query(10)):
        """
        Stream inference results for a video source.
        
        Args:
            source_id: Identifier for the video source
            max_fps: Maximum frames per second to process
        """
        if source_id not in video_sources:
            raise HTTPException(status_code=404, detail=f"Video source {source_id} not found")
        
        if not inference_engine.current_model_name:
            raise HTTPException(status_code=400, detail="No model selected for inference")
        
        async def generate():
            min_interval = 1.0 / max_fps
            last_frame_time = 0
            
            for frame_data in video_processor.get_frame_generator():
                current_time = time.time()
                
                # Limit FPS
                if current_time - last_frame_time < min_interval:
                    continue
                    
                last_frame_time = current_time
                
                # Process frame
                results = inference_engine.process_frame(frame_data)
                
                # Convert results to JSON and yield
                yield json.dumps(results) + "\n"
                
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    
    return app

def run_api_server(config_path: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Run the API server.
    
    Args:
        config_path: Path to configuration file
        host: Host address to bind to
        port: Port to listen on
    """
    app = create_api(config_path)
    uvicorn.run(app, host=host, port=port)

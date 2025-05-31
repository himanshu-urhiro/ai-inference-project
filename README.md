# PyTorch Video Inference Project

This project provides a complete framework for running AI inference on video feeds using PyTorch, optimized for edge deployment with a REST API interface.

## Project Structure

```
ai_inference_project/
├── main.py                  # Main entry point
├── src/                     # Source code
│   ├── models/              # Model management
│   │   ├── model_manager.py # Model loading and optimization
│   │   └── __init__.py
│   ├── video/               # Video processing
│   │   ├── video_processor.py # Video frame capture and preprocessing
│   │   └── __init__.py
│   ├── inference/           # Inference engine
│   │   ├── inference_engine.py # Core inference logic
│   │   └── __init__.py
│   ├── api/                 # REST API
│   │   ├── api.py           # FastAPI implementation
│   │   └── __init__.py
│   ├── config/              # Configuration management
│   │   ├── config_manager.py # Config loading and validation
│   │   └── __init__.py
│   ├── utils/               # Utility functions
│   │   ├── performance.py   # Performance monitoring for edge
│   │   └── __init__.py
│   └── __init__.py
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── data/                    # Data directory
│   ├── input/               # Input video files
│   └── output/              # Output results
└── logs/                    # Log files
```

## Features

- **PyTorch-based Inference**: Optimized for video processing
- **Edge Deployment**: Resource monitoring and optimization for edge devices
- **REST API**: FastAPI-based interface for remote access
- **Video Processing**: Efficient frame capture and preprocessing
- **Model Management**: Loading, optimization, and switching between models
- **Configuration**: Flexible configuration system

## Requirements

- Python 3.8+
- PyTorch 1.8+
- FastAPI
- OpenCV
- Additional dependencies in requirements.txt

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure the application in `config/config.json`
4. Run the application: `python main.py`

## API Endpoints

- `GET /status`: Get current API and inference engine status
- `POST /models/load`: Load a model for inference
- `POST /video/start/{source_id}`: Start a video source for inference
- `POST /video/stop/{source_id}`: Stop a video source
- `POST /inference/frame`: Process a single frame image for inference
- `GET /inference/stream/{source_id}`: Stream inference results for a video source

## Edge Optimization

The project includes several optimizations for edge deployment:

- Resource monitoring for CPU, memory, and GPU usage
- PyTorch optimizations for inference
- Efficient video frame processing with configurable frame skipping
- Model optimization techniques (quantization, pruning)

## Configuration

The application is configured through a JSON or YAML file with the following sections:

- `models`: Model paths and settings
- `video`: Video processing parameters
- `inference`: Inference engine settings
- `api`: API server configuration
- `monitoring`: Resource monitoring settings

## License

[MIT License](LICENSE)
# -ai-inference-project

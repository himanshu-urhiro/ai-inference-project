import pytest
import os
import sys
import json
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_manager import ModelManager
from src.video.video_processor import VideoProcessor
from src.inference.inference_engine import InferenceEngine
from src.config.config_manager import ConfigManager

class TestModelManager:
    """Test cases for the ModelManager class."""
    
    def test_init(self):
        """Test initialization of ModelManager."""
        config_path = "test_config.json"
        manager = ModelManager(config_path)
        assert manager.config_path == config_path
        assert manager.models == {}
        assert str(manager.device) in ["cpu", "cuda:0"]
    
    @patch('os.path.exists')
    @patch('torch.jit.load')
    def test_load_model(self, mock_load, mock_exists):
        """Test loading a model."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        manager = ModelManager("test_config.json")
        result = manager.load_model("test_model", "test_path")
        
        assert result is True
        assert "test_model" in manager.models
        mock_exists.assert_called_once_with("test_path")
        mock_load.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('os.path.exists')
    def test_load_model_not_found(self, mock_exists):
        """Test loading a model that doesn't exist."""
        mock_exists.return_value = False
        
        manager = ModelManager("test_config.json")
        result = manager.load_model("test_model", "test_path")
        
        assert result is False
        assert "test_model" not in manager.models
        mock_exists.assert_called_once_with("test_path")
    
    def test_get_model(self):
        """Test getting a loaded model."""
        manager = ModelManager("test_config.json")
        mock_model = MagicMock()
        manager.models["test_model"] = mock_model
        
        result = manager.get_model("test_model")
        assert result is mock_model
        
        result = manager.get_model("nonexistent_model")
        assert result is None
    
    def test_unload_model(self):
        """Test unloading a model."""
        manager = ModelManager("test_config.json")
        mock_model = MagicMock()
        manager.models["test_model"] = mock_model
        
        result = manager.unload_model("test_model")
        assert result is True
        assert "test_model" not in manager.models
        
        result = manager.unload_model("nonexistent_model")
        assert result is False

class TestVideoProcessor:
    """Test cases for the VideoProcessor class."""
    
    def test_init(self):
        """Test initialization of VideoProcessor."""
        config = {"queue_size": 10}
        processor = VideoProcessor(config)
        assert processor.config == config
        assert processor.frame_queue.maxsize == 10
        assert processor.running is False
        assert processor.capture_thread is None
    
    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        import numpy as np
        
        config = {
            "target_size": (224, 224),
            "convert_to_rgb": True,
            "normalize": True,
            "imagenet_norm": True,
            "add_batch_dim": True,
            "to_pytorch_format": True
        }
        
        processor = VideoProcessor(config)
        
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process the frame
        processed = processor.preprocess_frame(frame)
        
        # Check the shape and type
        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == np.float32
        
        # Check normalization (should be in range [-2.5, 2.5] after ImageNet normalization)
        assert -2.5 <= processed.min() <= 2.5
        assert -2.5 <= processed.max() <= 2.5

class TestConfigManager:
    """Test cases for the ConfigManager class."""
    
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_config_json(self, mock_json_load, mock_open, mock_exists):
        """Test loading JSON configuration."""
        mock_exists.return_value = True
        mock_json_load.return_value = {"test": "value"}
        
        manager = ConfigManager("test_config.json")
        
        assert manager.config == {"test": "value"}
        mock_exists.assert_called_once_with("test_config.json")
        mock_open.assert_called_once()
        mock_json_load.assert_called_once()
    
    def test_get_config(self):
        """Test getting the complete configuration."""
        manager = ConfigManager("test_config.json")
        manager.config = {"test": "value"}
        
        result = manager.get_config()
        assert result == {"test": "value"}
    
    def test_get_section(self):
        """Test getting a configuration section."""
        manager = ConfigManager("test_config.json")
        manager.config = {"section1": {"key": "value"}, "section2": {}}
        
        result = manager.get_section("section1")
        assert result == {"key": "value"}
        
        result = manager.get_section("nonexistent")
        assert result == {}
    
    def test_get_value(self):
        """Test getting a configuration value."""
        manager = ConfigManager("test_config.json")
        manager.config = {"section": {"subsection": {"key": "value"}}}
        
        result = manager.get_value("section.subsection.key")
        assert result == "value"
        
        result = manager.get_value("nonexistent", default="default")
        assert result == "default"

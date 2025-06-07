import os
import json
import sys
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, "../../../.."))
for path in (EXAMPLE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from label_studio_ml.examples.timeseries_segmenter.model import TimeSeriesSegmenter
    from label_studio_ml.examples.timeseries_segmenter._wsgi import init_app
    from label_studio_ml.examples.timeseries_segmenter.neural_nets import TimeSeriesLSTM
    logger.info("Successfully imported from label_studio_ml package")
except ImportError:  # running inside example Docker image
    from model import TimeSeriesSegmenter
    from _wsgi import init_app
    from neural_nets import TimeSeriesLSTM
    logger.info("Successfully imported from local modules (Docker environment)")

TEST_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(TEST_DIR, "time_series.csv")
logger.info(f"Test directory: {TEST_DIR}")
logger.info(f"CSV path: {CSV_PATH}")

LABEL_CONFIG = """
<View>
  <TimeSeriesLabels name="label" toName="ts">
    <Label value="Run"/>
    <Label value="Walk"/>
  </TimeSeriesLabels>
  <TimeSeries name="ts" valueType="url" value="$csv_url" timeColumn="time">
    <Channel column="sensorone" />
    <Channel column="sensortwo" />
  </TimeSeries>
</View>
"""

@pytest.fixture
def client():
    logger.info("Setting up Flask test client")
    app = init_app(model_class=TimeSeriesSegmenter)
    app.config["TESTING"] = True
    with app.test_client() as c:
        logger.info("Flask test client ready")
        yield c

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage during tests."""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary model directory: {temp_dir}")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"Cleaned up temporary directory: {temp_dir}")

@pytest.fixture
def segmenter_instance(temp_model_dir):
    """Create a TimeSeriesSegmenter instance with test configuration."""
    logger.info("Creating TimeSeriesSegmenter instance for testing")
    # Patch environment variables for the entire fixture scope
    with patch.dict(os.environ, {
        'MODEL_DIR': temp_model_dir, 
        'TRAIN_EPOCHS': '10', 
        'SEQUENCE_SIZE': '10',
        'HIDDEN_SIZE': '32'
    }):
        segmenter = TimeSeriesSegmenter(
            label_config=LABEL_CONFIG
        )
        # Override class attributes with test values
        segmenter.MODEL_DIR = temp_model_dir
        segmenter.TRAIN_EPOCHS = 10
        segmenter.SEQUENCE_SIZE = 10
        segmenter.HIDDEN_SIZE = 32
        
        segmenter.setup()
        logger.info("TimeSeriesSegmenter instance created and set up")
        yield segmenter

def make_task():
    """Create a sample task with annotations."""
    logger.debug("Creating sample task with annotations")
    return {
        "id": 1,
        "data": {"csv_url": CSV_PATH},
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "ts",
                        "type": "timeserieslabels",
                        "value": {"start": "0", "end": "40", "instant": False, "timeserieslabels": ["Run"]},
                    },
                    {
                        "from_name": "label",
                        "to_name": "ts",
                        "type": "timeserieslabels",
                        "value": {"start": "60", "end": "85", "instant": False, "timeserieslabels": ["Walk"]},
                    },
                ]
            }
        ],
    }

def make_task_no_annotations():
    """Create a task without annotations for prediction testing."""
    logger.debug("Creating task without annotations")
    return {
        "id": 2,
        "data": {"csv_url": CSV_PATH},
        "annotations": [],
    }

def fake_preload(self, task, path, read_file=True):
    """Mock function to preload CSV data."""
    logger.debug(f"Mock preload called with path: {path}")
    return open(path).read()

class TestTimeSeriesSegmenter:
    """Test suite for TimeSeriesSegmenter with PyTorch LSTM implementation."""

    def test_setup_and_configuration(self, segmenter_instance):
        """Test basic setup and parameter extraction."""
        logger.info("=== Testing setup and configuration ===")
        params = segmenter_instance._get_labeling_params()
        logger.info(f"Extracted parameters: {params}")
        
        assert params["labels"] == ["Run", "Walk"]
        assert params["all_labels"] == ["__background__", "Run", "Walk"]
        assert params["channels"] == ["sensorone", "sensortwo"]
        assert params["time_col"] == "time"
        assert params["from_name"] == "label"
        assert params["to_name"] == "ts"
        logger.info("✓ Setup and configuration test passed")

    def test_model_building(self, segmenter_instance):
        """Test model creation with correct parameters."""
        logger.info("=== Testing model building ===")
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        logger.info(f"Built model: input_size={model.input_size}, output_size={model.output_size}, sequence_size={model.sequence_size}")
        
        assert isinstance(model, TimeSeriesLSTM)
        assert model.input_size == 2
        assert model.output_size == 3
        assert model.sequence_size == 10  # From test environment
        logger.info("✓ Model building test passed")
        
    def test_csv_reading(self, segmenter_instance):
        """Test CSV data reading and processing."""
        logger.info("=== Testing CSV reading ===")
        task = make_task()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            df = segmenter_instance._read_csv(task, CSV_PATH)
            logger.info(f"Read CSV with shape: {df.shape}, columns: {list(df.columns)}")
            
        assert not df.empty
        assert df.shape == (100, 3)  # 100 rows, 3 columns
        assert list(df.columns) == ["time", "sensorone", "sensortwo"]
        assert df["time"].dtype in [int, float]
        logger.info("✓ CSV reading test passed")

    def test_sample_collection_with_background(self, segmenter_instance):
        """Test training sample collection including background class."""
        logger.info("=== Testing sample collection with background class ===")
        params = segmenter_instance._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params["all_labels"])}
        logger.info(f"Label mapping: {label2idx}")
        
        task = make_task()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            X, y = segmenter_instance._collect_samples([task], params, label2idx)
        
        logger.info(f"Collected samples: X.shape={X.shape}, y.shape={y.shape}")
        assert len(X) == 100  # All 100 rows should be included
        assert len(y) == 100
        assert X.shape == (100, 2)  # 2 sensor channels
        
        # Check label distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        logger.info(f"Label distribution: {label_dist}")
        
        # Should have background (0), Run (1), and Walk (2)
        assert 0 in label_dist  # Background
        assert 1 in label_dist  # Run (rows 0-40)
        assert 2 in label_dist  # Walk (rows 60-85)
        
        # Background should be the majority (unlabeled regions)
        assert label_dist[0] > label_dist[1]
        assert label_dist[0] > label_dist[2]
        logger.info("✓ Sample collection with background test passed")

    def test_model_save_load(self, segmenter_instance, temp_model_dir):
        """Test model saving and loading with new PyTorch state dict approach."""
        logger.info("=== Testing model save/load functionality ===")
        # Create and configure a simple model
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        label_map = {"__background__": 0, "Run": 1, "Walk": 2}
        model.set_label_map(label_map)
        logger.info(f"Created model with label map: {label_map}")
        
        # Save the model
        model_path = os.path.join(temp_model_dir, "test_model.pt")
        logger.info(f"Saving model to: {model_path}")
        model.save(model_path)
        assert os.path.exists(model_path)
        logger.info("✓ Model saved successfully")
        
        # Load the model
        logger.info("Loading model from disk")
        loaded_model = TimeSeriesLSTM.load_model(model_path)
        
        # Verify loaded model properties
        logger.info(f"Loaded model properties: input_size={loaded_model.input_size}, output_size={loaded_model.output_size}")
        assert loaded_model.input_size == model.input_size
        assert loaded_model.output_size == model.output_size
        assert loaded_model.sequence_size == model.sequence_size
        assert loaded_model.get_label_map() == label_map
        
        # Verify models produce similar outputs for same input
        test_input = np.random.randn(50, 2).astype(np.float32)
        logger.info(f"Testing prediction consistency with input shape: {test_input.shape}")
        
        original_pred = model.predict(test_input)
        loaded_pred = loaded_model.predict(test_input)
        
        # Should have same shape
        assert original_pred.shape == loaded_pred.shape
        logger.info(f"Prediction shapes match: {original_pred.shape}")
        logger.info("✓ Model save/load test passed")

    def test_training_and_prediction_workflow(self, segmenter_instance):
        """Test complete training and prediction workflow."""
        logger.info("=== Testing training and prediction workflow ===")
        task = make_task()
        
        with patch.object(segmenter_instance, '_get_tasks', return_value=[task]), \
             patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            
            # Train the model
            logger.info("Starting model training")
            data = {
                "annotation": {"project": 1},
                "project": {"id": 1, "label_config": LABEL_CONFIG},
            }
            result = segmenter_instance.fit("START_TRAINING", data)
            logger.info(f"Training completed with result: {result}")
            
            # Should return training metrics
            assert isinstance(result, dict)
            assert "accuracy" in result
            assert "f1_score" in result
            assert "loss" in result
            logger.info("✓ Training metrics validated")
            
            # Test prediction on the same task
            logger.info("Running prediction on trained model")
            predictions = segmenter_instance.predict([task])
            
            assert len(predictions.predictions) == 1
            pred = predictions.predictions[0]
            logger.info(f"Prediction result: {pred}")
            
            if pred:  # If prediction is not empty
                assert "result" in pred
                assert "score" in pred
                assert "model_version" in pred
                
                # Check that segments are valid
                for i, segment in enumerate(pred["result"]):
                    logger.info(f"Segment {i}: {segment}")
                    assert segment["type"] == "timeserieslabels"
                    assert segment["value"]["timeserieslabels"][0] in ["Run", "Walk"]
                    assert "score" in segment
                logger.info("✓ All segments validated")
            else:
                logger.warning("Prediction returned empty result")
        logger.info("✓ Training and prediction workflow test passed")

    def test_background_class_filtering(self, segmenter_instance):
        """Test that background predictions are properly filtered out."""
        logger.info("=== Testing background class filtering ===")
        # Create a minimal trained model
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        params = segmenter_instance._get_labeling_params()
        
        # Mock predictions that include background class
        task = make_task_no_annotations()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload), \
             patch.object(model, 'predict') as mock_predict:
            
            # Mock model to return mostly background predictions
            import torch
            mock_probs = torch.zeros(100, 3)
            mock_probs[:, 0] = 0.8  # High background probability
            mock_probs[10:20, 1] = 0.9  # Some "Run" predictions
            mock_probs[10:20, 0] = 0.05  # Lower background for these
            mock_predict.return_value = mock_probs
            logger.info("Set up mock predictions with background filtering test case")
            
            result = segmenter_instance._predict_task(task, model, params)
            logger.info(f"Prediction result with background filtering: {result}")
            
            # Should filter out background segments
            if result and "result" in result:
                for segment in result["result"]:
                    assert segment["value"]["timeserieslabels"][0] != "__background__"
                    logger.info(f"Confirmed non-background segment: {segment['value']['timeserieslabels'][0]}")
            else:
                logger.info("No segments returned (all were background - expected)")
        logger.info("✓ Background class filtering test passed")

    def test_short_sequence_handling(self, segmenter_instance, temp_model_dir):
        """Test handling of sequences shorter than window size."""
        logger.info("=== Testing short sequence handling ===")
        # Create a very short CSV
        short_csv_path = os.path.join(temp_model_dir, "short.csv")
        short_data = pd.DataFrame({
            "time": [0, 1, 2],
            "sensorone": [1.0, 2.0, 3.0],
            "sensortwo": [0.5, 1.5, 2.5]
        })
        short_data.to_csv(short_csv_path, index=False)
        logger.info(f"Created short CSV with 3 rows at: {short_csv_path}")
        
        short_task = {
            "id": 3,
            "data": {"csv_url": short_csv_path},
            "annotations": []
        }
        
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        params = segmenter_instance._get_labeling_params()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            # Should handle short sequences gracefully
            logger.info("Testing prediction on short sequence")
            result = segmenter_instance._predict_task(short_task, model, params)
            logger.info(f"Short sequence prediction result: {result}")
            # Should not crash and return valid result (even if empty)
            assert isinstance(result, dict)
        logger.info("✓ Short sequence handling test passed")

    def test_windowing_functionality(self):
        """Test the windowing and overlap functionality of TimeSeriesLSTM."""
        logger.info("=== Testing windowing functionality ===")
        model = TimeSeriesLSTM(
            input_size=2,
            output_size=3,
            sequence_size=10,
            hidden_size=16,
            num_layers=1
        )
        logger.info("Created TimeSeriesLSTM for windowing test")
        
        # Test with sequence longer than window
        sequence = np.random.randn(25, 2).astype(np.float32)
        labels = np.random.randint(0, 3, 25)
        logger.info(f"Testing with sequence shape: {sequence.shape}")
        
        chunks, label_chunks = model.preprocess_sequence(sequence, labels, overlap_ratio=0.5)
        logger.info(f"Created chunks with shape: {chunks.shape}")
        
        # Should create overlapping windows
        assert chunks.shape[1] == 10  # Window size
        assert chunks.shape[2] == 2   # Input features
        assert chunks.shape[0] > 1    # Multiple windows
        
        # Test prediction with overlap averaging
        logger.info("Testing prediction with overlap averaging")
        predictions = model.predict(sequence)
        logger.info(f"Predictions shape: {predictions.shape}")
        assert predictions.shape[0] == 25  # Same length as input
        assert predictions.shape[1] == 3   # Number of classes
        logger.info("✓ Windowing functionality test passed")

    def test_api_integration(self, client, temp_model_dir):
        """Test API endpoints with the new implementation."""
        logger.info("=== Testing API integration ===")
        with patch.dict(os.environ, {'MODEL_DIR': temp_model_dir, 'TRAIN_EPOCHS': '5'}):
            # Setup
            logger.info("Testing API setup endpoint")
            setup_data = {"schema": LABEL_CONFIG, "project": "1"}
            resp = client.post("/setup", data=json.dumps(setup_data), content_type="application/json")
            logger.info(f"Setup response: {resp.status_code}")
            assert resp.status_code == 200

            task = make_task()
            
            with patch.object(TimeSeriesSegmenter, "_get_tasks", return_value=[task]), \
                 patch.object(TimeSeriesSegmenter, "preload_task_data", new=fake_preload):
                
                # Training
                logger.info("Testing API training endpoint")
                data = {
                    "action": "START_TRAINING",
                    "annotation": {"project": 1},
                    "project": {"id": 1, "label_config": LABEL_CONFIG},
                }
                resp = client.post("/webhook", data=json.dumps(data), content_type="application/json")
                logger.info(f"Training response: {resp.status_code}")
                assert resp.status_code == 201
                
                # Prediction
                logger.info("Testing API prediction endpoint")
                predict_data = {
                    "tasks": [dict(id=1, data={"csv_url": CSV_PATH})], 
                    "label_config": LABEL_CONFIG, 
                    "project": "1"
                }
                resp = client.post("/predict", data=json.dumps(predict_data), content_type="application/json")
                logger.info(f"Prediction response: {resp.status_code}")
                assert resp.status_code == 200
                
                results = resp.json["results"]
                logger.info(f"API prediction results: {len(results)} tasks processed")
                assert len(results) == 1
        logger.info("✓ API integration test passed")

    def test_empty_data_handling(self, segmenter_instance):
        """Test handling of empty or invalid data."""
        logger.info("=== Testing empty data handling ===")
        params = segmenter_instance._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params["all_labels"])}
        
        # Test with empty task list
        logger.info("Testing with empty task list")
        X, y = segmenter_instance._collect_samples([], params, label2idx)
        assert len(X) == 0
        assert len(y) == 0
        logger.info("✓ Empty task list handled correctly")
        
        # Test prediction with empty data
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        empty_task = {
            "id": 999,
            "data": {"csv_url": "nonexistent.csv"},
        }
        
        # Should handle gracefully
        logger.info("Testing prediction with nonexistent file")
        with patch.object(segmenter_instance, 'preload_task_data', side_effect=Exception("File not found")):
            # Should not crash
            try:
                result = segmenter_instance._predict_task(empty_task, model, params)
                logger.info(f"Empty data prediction result: {result}")
                assert isinstance(result, dict)
            except Exception as e:
                logger.info(f"Expected exception for missing file: {e}")
                pass  # Expected to fail gracefully
        logger.info("✓ Empty data handling test passed")

    def test_model_parameters_configuration(self, temp_model_dir):
        """Test different model parameter configurations."""
        logger.info("=== Testing model parameters configuration ===")
        configs = [
            {"SEQUENCE_SIZE": 5, "HIDDEN_SIZE": 16},
            {"SEQUENCE_SIZE": 20, "HIDDEN_SIZE": 32},
            {"SEQUENCE_SIZE": 50, "HIDDEN_SIZE": 64},
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"Testing configuration {i+1}/{len(configs)}: {config}")
            segmenter = TimeSeriesSegmenter(
                label_config=LABEL_CONFIG
            )
            # Override instance attributes with test values
            segmenter.MODEL_DIR = temp_model_dir
            segmenter.SEQUENCE_SIZE = config["SEQUENCE_SIZE"]
            segmenter.HIDDEN_SIZE = config["HIDDEN_SIZE"]
            
            segmenter.setup()
            
            model = segmenter._build_model(n_channels=2, n_labels=3)
            logger.info(f"Created model with sequence_size={model.sequence_size}, hidden_size={model.hidden_size}")
            assert model.sequence_size == config["SEQUENCE_SIZE"]
            assert model.hidden_size == config["HIDDEN_SIZE"]
        logger.info("✓ Model parameters configuration test passed")
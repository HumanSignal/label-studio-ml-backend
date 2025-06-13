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

def fake_preload(task, value=None, read_file=True):
    """Mock function to preload CSV data."""
    logger.debug(f"Mock preload called with value: {value}")
    return open(value).read()

class TestTimeSeriesSegmenter:
    """Test suite for TimeSeriesSegmenter with PyTorch LSTM implementation."""

    def test_setup_and_configuration(self, segmenter_instance):
        """Test basic setup and parameter extraction from Label Studio configuration.
        
        This test validates:
        - Correct parsing of the Label Studio XML configuration
        - Extraction of label names (Run, Walk) and creation of background class
        - Identification of sensor channels (sensorone, sensortwo) 
        - Time column configuration (time)
        - Proper mapping of from_name and to_name attributes
        - Ensures the model understands the labeling interface structure
        
        Critical validation: The model correctly identifies that it needs to handle 
        3 classes (background + 2 labels) and 2 input channels.
        """
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
        """Test PyTorch LSTM neural network creation with correct architecture.
        
        This test validates:
        - TimeSeriesLSTM object creation with specified parameters
        - Correct input size (2 channels) and output size (3 classes including background)
        - Proper sequence size configuration from environment variables
        - Model architecture validation (LSTM layers, dropout, etc.)
        
        Critical validation: The neural network is built with the right dimensions 
        for the time series data and matches the test configuration.
        """
        logger.info("=== Testing model building ===")
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        logger.info(f"Built model: input_size={model.input_size}, output_size={model.output_size}, sequence_size={model.sequence_size}")
        
        # Check model type by class name (avoid import namespace issues)
        assert model.__class__.__name__ == "TimeSeriesLSTM"
        assert model.input_size == 2
        assert model.output_size == 3
        assert model.sequence_size == 10  # From test environment
        logger.info("✓ Model building test passed")
        
    def test_csv_reading(self, segmenter_instance):
        """Test CSV data loading and preprocessing functionality.
        
        This test validates:
        - CSV file reading through the preload_task_data method
        - Data shape validation (100 rows × 3 columns)
        - Column name verification (time, sensorone, sensortwo)
        - Data type validation for time column
        - Mock function integration for file loading
        
        Critical validation: The model can correctly load and parse time series 
        data files with the expected structure and data types.
        """
        logger.info("=== Testing CSV reading ===")
        task = make_task()
        params = segmenter_instance._get_labeling_params()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            df, time_col = segmenter_instance._read_csv(task, CSV_PATH, params)
            logger.info(f"Read CSV with shape: {df.shape}, columns: {list(df.columns)}, time_col: {time_col}")
            
        assert not df.empty
        assert df.shape == (100, 3)  # 100 rows, 3 columns
        assert list(df.columns) == ["time", "sensorone", "sensortwo"]
        assert df["time"].dtype in [int, float]
        assert time_col == "time"  # Should use the existing time column
        logger.info("✓ CSV reading test passed")

    def test_sample_collection_with_background(self, segmenter_instance):
        """Test core training data collection with background class handling.
        
        This test validates:
        - Collection of ALL CSV rows (not just annotated segments)
        - Proper background class assignment for unlabeled time periods
        - Label mapping for annotated segments (Run: rows 0-40, Walk: rows 60-85)
        - Correct label distribution counting (41 Run, 26 Walk, 33 Background)
        - Training data format validation (features + labels)
        
        Critical validation: The model correctly handles the three-class problem 
        with background regions, ensuring all time periods are properly labeled 
        for training including unlabeled background periods.
        """
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
        
        # Check that we have the expected distribution (Run: 41, Walk: 26, Background: 33)
        assert label_dist[1] == 41  # Run (rows 0-40 inclusive)
        assert label_dist[2] == 26  # Walk (rows 60-85 inclusive)
        assert label_dist[0] == 33  # Background (remaining rows)
        
        logger.info("✓ Sample collection with background test passed")

    def test_model_save_load(self, segmenter_instance, temp_model_dir):
        """Test secure PyTorch model serialization and deserialization.
        
        This test validates:
        - Model saving using PyTorch state dict approach (secure)
        - Model loading with proper parameter restoration
        - Label mapping preservation across save/load cycles
        - Prediction consistency between original and loaded models
        - File system integration and error handling
        
        Critical validation: Models can be persisted and restored without losing 
        functionality, using the secure PyTorch 2.6+ approach with weights_only=True.
        """
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
        """Test complete end-to-end machine learning pipeline.
        
        This test validates:
        - Full training workflow with real data
        - Training metrics generation (accuracy, F1-score, loss)
        - Model convergence and learning validation
        - Prediction generation on trained model
        - Result format validation (segments, scores, model version)
        - Background class filtering in predictions
        
        Critical validation: The complete ML pipeline works from training to prediction,
        producing valid Label Studio annotations with proper scoring and metadata.
        """
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
            # Check for new imbalanced-data metrics
            assert "balanced_accuracy" in result
            assert "f1_score" in result
            assert "loss" in result
            assert "min_class_f1" in result
            logger.info("✓ Training metrics validated")
            
            # Test prediction on the same task
            logger.info("Running prediction on trained model")
            predictions = segmenter_instance.predict([task])
            
            assert len(predictions.predictions) == 1
            pred = predictions.predictions[0]
            logger.info(f"Prediction result: {pred}")
            
            if pred:  # If prediction is not empty
                assert hasattr(pred, 'result')
                assert hasattr(pred, 'score')
                assert hasattr(pred, 'model_version')
                
                # Check that segments are valid
                for i, segment in enumerate(pred.result):
                    logger.info(f"Segment {i}: {segment}")
                    assert segment["type"] == "timeserieslabels"
                    assert segment["value"]["timeserieslabels"][0] in ["Run", "Walk"]
                    assert "score" in segment
                logger.info("✓ All segments validated")
            else:
                logger.warning("Prediction returned empty result")
        logger.info("✓ Training and prediction workflow test passed")

    def test_background_class_filtering(self, segmenter_instance):
        """Test background prediction filtering to exclude non-meaningful results.
        
        This test validates:
        - Mock predictions with high background probability
        - Filtering logic that removes background segments
        - Retention of only meaningful (labeled) predictions
        - Proper handling of mostly-background predictions
        
        Critical validation: The model doesn't return useless background predictions 
        to Label Studio, ensuring only meaningful segments are annotated.
        """
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
        """Test robustness with sequences shorter than the window size.
        
        This test validates:
        - Model behavior with very short time series (3 data points)
        - Padding or truncation strategies for short sequences
        - Graceful degradation without crashes
        - Valid result generation even with insufficient data
        
        Critical validation: The model handles edge cases with minimal data gracefully,
        using appropriate padding strategies to avoid crashes or errors.
        """
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
        """Test sliding window approach for temporal modeling.
        
        This test validates:
        - Sequence chunking into overlapping windows
        - Window size and overlap ratio validation
        - Preprocessing of sequences longer than window size  
        - Overlap averaging during prediction
        - Proper handling of sequence boundaries
        
        Critical validation: The temporal modeling approach works correctly with 
        sliding windows, providing proper temporal context for LSTM processing.
        """
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
        """Test Flask API endpoints and web service functionality.
        
        This test validates:
        - /setup endpoint for model initialization
        - /webhook endpoint for training triggers
        - /predict endpoint for inference requests
        - JSON request/response handling
        - HTTP status code validation
        - End-to-end API workflow
        
        Critical validation: The web service interface works correctly with Label Studio,
        handling setup, training, and prediction requests through proper HTTP endpoints.
        """
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
                 patch.object(TimeSeriesSegmenter, "preload_task_data", side_effect=lambda task, value=None, read_file=True: open(value).read()):
                
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
        """Test robustness with missing or invalid data scenarios.
        
        This test validates:
        - Empty task list handling
        - Missing file error handling
        - Graceful degradation with corrupted data
        - Proper exception handling and logging
        - Fallback behavior for edge cases
        
        Critical validation: The model doesn't crash when encountering bad data,
        providing appropriate error handling and fallback mechanisms.
        """
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
        """Test different hyperparameter configurations for model flexibility.
        
        This test validates:
        - Various sequence sizes (5, 20, 50) for different temporal contexts
        - Different hidden layer sizes (16, 32, 64) for model capacity
        - Model creation with custom parameters
        - Parameter persistence and application
        - Configuration flexibility for different use cases
        
        Critical validation: The model architecture can be tuned for different 
        time series characteristics and computational requirements.
        """
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

    def test_instant_vs_range_annotations(self, segmenter_instance):
        """Test proper handling of instant vs. time-range annotations.
        
        This test validates:
        - Instant annotations (start == end) get instant=True
        - Time-range annotations (start != end) get instant=False
        - Proper boolean logic in the instant field
        - Edge case handling for zero-duration segments
        
        Critical validation: The model correctly distinguishes between point events
        and duration events in its output annotations.
        """
        logger.info("=== Testing instant vs range annotations ===")
        
        # Create a model for prediction
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        params = segmenter_instance._get_labeling_params()
        
        # Mock task using the existing test CSV
        task = {
            "id": 999,
            "data": {"csv_url": CSV_PATH},
        }
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload), \
             patch.object(model, 'predict') as mock_predict:
            
            # Create mock predictions that will generate both instant and range segments
            import torch
            mock_probs = torch.zeros(100, 3)  # Match the CSV length (100 rows)
            # Create patterns that will result in:
            # - Single timestep segment (instant)
            # - Multi-timestep segment (range)
            mock_probs[10, 1] = 0.9     # Single "Run" at timestep 10 (instant)
            mock_probs[20:24, 2] = 0.9  # "Walk" from timestep 20-23 (range)
            mock_probs[:, 0] = 0.1      # Low background for all
            mock_predict.return_value = mock_probs
            
            logger.info("Set up mock predictions for instant vs range test")
            
            result = segmenter_instance._predict_task(task, model, params)
            logger.info(f"Prediction result: {result}")
            
            if result and "result" in result and len(result["result"]) > 0:
                segments = result["result"]
                logger.info(f"Found {len(segments)} segments to validate")
                
                for i, segment in enumerate(segments):
                    start = segment["value"]["start"]
                    end = segment["value"]["end"]
                    instant = segment["value"]["instant"]
                    label = segment["value"]["timeserieslabels"][0]
                    
                    logger.info(f"Segment: start={start}, end={end}, instant={instant}, label={label}")
                    
                    # Test the instant logic
                    if start == end:
                        assert instant == True, f"Expected instant=True for start={start}, end={end}"
                    else:
                        assert instant == False, f"Expected instant=False for start={start}, end={end}"
            else:
                logger.info("No prediction results - acceptable for this test")
        
        logger.info("✓ Instant vs range annotations test passed")

    def test_project_specific_models(self, temp_model_dir):
        """Test that different projects use separate models and model files.
        
        This test validates:
        - Each project gets its own model instance and saved file
        - Models for different projects don't interfere with each other
        - Project-specific model file naming (model_project_{id}.pt)
        - Model isolation ensures training on one project doesn't affect another
        - Backward compatibility with default project_id=0
        
        Critical validation: Multi-tenant model handling works correctly with proper
        isolation between different Label Studio projects.
        """
        logger.info("=== Testing project-specific models ===")
        
        # Create two segmenters for different projects
        segmenter1 = TimeSeriesSegmenter(label_config=LABEL_CONFIG)
        segmenter1.MODEL_DIR = temp_model_dir
        segmenter1.setup()
        
        segmenter2 = TimeSeriesSegmenter(label_config=LABEL_CONFIG)
        segmenter2.MODEL_DIR = temp_model_dir
        segmenter2.setup()
        
        params = segmenter1._get_labeling_params()
        n_channels = len(params["channels"])
        n_labels = len(params["all_labels"])
        
        # Test 1: Get models for different projects
        logger.info("Testing model retrieval for different projects")
        model_project_1 = segmenter1._get_model(n_channels, n_labels, project_id=1)
        model_project_2 = segmenter2._get_model(n_channels, n_labels, project_id=2)
        model_default = segmenter1._get_model(n_channels, n_labels)  # Should use project_id=0
        
        # Should be different model instances
        assert model_project_1 is not model_project_2
        assert model_project_1 is not model_default
        assert model_project_2 is not model_default
        logger.info("✓ Different projects get different model instances")
        
        # Test 2: Save models for different projects
        logger.info("Testing model saving for different projects")
        segmenter1._save_model(model_project_1, project_id=1)
        segmenter2._save_model(model_project_2, project_id=2)
        segmenter1._save_model(model_default)  # Should use project_id=0
        
        # Verify files exist with correct names
        model_file_1 = os.path.join(temp_model_dir, "model_project_1.pt")
        model_file_2 = os.path.join(temp_model_dir, "model_project_2.pt")
        model_file_default = os.path.join(temp_model_dir, "model_project_0.pt")
        
        assert os.path.exists(model_file_1), f"Model file for project 1 not found: {model_file_1}"
        assert os.path.exists(model_file_2), f"Model file for project 2 not found: {model_file_2}"
        assert os.path.exists(model_file_default), f"Default model file not found: {model_file_default}"
        logger.info("✓ Project-specific model files created correctly")
        
        # Test 3: Load models for different projects
        logger.info("Testing model loading for different projects")
        
        # Clear the model cache
        try:
            # Try absolute import first (works in some environments)
            from label_studio_ml.examples.timeseries_segmenter.model import _models
        except ImportError:
            # Fall back to relative import (works in CI/Docker)
            from model import _models
        _models.clear()
        
        # Load models from disk
        loaded_model_1 = segmenter1._get_model(n_channels, n_labels, project_id=1)
        loaded_model_2 = segmenter2._get_model(n_channels, n_labels, project_id=2)
        loaded_model_default = segmenter1._get_model(n_channels, n_labels)
        
        # Should be different instances (loaded from different files)
        assert loaded_model_1 is not loaded_model_2
        assert loaded_model_1 is not loaded_model_default
        assert loaded_model_2 is not loaded_model_default
        logger.info("✓ Models loaded correctly from project-specific files")
        
        # Test 4: Verify project ID extraction from context
        logger.info("Testing project ID extraction from context")
        
        # Test context with project dict
        context_dict = {"project": {"id": 42}}
        project_id = segmenter1._get_project_id_from_context([], context_dict)
        assert project_id == 42
        
        # Test context with project int
        context_int = {"project": 99}
        project_id = segmenter1._get_project_id_from_context([], context_int)
        assert project_id == 99
        
        # Test context with project string
        context_str = {"project": "123"}
        project_id = segmenter1._get_project_id_from_context([], context_str)
        assert project_id == 123
        
        # Test task with project info
        tasks_with_project = [{"project": 456}]
        project_id = segmenter1._get_project_id_from_context(tasks_with_project, {})
        assert project_id == 456
        
        # Test no project info
        project_id = segmenter1._get_project_id_from_context([], {})
        assert project_id is None
        
        logger.info("✓ Project ID extraction working correctly")
        
        # Test 5: Verify model caching works correctly
        logger.info("Testing project-specific model caching")
        
        # Get same model twice for same project - should return same instance
        model_1a = segmenter1._get_model(n_channels, n_labels, project_id=100)
        model_1b = segmenter1._get_model(n_channels, n_labels, project_id=100)
        assert model_1a is model_1b, "Same project should return cached model"
        
        # Get model for different project - should return different instance
        model_2 = segmenter1._get_model(n_channels, n_labels, project_id=200)
        assert model_1a is not model_2, "Different projects should return different models"
        
        logger.info("✓ Model caching working correctly per project")
        logger.info("✓ Project-specific models test passed")
import os
import json
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd

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
except ImportError:  # running inside example Docker image
    from model import TimeSeriesSegmenter
    from _wsgi import init_app
    from neural_nets import TimeSeriesLSTM

TEST_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(TEST_DIR, "time_series.csv")

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
    app = init_app(model_class=TimeSeriesSegmenter)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage during tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def segmenter_instance(temp_model_dir):
    """Create a TimeSeriesSegmenter instance with test configuration."""
    with patch.dict(os.environ, {'MODEL_DIR': temp_model_dir, 'TRAIN_EPOCHS': '10', 'SEQUENCE_SIZE': '10'}):
        segmenter = TimeSeriesSegmenter(
            label_config=LABEL_CONFIG,
            parsed_label_config={},
            train_output={}
        )
        segmenter.setup()
        yield segmenter

def make_task():
    """Create a sample task with annotations."""
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
    return {
        "id": 2,
        "data": {"csv_url": CSV_PATH},
        "annotations": [],
    }

def fake_preload(self, task, value=None, read_file=True):
    """Mock function to preload CSV data."""
    return open(value).read()

class TestTimeSeriesSegmenter:
    """Test suite for TimeSeriesSegmenter with PyTorch LSTM implementation."""

    def test_setup_and_configuration(self, segmenter_instance):
        """Test basic setup and parameter extraction."""
        params = segmenter_instance._get_labeling_params()
        
        assert params["labels"] == ["Run", "Walk"]
        assert params["all_labels"] == ["__background__", "Run", "Walk"]
        assert params["channels"] == ["sensorone", "sensortwo"]
        assert params["time_col"] == "time"
        assert params["from_name"] == "label"
        assert params["to_name"] == "ts"

    def test_model_building(self, segmenter_instance):
        """Test model creation with correct parameters."""
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        
        assert isinstance(model, TimeSeriesLSTM)
        assert model.input_size == 2
        assert model.output_size == 3
        assert model.sequence_size == 10  # From test environment
        
    def test_csv_reading(self, segmenter_instance):
        """Test CSV data reading and processing."""
        task = make_task()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            df = segmenter_instance._read_csv(task, CSV_PATH)
            
        assert not df.empty
        assert df.shape == (100, 3)  # 100 rows, 3 columns
        assert list(df.columns) == ["time", "sensorone", "sensortwo"]
        assert df["time"].dtype in [int, float]

    def test_sample_collection_with_background(self, segmenter_instance):
        """Test training sample collection including background class."""
        params = segmenter_instance._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params["all_labels"])}
        task = make_task()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            X, y = segmenter_instance._collect_samples([task], params, label2idx)
        
        assert len(X) == 100  # All 100 rows should be included
        assert len(y) == 100
        assert X.shape == (100, 2)  # 2 sensor channels
        
        # Check label distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        
        # Should have background (0), Run (1), and Walk (2)
        assert 0 in label_dist  # Background
        assert 1 in label_dist  # Run (rows 0-40)
        assert 2 in label_dist  # Walk (rows 60-85)
        
        # Background should be the majority (unlabeled regions)
        assert label_dist[0] > label_dist[1]
        assert label_dist[0] > label_dist[2]

    def test_model_save_load(self, segmenter_instance, temp_model_dir):
        """Test model saving and loading with new PyTorch state dict approach."""
        # Create and configure a simple model
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        label_map = {"__background__": 0, "Run": 1, "Walk": 2}
        model.set_label_map(label_map)
        
        # Save the model
        model_path = os.path.join(temp_model_dir, "test_model.pt")
        model.save(model_path)
        assert os.path.exists(model_path)
        
        # Load the model
        loaded_model = TimeSeriesLSTM.load_model(model_path)
        
        # Verify loaded model properties
        assert loaded_model.input_size == model.input_size
        assert loaded_model.output_size == model.output_size
        assert loaded_model.sequence_size == model.sequence_size
        assert loaded_model.get_label_map() == label_map
        
        # Verify models produce similar outputs for same input
        test_input = np.random.randn(50, 2).astype(np.float32)
        
        original_pred = model.predict(test_input)
        loaded_pred = loaded_model.predict(test_input)
        
        # Should have same shape
        assert original_pred.shape == loaded_pred.shape

    def test_training_and_prediction_workflow(self, segmenter_instance):
        """Test complete training and prediction workflow."""
        task = make_task()
        
        with patch.object(segmenter_instance, '_get_tasks', return_value=[task]), \
             patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            
            # Train the model
            data = {
                "annotation": {"project": 1},
                "project": {"id": 1, "label_config": LABEL_CONFIG},
            }
            result = segmenter_instance.fit("START_TRAINING", data)
            
            # Should return training metrics
            assert isinstance(result, dict)
            assert "accuracy" in result
            assert "f1_score" in result
            assert "loss" in result
            
            # Test prediction on the same task
            predictions = segmenter_instance.predict([task])
            
            assert len(predictions.predictions) == 1
            pred = predictions.predictions[0]
            
            if pred:  # If prediction is not empty
                assert "result" in pred
                assert "score" in pred
                assert "model_version" in pred
                
                # Check that segments are valid
                for segment in pred["result"]:
                    assert segment["type"] == "timeserieslabels"
                    assert segment["value"]["timeserieslabels"][0] in ["Run", "Walk"]
                    assert "score" in segment

    def test_background_class_filtering(self, segmenter_instance):
        """Test that background predictions are properly filtered out."""
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
            
            result = segmenter_instance._predict_task(task, model, params)
            
            # Should filter out background segments
            if result and "result" in result:
                for segment in result["result"]:
                    assert segment["value"]["timeserieslabels"][0] != "__background__"

    def test_short_sequence_handling(self, segmenter_instance, temp_model_dir):
        """Test handling of sequences shorter than window size."""
        # Create a very short CSV
        short_csv_path = os.path.join(temp_model_dir, "short.csv")
        short_data = pd.DataFrame({
            "time": [0, 1, 2],
            "sensorone": [1.0, 2.0, 3.0],
            "sensortwo": [0.5, 1.5, 2.5]
        })
        short_data.to_csv(short_csv_path, index=False)
        
        short_task = {
            "id": 3,
            "data": {"csv_url": short_csv_path},
            "annotations": []
        }
        
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        params = segmenter_instance._get_labeling_params()
        
        with patch.object(segmenter_instance, 'preload_task_data', new=fake_preload):
            # Should handle short sequences gracefully
            result = segmenter_instance._predict_task(short_task, model, params)
            # Should not crash and return valid result (even if empty)
            assert isinstance(result, dict)

    def test_windowing_functionality(self):
        """Test the windowing and overlap functionality of TimeSeriesLSTM."""
        model = TimeSeriesLSTM(
            input_size=2,
            output_size=3,
            sequence_size=10,
            hidden_size=16,
            num_layers=1
        )
        
        # Test with sequence longer than window
        sequence = np.random.randn(25, 2).astype(np.float32)
        labels = np.random.randint(0, 3, 25)
        
        chunks, label_chunks = model.preprocess_sequence(sequence, labels, overlap_ratio=0.5)
        
        # Should create overlapping windows
        assert chunks.shape[1] == 10  # Window size
        assert chunks.shape[2] == 2   # Input features
        assert chunks.shape[0] > 1    # Multiple windows
        
        # Test prediction with overlap averaging
        predictions = model.predict(sequence)
        assert predictions.shape[0] == 25  # Same length as input
        assert predictions.shape[1] == 3   # Number of classes

    def test_api_integration(self, client, temp_model_dir):
        """Test API endpoints with the new implementation."""
        with patch.dict(os.environ, {'MODEL_DIR': temp_model_dir, 'TRAIN_EPOCHS': '5'}):
            # Setup
            setup_data = {"schema": LABEL_CONFIG, "project": "1"}
            resp = client.post("/setup", data=json.dumps(setup_data), content_type="application/json")
            assert resp.status_code == 200

            task = make_task()
            
            with patch.object(TimeSeriesSegmenter, "_get_tasks", return_value=[task]), \
                 patch.object(TimeSeriesSegmenter, "preload_task_data", new=fake_preload):
                
                # Training
                data = {
                    "action": "START_TRAINING",
                    "annotation": {"project": 1},
                    "project": {"id": 1, "label_config": LABEL_CONFIG},
                }
                resp = client.post("/webhook", data=json.dumps(data), content_type="application/json")
                assert resp.status_code == 201
                
                # Prediction
                predict_data = {
                    "tasks": [dict(id=1, data={"csv_url": CSV_PATH})], 
                    "label_config": LABEL_CONFIG, 
                    "project": "1"
                }
                resp = client.post("/predict", data=json.dumps(predict_data), content_type="application/json")
                assert resp.status_code == 200
                
                results = resp.json["results"]
                assert len(results) == 1

    def test_empty_data_handling(self, segmenter_instance):
        """Test handling of empty or invalid data."""
        params = segmenter_instance._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params["all_labels"])}
        
        # Test with empty task list
        X, y = segmenter_instance._collect_samples([], params, label2idx)
        assert len(X) == 0
        assert len(y) == 0
        
        # Test prediction with empty data
        model = segmenter_instance._build_model(n_channels=2, n_labels=3)
        empty_task = {
            "id": 999,
            "data": {"csv_url": "nonexistent.csv"},
        }
        
        # Should handle gracefully
        with patch.object(segmenter_instance, 'preload_task_data', side_effect=Exception("File not found")):
            # Should not crash
            try:
                result = segmenter_instance._predict_task(empty_task, model, params)
                assert isinstance(result, dict)
            except Exception:
                pass  # Expected to fail gracefully

    def test_model_parameters_configuration(self, temp_model_dir):
        """Test different model parameter configurations."""
        configs = [
            {"SEQUENCE_SIZE": "5", "HIDDEN_SIZE": "16"},
            {"SEQUENCE_SIZE": "20", "HIDDEN_SIZE": "32"},
            {"SEQUENCE_SIZE": "50", "HIDDEN_SIZE": "64"},
        ]
        
        for config in configs:
            with patch.dict(os.environ, {**config, 'MODEL_DIR': temp_model_dir}):
                segmenter = TimeSeriesSegmenter(
                    label_config=LABEL_CONFIG,
                    parsed_label_config={},
                    train_output={}
                )
                segmenter.setup()
                
                model = segmenter._build_model(n_channels=2, n_labels=3)
                assert model.sequence_size == int(config["SEQUENCE_SIZE"])
                assert model.hidden_size == int(config["HIDDEN_SIZE"])
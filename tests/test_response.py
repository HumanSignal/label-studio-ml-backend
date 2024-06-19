
import pytest
from pydantic import BaseModel
from typing import Optional

from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

MODEL_VERSION = "0.1"

@pytest.fixture
def predictions():
    # Replace 'Example Prediction' with real Predictions
    return [
        PredictionValue(model_version=MODEL_VERSION, score=0.1, result=[{}]),
        PredictionValue(score=0.1, result=[{}]),
        [
            PredictionValue(model_version='prediction_1', score=0.2, result=[{}]),
            PredictionValue(score=0.3, result=[{}]),
        ]
    ]

def test_has_model_version(predictions):
    res = ModelResponse(predictions=predictions)
    assert not res.has_model_version() # If model_version is not set, should return False

    res.model_version = "1.0"
    assert res.has_model_version() # If model_version is set, should return True

def test_update_predictions_version(predictions):
    res = ModelResponse(predictions=predictions)

    # Assuming model_version is not set initially in predictions 
    res.model_version = "1.0"
    res.update_predictions_version()

    assert predictions[0].model_version == MODEL_VERSION
    assert predictions[1].model_version == "1.0"
    

def test_set_version(predictions):
    res = ModelResponse(predictions=predictions)
    res.set_version("2.0")
    
    assert res.model_version == "2.0" # model_version should be updated to "2.0"

    assert predictions[0].model_version == MODEL_VERSION
    assert predictions[1].model_version == "2.0"
        

def test_serialize(predictions):
    # Assuming PredictionValue has method .serialize() which returns a dict 
    res = ModelResponse(model_version="1.0", predictions=predictions)
    serialized_res = res.model_dump()

    assert serialized_res['model_version'] == "1.0"
    assert serialized_res['predictions'] == [
        {'model_version': MODEL_VERSION, 'score': 0.1, 'result': [{}]},
        {'model_version': None, 'score': 0.1, 'result': [{}]},
        [
            {'model_version': 'prediction_1', 'score': 0.2, 'result': [{}]},
            {'model_version': None, 'score': 0.3, 'result': [{}]},
        ]
    ]

    # update model_version
    res.update_predictions_version()
    serialized_res = res.model_dump()

    assert serialized_res['predictions'] == [
        {'model_version': "0.1", 'score': 0.1, 'result': [{}]},
        {'model_version': "1.0", 'score': 0.1, 'result': [{}]},
        [
            {'model_version': 'prediction_1', 'score': 0.2, 'result': [{}]},
            {'model_version': '1.0', 'score': 0.3, 'result': [{}]},
        ]
    ]

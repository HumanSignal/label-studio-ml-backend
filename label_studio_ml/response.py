
from typing import Type, Dict, Optional, List, Tuple, Any, Union
from pydantic import BaseModel, confloat

from label_studio_sdk._legacy.objects import PredictionValue


class ModelResponse(BaseModel):
    """
    """
    model_version: Optional[str] = None
    predictions: List[PredictionValue]

    def has_model_version(self) -> bool:
        return bool(self.model_version)

    def update_predictions_version(self) -> None:
        """
        """
        for prediction in self.predictions:
            if not prediction.model_version:
                prediction.model_version = self.model_version
    
    def set_version(self, version: str) -> None:
        """
        """
        self.model_version = version
        # Set the version for each prediction
        self.update_predictions_version()

    def serialize(self):
        """
        """
        return {
            "model_version": self.model_version,
            "predictions": [ p.serialize() for p in self.predictions ]
        }
        

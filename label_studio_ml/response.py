
from typing import Type, Dict, Optional, List, Tuple, Any, Union
from pydantic import BaseModel, confloat, Field
from label_studio_sdk.label_interface.objects import PredictionValue
from typing import Union, List


# one or multiple predictions per task
SingleTaskPredictions = Union[List[PredictionValue], PredictionValue]


class ModelResponse(BaseModel):
    """ Model response with predictions for Label Studio, used in /predict API endpoint
    """
    class Config:
        protected_namespaces = ('__.*__', '_.*')  # Excludes 'model_'

    model_version: Optional[str] = None
    predictions: List[SingleTaskPredictions]

    def has_model_version(self) -> bool:
        return bool(self.model_version)

    def update_predictions_version(self) -> None:
        """Attach model_version to each prediction (SDK objects or plain dicts)."""
        mv = self.model_version
        if not mv:
            return
        for prediction in self.predictions:
            if isinstance(prediction, dict):
                if not prediction.get("model_version"):
                    prediction["model_version"] = mv
                continue
            if isinstance(prediction, PredictionValue):
                pred_list: List = [prediction]
            elif isinstance(prediction, list):
                pred_list = prediction
            else:
                continue
            for p in pred_list:
                if isinstance(p, PredictionValue) and not p.model_version:
                    p.model_version = mv
    
    def set_version(self, version: str) -> None:
        """
        """
        self.model_version = version
        # Set the version for each prediction
        self.update_predictions_version()
        

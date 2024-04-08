import os
import nemo
import nemo.collections.asr as nemo_asr

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_local_path


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    MODEL_NAME = os.getenv('MODEL_NAME', 'QuartzNet15x5Base-En')

    _model = None


    def _lazy_init(self):
        if self._model is not None:
            return

        # This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
        self._model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=self.MODEL_NAME)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        self._lazy_init()

        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Audio')

        audio_paths = []
        for task in tasks:
            audio_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
            audio_path = get_local_path(audio_url)
            audio_paths.append(audio_path)

        # run ASR
        transcriptions = self._model.transcribe(paths2audio_files=audio_paths)

        predictions = []
        for transcription in transcriptions:
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'textarea',
                    'value': {
                        'text': [transcription]
                    }
                }],
                'score': 1.0,
                'model_version': self.MODEL_NAME
            })
        
        return ModelResponse(predictions=predictions)

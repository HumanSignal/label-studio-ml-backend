import os
import nemo
import nemo.collections.asr as nemo_asr

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path


MODEL_NAME = os.getenv('MODEL_NAME', 'QuartzNet15x5Base-En')
_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=MODEL_NAME)


class NemoASR(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:

        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Audio')

        audio_paths = []
        for task in tasks:
            audio_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
            audio_path = get_local_path(audio_url, task_id=task.get('id'))
            audio_paths.append(audio_path)

        # run ASR
        transcriptions = _model.transcribe(paths2audio_files=audio_paths)

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
                'model_version': self.get('model_version')
            })
        
        return ModelResponse(predictions=predictions)

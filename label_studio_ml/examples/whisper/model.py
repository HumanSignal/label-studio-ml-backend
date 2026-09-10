import logging
import math
import os
import uuid
from typing import Dict, List, Optional

from faster_whisper import WhisperModel
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv('WHISPER_MODEL', 'base')
DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
LANGUAGE = os.getenv('WHISPER_LANGUAGE') or None
BEAM_SIZE = int(os.getenv('WHISPER_BEAM_SIZE', 5))
SEGMENT_LABEL = os.getenv('WHISPER_SEGMENT_LABEL', 'Speech')

_model = None


def _get_model():
    """Loaded on first use so importing this module stays cheap."""
    global _model
    if _model is None:
        logger.info('loading whisper model %s on %s (%s)', MODEL_NAME, DEVICE, COMPUTE_TYPE)
        _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


class WhisperASR(LabelStudioMLBase):
    """Speech recognition with faster-whisper.

    Two labeling configs are supported and picked automatically:

    * ``Audio`` + ``TextArea`` gives one transcript for the whole file.
    * ``Audio`` + ``Labels`` + per-region ``TextArea`` gives one region per
      Whisper segment, each carrying its own transcript.
    """

    def setup(self):
        self.set('model_version', f'{self.__class__.__name__}-{MODEL_NAME}-v0.0.1')

    def _segment_control(self):
        """Return (labels_from_name, textarea_from_name, to_name, value) when the
        config asks for per-region transcription, otherwise None."""
        try:
            labels_from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Audio')
            textarea_from_name, _, _ = self.label_interface.get_first_tag_occurence('TextArea', 'Audio')
        except Exception:
            return None
        return labels_from_name, textarea_from_name, to_name, value

    def _label_name(self, from_name):
        labels = self.label_interface.get_tag(from_name).labels
        if SEGMENT_LABEL in labels:
            return SEGMENT_LABEL
        return labels[0] if labels else SEGMENT_LABEL

    def _audio_path(self, task, value):
        audio_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
        return self.get_local_path(audio_url, task_id=task.get('id'))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        segmented = self._segment_control()
        if segmented:
            return self._predict_segments(tasks, *segmented)

        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Audio')
        predictions = []
        for task in tasks:
            segments, _ = _get_model().transcribe(
                self._audio_path(task, value), beam_size=BEAM_SIZE, language=LANGUAGE
            )
            segments = list(segments)
            text = ''.join(segment.text for segment in segments).strip()
            predictions.append(
                {
                    'result': [
                        {
                            'from_name': from_name,
                            'to_name': to_name,
                            'type': 'textarea',
                            'value': {'text': [text]},
                        }
                    ],
                    'score': _mean_score(segments),
                    'model_version': self.get('model_version'),
                }
            )
        return ModelResponse(predictions=predictions)

    def _predict_segments(self, tasks, labels_from_name, textarea_from_name, to_name, value):
        label = self._label_name(labels_from_name)
        predictions = []
        for task in tasks:
            segments, _ = _get_model().transcribe(
                self._audio_path(task, value), beam_size=BEAM_SIZE, language=LANGUAGE
            )
            segments = list(segments)
            result = []
            for segment in segments:
                # both entries share one id so Label Studio attaches the
                # transcript to the region instead of creating a second one
                region_id = uuid.uuid4().hex[:10]
                span = {'start': segment.start, 'end': segment.end}
                result.append(
                    {
                        'id': region_id,
                        'from_name': labels_from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        'value': dict(span, labels=[label]),
                    }
                )
                result.append(
                    {
                        'id': region_id,
                        'from_name': textarea_from_name,
                        'to_name': to_name,
                        'type': 'textarea',
                        'value': dict(span, text=[segment.text.strip()]),
                    }
                )
            predictions.append(
                {
                    'result': result,
                    'score': _mean_score(segments),
                    'model_version': self.get('model_version'),
                }
            )
        return ModelResponse(predictions=predictions)


def _mean_score(segments):
    """Whisper reports avg_logprob per segment; turn it into a 0..1 score."""
    if not segments:
        return 0.0
    total = sum(math.exp(segment.avg_logprob) for segment in segments)
    return round(min(1.0, total / len(segments)), 4)

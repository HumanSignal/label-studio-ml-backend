import io
import logging
import os
import base64

import pytesseract as pt
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)
global OCR_config
OCR_config = "--psm 6 -l chi_sim+eng+deu"

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")


class BBOXOCR(LabelStudioMLBase):
    MODEL_DIR = os.environ.get('MODEL_DIR', '.')

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    @staticmethod
    def _load_image_from_context_b64(image_b64: str) -> Image.Image:
        if not image_b64:
            raise ValueError("Empty image payload")
        try:
            binary = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(binary))
        except Exception as e:
            logger.exception("Failed to decode base64 image from context")
            raise e

    def predict(self, tasks, **kwargs):
        context = kwargs.get('context')

        if not context:
            return []

        image_b64 = context.get('image')
        if not image_b64:
            return []

        try:
            image = self._load_image_from_context_b64(image_b64)
        except Exception:
            return []

        # Perform OCR on the provided region image
        result_text = pt.image_to_string(image, config=OCR_config).strip()

        # For now we don't need to construct LS results; return minimal payload
        return [{
            'result': [],
            'score': 0,
            'text': result_text,
            'model_version': self.get('model_version')
        }]

    @staticmethod
    def _extract_meta(task):
        # Deprecated in simplified flow; keep for backward compatibility if needed
        return {}

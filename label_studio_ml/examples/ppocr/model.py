"""
PP-OCR ML Backend for Label Studio

This module implements a Label Studio ML backend using PaddleX's PP-OCR pipeline
for text detection and recognition with support for 100+ languages.
Tested with PP-OCRv4 and PP-OCRv5, expected to work with future versions.
"""

import logging
import os
from typing import Dict, List, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)


class PPOCR(LabelStudioMLBase):
    """
    PP-OCR ML Backend for Label Studio.

    Uses PaddleX's OCR pipeline for text detection and recognition.
    Supports multiple languages, model versions (v4/v5), and both
    mobile (fast) and server (accurate) variants.
    """

    # PP-OCR version: 'v4', 'v5', etc.
    PPOCR_VERSION = os.getenv('PPOCR_VERSION', 'v5')

    # Model configuration
    MODEL_VARIANT = os.getenv('MODEL_VARIANT', 'server')  # 'mobile' or 'server'
    LANG = os.getenv('OCR_LANG', 'ch')  # Language code (use OCR_LANG to avoid conflict with system LANG)

    # Device configuration
    DEVICE = os.getenv('DEVICE', 'gpu:0')  # 'cpu', 'gpu:0', 'gpu:1', etc.

    # Score thresholds
    SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.5))
    DET_SCORE_THRESHOLD = float(os.getenv('DET_SCORE_THRESHOLD', 0.3))

    # Output options
    OUTPUT_TYPE = os.getenv('OUTPUT_TYPE', 'polygon')  # 'polygon' or 'rectangle'
    INCLUDE_TRANSCRIPTION = os.getenv('INCLUDE_TRANSCRIPTION', 'true').lower() == 'true'

    # Document preprocessing options (disabled by default)
    USE_DOC_ORIENTATION = os.getenv('USE_DOC_ORIENTATION', 'false').lower() == 'true'
    USE_DOC_UNWARPING = os.getenv('USE_DOC_UNWARPING', 'false').lower() == 'true'
    USE_TEXTLINE_ORIENTATION = os.getenv('USE_TEXTLINE_ORIENTATION', 'false').lower() == 'true'

    # Label Studio connection
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    # Model directory for caching
    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    # Languages supported by base PP-OCR model (both server and mobile variants)
    # These use PP-OCR{version}_{variant}_rec directly
    BASE_MODEL_LANGUAGES = {'ch', 'en'}  # Chinese, English, Japanese, Traditional Chinese

    # Languages with dedicated mobile-only models
    # These use {lang}_PP-OCR{version}_mobile_rec
    DEDICATED_MODEL_LANGUAGES = {
        'arabic',  # Arabic
        'cyrillic',  # Cyrillic (Russian, etc.)
        'devanagari',  # Devanagari (Hindi, etc.)
        'el',  # Greek
        'eslav',  # East Slavic
        'korean',  # Korean
        'latin',  # Latin-based
        'ta',  # Tamil
        'te',  # Telugu
        'th',  # Thai
    }

    # Process-wide pipeline cache. The API creates a new PPOCR instance per /predict
    # request, so this must be stored on the class (not the instance) to load once.
    _pipeline = None

    def _lazy_init(self):
        """Initialize the PaddleX OCR pipeline lazily on first use."""
        if self._pipeline is not None:
            return

        from paddlex import create_pipeline

        version = self.PPOCR_VERSION.lower()
        # Normalize version string: accept both 'v5' and '5'
        if not version.startswith('v'):
            version = f'v{version}'
        # PaddleX uses lowercase 'v' in model names: PP-OCRv5, PP-OCRv4
        version_tag = f"PP-OCR{version}"  # e.g., 'PP-OCRv5'

        # Build recognition model name based on language and variant
        if self.LANG in self.BASE_MODEL_LANGUAGES:
            rec_model = f"{version_tag}_{self.MODEL_VARIANT}_rec"
        elif self.LANG in self.DEDICATED_MODEL_LANGUAGES:
            rec_model = f"{self.LANG}_{version_tag}_mobile_rec"
            if self.MODEL_VARIANT == 'server':
                logger.warning(
                    f"Server variant not available for language '{self.LANG}'. "
                    f"Using mobile variant instead."
                )
        else:
            # Unknown language - try base model with warning
            logger.warning(
                f"Unknown language '{self.LANG}'. Using base {version_tag} model."
            )
            rec_model = f"{version_tag}_{self.MODEL_VARIANT}_rec"

        # Detection model based on variant
        det_model = f"{version_tag}_{self.MODEL_VARIANT}_det"

        # Determine if document preprocessor is needed
        use_doc_preprocessor = self.USE_DOC_ORIENTATION or self.USE_DOC_UNWARPING

        logger.info(f"Initializing {version_tag} pipeline:")
        logger.info(f"  Detection model: {det_model}")
        logger.info(f"  Recognition model: {rec_model}")
        logger.info(f"  Device: {self.DEVICE}")
        logger.info(f"  Language: {self.LANG}")

        # Build config dict for pipeline
        # Reference: PaddleX/paddlex/configs/pipelines/OCR.yaml
        config = {
            "pipeline_name": "OCR",
            "text_type": "general",
            "use_doc_preprocessor": use_doc_preprocessor,
            "use_textline_orientation": self.USE_TEXTLINE_ORIENTATION,
            "SubModules": {
                "TextDetection": {
                    "module_name": "text_detection",
                    "model_name": det_model,
                    "thresh": self.DET_SCORE_THRESHOLD,
                },
                "TextRecognition": {
                    "module_name": "text_recognition",
                    "model_name": rec_model,
                    "score_thresh": self.SCORE_THRESHOLD,
                }
            }
        }

        # Add TextLineOrientation config if enabled
        if self.USE_TEXTLINE_ORIENTATION:
            config["SubModules"]["TextLineOrientation"] = {
                "module_name": "textline_orientation",
                "model_name": "PP-LCNet_x1_0_textline_ori",
            }

        # Add document preprocessor config if needed (uses SubPipelines, not SubModules)
        if use_doc_preprocessor:
            config["SubPipelines"] = {
                "DocPreprocessor": {
                    "pipeline_name": "doc_preprocessor",
                    "use_doc_orientation_classify": self.USE_DOC_ORIENTATION,
                    "use_doc_unwarping": self.USE_DOC_UNWARPING,
                    "SubModules": {
                        "DocOrientationClassify": {
                            "module_name": "doc_text_orientation",
                            "model_name": "PP-LCNet_x1_0_doc_ori",
                        },
                        "DocUnwarping": {
                            "module_name": "image_unwarping",
                            "model_name": "UVDoc",
                        }
                    }
                }
            }

        # Configure predictor options
        # Use paddle_fp32 run_mode on CPU to avoid MKL-DNN/PIR issues on Windows
        from paddlex.inference.utils.pp_option import PaddlePredictorOption
        pp_option = PaddlePredictorOption()

        if self.DEVICE == 'cpu' or self.DEVICE.startswith('cpu'):
            pp_option.run_mode = 'paddle_fp32'
            logger.info("  Run mode: paddle_fp32 (CPU without MKL-DNN)")
        else:
            pp_option.run_mode = 'paddle'
            logger.info(f"  Run mode: paddle (GPU)")

        # Cache on the class so every per-request instance reuses the same pipeline.
        PPOCR._pipeline = create_pipeline(config=config, device=self.DEVICE, pp_option=pp_option)
        logger.info(f"{version_tag} pipeline initialized successfully")

    def setup(self):
        """Configure model parameters."""
        self.set("model_version", f"PPOCR{self.PPOCR_VERSION}-{self.MODEL_VARIANT}-{self.LANG}-v0.0.1")

    def _get_image_url(self, task: Dict, value: str) -> str:
        """
        Get the image URL from task data, handling S3 presigned URLs.

        Args:
            task: The task dictionary containing image data.
            value: The key for the image URL in task data.

        Returns:
            The resolved image URL.
        """
        image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
        if not image_url:
            raise ValueError(
                f"Could not find image data in task {task.get('id')} "
                f"(looked for key '{value}' and '{DATA_UNDEFINED_NAME}')"
            )

        if image_url.startswith('s3://'):
            # Generate presigned URL for S3
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f"Can't generate presigned URL for {image_url}. Reason: {exc}")

        return image_url

    def predict_single(self, task: Dict) -> Optional[Dict]:
        """
        Process a single task and return OCR predictions.

        Args:
            task: A task dictionary containing image data.

        Returns:
            A dictionary with prediction results or None if no predictions.
        """
        logger.debug('Task data: %s', task['data'])

        # Get tag information from label config
        # Try to get Polygon first, then Rectangle as fallback
        from_name_poly = None
        from_name_rect = None
        from_name_trans = None
        to_name = None
        value = None

        try:
            from_name_poly, to_name, value = self.get_first_tag_occurence('Polygon', 'Image')
        except Exception:
            pass

        try:
            from_name_rect, to_name_rect, value_rect = self.get_first_tag_occurence('Rectangle', 'Image')
            if to_name is None:
                to_name = to_name_rect
            if value is None:
                value = value_rect
        except Exception:
            pass

        try:
            from_name_trans, to_name_trans, value_trans = self.get_first_tag_occurence('TextArea', 'Image')
            if to_name is None:
                to_name = to_name_trans
            if value is None:
                value = value_trans
        except Exception:
            pass

        # If no suitable tags found, try to get the Image tag directly
        if to_name is None or value is None:
            try:
                _, to_name, value = self.get_first_tag_occurence('Labels', 'Image')
            except Exception:
                # Last resort: use default values
                to_name = 'image'
                value = 'image'
                logger.warning("Could not find Image tag in label config, using defaults")

        # Get labels from config
        labels = self.label_interface.labels
        labels = sum([list(l) for l in labels], [])
        if len(labels) > 1:
            logger.warning(
                'More than one label in the tag. Only the first one will be used: %s',
                labels[0]
            )
        label = labels[0] if labels else 'Text'

        # Get image path
        image_url = self._get_image_url(task, value)
        cache_dir = os.path.join(self.MODEL_DIR, '.file-cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f'Using cache dir: {cache_dir}')

        image_path = get_local_path(
            image_url,
            cache_dir=cache_dir,
            hostname=self.LABEL_STUDIO_HOST,
            access_token=self.LABEL_STUDIO_ACCESS_TOKEN,
            task_id=task.get('id')
        )

        # Run OCR pipeline
        output = self._pipeline.predict(
            image_path,
            use_doc_orientation_classify=self.USE_DOC_ORIENTATION,
            use_doc_unwarping=self.USE_DOC_UNWARPING,
            use_textline_orientation=self.USE_TEXTLINE_ORIENTATION,
            text_rec_score_thresh=self.SCORE_THRESHOLD,
        )

        # Get image dimensions
        img_width, img_height = get_image_size(image_path)

        result = []
        all_scores = []

        # Process OCR results
        for ocr_result in output:
            rec_texts = ocr_result.get("rec_texts", [])
            rec_scores = ocr_result.get("rec_scores", [])
            rec_polys = ocr_result.get("rec_polys", [])
            rec_boxes = ocr_result.get("rec_boxes", [])

            for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                # Convert numpy types to Python types
                score = float(score)

                if score < self.SCORE_THRESHOLD:
                    logger.debug(f'Skipping result with low score: {score}')
                    continue

                # Generate unique ID for linking polygon and textarea
                region_id = str(uuid4())[:4]

                if self.OUTPUT_TYPE == 'polygon':
                    # Convert polygon coordinates to percentage format
                    points = []
                    for point in poly:
                        px = float(point[0]) / img_width * 100
                        py = float(point[1]) / img_height * 100
                        # Ensure points are within bounds
                        px = max(0, min(100, px))
                        py = max(0, min(100, py))
                        points.append([px, py])

                    # Add polygon annotation
                    result.append({
                        'original_width': img_width,
                        'original_height': img_height,
                        'image_rotation': 0,
                        'value': {
                            'points': points,
                        },
                        'id': region_id,
                        'from_name': from_name_poly,
                        'to_name': to_name,
                        'type': 'polygon',
                        'origin': 'manual',
                        'score': score,
                    })
                else:
                    # Rectangle mode - use bounding box
                    if i < len(rec_boxes):
                        box = rec_boxes[i]
                        x = float(box[0]) / img_width * 100
                        y = float(box[1]) / img_height * 100
                        width = (float(box[2]) - float(box[0])) / img_width * 100
                        height = (float(box[3]) - float(box[1])) / img_height * 100

                        # Ensure values are within bounds
                        x = max(0, min(100, x))
                        y = max(0, min(100, y))
                        width = max(0, min(100 - x, width))
                        height = max(0, min(100 - y, height))

                        result.append({
                            'original_width': img_width,
                            'original_height': img_height,
                            'image_rotation': 0,
                            'value': {
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height,
                                'rotation': 0,
                            },
                            'id': region_id,
                            'from_name': from_name_rect,
                            'to_name': to_name,
                            'type': 'rectangle',
                            'origin': 'manual',
                            'score': score,
                        })

                # Add transcription annotation if enabled
                if self.INCLUDE_TRANSCRIPTION and from_name_trans:
                    # For polygon output, include points in textarea for region linking
                    textarea_value = {
                        'text': [text],
                        'labels': [label],
                    }

                    if self.OUTPUT_TYPE == 'polygon':
                        textarea_value['points'] = points

                    result.append({
                        'original_width': img_width,
                        'original_height': img_height,
                        'image_rotation': 0,
                        'value': textarea_value,
                        'id': region_id,
                        'from_name': from_name_trans,
                        'to_name': to_name,
                        'type': 'textarea',
                        'origin': 'manual',
                        'score': score,
                    })

                all_scores.append(score)

        if not result:
            logger.info('No text detected in image')
            return None

        return {
            'result': result,
            'score': sum(all_scores) / len(all_scores),
            'model_version': self.get('model_version'),
        }

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Process multiple tasks and return OCR predictions.

        Args:
            tasks: A list of task dictionaries.
            context: Optional context dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelResponse containing predictions for all tasks.
        """
        self._lazy_init()

        predictions = []
        for task in tasks:
            try:
                prediction = self.predict_single(task)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error processing task {task.get('id')}: {e}")
                continue

        return ModelResponse(
            predictions=predictions,
            model_versions=self.get('model_version')
        )


# Backward compatibility alias
PPOCRv5 = PPOCR

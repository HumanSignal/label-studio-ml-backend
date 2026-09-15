"""Single-project, interactive deployment of the existing SAM2 image backend."""

import logging
import math
import os
from urllib.parse import urlsplit

from flask import jsonify, request

MAX_PIXELS = 2048 * 2048
# Label Studio converts box offsets and sizes separately, so an edge-aligned box can sum to 100.00000000000001.
MAX_PERCENT = 100 + 1e-6


def validate_media_url(url: str, origin: str) -> None:
    """Only allow uploads served by the configured Label Studio HTTPS origin."""
    if not isinstance(url, str) or '\\' in url or any(ord(c) < 32 for c in url):
        raise ValueError('Expected a Label Studio upload URL')
    parsed, host = urlsplit(url), urlsplit(origin)
    if parsed.scheme or parsed.netloc:
        if (parsed.scheme, parsed.netloc) != (host.scheme, host.netloc):
            raise ValueError('Media must belong to the configured Label Studio origin')
    if not parsed.path.startswith(('/data/upload/', '/storage-data/uploaded/')):
        raise ValueError('This example supports Label Studio uploads only')
    if parsed.username or parsed.password or parsed.fragment or '%' in parsed.path:
        raise ValueError('Unsupported upload URL')
    # The SDK reads '/data/...?d=<path>' from the local filesystem instead of Label Studio.
    if parsed.query and not parsed.path.startswith('/storage-data/uploaded/'):
        raise ValueError('Unsupported upload URL')
    if any(part in ('.', '..') for part in parsed.path.split('/')):
        raise ValueError('Unsupported upload path')


def valid_context(context: dict) -> bool:
    results = context.get('result', [])
    if not isinstance(results, list) or len(results) > 32:
        return False
    for item in results:
        if not isinstance(item, dict) or item.get('type') not in ('keypointlabels', 'rectanglelabels'):
            return False
        width, height = item.get('original_width'), item.get('original_height')
        if any(type(n) is not int or n < 1 for n in (width, height)) or width * height > MAX_PIXELS:
            return False
        value = item.get('value')
        if not isinstance(value, dict):
            return False
        labels = value.get(item['type'])
        if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str) or not labels[0]:
            return False
        keys = ('x', 'y', 'width', 'height') if item['type'] == 'rectanglelabels' else ('x', 'y')
        if any(
            type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) or not 0 <= value[k] <= MAX_PERCENT
            for k in keys
        ):
            return False
        if (width, height) != (results[0]['original_width'], results[0]['original_height']):
            return False
        if item['type'] == 'rectanglelabels' and (
            value['width'] <= 0
            or value['height'] <= 0
            or value['x'] + value['width'] > MAX_PERCENT
            or value['y'] + value['height'] > MAX_PERCENT
        ):
            return False
    return True


def create_app(model_class=None):
    """Load the model inside a Gunicorn sync worker, never in a preloaded master."""
    from label_studio_ml.api import init_app

    origin = os.environ['LABEL_STUDIO_URL'].rstrip('/')
    parsed = urlsplit(origin)
    if (
        parsed.scheme != 'https'
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError('LABEL_STUDIO_URL must be an HTTPS origin without a path or credentials')
    if not os.getenv('LABEL_STUDIO_API_KEY'):
        raise ValueError('Set the Label Studio media credential through a runtime secret')
    project = os.environ['LABEL_STUDIO_PROJECT_ID']
    if not project.isdecimal():
        raise ValueError('LABEL_STUDIO_PROJECT_ID must be numeric')
    if os.getenv('BASIC_AUTH_USER') or os.getenv('BASIC_AUTH_PASS'):
        raise ValueError('Use native Endpoint auth; leave backend BASIC_AUTH variables unset')
    # The shared API logs full headers/bodies at DEBUG; the SDK can log media URLs.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('label_studio_ml.api').setLevel(logging.WARNING)
    logging.getLogger('label_studio_sdk').setLevel(logging.WARNING)

    if model_class is None:
        from model import NewModel
        from PIL import Image

        class EndpointModel(NewModel):
            def setup(self):
                self.set('model_version', os.environ['MODEL_VERSION'])

            def predict(self, tasks, context=None, **kwargs):
                results = (context or {}).get('result', [])
                self.expected_size = (results[0]['original_width'], results[0]['original_height']) if results else None
                try:
                    return super().predict(tasks, context=context, **kwargs)
                except Exception:
                    # The shared API serializes tracebacks; don't include signed media URLs.
                    raise RuntimeError('Prediction failed; check input, media access and model readiness') from None

            def set_image(self, image_url, task_id):
                validate_media_url(image_url, origin)
                image_path = self.get_local_path(image_url, task_id=task_id)
                with Image.open(image_path) as image:
                    if self.expected_size and image.size != self.expected_size:
                        raise ValueError('Prompt dimensions do not match the uploaded image')
                    if image.width * image.height > MAX_PIXELS:
                        raise ValueError('Use images with at most 2048 x 2048 pixels')
                # The existing helper reuses the cached local file.
                super().set_image(image_url, task_id)

        model_class = EndpointModel

    app = init_app(model_class)
    # Interactive requests include the task's saved annotations, drafts and predictions (brush RLE).
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    @app.before_request
    def validate_request():
        if request.path == '/health' and request.method == 'GET':
            return None
        if request.path not in ('/setup', '/predict') or request.method != 'POST':
            return jsonify(error='Unsupported route for this inference-only example'), 404
        payload = request.get_json(silent=True)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get('project'), str)
            or payload['project'].split('.')[0] != project
        ):
            return jsonify(error='Invalid project or JSON body'), 400
        schema = 'schema' if request.path == '/setup' else 'label_config'
        if not isinstance(payload.get(schema), str) or not payload[schema]:
            return jsonify(error='Missing labeling configuration'), 400
        if request.path == '/predict':
            tasks, params = payload.get('tasks'), payload.get('params', {})
            if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
                return jsonify(error='Exactly one interactive task is required'), 400
            if not isinstance(tasks[0].get('data'), dict) or not isinstance(params, dict):
                return jsonify(error='Invalid task data or prediction parameters'), 400
            context = params.get('context')
            if context is None:
                context = {}
            if not isinstance(context, dict) or not valid_context(context):
                return jsonify(error='Invalid interactive context'), 400
        return None

    return app

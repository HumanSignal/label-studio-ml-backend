import hmac
import logging
import os

from flask import Flask, request, jsonify, Response

from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import exception_handler

logger = logging.getLogger(__name__)

_server = Flask(__name__)
MODEL_CLASS = LabelStudioMLBase
BASIC_AUTH = None


def init_app(model_class, basic_auth_user=None, basic_auth_pass=None):
    global MODEL_CLASS
    global BASIC_AUTH

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    return _server


@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    """
    Predict tasks

    Example request:
    request = {
            'tasks': tasks,
            'model_version': model_version,
            'project': '{project.id}.{int(project.created_at.timestamp())}',
            'label_config': project.label_config,
            'params': {
                'login': project.task_data_login,
                'password': project.task_data_password,
                'context': context,
            },
        }

    @return:
    Predictions in LS format
    """
    data = request.json
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    # model.use_label_config(label_config)

    response = model.predict(tasks, context=context, **params)

    # if there is no model version we will take the default
    if isinstance(response, ModelResponse):
        if not response.has_model_version():
            mv = model.model_version
            if mv:
                response.set_version(str(mv))
        else:
            response.update_predictions_version()

        response = response.model_dump()

    res = response
    if res is None:
        res = []

    if isinstance(res, dict):
        res = response.get("predictions", response)

    return jsonify({'results': res})


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('schema')
    extra_params = data.get('extra_params')
    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    if extra_params:
        model.set_extra_params(extra_params)

    model_version = model.get('model_version')
    return jsonify({'model_version': model_version})


TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'START_TRAINING'
)


@_server.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event = data.pop('action')
    if event not in TRAIN_EVENTS:
        return jsonify({'status': 'Unknown event'}), 200
    project_id = str(data['project']['id'])
    label_config = data['project']['label_config']
    model = MODEL_CLASS(project_id, label_config=label_config)
    result = model.fit(event, data)

    try:
        response = jsonify({'result': result, 'status': 'ok'})
    except Exception as e:
        response = jsonify({'error': str(e), 'status': 'error'})

    return response, 201


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    return jsonify({
        'status': 'UP',
        'model_class': MODEL_CLASS.__name__
    })


@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    return jsonify({})


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


@_server.before_request
def check_auth():
    if BASIC_AUTH is not None:

        auth = request.authorization
        if not auth or not (safe_str_cmp(auth.username, BASIC_AUTH[0]) and safe_str_cmp(auth.password, BASIC_AUTH[1])):
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response

import logging
import os

from flask import Flask, request, jsonify
from rq.exceptions import NoSuchJobError

from .model import LabelStudioMLManager, LABEL_STUDIO_ML_BACKEND_V2_DEFAULT
from .exceptions import exception_handler

logger = logging.getLogger(__name__)

_server = Flask(__name__)
_manager = LabelStudioMLManager()


def init_app(model_class, **kwargs):
    global _manager
    _manager.initialize(model_class, **kwargs)
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
    project = data.get('project')
    label_config = data.get('label_config')
    force_reload = data.get('force_reload', False)
    try_fetch = data.get('try_fetch', True)
    params = data.get('params') or {}
    predictions, model = _manager.predict(tasks, project, label_config, force_reload, try_fetch, **params)
    response = {
        'results': predictions,
        'model_version': model.model_version
    }
    return jsonify(response)


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    data = request.json
    logger.debug(data)
    project = data.get('project')
    schema = data.get('schema')
    force_reload = data.get('force_reload', False)
    hostname = data.get('hostname', '')  # host name for uploaded files and building urls
    access_token = data.get('access_token', '')  # user access token to retrieve data
    model_version = data.get('model_version')
    model = _manager.fetch(project, schema, force_reload,
                           hostname=hostname,
                           access_token=access_token,
                           model_version=model_version)
    logger.debug('Fetch model version: {}'.format(model.model_version))
    return jsonify({'model_version': model.model_version})


@_server.route('/train', methods=['POST'])
@exception_handler
def _train():
    logger.warning("=> Warning: API /train is deprecated since Label Studio 1.4.1. "
                   "ML backend used API /train for training previously, "
                   "but since 1.4.1 Label Studio backend and ML backend use /webhook for the training run.")
    data = request.json
    annotations = data.get('annotations', 'No annotations provided')
    project = data.get('project')
    label_config = data.get('label_config')
    params = data.get('params', {})
    if isinstance(project, dict):
        project = ""
    if len(annotations) == 0:
        return jsonify('No annotations found.'), 400
    job = _manager.train(annotations, project, label_config, **params)
    response = {'job': job.id} if job else {}
    return jsonify(response), 201


@_server.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event = data.pop('action')
    run = _manager.webhook(event, data)
    return jsonify(run), 201


@_server.route('/is_training', methods=['GET'])
@exception_handler
def _is_training():
    project = request.args.get('project')
    output = _manager.is_training(project)
    return jsonify(output)


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    return jsonify({
        'status': 'UP',
        'model_dir': _manager.model_dir,
        'v2': os.getenv('LABEL_STUDIO_ML_BACKEND_V2', default=LABEL_STUDIO_ML_BACKEND_V2_DEFAULT)
    })


@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    return jsonify({})


@_server.errorhandler(NoSuchJobError)
def no_such_job_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 410


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


@_server.route('/versions', methods=['POST'])
@exception_handler
def get_version():
    """
    Get model versions from ML backend
    @return: A list of versions
    """
    data = request.json
    project = data.get('project')
    if project is None:
        return str('No project provided'), 400
    versions = list(_manager._get_models_from_workdir(project))
    return jsonify({
        'versions': versions,
        'model_dir': _manager.model_dir,
        'v2': os.getenv('LABEL_STUDIO_ML_BACKEND_V2', default=True)
    })

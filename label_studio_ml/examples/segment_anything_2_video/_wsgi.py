import os
import argparse
import json
import logging
import logging.config

logging.config.dictConfig({
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "standard": {
      "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": os.getenv('LOG_LEVEL'),
      "stream": "ext://sys.stdout",
      "formatter": "standard"
    }
  },
  "root": {
    "level": os.getenv('LOG_LEVEL'),
    "handlers": [
      "console"
    ],
    "propagate": True
  }
})

from label_studio_ml.api import init_app
from model import NewModel


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def validate_startup_environment():
    """Validate environment before starting server"""
    logger = logging.getLogger(__name__)
    logger.info('🔍 Running startup validation...')

    # Check GPU availability if CUDA is configured
    device = os.getenv('DEVICE', 'cuda')
    if device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                logger.error('❌ CUDA not available but DEVICE=cuda')
                raise RuntimeError('GPU required but not available')
            logger.info(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
        except ImportError:
            logger.error('❌ PyTorch not installed')
            raise RuntimeError('PyTorch is required')

    # Check SAM2 model checkpoint exists
    model_checkpoint = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')
    checkpoint_path = f'/sam2/checkpoints/{model_checkpoint}'
    if not os.path.exists(checkpoint_path):
        logger.warning(f'⚠️  SAM2 checkpoint not found at {checkpoint_path}')
        logger.info(f'Model will attempt to download on first use')
    else:
        logger.info(f'✅ SAM2 checkpoint found: {checkpoint_path}')

    # Check Label Studio connectivity
    ls_host = os.getenv('LABEL_STUDIO_HOST') or os.getenv('LABEL_STUDIO_URL')
    ls_api_key = os.getenv('LABEL_STUDIO_API_KEY')

    if not ls_host:
        logger.warning('⚠️  LABEL_STUDIO_HOST/LABEL_STUDIO_URL not set')
    else:
        logger.info(f'✅ Label Studio URL configured: {ls_host}')

    if not ls_api_key:
        logger.warning('⚠️  LABEL_STUDIO_API_KEY not set')
    else:
        logger.info(f'✅ Label Studio API key configured')

    logger.info('✅ Startup validation complete')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label studio')
    parser.add_argument(
        '-p', '--port', dest='port', type=int, default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0',
        help='Server host')
    parser.add_argument(
        '--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='),
        help='Additional LabelStudioMLBase model initialization kwargs')
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None,
        help='Logging level')
    parser.add_argument(
        '--model-dir', dest='model_dir', default=os.path.dirname(__file__),
        help='Directory where models are stored (relative to the project directory)')
    parser.add_argument(
        '--check', dest='check', action='store_true',
        help='Validate model instance before launching server')
    parser.add_argument('--basic-auth-user',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None),
                        help='Basic auth user')
    
    parser.add_argument('--basic-auth-pass',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None),
                        help='Basic auth pass')    
    
    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == 'True' or v == 'true':
                param[k] = True
            elif v == 'False' or v == 'false':
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()

    if args.kwargs:
        kwargs.update(parse_kwargs())

    if args.check:
        print('Check "' + NewModel.__name__ + '" instance creation..')
        validate_startup_environment()
        model = NewModel(**kwargs)

    app = init_app(model_class=NewModel, basic_auth_user=args.basic_auth_user, basic_auth_pass=args.basic_auth_pass)

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI use
    try:
        validate_startup_environment()
    except Exception as e:
        logging.getLogger(__name__).error(f'❌ Startup validation failed: {e}')
        # Continue anyway for uWSGI - errors will be caught on first request

    app = init_app(model_class=NewModel)

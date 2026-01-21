import os
import argparse
import json
import logging
import logging.config

# Set a default log level if LOG_LEVEL is not defined
log_level = os.getenv("LOG_LEVEL", "INFO")

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
      "level": log_level,
      "stream": "ext://sys.stdout",
      "formatter": "standard"
    }
  },
  "root": {
    "level": log_level,
    "handlers": [
      "console"
    ],
    "propagate": True
  }
})

from label_studio_ml.api import init_app
from model import DFINEModel  # Changed from NewModel to DFINEModel


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label Studio ML Backend for D-FINE')
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
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=log_level,
        help='Logging level')
    parser.add_argument(
        '--model-dir', dest='model_dir', default=os.getenv('MODEL_DIR', '/data/models'), # Default from Docker env
        help='Directory where models (.pth weights) are stored')
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
        if args.kwargs:
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

    kwargs_parsed = get_kwargs_from_config()
    kwargs_parsed.update(parse_kwargs())

    # Pass MODEL_DIR to the model constructor if needed, or rely on env vars within the model
    if args.model_dir:
        kwargs_parsed['model_dir'] = args.model_dir
        # Also update environment variable if model relies on it directly and it's not already set
        if not os.getenv('MODEL_DIR'):
            os.environ['MODEL_DIR'] = args.model_dir


    if args.check:
        print('Check "' + DFINEModel.__name__ + '" instance creation..')
        model = DFINEModel(**kwargs_parsed)

    app = init_app(model_class=DFINEModel, **kwargs_parsed) # Pass parsed kwargs here

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI use
    # Ensure MODEL_DIR is available for the model initialization
    kwargs_for_init = get_kwargs_from_config()
    if not os.getenv('MODEL_DIR') and 'model_dir' not in kwargs_for_init:
         kwargs_for_init['model_dir'] = os.getenv('MODEL_DIR', '/data/models')
         if not os.getenv('MODEL_DIR'):
            os.environ['MODEL_DIR'] = kwargs_for_init['model_dir']
            
    app = init_app(model_class=DFINEModel, **kwargs_for_init)
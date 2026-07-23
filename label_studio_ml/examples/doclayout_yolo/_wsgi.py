import os
import argparse
import json
import logging
import logging.config

# Set a default log level if LOG_LEVEL is not defined
log_level = os.getenv("LOG_LEVEL", "INFO")

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,  # Prevent overriding existing loggers
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
                "formatter": "standard",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
            "propagate": True,
        },
    }
)

from label_studio_ml.api import init_app
# --- MODIFIED IMPORT ---
from model import DocLayoutYOLO # Import the adapted class
# --- END MODIFICATION ---


# Config path is optional, can be used for fixed model params
# _DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
#     if not os.path.exists(config_path):
#         return dict()
#     with open(config_path) as f:
#         config = json.load(f)
#     assert isinstance(config, dict)
#     return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Studio ML Backend for DocLayout-YOLO") # Updated description
    parser.add_argument(
        "-p", "--port", dest="port", type=int, default=9090, help="Server port"
    )
    parser.add_argument(
        "--host", dest="host", type=str, default="0.0.0.0", help="Server host"
    )
    parser.add_argument(
        "--kwargs",
        "--with",
        dest="kwargs",
        metavar="KEY=VAL",
        nargs="+",
        type=lambda kv: kv.split("="),
        help="Additional LabelStudioMLBase model initialization kwargs",
    )
    parser.add_argument(
        "-d", "--debug", dest="debug", action="store_true", help="Switch debug mode"
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=log_level,
        help="Logging level",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=os.getenv('MODEL_ROOT', os.path.join(os.path.dirname(__file__), 'models')), # Use MODEL_ROOT env var
        help="Directory where models are stored",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Validate model instance before launching server",
    )
    parser.add_argument(
        "--basic-auth-user",
        default=os.environ.get("ML_SERVER_BASIC_AUTH_USER", None),
        help="Basic auth user",
    )

    parser.add_argument(
        "--basic-auth-pass",
        default=os.environ.get("ML_SERVER_BASIC_AUTH_PASS", None),
        help="Basic auth pass",
    )
    # --- ADDED ARGUMENTS ---
    parser.add_argument(
        '--model-name', dest='model_name', default=os.environ.get('MODEL_NAME'),
        help='Default model name to load (e.g., doclayout_yolo_docstructbench_imgsz1024.pt)'
    )
    parser.add_argument(
        '--threshold', dest='score_threshold', type=float, default=os.environ.get('MODEL_SCORE_THRESHOLD', 0.5),
        help='Default confidence threshold'
    )
    parser.add_argument(
        '--imgsz', dest='imgsz', type=int, default=os.environ.get('DEFAULT_IMGSZ', 1024),
        help='Default image size for prediction'
    )
    # --- END ADDITION ---

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
        if not args.kwargs:
             return param
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == "True" or v == "true":
                param[k] = True
            elif v == "False" or v == "false":
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    # Prioritize command line args, then environment vars for model params
    init_kwargs = {
        'model_dir': args.model_dir,
        'model_name': args.model_name,
        'score_threshold': args.score_threshold,
        'imgsz': args.imgsz
    }
    # Add kwargs from command line potentially overriding others
    init_kwargs.update(parse_kwargs())

    # Update environment variables based on args for consistency if needed elsewhere
    os.environ['MODEL_ROOT'] = args.model_dir
    if args.model_name:
        os.environ['MODEL_NAME'] = args.model_name
    os.environ['MODEL_SCORE_THRESHOLD'] = str(args.score_threshold)
    os.environ['DEFAULT_IMGSZ'] = str(args.imgsz)


    if args.check:
        print(f'Check "{DocLayoutYOLO.__name__}" instance creation..')
        # Pass relevant args to the constructor if needed, or rely on environment variables set above
        model = DocLayoutYOLO(**init_kwargs)
        print("Model instance created successfully.")


    # Initialize the app with the specific model class and auth details
    app = init_app(
        model_class=DocLayoutYOLO,
        basic_auth_user=args.basic_auth_user,
        basic_auth_pass=args.basic_auth_pass,
        # Pass initialization kwargs to the model class constructor
        **init_kwargs
    )

    # Start the Flask development server (for local testing)
    # For production, use Gunicorn via start.sh
    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI/Gunicorn use
    # Initialize with kwargs from environment variables (handled within the class now)
    app = init_app(model_class=DocLayoutYOLO)
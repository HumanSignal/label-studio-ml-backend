import os
import argparse
import json
import logging
import logging.config

from flask_ngrok import run_with_ngrok

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
from .model import LabelStudioMLBase

app = None


def run_ml_backend(port=9090, host='0.0.0.0', debug=False):
    global app
    app = init_app(model_class=LabelStudioMLBase)
    run_with_ngrok(app)
    app.run(host=host, port=port, debug=debug)

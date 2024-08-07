from werkzeug.middleware.dispatcher import DispatcherMiddleware

from .data_wsgi._wsgi import application as data
from .model_wsgi import _wsgi as model


app = DispatcherMiddleware(model.application, {
    '/data': data
})
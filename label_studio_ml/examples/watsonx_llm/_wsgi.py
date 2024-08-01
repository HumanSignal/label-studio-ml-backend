from werkzeug.middleware.dispatcher import DispatcherMiddleware

from wsgi_data._wsgi import application as data
from wsgi_model import _wsgi as model


app = DispatcherMiddleware(model.application, {
    '/data': data
})
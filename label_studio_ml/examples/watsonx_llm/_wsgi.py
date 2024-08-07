from werkzeug.middleware.dispatcher import DispatcherMiddleware

from data_wsgi import application as data
import model_wsgi as model

app = DispatcherMiddleware(model.application, {
    '/data': data
})
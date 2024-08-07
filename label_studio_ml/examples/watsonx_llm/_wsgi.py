from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flask import Flask

from data_wsgi import application as data
import model_wsgi as model

app = Flask(__name__)

app.wsgi_app = DispatcherMiddleware(model.application.wsgi_app, {
    '/data': data.wsgi_app
})
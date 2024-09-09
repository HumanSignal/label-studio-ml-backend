from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flask import Flask

from data_wsgi import application as data
import model_wsgi as model

"""
Here, we create a Flask app to serve as a wrapper for both the ml-backend model api and the webhook api. By doing this, 
we can host both behind the same endpoint, with the model accessible at <host_url>/ and the webhook accessible at 
<host_url>/data/. We set app.wsgi_app in this way so that we can run our tests. 
"""
app = Flask(__name__)

app.wsgi_app = DispatcherMiddleware(model.application.wsgi_app, {
    '/data': data.wsgi_app
})
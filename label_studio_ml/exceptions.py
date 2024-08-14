import traceback as tb
import logging

from flask import request, jsonify, make_response

logger = logging.getLogger(__name__)


# make an answer to client
def answer(status=0, msg='', result=None):
    if status == 0 and not msg and result is None:
        status = -1000
        msg = "nothing happened"

    if status == 200 and not msg:
        msg = 'ok'

    a = {"status": status, "detail": msg}
    a.update({'request': request.args})

    if result is not None:
        a.update({"result": result})

    return make_response(jsonify(a), status)


# make an answer as exception
class AnswerException(Exception):
    def __init__(self, status, msg='', result=None):
        self.status, self.msg, self.result = status, msg, result
        self.answer = answer(status, msg, result)
        Exception.__init__(self, self.answer)


# standard exception treatment for any api function
def exception_handler(f):
    def exception_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)

        except AnswerException as e:
            traceback = tb.format_exc()
            logger.error(traceback)
            if 'traceback' not in e.result:
                e.result['traceback'] = traceback
            if hasattr(exception_f, 'request_id') and not e.result['request_id']:
                e.result['request_id'] = exception_f.request_id

            return answer(e.status, e.msg, e.result)

        except Exception as e:
            traceback = tb.format_exc()
            logger.error(traceback)
            print(traceback)
            body = {'traceback': traceback}
            if hasattr(exception_f, 'request_id'):
                body['request_id'] = exception_f.request_id
            return answer(500, e.__class__.__name__ + ': ' + str(e), body)

    exception_f.__name__ = f.__name__
    return exception_f
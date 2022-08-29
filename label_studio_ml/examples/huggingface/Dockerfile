FROM python:3.7

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY *.py /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

FROM python:3.7

WORKDIR /tmp
COPY requirements.txt .

RUN pip install --no-cache \
                -r requirements.txt \
                uwsgi==2.0.19.1 \
                supervisor==4.2.2 \
                label-studio==1.0.1 \
                git+https://github.com/heartexlabs/label-studio-ml-backend

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY *.py /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

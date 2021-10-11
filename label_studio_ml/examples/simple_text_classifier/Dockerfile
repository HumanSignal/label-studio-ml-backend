FROM python:3.7

WORKDIR /tmp
COPY requirements.txt .

RUN pip install --no-cache \
                -r requirements.txt \
                uwsgi==2.0.19.1 \
                supervisor==4.2.2 \
                click==7.1.2 \
                -e git+https://github.com/heartexlabs/label-studio-ml-backend@9f19155f622448cdf8bb35c3c27ea9b62afa6d74#egg=label-studio-ml

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY *.py /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

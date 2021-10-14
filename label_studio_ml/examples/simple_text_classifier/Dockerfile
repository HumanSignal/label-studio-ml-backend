FROM python:3.7

WORKDIR /tmp
COPY requirements.txt .

RUN pip install uwsgi==2.0.19.1 supervisor==4.2.2 click==7.1.2

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

RUN pip install -r requirements.txt

WORKDIR /app

RUN pip install -e git+https://github.com/heartexlabs/label-studio-ml-backend@32ab10cd86a6da96fdc795c349e9b4854c40ac7e#egg=label-studio-ml

COPY *.py /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

#!/bin/bash

# Execute the gunicorn command
exec gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime.app._wsgi:app

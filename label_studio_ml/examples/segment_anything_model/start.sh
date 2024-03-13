#!/bin/bash

# Execute the gunicorn command
exec gunicorn --preload --bind :${PORT:-9090} --workers 1 --threads 8 --timeout 0 _wsgi:app

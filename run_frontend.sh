#!/usr/bin/env bash

export PYTHONPATH="$( dirname "${BASH_SOURCE[0]}" )"
${PYTHONPATH}/venv/bin/gunicorn --certfile $1 --keyfile $2 -b 0.0.0.0:4000 wsgi:app

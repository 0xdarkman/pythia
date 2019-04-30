#!/usr/bin/env bash

gunicorn --certfile $1 --keyfile $2 -b 0.0.0.0:4000 wsgi:app

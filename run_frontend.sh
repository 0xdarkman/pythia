#!/usr/bin/env bash

gunicorn -b 127.0.0.1:8000 wsgi:app

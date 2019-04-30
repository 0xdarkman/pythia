#!/usr/bin/env bash

export PYTHONPATH="$( dirname "${BASH_SOURCE[0]}" )"
${PYTHONPATH}/venv/bin/python ${PYTHONPATH}/service/fpm_service.py
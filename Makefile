#Makefile for hume project

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH))))

VENV_NAME := venv_${CURRENT_DIR}
PYTHON=${VENV_NAME}/bin/python

sync-venv:
	: # Create or update default virtual environment to latest pinned
	: # dependencies
	test -d $(VENV_NAME) || \
				python3.11 -m virtualenv $(VENV_NAME); \
				${PYTHON} -m pip install -U uv; \
	. $(VENV_NAME)/bin/activate && uv pip sync requirements_dev.txt
	. $(VENV_NAME)/bin/activate && uv pip install --no-deps -e .

requirements:
	: # Update requirements_dev.txt if only new library is added
	: # Assumes virtual environment with pip-tools installed is activated
	uv pip compile pyproject.toml --output-file=requirements.txt  --annotation-style=line
	uv pip compile pyproject.toml --extra=dev --output-file=requirements_dev.txt  --annotation-style=line

update-requirements:
	: # Update requirements_dev.txt if dependencies should be updated
	: # Assumes virtual environment with pip-tools installed is activated
	uv pip compile pyproject.toml --output-file=requirements.txt  --annotation-style=line --upgrade
	uv pip compile pyproject.toml --extra=dev --output-file=requirements_dev.txt  --annotation-style=line --upgrade

precommit:
	: # Run precommit on all files locally (this runs only the pre-commit and not the pre-push hooks)
	pre-commit install
	pre-commit run --all-files

test:
	: # Run pytest
	python -m pytest

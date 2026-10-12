PYTHON ?= python3

.PHONY: format lint test coverage build check

format:
	$(PYTHON) -m ruff format imagedt tests scripts test setup.py
	$(PYTHON) -m ruff check imagedt tests scripts test --fix

lint:
	$(PYTHON) -m ruff check imagedt tests scripts test
	$(PYTHON) -m ruff format --check imagedt tests scripts test setup.py

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=imagedt --cov-report=term-missing --cov-fail-under=88

build:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
	$(PYTHON) scripts/check_release.py

check: lint test

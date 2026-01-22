##################################################################
# MultiNEAs Makefile
##################################################################

.PHONY: help install install-dev test clean build upload docs

help:
	@echo "MultiNEAs Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  test         - Run tests with pytest"
	@echo "  clean        - Remove build artifacts and cache files"
	@echo "  build        - Build distribution packages"
	@echo "  upload       - Upload package to PyPI"
	@echo "  upload-test  - Upload package to TestPyPI"
	@echo "  docs         - Build documentation"

install:
	pip install .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	pytest

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*

docs:
	cd docs && make html

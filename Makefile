all: build
format:
	black . -l 79
	linecheck . --fix
install:
	pip install -e .
test:
	pytest -m 'not local' --cov=./ --cov-report=xml
documentation:
	jupyter-book clean docs/book
	jupyter-book build docs/book
pip-package:
	pip install wheel
	python setup.py sdist bdist_wheel

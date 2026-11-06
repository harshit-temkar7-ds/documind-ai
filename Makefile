.PHONY: install run test lint
install:
	pip install -r requirements-dev.txt
run:
	python main.py
test:
	pytest tests/ -v
lint:
	ruff check .

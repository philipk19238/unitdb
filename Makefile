.PHONY: format lint

format:
	poetry run isort . --settings-path isort.cfg 
	poetry run black . --config black.cfg 
lint:
	poetry run flake8
	poetry run mypy -p unitdb

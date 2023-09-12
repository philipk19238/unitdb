.PHONY: format check-format lint

format:
	poetry run isort . --settings-path isort.cfg 
	poetry run black . --config black.cfg
check-format:
	poetry run isort . --settings-path isort.cfg --check-only 
	poetry run black . --config black.cfg --check 
lint:
	poetry run flake8
	poetry run mypy -p unitdb

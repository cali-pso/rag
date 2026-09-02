PYTHON = uv run python3
FLAKE = uv run flake8
MYPY = uv run mypy
SRC = src
RUN = $(PYTHON) -m $(SRC)

install:
	uv sync

run:
	$(RUN)

index:
	$(RUN) index

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf data/processed/
	find . -type d -name "__pycache__" -exec rm -rf {} +

fclean: clean
	rm -rf .venv

lint:
	- $(FLAKE) $(SRC)
	- $(MYPY) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --explicit-package-bases $(SRC)

lint-strict:
	- $(FLAKE) $(SRC)
	- $(MYPY) --strict $(SRC)

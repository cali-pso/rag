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

search_code:
	$(RUN) search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions

search_docs:
	$(RUN) search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions

eval_code:
	$(RUN) evaluate_student_search_results --answer_path data/output/search_results/UnansweredQuestions/student_search_results.json --truth_path data/datasets/AnsweredQuestions/dataset_code_public.json

eval_docs:
	$(RUN) evaluate_student_search_results --answer_path data/output/search_results/UnansweredQuestions/student_search_results.json --truth_path data/datasets/AnsweredQuestions/dataset_docs_public.json

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

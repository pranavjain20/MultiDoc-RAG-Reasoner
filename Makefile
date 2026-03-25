.PHONY: install test index ui eval

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

index:
	python examples/build_index.py

ui:
	python examples/build_UI.py

eval:
	python examples/export_evaluation_chunks.py
	python examples/evaluation.py

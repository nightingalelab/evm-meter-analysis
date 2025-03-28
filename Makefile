setup:
	python -m venv .venv && \
		. .venv/bin/activate && pip install -r requirements.txt && \
		pre-commit install

data: src/data.py
	. .venv/bin/activate && python src/data.py

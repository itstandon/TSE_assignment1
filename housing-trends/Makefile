.PHONY: install test mr mr-quick app qwen-server

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# One-command reproduction of the headline results table (results/mr/summary.md) on the frozen snapshot.
mr:
	python scripts/run_mrs.py

mr-quick:
	python scripts/run_mrs.py --quick

app:
	python -m streamlit run app_input.py

# Local Qwen 2.5 7B (llama.cpp) for the LLM agents; run on a GPU node, keep it running.
qwen-server:
	bash scripts/start_qwen_server.sh

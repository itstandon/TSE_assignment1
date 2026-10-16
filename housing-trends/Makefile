.PHONY: install test mr mr-quick mr-llm mr-diagnose app qwen-server ollama-server

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# One-command reproduction of the headline results table (results/mr/summary.md) on the frozen snapshot.
mr:
	python scripts/run_mrs.py

mr-quick:
	python scripts/run_mrs.py --quick

# Qwen 2.5 7B table for the top-3 MRs (results/mr_llm/summary.md). Replays from results/llm_cache/ (no GPU needed when
# every reply is cached); otherwise it needs `make ollama-server` running. The env vars override .env.local (Groq).
QWEN_ENV = LLM_PROVIDER=qwen LLM2_BASE_URL=http://127.0.0.1:11434/v1 LLM2_MODEL=qwen2.5:7b
mr-llm:
	$(QWEN_ENV) python scripts/run_mrs.py --llm real --only MR2,MR4,MR7 --workers 5 --out results/mr_llm

# Root-cause evidence for the top-3 MRs: ablations + bucket attribution (results/mr_diagnosis/, offline).
mr-diagnose:
	python scripts/diagnose_top3.py 2

app:
	python -m streamlit run app_input.py

# Local Qwen 2.5 7B (llama.cpp) for the LLM agents; run on a GPU node, keep it running.
qwen-server:
	bash scripts/start_qwen_server.sh

# The server actually used for results/mr_llm/: Ollama with the qwen2.5:7b model (Q4_K_M), 4 parallel request slots.
ollama-server:
	OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=24h ollama serve

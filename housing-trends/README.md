# housing-trends

## LLM: local Qwen 2.5 7B

The LLM agents (Parser for free-text input, Scanner's choice of comparable neighbour ZIPs, Critic and Pooling
decisions, Anomaly Diagnoser) use **Qwen 2.5 7B
Instruct** (GGUF, Q4_K_M) served locally by llama.cpp's `llama-server`. No API key and no cloud are involved.
`LLM_PROVIDER=qwen` in `.env.local` is the default.

```bash
srun --jobid=<JOBID> --overlap --pty bash          # or: srun -p u22 --gres=gpu:1 --pty bash
cd ~/TSE_assn/TSE_assignment1/housing-trends
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt     # once
bash scripts/start_qwen_server.sh > qwen.log 2>&1 &                       # = make qwen-server, http://127.0.0.1:8081/v1
until curl -s localhost:8081/health | grep -q ok; do sleep 2; done; echo "Qwen ready"
.venv/bin/streamlit run app_input.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

From laptop: `ssh -L 8501:<node>:8501 <user>@ada-gw1`, then open http://localhost:8501 and pick
"Qwen 7B (local)". 

Scripts work the same way (`scripts/run_pipeline.py`, `scripts/try_parser.py`,
`scripts/run_mrs.py --llm real`). Override `QWEN_GGUF`, `QWEN_PORT`, `QWEN_HOST`, `LLAMA_SERVER` if needed; port 8081
avoids vrspecml's 8080. If the server runs on another host, set `LLM2_BASE_URL=http://<host>:8081/v1` in `.env.local`.

Replies are cached in `results/llm_cache/`, keyed by model + seed (`qwen2.5-7b-instruct@seed42`). To go back to a
hosted model, set `LLM_PROVIDER=groq` or `google` and its key in `.env.local`. Tests use fake LLMs: `make test`.

## Reproducing the results

| Command | Produces | Needs |
|---|---|---|
| `make mr` | **Headline table** `results/mr/summary.md`: 10 MRs x 30 base inputs x seeds 101, 202, 303, rule-based stand-in for the LLM nodes. Also regenerates `docs/mr_catalog.md` | CPU only, about 15 min on 8 cores |
| `make mr-llm` | Qwen 2.5 7B table for the top-3 MRs (MR2, MR4, MR7) `results/mr_llm/summary.md`, same bases and seeds | Replays from `results/llm_cache/`, so no GPU or server is needed when every reply is cached. Otherwise run `make ollama-server` first (GPU; about 4.5 h live on a GTX 1080 Ti) |
| `make mr-diagnose` | Root-cause evidence `results/mr_diagnosis/` (loop ablations + per-bucket attribution) for `docs/root_cause_top3.md` | CPU only |
| `python scripts/mr_replay.py MR2 0 101` | One trial replayed with both contracts and the loops that fired | CPU only |

The Qwen MR run used **Ollama** (`qwen2.5:7b`, Q4_K_M, `OLLAMA_NUM_PARALLEL=4`) at `http://127.0.0.1:11434/v1`. The model's seed is the trial seed, at temperature 0. Without the cache
a live re-run is not guaranteed to be byte-identical (batched parallel requests, GPU arithmetic, Ollama version).
Ask for enough memory: 8 harness workers plus the model server exceeded a 10 GB job and got the server OOM-killed.

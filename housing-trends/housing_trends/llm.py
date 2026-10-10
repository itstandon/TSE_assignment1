"""LLM adapter. An 'llm' is any callable (system, user) -> str, so tests can inject a fake.
Configuration comes from .env.local at the repo root (LLM2_MODEL, LLM2_API_KEY, optional LLM2_BASE_URL, LLM_SEED)."""
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request

try:                                            # optional: only needed to read .env.local
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env.local")
except ImportError:
    pass


def openai_compat_llm(model: str, api_key: str, base_url: str = "https://api.groq.com/openai/v1",
                      seed: int = 42, timeout: int = 120):
    """Any OpenAI-compatible chat endpoint (Groq, vLLM, ...). Temperature 0, fixed seed, JSON mode.
    Record `model` and `seed` in your run log: they are part of the experiment."""

    def post(payload):
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                     "User-Agent": "housing-trends/0.1"})   # default urllib UA is blocked by some gateways
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def call(system: str, user: str) -> str:
        payload = {"model": model, "temperature": 0, "seed": seed,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "response_format": {"type": "json_object"}}
        try:
            return post(payload)
        except urllib.error.HTTPError as e:
            if e.code != 400:                       # 429 rate limit, 401 bad key, ... -> surface it
                raise RuntimeError(f"LLM API error {e.code}: {e.read().decode()[:300]}") from e
            payload.pop("response_format")          # endpoint rejected JSON mode: retry without it
            return post(payload)
    call.name = f"{model}@seed{seed}"
    return call


def llm_from_env():
    """Build an llm from env vars LLM2_MODEL, LLM2_API_KEY, optional LLM2_BASE_URL, LLM_SEED."""
    key = os.environ.get("LLM2_API_KEY")
    if not key:
        raise RuntimeError("LLM2_API_KEY is not set")
    return openai_compat_llm(
        model=os.environ.get("LLM2_MODEL", "openai/gpt-oss-120b"), api_key=key,
        base_url=os.environ.get("LLM2_BASE_URL", "https://api.groq.com/openai/v1"),
        seed=int(os.environ.get("LLM_SEED", "42")))


class CachedLLM:
    """Record/replay wrapper: identical (system, user) prompts are answered from disk, so a run that used a hosted
    model can be reproduced later without an API key. `tag` should name the model + seed (it is part of the key)."""

    def __init__(self, llm, directory, tag: str = ""):
        self.llm, self.dir, self.tag = llm, pathlib.Path(directory), tag
        self.hits = self.misses = 0
        self.name = f"cached:{getattr(llm, 'name', 'llm')}"

    def __call__(self, system: str, user: str) -> str:
        key = hashlib.sha256(f"{self.tag}\x00{system}\x00{user}".encode()).hexdigest()
        path = self.dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["reply"]
        self.misses += 1
        reply = self.llm(system, user)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tag": self.tag, "reply": reply}))
        return reply
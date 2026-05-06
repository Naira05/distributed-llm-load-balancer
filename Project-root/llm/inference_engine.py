llm/inference_engine.py

Copy
# llm/inference_engine.py
# HYBRID LLM — 3 backends
#   "simulated" → fast simulator       (default, 1000+ users safe)
#   "claude"    → Anthropic Claude API (real responses, needs internet)
#   "ollama"    → local Llama3         (offline, needs Ollama installed)
#
# HOW TO USE IN main.py:
#   from llm.inference_engine import set_backend
#   set_backend("simulated")   # load test 1000 users
#   set_backend("claude")      # real Claude API
#   set_backend("ollama")      # local Llama3 (run: ollama serve)
#
# gpu_worker.py needs NO changes — still calls infer() as before.

import time
import random
import threading
import requests
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Literal

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "llama3"

Backend = Literal["simulated", "claude", "ollama"]
_BACKEND: Backend = "simulated"   # ← change default here if needed

def set_backend(backend: Backend):
    global _BACKEND
    _BACKEND = backend
    labels = {
        "simulated": "SIMULATOR (safe for 1000+ users)",
        "claude":    "REAL Claude API",
        "ollama":    f"LOCAL Ollama / {OLLAMA_MODEL}",
    }
    print(f"[InferenceEngine] Backend → {labels.get(backend, backend)}")

def set_real_llm(enabled: bool):
    """Legacy helper — kept for backward compatibility."""
    set_backend("claude" if enabled else "simulated")

# ─────────────────────────────────────────────
# MODEL SPECS  (used by simulator + stats)
# ─────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    speed: float
    accuracy: float
    context_window: int
    base_latency_ms: float

MODELS = {
    "tiny":   ModelSpec("TinyLLM-1B",   100, 0.65,  512,  50),
    "small":  ModelSpec("SmallLLM-7B",   50, 0.78, 2048, 120),
    "medium": ModelSpec("MediumLLM-13B", 25, 0.85, 4096, 200),
    "large":  ModelSpec("LargeLLM-70B",  10, 0.92, 8192, 500),
}

# ─────────────────────────────────────────────
# ENGINE STATE  (stats — works for all backends)
# ─────────────────────────────────────────────

class InferenceEngine:
    def __init__(self):
        self.current_model = "medium"
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_latency = 0.0
        self.lock = threading.Lock()

    def get_stats(self):
        with self.lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            name = {"claude": CLAUDE_MODEL, "ollama": OLLAMA_MODEL}.get(
                _BACKEND, self.current_model)
            return {
                "model": name,
                "mode": _BACKEND,
                "total_requests": self.total_requests,
                "avg_latency_ms": round(avg * 1000, 2),
                "total_tokens": self.total_tokens_generated,
                "throughput_req_s": (self.total_requests / self.total_latency
                                     if self.total_latency > 0 else 0),
            }

    def reset_stats(self):
        with self.lock:
            self.total_requests = 0
            self.total_tokens_generated = 0
            self.total_latency = 0.0

    def _update_stats(self, latency_s: float, tokens: int):
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency_s
            self.total_tokens_generated += tokens

    def set_model(self, name: str) -> bool:
        if name not in MODELS:
            return False
        with self.lock:
            self.current_model = name
        return True

    def get_current_model(self) -> ModelSpec:
        return MODELS[self.current_model]

_engine = InferenceEngine()

# ─────────────────────────────────────────────
# BACKEND 1 — Claude API
# ─────────────────────────────────────────────

def _real_infer(prompt: str, max_tokens: int, context: str) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    content = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": min(max_tokens, 1024),
        "system": ("You are a helpful AI assistant in a distributed GPU cluster. "
                   "Answer concisely in 2-4 sentences."),
        "messages": [{"role": "user", "content": content}],
    }
    try:
        resp = requests.post(ANTHROPIC_API_URL,
                             headers={"Content-Type": "application/json"},
                             json=payload, timeout=30)
        data = resp.json()
        if resp.status_code != 200:
            raise Exception(f"API {resp.status_code}: {data.get('error', data)}")
        text   = data["content"][0]["text"]
        in_tok = data["usage"]["input_tokens"]
        out_tok = data["usage"]["output_tokens"]
        latency = time.time() - start
        _engine._update_stats(latency, out_tok)
        return {
            "request_id": request_id, "model": CLAUDE_MODEL,
            "prompt": prompt[:100], "response": text,
            "tokens": {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
            "performance": {"latency_ms": round(latency * 1000, 2),
                            "throughput_tokens_s": round(out_tok / latency, 2)},
            "mode": "claude", "status": "success",
        }
    except Exception as e:
        latency = time.time() - start
        _engine._update_stats(latency, 0)
        return {
            "request_id": request_id, "model": CLAUDE_MODEL,
            "prompt": prompt[:100], "response": f"[Claude Error: {str(e)[:100]}]",
            "tokens": {"input": 0, "output": 0, "total": 0},
            "performance": {"latency_ms": round(latency * 1000, 2), "throughput_tokens_s": 0},
            "mode": "claude_error", "status": "error", "error": str(e),
        }

# ─────────────────────────────────────────────
# BACKEND 2 — Ollama (local Llama3)
# ─────────────────────────────────────────────

def _ollama_infer(prompt: str, max_tokens: int, context: str) -> Dict[str, Any]:
    """
    Requires:
      1. ollama pull llama3
      2. ollama serve
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    content = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": content,
        "system": ("You are a helpful AI assistant in a distributed GPU cluster. "
                   "Answer concisely in 2-4 sentences."),
        "stream": False,
        "options": {"num_predict": min(max_tokens, 512), "temperature": 0.7},
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                             json=payload, timeout=120)
        data = resp.json()
        if resp.status_code != 200:
            raise Exception(f"Ollama {resp.status_code}: {data}")
        text    = data.get("response", "")
        out_tok = data.get("eval_count", len(text.split()))
        in_tok  = data.get("prompt_eval_count", len(content.split()))
        latency = time.time() - start
        _engine._update_stats(latency, out_tok)
        return {
            "request_id": request_id, "model": OLLAMA_MODEL,
            "prompt": prompt[:100], "response": text,
            "tokens": {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
            "performance": {"latency_ms": round(latency * 1000, 2),
                            "throughput_tokens_s": round(out_tok / latency, 2) if latency else 0},
            "mode": "ollama", "status": "success",
        }
    except requests.exceptions.ConnectionError:
        latency = time.time() - start
        _engine._update_stats(latency, 0)
        return {
            "request_id": request_id, "model": OLLAMA_MODEL,
            "prompt": prompt[:100],
            "response": "[Ollama not running — start with: ollama serve]",
            "tokens": {"input": 0, "output": 0, "total": 0},
            "performance": {"latency_ms": round(latency * 1000, 2), "throughput_tokens_s": 0},
            "mode": "ollama_error", "status": "error",
            "error": "Ollama not reachable at localhost:11434",
        }
    except Exception as e:
        latency = time.time() - start
        _engine._update_stats(latency, 0)
        return {
            "request_id": request_id, "model": OLLAMA_MODEL,
            "prompt": prompt[:100], "response": f"[Ollama Error: {str(e)[:100]}]",
            "tokens": {"input": 0, "output": 0, "total": 0},
            "performance": {"latency_ms": round(latency * 1000, 2), "throughput_tokens_s": 0},
            "mode": "ollama_error", "status": "error", "error": str(e),
        }

def ollama_health_check() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        ready  = any(OLLAMA_MODEL in m for m in models)
        return {
            "status": "running", "model_ready": ready,
            "available_models": models,
            "message": f"✅ {OLLAMA_MODEL} ready" if ready
                       else f"⚠️  Run: ollama pull {OLLAMA_MODEL}",
        }
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "model_ready": False,
                "message": "❌ Ollama not running — start with: ollama serve"}

# ─────────────────────────────────────────────
# BACKEND 3 — Simulator (calibrated)
# ─────────────────────────────────────────────

_SIM_RESPONSES = [
    "Based on distributed systems principles, {s} involves coordinating multiple nodes to ensure consistency and availability under high concurrency.",
    "Regarding {s}: the GPU cluster applies load-aware routing so no single worker becomes a bottleneck. Parallel inference pipelines maximize throughput.",
    "For {s}, the RAG pipeline retrieves relevant context from the vector store before passing it to the LLM, improving response accuracy significantly.",
    "Analyzing {s} using the distributed scheduler: tasks are assigned via least-connections policy when nodes have heterogeneous capacity.",
    "The answer to {s} depends on fault tolerance configuration. Active heartbeats detect failed nodes and reassign tasks within milliseconds.",
]

def _simulated_infer(prompt: str, max_tokens: int,
                     temperature: float, context: str) -> Dict[str, Any]:
    start = time.time()
    request_id = str(uuid.uuid4())[:8]
    model = _engine.get_current_model()
    p_tok = len(prompt.split())
    c_tok = len(context.split()) if context else 0
    total_in = p_tok + c_tok
    if total_in > model.context_window:
        return {"error": f"Input exceeds context window ({total_in} > {model.context_window})",
                "request_id": request_id, "status": "error"}
    ms = max(random.gauss(320, 70) + (total_in / model.context_window) * 180
             + abs(temperature - 1.0) * 40, 120) * random.uniform(0.92, 1.08)
    time.sleep(ms / 1000)
    out_tok = max(40, min(int(random.gauss(90, 20)), max_tokens))
    snippet  = " ".join(prompt.split()[:5])
    response = random.choice(_SIM_RESPONSES).format(s=snippet)
    if context:
        response += f" [RAG: {context[:50]}...]"
    latency = time.time() - start
    _engine._update_stats(latency, out_tok)
    return {
        "request_id": request_id, "model": model.name,
        "prompt": prompt[:100], "response": response,
        "tokens": {"input": total_in, "output": out_tok, "total": total_in + out_tok},
        "performance": {"latency_ms": round(latency * 1000, 2),
                        "throughput_tokens_s": round(out_tok / latency, 2)},
        "mode": "simulated", "status": "success",
    }

# ─────────────────────────────────────────────
# PUBLIC API  (same interface as original)
# ─────────────────────────────────────────────

def infer(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    context: str = "",
    backend: Backend = None,
    use_real: bool = None,      # legacy — True = "claude"
) -> Dict[str, Any]:
    if use_real is not None:
        active = "claude" if use_real else "simulated"
    elif backend is not None:
        active = backend
    else:
        active = _BACKEND

    if active == "claude":
        return _real_infer(prompt, max_tokens, context)
    elif active == "ollama":
        return _ollama_infer(prompt, max_tokens, context)
    else:
        return _simulated_infer(prompt, max_tokens, temperature, context)

def batch_infer(prompts: List[str], num_workers: int = 4,
                backend: Backend = None) -> List[Dict[str, Any]]:
    return [infer(p, backend=backend) for p in prompts]

def stream_infer(prompt: str, max_tokens: int = 256,
                 chunk_size: int = 50) -> Dict[str, Any]:
    if _BACKEND in ("claude", "ollama"):
        fn = _real_infer if _BACKEND == "claude" else _ollama_infer
        r  = fn(prompt, max_tokens, "")
        r["chunks"] = [r["response"]]
        r["tokens_generated"] = r["tokens"]["output"]
        r["latency_ms"] = r["performance"]["latency_ms"]
        return r
    start = time.time()
    model  = _engine.get_current_model()
    chunks, tokens_generated = [], 0
    for i in range((max_tokens // chunk_size) + 1):
        ct = min(chunk_size, max_tokens - tokens_generated)
        if ct == 0: break
        time.sleep(max((model.base_latency_ms / 1000) * (1 - i * 0.1), 0.01))
        chunks.append(f"[chunk-{i}: {ct} tokens] ")
        tokens_generated += ct
    latency = time.time() - start
    _engine._update_stats(latency, tokens_generated)
    return {
        "model": model.name, "prompt": prompt, "response": "".join(chunks),
        "chunks": chunks, "tokens_generated": tokens_generated,
        "latency_ms": round(latency * 1000, 2), "mode": "simulated", "status": "success",
    }

def health_check() -> Dict[str, Any]:
    stats  = _engine.get_stats()
    avg_ms = stats["avg_latency_ms"]
    status = ("IDLE" if avg_ms == 0 else "HEALTHY" if avg_ms < 500
              else "DEGRADED" if avg_ms < 1500 else "SLOW")
    result = {"status": status, "backend": _BACKEND, "model": stats["model"],
              "avg_latency_ms": avg_ms, "total_requests": stats["total_requests"],
              "throughput_req_s": round(stats["throughput_req_s"], 2)}
    if _BACKEND == "ollama":
        result["ollama"] = ollama_health_check()
    return result

def set_model(name: str) -> bool:      return _engine.set_model(name)
def get_stats() -> Dict[str, Any]:     return _engine.get_stats()
def reset_stats():                     _engine.reset_stats()

def get_model_info() -> Dict[str, Any]:
    model = _engine.get_current_model()
    return {
        "name": {"claude": CLAUDE_MODEL, "ollama": OLLAMA_MODEL}.get(_BACKEND, model.name),
        "backend": _BACKEND,
        "speed": f"{model.speed} req/s",
        "accuracy": f"{model.accuracy:.0%}",
        "context_window": model.context_window,
        "base_latency_ms": model.base_latency_ms,
    }

def list_models() -> Dict[str, Dict[str, Any]]:
    return {n: {"name": s.name, "accuracy": f"{s.accuracy:.0%}",
                "speed": f"{s.speed} req/s", "context": s.context_window}
            for n, s in MODELS.items()}

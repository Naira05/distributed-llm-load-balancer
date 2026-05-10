import time
import threading
import uuid
import ollama
from dataclasses import dataclass
from typing import Dict, Any, List

# CONFIG

DEFAULT_MODEL = "tinyllama"  # MUST match: `ollama run tinyllama`

@dataclass
class ModelSpec:
    name: str
    speed: float
    accuracy: float
    context_window: int
    base_latency_ms: float


MODELS = {
    "tiny": ModelSpec("tinyllama", 100, 0.65, 4096, 80),
    "small": ModelSpec("small", 50, 0.75, 4096, 120),
    "medium": ModelSpec("medium", 25, 0.85, 4096, 200),
    "large": ModelSpec("large", 10, 0.92, 8192, 500),
}

# ENGINE

class InferenceEngine:
    def __init__(self):
        self.current_model = "tiny"
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_latency = 0.0
        self.lock = threading.Lock()

    def get_current_model(self):
        return MODELS[self.current_model]

    def set_model(self, model_name: str):
        if model_name in MODELS:
            self.current_model = model_name
            return True
        return False

    def _update(self, latency, tokens):
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency
            self.total_tokens_generated += tokens

    def get_stats(self):
        with self.lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            return {
                "model": self.current_model,
                "requests": self.total_requests,
                "avg_latency_ms": round(avg * 1000, 2),
                "throughput_req_s": self.total_requests / self.total_latency if self.total_latency else 0
            }


_engine = InferenceEngine()

# REAL OLLAMA INFERENCE

def infer(
    prompt: str,
    max_tokens: int = 256,
    context: str = "",
    temperature: float = 0.7
) -> Dict[str, Any]:

    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    model = MODELS[DEFAULT_MODEL].name  # "tinyllama"

    full_prompt = f"{context}\n\n{prompt}" if context else prompt

    try:
        response = ollama.generate(
            model=model,
            prompt=full_prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        output = response["response"]

    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e)
        }

    latency = time.time() - start
    tokens = len(output.split())

    _engine._update(latency, tokens)

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
        "request_id": request_id,
        "model": model,
        "prompt": prompt[:100],
        "response": output,
        "tokens": {
            "output": tokens
        },
        "performance": {
            "latency_ms": round(latency * 1000, 2),
            "throughput_tokens_s": round(tokens / latency, 2)
        },
        "status": "success"
    }

# BATCH INFERENCE

def batch_infer(prompts: List[str]) -> List[Dict[str, Any]]:
    results = []
    for p in prompts:
        results.append(infer(p))
    return results

# STREAMING (SIMULATED)

def stream_infer(prompt: str) -> Dict[str, Any]:
    start = time.time()

    result = infer(prompt)
    text = result.get("response", "")

    chunks = []
    words = text.split()

    for i in range(0, len(words), 5):
        chunk = " ".join(words[i:i+5])
        time.sleep(0.02)
        chunks.append(chunk)

    return {
        "request_id": result.get("request_id"),
        "response": text,
        "chunks": chunks,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "status": "success"
    }

# STATS

def get_stats():
    return _engine.get_stats()

def reset_stats():
    global _engine
    _engine = InferenceEngine()

def set_model(name: str):
    return _engine.set_model(name)

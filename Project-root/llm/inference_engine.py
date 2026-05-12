import time
import threading
import uuid
import ollama
from dataclasses import dataclass
from typing import Dict, Any, List

# ── CONFIG ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "tiny"   # must match: ollama pull tinyllama

@dataclass
class ModelSpec:
    name: str
    speed: float
    accuracy: float
    context_window: int
    base_latency_ms: float


MODELS = {
    "tiny":   ModelSpec("tinyllama:latest", 100, 0.65, 4096,  80),
    "small":  ModelSpec("small:latest",      50, 0.75, 4096, 120),
    "medium": ModelSpec("medium:latest",     25, 0.85, 4096, 200),
    "large":  ModelSpec("large:latest",      10, 0.92, 8192, 500),
}

# ── ENGINE ────────────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self):
        self.current_model   = "tiny"
        self.total_requests  = 0
        self.total_tokens    = 0
        self.total_latency   = 0.0
        self.lock            = threading.Lock()

    def get_current_model(self) -> ModelSpec:
        return MODELS[self.current_model]

    def set_model(self, model_name: str) -> bool:
        if model_name in MODELS:
            self.current_model = model_name
            return True
        return False

    def _update(self, latency: float, tokens: int):
        with self.lock:
            self.total_requests += 1
            self.total_latency  += latency
            self.total_tokens   += tokens

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            tps = self.total_requests / self.total_latency if self.total_latency  else 0
            return {
                "model":              self.current_model,
                "requests":           self.total_requests,
                "avg_latency_ms":     round(avg * 1000, 2),
                "throughput_req_s":   round(tps, 4),
            }


_engine = InferenceEngine()

# ── REAL OLLAMA INFERENCE ─────────────────────────────────────────────────────

def infer(
    prompt:      str,
    max_tokens:  int   = 256,
    context:     str   = "",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Call tinyllama via Ollama and return a structured result dict.
    This is the ONLY function that actually hits the model.
    """
    start      = time.time()
    request_id = str(uuid.uuid4())[:8]
    model_name = MODELS[DEFAULT_MODEL].name          # "tinyllama"
    full_prompt = f"{context}\n\n{prompt}" if context else prompt

    try:
        response = ollama.generate(
            model=model_name,
            prompt=full_prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        output = response["response"]
        status = "success"

    except Exception as exc:
        return {
            "request_id": request_id,
            "model":       model_name,
            "prompt":      prompt[:100],
            "response":    "",
            "tokens":      {"output": 0},
            "performance": {"latency_ms": 0, "throughput_tokens_s": 0},
            "status":      "error",
            "error":       str(exc),
        }

    latency = time.time() - start
    tokens  = len(output.split())

    _engine._update(latency, tokens)

    # ── THE BUG FIX: the original infer() forgot to return anything ──
    return {
        "request_id": request_id,
        "model":      model_name,
        "prompt":     prompt[:100],
        "response":   output,
        "tokens":     {"output": tokens},
        "performance": {
            "latency_ms":          round(latency * 1000, 2),
            "throughput_tokens_s": round(tokens / latency, 2) if latency else 0,
        },
        "status": status,
    }

# ── BATCH INFERENCE ───────────────────────────────────────────────────────────

def batch_infer(prompts: List[str]) -> List[Dict[str, Any]]:
    """Process a list of prompts sequentially and return all results."""
    return [infer(p) for p in prompts]

# ── STREAMING (word-by-word simulation over real output) ─────────────────────

def stream_infer(
    prompt:     str,
    max_tokens: int = 64,
    chunk_size: int = 5,
) -> Dict[str, Any]:
    """
    Calls real Ollama inference, then yields the response in word-chunks
    to simulate streaming. Returns the full result plus a `chunks` list.
    """
    start  = time.time()
    result = infer(prompt, max_tokens=max_tokens)
    text   = result.get("response", "")
    words  = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        time.sleep(0.02)          # simulate network/streaming delay
        chunks.append(chunk)

    return {
        "request_id": result.get("request_id"),
        "model":      result.get("model"),
        "response":   text,
        "chunks":     chunks,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "status":     result.get("status", "success"),
    }

# ── STATS / CONTROL ───────────────────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    return _engine.get_stats()

def reset_stats():
    global _engine
    _engine = InferenceEngine()

def set_model(name: str) -> bool:
    return _engine.set_model(name)
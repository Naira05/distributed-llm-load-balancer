# llm/inference_engine.py
# NEW LLM IMPLEMENTATION - INDEPENDENT OF LEGACY CODE

import time
import random
import threading
from dataclasses import dataclass
from typing import Dict, Any, List
import uuid

# =========================================================
# CONFIGURATION & MODELS
# =========================================================

@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    name: str
    speed: float  # requests per second capacity
    accuracy: float  # 0.0-1.0
    context_window: int  # max tokens
    base_latency_ms: float


# Pre-defined models
MODELS = {
    "tiny": ModelSpec(
        name="TinyLLM-1B",
        speed=100,
        accuracy=0.65,
        context_window=512,
        base_latency_ms=50
    ),
    "small": ModelSpec(
        name="SmallLLM-7B",
        speed=50,
        accuracy=0.78,
        context_window=2048,
        base_latency_ms=120
    ),
    "medium": ModelSpec(
        name="MediumLLM-13B",
        speed=25,
        accuracy=0.85,
        context_window=4096,
        base_latency_ms=200
    ),
    "large": ModelSpec(
        name="LargeLLM-70B",
        speed=10,
        accuracy=0.92,
        context_window=8192,
        base_latency_ms=500
    ),
}

# =========================================================
# GLOBAL INFERENCE ENGINE STATE
# =========================================================

class InferenceEngine:
    """Global LLM inference engine with stats."""

    def __init__(self):
        self.current_model = "medium"
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_latency = 0.0
        self.lock = threading.Lock()
        self.request_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self.lock:
            avg_latency = (
                self.total_latency / self.total_requests 
                if self.total_requests > 0 
                else 0
            )
            return {
                "model": self.current_model,
                "total_requests": self.total_requests,
                "avg_latency_ms": round(avg_latency * 1000, 2),
                "total_tokens": self.total_tokens_generated,
                "throughput_req_s": (
                    self.total_requests / self.total_latency 
                    if self.total_latency > 0 
                    else 0
                )
            }

    def reset_stats(self):
        """Reset all statistics."""
        with self.lock:
            self.total_requests = 0
            self.total_tokens_generated = 0
            self.total_latency = 0.0
            self.request_history = []

    def _update_stats(self, latency_s: float, tokens: int):
        """Internal: update statistics."""
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency_s
            self.total_tokens_generated += tokens

    def set_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if model_name not in MODELS:
            return False
        with self.lock:
            self.current_model = model_name
        return True

    def get_current_model(self) -> ModelSpec:
        """Get current model specification."""
        return MODELS[self.current_model]


# Global engine instance
_engine = InferenceEngine()


# =========================================================
# CORE INFERENCE FUNCTIONS
# =========================================================

def infer(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    context: str = ""
) -> Dict[str, Any]:
    """
    Execute LLM inference on a prompt.

    Args:
        prompt: Input text to generate from
        max_tokens: Maximum output tokens
        temperature: Sampling temperature (0.0-2.0)
        context: Optional RAG context

    Returns:
        Dictionary with response, metadata, and performance info
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    model = _engine.get_current_model()

    # -----------------------------------------------
    # Simulate inference complexity
    # -----------------------------------------------

    # Input complexity factor
    prompt_length = len(prompt.split())
    context_length = len(context.split()) if context else 0
    total_input_tokens = prompt_length + context_length

    # Constraint checking
    if total_input_tokens > model.context_window:
        return {
            "error": f"Input exceeds context window ({total_input_tokens} > {model.context_window})",
            "request_id": request_id,
            "status": "error"
        }

    # Output tokens (simulate)
    output_tokens = min(int(max_tokens * random.uniform(0.7, 1.0)), max_tokens)

    # Latency calculation
    base_latency = model.base_latency_ms / 1000.0
    input_factor = (total_input_tokens / model.context_window) * 0.3
    output_factor = (output_tokens / 256) * 0.5
    temperature_factor = abs(temperature - 1.0) * 0.1

    total_latency = base_latency * (1 + input_factor + output_factor + temperature_factor)

    # Add random jitter (GPU variance)
    total_latency *= random.uniform(0.9, 1.1)

    # Simulate computation
    time.sleep(total_latency)

    # -----------------------------------------------
    # Generate response
    # -----------------------------------------------
    response_text = _generate_llm_response(prompt, context, model, output_tokens)

    actual_latency = time.time() - start_time

    # Update stats
    _engine._update_stats(actual_latency, output_tokens)

    return {
        "request_id": request_id,
        "model": model.name,
        "prompt": prompt[:100],  # truncate for display
        "response": response_text,
        "tokens": {
            "input": total_input_tokens,
            "output": output_tokens,
            "total": total_input_tokens + output_tokens
        },
        "performance": {
            "latency_ms": round(actual_latency * 1000, 2),
            "throughput_tokens_s": round(output_tokens / actual_latency, 2)
        },
        "status": "success"
    }


def batch_infer(prompts: List[str], num_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Execute inference on multiple prompts sequentially or in batches.

    Args:
        prompts: List of prompts to process
        num_workers: Number of parallel processing threads (hint)

    Returns:
        List of inference results
    """
    results = []
    for prompt in prompts:
        result = infer(prompt)
        results.append(result)
    return results


def stream_infer(
    prompt: str,
    max_tokens: int = 256,
    chunk_size: int = 50
) -> Dict[str, Any]:
    """
    Simulate streaming inference (token-by-token generation).

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        chunk_size: Tokens per chunk

    Returns:
        Full response with simulated chunks
    """
    start_time = time.time()
    model = _engine.get_current_model()

    chunks = []
    tokens_generated = 0

    # Simulate streaming chunks
    num_chunks = (max_tokens // chunk_size) + 1

    for i in range(num_chunks):
        chunk_tokens = min(chunk_size, max_tokens - tokens_generated)
        if chunk_tokens == 0:
            break

        # Simulate chunk latency (with decreasing latency per chunk)
        chunk_latency = (model.base_latency_ms / 1000) * (1 - i * 0.1)
        time.sleep(max(chunk_latency, 0.01))

        chunk_text = _generate_chunk(i, chunk_tokens)
        chunks.append(chunk_text)
        tokens_generated += chunk_tokens

    total_latency = time.time() - start_time
    _engine._update_stats(total_latency, tokens_generated)

    full_response = "".join(chunks)

    return {
        "model": model.name,
        "prompt": prompt,
        "response": full_response,
        "chunks": chunks,
        "tokens_generated": tokens_generated,
        "latency_ms": round(total_latency * 1000, 2),
        "status": "success"
    }


def health_check() -> Dict[str, Any]:
    """
    Check engine health and readiness.

    Returns:
        Health status dictionary
    """
    stats = _engine.get_stats()

    if stats["avg_latency_ms"] < 150:
        status = "HEALTHY"
    elif stats["avg_latency_ms"] < 300:
        status = "DEGRADED"
    else:
        status = "SLOW"

    return {
        "status": status,
        "model": stats["model"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "total_requests": stats["total_requests"],
        "throughput_req_s": round(stats["throughput_req_s"], 2)
    }


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _generate_llm_response(prompt: str, context: str, model: ModelSpec, tokens: int) -> str:
    """Generate realistic LLM-style response."""
    
    responses = [
        f"Based on the context, here's a comprehensive answer about '{prompt[:50]}...'",
        f"Analyzing your query using {model.name}. The response suggests...",
        f"After processing with accuracy {model.accuracy:.0%}, the answer is:",
        f"Using advanced reasoning: {context[:50] if context else 'general knowledge'}...",
        f"Model {model.name} generated: This is a thoughtful response to your query.",
    ]

    base_response = random.choice(responses)

    # Add synthetic content based on token count
    token_filler = " ".join([f"token{i}" for i in range(min(tokens - 20, 50))])

    return f"{base_response} {token_filler}"


def _generate_chunk(chunk_idx: int, tokens: int) -> str:
    """Generate a single streaming chunk."""
    chunk_templates = [
        "Processing ",
        "Analyzing ",
        "Computing ",
        "Inferring ",
        "Reasoning "
    ]
    base = random.choice(chunk_templates)
    return base + f"chunk-{chunk_idx} ({tokens} tokens) "


# =========================================================
# CONFIGURATION / CONTROL
# =========================================================

def set_model(model_name: str) -> bool:
    """Switch to a different LLM model."""
    return _engine.set_model(model_name)


def get_model_info() -> Dict[str, Any]:
    """Get information about the current model."""
    model = _engine.get_current_model()
    return {
        "name": model.name,
        "speed": f"{model.speed} req/s",
        "accuracy": f"{model.accuracy:.0%}",
        "context_window": model.context_window,
        "base_latency_ms": model.base_latency_ms
    }


def list_models() -> Dict[str, Dict[str, Any]]:
    """List all available models."""
    return {
        name: {
            "name": spec.name,
            "accuracy": f"{spec.accuracy:.0%}",
            "speed": f"{spec.speed} req/s",
            "context": spec.context_window,
        }
        for name, spec in MODELS.items()
    }


def get_stats() -> Dict[str, Any]:
    """Get engine statistics."""
    return _engine.get_stats()


def reset_stats():
    """Reset engine statistics."""
    _engine.reset_stats()

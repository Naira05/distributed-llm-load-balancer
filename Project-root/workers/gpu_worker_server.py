
import argparse
import threading
import time
import random
import sys
import os

from flask import Flask, request, jsonify

# ── allow "from llm.inference_engine import infer" when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.inference_engine import infer

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Worker state (one per process) ───────────────────────────────────────────

class GPUWorkerState:
    def __init__(self, worker_id: int):
        self.worker_id       = worker_id
        self.active_tasks    = 0
        self.total_processed = 0
        self.failed          = False
        self.gpu_utilization = 0
        self.gpu_memory_used = 0.0
        self.lock            = threading.Lock()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _enter_task(self):
        with self.lock:
            self.active_tasks    += 1
            self.gpu_utilization  = min(100, self.gpu_utilization + random.randint(5, 15))
            self.gpu_memory_used  = min(24,  self.gpu_memory_used + random.uniform(0.5, 2.0))

    def _leave_task(self):
        with self.lock:
            self.active_tasks    = max(0,  self.active_tasks    - 1)
            self.gpu_utilization = max(0,  self.gpu_utilization - random.randint(3, 10))
            self.gpu_memory_used = max(0.0, self.gpu_memory_used - random.uniform(0.3, 1.5))

    # ── process one inference request ─────────────────────────────────────────

    def process(self, payload: dict) -> dict:
        if self.failed:
            return {
                "request_id": payload.get("id"),
                "worker_id":  self.worker_id,
                "error":      f"Worker {self.worker_id} is unavailable (failed)",
                "status":     "error",
            }

        request_id = payload.get("id")
        query      = payload.get("query")
        context    = payload.get("context", "")

        if not query:
            return {
                "request_id": request_id,
                "worker_id":  self.worker_id,
                "error":      "Missing 'query' field in request",
                "status":     "error",
            }

        self._enter_task()
        start = time.time()

        try:
            print(
                f"[GPU-{self.worker_id}] ▶ request={request_id} | "
                f"gpu={self.gpu_utilization}% | active={self.active_tasks}"
            )

            response = infer(prompt=query, context=context)
            latency  = time.time() - start

            output_text = response.get("response", "") if isinstance(response, dict) else str(response)
            status      = response.get("status", "success") if isinstance(response, dict) else "success"

            with self.lock:
                self.total_processed += 1

            print(
                f"[GPU-{self.worker_id}] ✔ request={request_id} | "
                f"latency={latency:.2f}s | tokens≈{len(output_text.split())}"
            )

            return {
                "request_id":       request_id,
                "worker_id":        self.worker_id,
                "response":         output_text,
                "latency_s":        round(latency, 3),
                "gpu_utilization":  self.gpu_utilization,
                "gpu_memory_used":  round(self.gpu_memory_used, 2),
                "status":           status,
            }

        except Exception as exc:
            print(f"[GPU-{self.worker_id}] ✖ request={request_id} | error={exc}")
            return {
                "request_id": request_id,
                "worker_id":  self.worker_id,
                "error":      str(exc),
                "status":     "error",
            }

        finally:
            self._leave_task()

    # ── load snapshot ─────────────────────────────────────────────────────────

    def get_load(self) -> dict:
        with self.lock:
            return {
                "worker_id":       self.worker_id,
                "active_tasks":    self.active_tasks,
                "gpu_utilization": self.gpu_utilization,
                "gpu_memory_used": round(self.gpu_memory_used, 2),
                "total_processed": self.total_processed,
                "failed":          self.failed,
            }


# Created after argument parsing (see __main__ block)
_worker: GPUWorkerState = None


# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.route("/process", methods=["POST"])
def process_request():
    """
    Accepts JSON:  { "id": "...", "query": "...", "context": "..." }
    Returns JSON result dict.
    """
    payload = request.get_json(force=True, silent=True) or {}
    result  = _worker.process(payload)
    code    = 200 if result.get("status") == "success" else 500
    return jsonify(result), code


@app.route("/load", methods=["GET"])
def get_load():
    """
    Returns current load metrics.
    Used by the load balancer for least-connections and load-aware routing.
    """
    return jsonify(_worker.get_load()), 200


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    if _worker.failed:
        return jsonify({"status": "failed", "worker_id": _worker.worker_id}), 503
    return jsonify({"status": "ok", "worker_id": _worker.worker_id}), 200


@app.route("/fail", methods=["POST"])
def fail_worker():
    """Trigger a simulated failure (for fault-tolerance testing)."""
    _worker.failed = True
    print(f"[GPU-{_worker.worker_id}] ⚠ SIMULATED FAILURE TRIGGERED")
    return jsonify({"status": "failed"}), 200


@app.route("/recover", methods=["POST"])
def recover_worker():
    """Recover from a simulated failure."""
    _worker.failed = False
    print(f"[GPU-{_worker.worker_id}] ✔ RECOVERED")
    return jsonify({"status": "recovered"}), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Worker HTTP Server")
    parser.add_argument("--worker-id", type=int, default=1,   help="Unique worker ID")
    parser.add_argument("--port",      type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host",      type=str, default="127.0.0.1",
                        help="Host to bind (use 0.0.0.0 for multi-machine)")
    args = parser.parse_args()

    _worker = GPUWorkerState(worker_id=args.worker_id)

    print(f"[GPU-{args.worker_id}] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
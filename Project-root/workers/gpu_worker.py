# workers/gpu_worker.py
# GPU Worker implementation using new inference engine

import time
import random
import threading
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.inference_engine import infer, health_check, get_stats
from rag.retriever import retrieve_context


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="[GPU-WORKER-%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger("GPUWorker")


# =========================================================
# GPU WORKER NODE
# =========================================================

class GPUWorker:
    """
    Simulates a GPU inference node in a distributed LLM system.

    Pipeline:
    Request → RAG Retrieval → GPU Inference → Response
    """

    def __init__(self, worker_id: int, capacity: int = 10):
        """
        Initialize a GPU worker.

        Args:
            worker_id: Unique identifier for this worker
            capacity: Maximum concurrent tasks
        """
        self.id = worker_id
        self.capacity = capacity

        # -------------------------
        # STATE
        # -------------------------
        self.active_tasks = 0
        self.is_alive = True

        # -------------------------
        # METRICS
        # -------------------------
        self.total_requests = 0
        self.total_latency = 0.0
        self.successful_requests = 0
        self.failed_requests = 0

        # -------------------------
        # THREAD SAFETY
        # -------------------------
        self.lock = threading.Lock()

        log.info(f"GPU Worker {self.id} initialized (capacity={capacity})")

    # =====================================================
    # MAIN EXECUTION PIPELINE
    # =====================================================

    def process(self, request):
        """
        Executes full inference pipeline for a request.

        Args:
            request: Dictionary with 'id' and 'query' keys

        Returns:
            Result dictionary with response and metadata
        """

        start_time = time.time()

        with self.lock:
            if not self.is_alive:
                raise Exception(f"Worker {self.id} is dead")
            if self.active_tasks >= self.capacity:
                raise Exception(f"Worker {self.id} at capacity")
            self.active_tasks += 1

        request_id = request.get("id", -1)
        query = request.get("query", "")

        log.info(f"Worker {self.id} started request {request_id}")

        try:
            # -------------------------
            # 1. RAG RETRIEVAL
            # -------------------------
            try:
                context = retrieve_context(query)
            except Exception as e:
                log.warning(f"RAG retrieval failed: {e}, continuing without context")
                context = ""

            # -------------------------
            # 2. GPU INFERENCE
            # -------------------------
            result = infer(query, context=context, max_tokens=256)

            # -------------------------
            # 3. POST-PROCESSING
            # -------------------------
            self._post_process()

            latency = time.time() - start_time

            with self.lock:
                self.total_requests += 1
                self.total_latency += latency
                self.successful_requests += 1
                self.active_tasks -= 1

            log.info(f"Worker {self.id} completed request {request_id} ({latency:.3f}s)")

            return {
                "worker_id": self.id,
                "request_id": request_id,
                "result": result,
                "status": "success",
                "latency": latency
            }

        except Exception as e:
            log.error(f"Worker {self.id} failed on request {request_id}: {e}")

            with self.lock:
                self.total_requests += 1
                self.failed_requests += 1
                self.active_tasks -= 1

            return {
                "worker_id": self.id,
                "request_id": request_id,
                "error": str(e),
                "status": "failed"
            }

    # =====================================================
    # HELPER METHODS
    # =====================================================

    def _post_process(self):
        """Simulate post-processing delay (output formatting, etc)."""
        post_process_delay = random.uniform(0.01, 0.05)
        time.sleep(post_process_delay)

    def get_metrics(self):
        """Get worker metrics."""
        with self.lock:
            avg_latency = (
                self.total_latency / self.successful_requests
                if self.successful_requests > 0
                else 0
            )
            return {
                "worker_id": self.id,
                "total_requests": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "avg_latency": avg_latency,
                "active_tasks": self.active_tasks
            }

    def is_healthy(self):
        """Check if worker is healthy and accepting requests."""
        with self.lock:
            return self.is_alive and self.active_tasks < self.capacity

    def shutdown(self):
        """Gracefully shutdown the worker."""
        with self.lock:
            self.is_alive = False
        log.info(f"Worker {self.id} shutdown")

    def restart(self):
        """Restart the worker."""
        with self.lock:
            self.is_alive = True
        log.info(f"Worker {self.id} restarted")

    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.total_requests = 0
            self.total_latency = 0.0
            self.successful_requests = 0
            self.failed_requests = 0


# =========================================================
# WORKER POOL MANAGER
# =========================================================

class GPUWorkerPool:
    """Manages a pool of GPU workers."""

    def __init__(self, num_workers: int = 4):
        """Initialize worker pool."""
        self.workers = [GPUWorker(i) for i in range(num_workers)]
        self.num_workers = num_workers
        self.lock = threading.Lock()

    def dispatch(self, request):
        """
        Dispatch request to the least loaded worker.

        Args:
            request: Request dictionary

        Returns:
            Result from worker
        """
        # Find worker with fewest active tasks
        best_worker = min(
            self.workers,
            key=lambda w: w.active_tasks if w.is_healthy() else float('inf')
        )

        if not best_worker.is_healthy():
            raise Exception("No healthy workers available")

        return best_worker.process(request)

    def get_pool_metrics(self):
        """Get metrics for all workers in pool."""
        return {
            "num_workers": self.num_workers,
            "workers": [w.get_metrics() for w in self.workers],
            "health": [w.is_healthy() for w in self.workers]
        }

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            worker.shutdown()

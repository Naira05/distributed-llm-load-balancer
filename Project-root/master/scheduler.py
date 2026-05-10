# master/scheduler.py

import time
import threading
import logging
from dataclasses import asdict

log = logging.getLogger("Master")


class Scheduler:
    """
    Thin coordinator that sits between the load generator and the LoadBalancer.
    All worker selection and fault-tolerance logic lives in LoadBalancer —
    the scheduler just formats the request, calls lb.dispatch(), and tracks metrics.
    """

    def __init__(self, load_balancer, strategy: str = "load_aware", **kwargs):
        self.lb       = load_balancer
        self.strategy = strategy          # "round_robin" | "least_connections" | "load_aware"

        self._lock            = threading.Lock()
        self.total_requests   = 0
        self.failed_requests  = 0
        self.total_latency    = 0.0

        log.info(f"Scheduler started | strategy={strategy}")

    # ── main entry point called by LoadGenerator ──────────────────────────────

    def handle_request(self, request) -> dict:
        return self.assign_task(request)

    # ── dispatch through HTTP load balancer ───────────────────────────────────

    def assign_task(self, request) -> dict:
        start = time.time()

        with self._lock:
            self.total_requests += 1

        # Build the JSON payload the worker server expects
        if hasattr(request, "__dict__"):
            payload = {
                "id":      getattr(request, "id",      None),
                "query":   getattr(request, "query",   ""),
                "context": getattr(request, "context", ""),
            }
        elif isinstance(request, dict):
            payload = {
                "id":      request.get("id"),
                "query":   request.get("query", ""),
                "context": request.get("context", ""),
            }
        else:
            payload = {"id": None, "query": str(request), "context": ""}

        # Send to a real HTTP worker via the load balancer
        response = self.lb.dispatch(payload, strategy=self.strategy)

        latency = time.time() - start

        with self._lock:
            self.total_latency += latency
            if response.get("status") != "success":
                self.failed_requests += 1

        response["latency"] = latency
        return response

    # ── metrics ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            return {
                "total_requests":  self.total_requests,
                "failed_requests": self.failed_requests,
                "avg_latency_ms":  round(avg * 1000, 2),
            }
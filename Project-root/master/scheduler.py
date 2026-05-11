# master/scheduler.py

import time
import threading
import logging
from typing import Optional

log = logging.getLogger("Master")


class Scheduler:
    """
    Coordinator between the load generator and the LoadBalancer.
    Tracks per-worker GPU stats, request routing, and live status.
    """

    def __init__(self, load_balancer, strategy: str = "load_aware", **kwargs):
        self.lb       = load_balancer
        self.strategy = strategy

        self._lock           = threading.Lock()
        self.total_requests  = 0
        self.failed_requests = 0
        self.total_latency   = 0.0

        # per-worker tracking: { url -> { requests, failed, total_latency, last_gpu } }
        self._worker_stats = {url: {
            "requests":      0,
            "failed":        0,
            "total_latency": 0.0,
            "last_gpu_util": None,
            "last_gpu_mem":  None,
            "last_worker_id": None,
        } for url in load_balancer.worker_urls}

        log.info(f"Scheduler started | strategy={strategy}")

    # ── GPU status helpers ────────────────────────────────────────────────────

    def fetch_worker_gpu(self, url: str) -> Optional[dict]:
        """Fetch /load from a single worker and return GPU info."""
        try:
            import requests as req
            data = req.get(f"{url}/load", timeout=3).json()
            return {
                "worker_id":       data.get("worker_id"),
                "gpu_utilization": data.get("gpu_utilization", 0),
                "gpu_memory_used": data.get("gpu_memory_used", 0),
                "active_tasks":    data.get("active_tasks", 0),
                "total_processed": data.get("total_processed", 0),
                "failed":          data.get("failed", False),
            }
        except Exception:
            return None

    def print_gpu_status(self, header: str = "GPU Worker Status"):
        """Print a live snapshot of all workers' GPU stats."""
        print(f"\n{'─'*55}")
        print(f"  {header}")
        print(f"{'─'*55}")
        alive = self.lb.status()
        for url in self.lb.worker_urls:
            is_alive = alive.get(url, {}).get("alive", False)
            if not is_alive:
                print(f"  ✖ {url}")
                print(f"      Status: DOWN")
                continue

            gpu = self.fetch_worker_gpu(url)
            if gpu:
                print(f"  ✔ {url}")
                print(f"      Worker ID      : {gpu['worker_id']}")
                print(f"      GPU Utilization: {gpu['gpu_utilization']}%")
                print(f"      GPU Memory Used: {gpu['gpu_memory_used']} MB")
                print(f"      Active Tasks   : {gpu['active_tasks']}")
                print(f"      Total Processed: {gpu['total_processed']}")
                print(f"      Failed         : {gpu['failed']}")
            else:
                print(f"  ? {url}  (could not fetch /load)")
        print(f"{'─'*55}\n")

    # ── main entry point ──────────────────────────────────────────────────────

    def handle_request(self, request) -> dict:
        return self.assign_task(request)

    # ── dispatch ──────────────────────────────────────────────────────────────

    def assign_task(self, request) -> dict:
        start = time.time()

        with self._lock:
            self.total_requests += 1

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

        # Ask LB which worker it will use BEFORE dispatching (for logging)
        chosen_url = self.lb.load_aware() if self.strategy == "load_aware" else \
                     self.lb.round_robin() if self.strategy == "round_robin" else \
                     self.lb.least_connections()

        response = self.lb.dispatch(payload, strategy=self.strategy)
        latency  = time.time() - start

        # ── update global stats ───────────────────────────────────────────────
        with self._lock:
            self.total_latency += latency
            if response.get("status") != "success":
                self.failed_requests += 1

        # ── update per-worker stats + snapshot GPU ────────────────────────────
        if chosen_url and chosen_url in self._worker_stats:
            gpu = self.fetch_worker_gpu(chosen_url)
            with self._lock:
                ws = self._worker_stats[chosen_url]
                ws["requests"]      += 1
                ws["total_latency"] += latency
                if response.get("status") != "success":
                    ws["failed"] += 1
                if gpu:
                    ws["last_gpu_util"]  = gpu["gpu_utilization"]
                    ws["last_gpu_mem"]   = gpu["gpu_memory_used"]
                    ws["last_worker_id"] = gpu["worker_id"]

            log.info(
                f"[Request {payload['id']}] → {chosen_url} | "
                f"latency={latency*1000:.1f}ms | "
                f"status={response.get('status')} | "
                f"GPU={gpu['gpu_utilization'] if gpu else '?'}% | "
                f"mem={gpu['gpu_memory_used'] if gpu else '?'}MB"
            )

        response["latency"]     = latency
        response["routed_to"]   = chosen_url
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

    def print_worker_summary(self):
        """Print per-worker breakdown after the load test."""
        print(f"\n{'═'*55}")
        print("  PER-WORKER SUMMARY")
        print(f"{'═'*55}")
        with self._lock:
            for url, ws in self._worker_stats.items():
                reqs = ws["requests"]
                avg  = (ws["total_latency"] / reqs * 1000) if reqs else 0
                print(f"\n  Worker : {url}")
                print(f"  ID     : {ws['last_worker_id']}")
                print(f"  Handled: {reqs} requests  |  Failed: {ws['failed']}")
                print(f"  Avg Lat: {avg:.1f} ms")
                print(f"  Last GPU Util : {ws['last_gpu_util']}%")
                print(f"  Last GPU Mem  : {ws['last_gpu_mem']} MB")
        print(f"{'═'*55}\n")
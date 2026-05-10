import threading
import time
import requests
from typing import List, Optional, Dict, Any


class LoadBalancer:

    def __init__(
        self,
        worker_urls: List[str],
        health_check_interval: float = 5.0,
        request_timeout: float = 120.0,
    ):
        self.worker_urls = list(worker_urls)
        self.request_timeout = request_timeout
        self._current_rr = 0
        self._lock = threading.Lock()

        self._alive: Dict[str, bool] = {url: True for url in worker_urls}

        self._health_thread = threading.Thread(
            target=self._health_loop,
            args=(health_check_interval,),
            daemon=True,
        )
        self._health_thread.start()

        print(f"[LB] Started with {len(worker_urls)} workers: {worker_urls}")

    # ── alive workers ────────────────────────────────────────────────
    def _live_workers(self) -> List[str]:
        with self._lock:
            return [u for u, alive in self._alive.items() if alive]

    # ── health check ────────────────────────────────────────────────
    def _health_loop(self, interval: float):
        while True:
            time.sleep(interval)
            for url in self.worker_urls:
                self._check_health(url)

    def _check_health(self, url: str):
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            alive = resp.status_code == 200
        except Exception:
            alive = False

        with self._lock:
            self._alive[url] = alive

    # ── Round Robin ────────────────────────────────────────────────
    def round_robin(self) -> Optional[str]:
        workers = self._live_workers()
        if not workers:
            return None

        with self._lock:
            url = workers[self._current_rr % len(workers)]
            self._current_rr += 1
        return url

    # ── Least Connections ───────────────────────────────────────────
    def least_connections(self) -> Optional[str]:
        workers = self._live_workers()
        if not workers:
            return None

        best, best_load = None, float("inf")

        for url in workers:
            try:
                data = requests.get(f"{url}/load", timeout=2).json()
                load = data.get("active_tasks", 0)

                if load < best_load:
                    best_load = load
                    best = url

            except Exception:
                with self._lock:
                    self._alive[url] = False

        return best

    # ── Load Aware ────────────────────────────────────────────────
    def load_aware(self) -> Optional[str]:
        workers = self._live_workers()
        if not workers:
            return None

        best, best_score = None, float("inf")

        for url in workers:
            try:
                data = requests.get(f"{url}/load", timeout=2).json()
                score = data.get("active_tasks", 0) + data.get("gpu_utilization", 0) / 100

                if score < best_score:
                    best_score = score
                    best = url

            except Exception:
                with self._lock:
                    self._alive[url] = False

        return best

    # ── Dispatch ────────────────────────────────────────────────
    def dispatch(self, request_payload: Dict[str, Any], strategy: str = "load_aware"):
        strategy_fn = {
            "round_robin": self.round_robin,
            "least_connections": self.least_connections,
            "load_aware": self.load_aware,
        }.get(strategy, self.load_aware)

        url = strategy_fn()

        if url is None:
            return {"status": "error", "error": "No workers available"}

        try:
            resp = requests.post(f"{url}/process", json=request_payload, timeout=self.request_timeout)
            return resp.json()

        except Exception as e:
            with self._lock:
                self._alive[url] = False
            return {"status": "error", "error": str(e)}

    def status(self):
        with self._lock:
            return {u: {"alive": a} for u, a in self._alive.items()}
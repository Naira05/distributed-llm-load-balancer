import threading
import time
import requests
from typing import List, Optional, Dict, Any


class LoadBalancer:

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        worker_urls: List[str],
        health_check_interval: float = 5.0,   # seconds between health checks
        request_timeout:       float = 120.0,  # seconds before a /process call times out
    ):
        self.worker_urls           = list(worker_urls)
        self.request_timeout       = request_timeout
        self._current_rr           = 0
        self._lock                 = threading.Lock()

        # track which workers are currently alive
        self._alive: Dict[str, bool] = {url: True for url in worker_urls}

        # start background health-checker
        self._health_thread = threading.Thread(
            target=self._health_loop,
            args=(health_check_interval,),
            daemon=True,
        )
        self._health_thread.start()
        print(f"[LB] Started with {len(worker_urls)} workers: {worker_urls}")

    # ── alive workers ─────────────────────────────────────────────────────────

    def _live_workers(self) -> List[str]:
        with self._lock:
            return [u for u, alive in self._alive.items() if alive]

    # ── background health checks ──────────────────────────────────────────────

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
            was_alive = self._alive.get(url, True)
            self._alive[url] = alive

        if was_alive and not alive:
            print(f"[LB] ⚠  Worker {url} is DOWN")
        elif not was_alive and alive:
            print(f"[LB] ✔  Worker {url} is back UP")

    # ── strategy: Round Robin ─────────────────────────────────────────────────

    def round_robin(self) -> Optional[str]:
        workers = self._live_workers()
        if not workers:
            return None
        with self._lock:
            url = workers[self._current_rr % len(workers)]
            self._current_rr += 1
        return url

    # ── strategy: Least Connections ───────────────────────────────────────────

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
                    best      = url
            except Exception:
                # worker unreachable → skip it
                with self._lock:
                    self._alive[url] = False
                print(f"[LB] ⚠  {url} unreachable during least-connections probe")

        return best

    # ── strategy: Load-Aware Routing ─────────────────────────────────────────

    def load_aware(self) -> Optional[str]:
        """
        Score = active_tasks + gpu_utilization / 100
        Lower score → preferred worker.
        """
        workers = self._live_workers()
        if not workers:
            return None

        best, best_score = None, float("inf")
        for url in workers:
            try:
                data  = requests.get(f"{url}/load", timeout=2).json()
                score = data.get("active_tasks", 0) + data.get("gpu_utilization", 0) / 100
                if score < best_score:
                    best_score = score
                    best       = url
            except Exception:
                with self._lock:
                    self._alive[url] = False
                print(f"[LB] ⚠  {url} unreachable during load-aware probe")

        return best

    # ── dispatch ──────────────────────────────────────────────────────────────

    def dispatch(
        self,
        request_payload: Dict[str, Any],
        strategy: str = "load_aware",
    ) -> Dict[str, Any]:
        """
        Send *request_payload* to the best available worker using *strategy*.

        strategy options: "round_robin" | "least_connections" | "load_aware"

        Returns the worker's response dict, or an error dict if all workers
        are unavailable or the request fails.
        """
        strategy_fn = {
            "round_robin":       self.round_robin,
            "least_connections": self.least_connections,
            "load_aware":        self.load_aware,
        }.get(strategy, self.load_aware)

        url = strategy_fn()

        if url is None:
            return {
                "request_id": request_payload.get("id"),
                "status":     "error",
                "error":      "No workers available",
            }

        try:
            print(f"[LB] → dispatch request={request_payload.get('id')} to {url}")
            resp = requests.post(
                f"{url}/process",
                json=request_payload,
                timeout=self.request_timeout,
            )
            return resp.json()

        except requests.exceptions.Timeout:
            # mark worker suspect; caller can retry
            with self._lock:
                self._alive[url] = False
            return {
                "request_id": request_payload.get("id"),
                "status":     "error",
                "error":      f"Worker {url} timed out",
            }

        except Exception as exc:
            with self._lock:
                self._alive[url] = False
            return {
                "request_id": request_payload.get("id"),
                "status":     "error",
                "error":      str(exc),
            }

    # ── status snapshot ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a summary of all workers and their alive/dead state."""
        with self._lock:
            return {url: {"alive": alive} for url, alive in self._alive.items()}
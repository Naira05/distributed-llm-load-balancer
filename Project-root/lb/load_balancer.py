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
        max_retries: int = 3,
    ):
        self.worker_urls    = list(worker_urls)
        self.request_timeout = request_timeout
        self.max_retries    = max_retries
        self._current_rr    = 0
        self._lock          = threading.Lock()

        self._alive: Dict[str, bool]  = {url: True for url in worker_urls}
        # track total processed per worker for better scoring
        self._processed: Dict[str, int] = {url: 0 for url in worker_urls}

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

    # ── health check ──────────────────────────────────────────────────────────
    def _health_loop(self, interval: float):
        while True:
            time.sleep(interval)
            for url in self.worker_urls:
                self._check_health(url)

    def _check_health(self, url: str):
        try:
            resp  = requests.get(f"{url}/health", timeout=2)
            alive = resp.status_code == 200
        except Exception:
            alive = False
        with self._lock:
            self._alive[url] = alive

    # ── Round Robin ───────────────────────────────────────────────────────────
    def round_robin(self, exclude: Optional[str] = None) -> Optional[str]:
        workers = [w for w in self._live_workers() if w != exclude]
        if not workers:
            return None
        with self._lock:
            url = workers[self._current_rr % len(workers)]
            self._current_rr += 1
        return url

    # ── Least Connections ─────────────────────────────────────────────────────
    def least_connections(self, exclude: Optional[str] = None) -> Optional[str]:
        workers = [w for w in self._live_workers() if w != exclude]
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

    # ── Load Aware (improved) ─────────────────────────────────────────────────
    def load_aware(self, exclude: Optional[str] = None) -> Optional[str]:
        """
        Score = active_tasks * 3          (heaviest weight — real-time pressure)
              + gpu_utilization / 100      (GPU busy-ness)
              + gpu_memory_used / 1000     (memory pressure)
              + total_processed / 50       (historical load — prevents funneling)
        Lower score = better worker.
        """
        workers = [w for w in self._live_workers() if w != exclude]
        if not workers:
            return None

        best, best_score = None, float("inf")
        for url in workers:
            try:
                data  = requests.get(f"{url}/load", timeout=2).json()
                score = (
                    data.get("active_tasks",    0) * 3
                    + data.get("gpu_utilization", 0) / 100
                    + data.get("gpu_memory_used", 0) / 1000
                    + data.get("total_processed", 0) / 50
                )
                if score < best_score:
                    best_score = score
                    best = url
            except Exception:
                with self._lock:
                    self._alive[url] = False
        return best

    # ── Dispatch with built-in failover ───────────────────────────────────────
    def dispatch(
        self,
        request_payload: Dict[str, Any],
        strategy: str = "load_aware",
    ) -> Dict[str, Any]:

        strategy_fn = {
            "round_robin":       self.round_robin,
            "least_connections": self.least_connections,
            "load_aware":        self.load_aware,
        }.get(strategy, self.load_aware)

        last_failed_url = None

        for attempt in range(1, self.max_retries + 1):
            # exclude the last worker that failed so we pick a different one
            url = strategy_fn(exclude=last_failed_url)

            if url is None:
                return {"status": "error", "error": "No healthy workers available"}

            try:
                resp   = requests.post(
                    f"{url}/process",
                    json=request_payload,
                    timeout=self.request_timeout,
                )
                result = resp.json()

                if result.get("status") == "success":
                    with self._lock:
                        self._processed[url] = self._processed.get(url, 0) + 1
                    result["routed_to"] = url
                    return result

                # worker returned non-success — treat as soft failure
                print(f"[LB] Attempt {attempt}: {url} returned status={result.get('status')}, retrying...")
                with self._lock:
                    self._alive[url] = False
                last_failed_url = url
                time.sleep(0.5)

            except Exception as e:
                print(f"[LB] Attempt {attempt}: {url} threw exception ({e}), retrying...")
                with self._lock:
                    self._alive[url] = False
                last_failed_url = url
                time.sleep(0.5)

        return {
            "status": "error",
            "error":  f"All {self.max_retries} attempts failed",
            "routed_to": last_failed_url,
        }

    def status(self):
        with self._lock:
            return {u: {"alive": a} for u, a in self._alive.items()}
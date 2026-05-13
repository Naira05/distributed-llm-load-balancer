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
        self.worker_urls = list(worker_urls)
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self._lock = threading.Lock()
        self._rr_index = 0

        # worker state
        self._alive = {u: True for u in worker_urls}
        self._fail_counts = {u: 0 for u in worker_urls}

        # simple tracking (optional stats only)
        self._session_processed = {u: 0 for u in worker_urls}

        # health thread
        self._health_thread = threading.Thread(
            target=self._health_loop,
            args=(health_check_interval,),
            daemon=True,
        )
        self._health_thread.start()

        print(f"[LB] Round Robin Load Balancer started with {len(worker_urls)} workers")

    # ─────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────

    def _health_loop(self, interval: float):
        while True:
            time.sleep(interval)

            for url in self.worker_urls:
                try:
                    r = requests.get(f"{url}/health", timeout=5)
                    alive = r.status_code == 200
                except:
                    alive = False

                with self._lock:
                    if alive:
                        self._alive[url] = True
                        self._fail_counts[url] = 0
                    else:
                        self._fail_counts[url] += 1
                        if self._fail_counts[url] >= 5:
                            self._alive[url] = False

    # ─────────────────────────────
    # LIVE WORKERS
    # ─────────────────────────────

    def _live_workers(self):
        with self._lock:
            return [w for w, a in self._alive.items() if a]

    # ─────────────────────────────
    # PURE ROUND ROBIN
    # ─────────────────────────────

    def round_robin(self, exclude: Optional[str] = None):

        workers = [w for w in self._live_workers() if w != exclude]

        if not workers:
            return None

        with self._lock:

            # safe modulo RR index
            self._rr_index %= len(workers)
            url = workers[self._rr_index]
            self._rr_index += 1

        return url

    # ─────────────────────────────
    # DISPATCH
    # ─────────────────────────────

    def dispatch(self, request_payload: Dict[str, Any]):

        last_failed = None

        for attempt in range(1, self.max_retries + 1):

            url = self.round_robin(exclude=last_failed)

            if url is None:
                time.sleep(0.5)
                continue

            try:
                start = time.time()

                resp = requests.post(
                    f"{url}/process",
                    json=request_payload,
                    timeout=self.request_timeout,
                )

                latency = time.time() - start
                result = resp.json()

                if result.get("status") == "success":

                    with self._lock:
                        self._session_processed[url] += 1
                        self._fail_counts[url] = 0

                    result["routed_to"] = url
                    result["latency"] = latency
                    return result

                last_failed = url
                time.sleep(0.3)

            except Exception:
                with self._lock:
                    self._fail_counts[url] += 1
                    if self._fail_counts[url] >= 5:
                        self._alive[url] = False

                last_failed = url
                time.sleep(0.3)

        return {
            "status": "error",
            "error": "All retries failed",
            "routed_to": last_failed,
        }

    # ─────────────────────────────
    # STATUS
    # ─────────────────────────────

    def status(self):
        with self._lock:
            return {
                u: {
                    "alive": a,
                    "processed": self._session_processed[u],
                    "fail_count": self._fail_counts[u],
                }
                for u, a in self._alive.items()
            }
import threading
import time
import requests
from typing import List, Optional, Dict, Any


# Strategy constants
ROUND_ROBIN    = "round_robin"
LEAST_CONN     = "least_connections"
LOAD_AWARE     = "load_aware"


class LoadBalancer:

    def __init__(
        self,
        worker_urls: List[str],
        strategy: str = ROUND_ROBIN,
        health_check_interval: float = 5.0,
        request_timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.worker_urls = list(worker_urls)
        self.strategy = strategy
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self._lock = threading.Lock()

        # Round Robin index
        self._rr_index = 0

        # worker state
        self._alive            = {u: True for u in worker_urls}
        self._fail_counts      = {u: 0    for u in worker_urls}
        self._session_processed = {u: 0   for u in worker_urls}

        # Least Connections — tracks how many requests are in-flight per worker
        self._active_connections = {u: 0 for u in worker_urls}

        self._worker_stats = {
            u: {"gpu_util": 0, "gpu_mem": 0, "queue": 0}
            for u in worker_urls
        }

        # start background health-check thread
        self._health_thread = threading.Thread(
            target=self._health_loop,
            args=(health_check_interval,),
            daemon=True,
        )
        self._health_thread.start()

        print(f"[LB] Load Balancer started | strategy={strategy} | workers={len(worker_urls)}")



    def _health_loop(self, interval: float):
        while True:
            time.sleep(interval)

            for url in self.worker_urls:
                try:
                    r = requests.get(
                        f"{url}/health",
                        headers={"ngrok-skip-browser-warning": "true"},
                        timeout=5,
                    )
                    alive = r.status_code == 200

                    # Load-Aware: parse GPU/queue stats from health response if present
                    if alive:
                        try:
                            data = r.json()
                            with self._lock:
                                self._worker_stats[url]["gpu_util"] = data.get("gpu_utilization", 0)
                                self._worker_stats[url]["gpu_mem"]  = data.get("gpu_memory_used", 0)
                                self._worker_stats[url]["queue"]    = data.get("queue_depth", 0)
                        except Exception:
                            pass  # health endpoint returned non-JSON — still alive, stats unknown

                except Exception:
                    alive = False

                with self._lock:
                    prev_state = self._alive[url]

                    if alive:
                        self._alive[url] = True
                        self._fail_counts[url] = 0

                        if not prev_state:
                            print(f"[LB] ♻  Worker RECOVERED: {url}")

                    else:
                        self._fail_counts[url] += 1

                        if self._fail_counts[url] >= 5:
                            if self._alive[url]:
                                print(f"[LB] ☠  Worker DEAD: {url}")
                            self._alive[url] = False


    #  HELPERS                                                             
    def _live_workers(self) -> List[str]:
        with self._lock:
            return [u for u, alive in self._alive.items() if alive]

    def _candidates(self, exclude: Optional[str]) -> List[str]:
        return [w for w in self._live_workers() if w != exclude]


    def _round_robin(self, exclude: Optional[str] = None) -> Optional[str]:
        workers = self._candidates(exclude)

        if not workers:
            return None

        with self._lock:
            self._rr_index %= len(workers)
            url = workers[self._rr_index]
            self._rr_index += 1

        return url


    def _least_connections(self, exclude: Optional[str] = None) -> Optional[str]:
        workers = self._candidates(exclude)

        if not workers:
            return None

        with self._lock:
            url = min(workers, key=lambda w: self._active_connections[w])

        return url



    def _load_aware(self, exclude: Optional[str] = None) -> Optional[str]:
        workers = self._candidates(exclude)

        if not workers:
            return None

        with self._lock:
            def score(w):
                s = self._worker_stats[w]
                return s["gpu_util"] + s["queue"] * 10

            url = min(workers, key=score)

        return url


    #  PICK WORKER  (routes to the right strategy)                        #

    def _pick_worker(self, exclude: Optional[str] = None) -> Optional[str]:
        if self.strategy == LEAST_CONN:
            return self._least_connections(exclude)
        elif self.strategy == LOAD_AWARE:
            return self._load_aware(exclude)
        else:
            return self._round_robin(exclude)   # default


    #  DISPATCH  (main entry point — send one request, retry on failure)  #

    def dispatch(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        last_failed = None

        for attempt in range(1, self.max_retries + 1):

            url = self._pick_worker(exclude=last_failed)

            if url is None:
                print(f"[LB] ⚠  No live workers (attempt {attempt}), waiting...")
                time.sleep(0.5)
                continue

            print(f"[LB] → Attempt {attempt} | strategy={self.strategy} | "
                  f"req={request_payload.get('id')} | worker={url}")

            # track in-flight for Least Connections
            with self._lock:
                self._active_connections[url] += 1

            try:
                start = time.time()

                resp = requests.post(
                    f"{url}/process",
                    json=request_payload,
                    headers={"ngrok-skip-browser-warning": "true"},
                    timeout=self.request_timeout,
                )

                latency = time.time() - start
                result  = resp.json()

                if result.get("status") == "success":
                    with self._lock:
                        self._session_processed[url] += 1
                        self._fail_counts[url] = 0

                    print(f"[LB] ✅ Success | worker={url} | latency={latency:.2f}s")

                    result["routed_to"] = url
                    result["latency"]   = latency
                    return result

                # ---- soft failure (worker ran but returned an error status) ----
                print(f"[LB] ⚠  Soft failure from {url}, retrying elsewhere...")
                last_failed = url
                time.sleep(0.3)

            except Exception as e:
                # ---- hard failure (network down, timeout, crash) ----
                latency = time.time() - start
                print(f"[LB] ❌ Hard failure | worker={url} | error={e}")

                with self._lock:
                    self._fail_counts[url] += 1
                    if self._fail_counts[url] >= 5:
                        self._alive[url] = False
                        print(f"[LB] ☠  Worker marked DEAD: {url}")

                last_failed = url
                time.sleep(0.3)

            finally:
                # always decrement in-flight counter whether success or failure
                with self._lock:
                    self._active_connections[url] -= 1
                    # clamp to 0 — should never go negative but safety net
                    if self._active_connections[url] < 0:
                        self._active_connections[url] = 0

        print("[LB] ❌ All retries exhausted")
        return {
            "status":     "error",
            "error":      "All retries failed",
            "routed_to":  last_failed,
        }


    #  DISPATCH TO ALL  (send to every live worker in parallel)           #
    def dispatch_to_all(
        self,
        base_payload: Dict[str, Any],
        contexts: List[str],
    ) -> List[Dict[str, Any]]:
        workers = self._live_workers()

        if not workers:
            return [{"status": "error", "error": "No live workers"}]

        results  = [None] * len(workers)   # pre-allocate so order is preserved
        threads  = []

        def call_worker(index: int, url: str, payload: Dict[str, Any]):
            with self._lock:
                self._active_connections[url] += 1
            try:
                start = time.time()
                resp  = requests.post(
                    f"{url}/process",
                    json=payload,
                    headers={"ngrok-skip-browser-warning": "true"},
                    timeout=self.request_timeout,
                )
                latency = time.time() - start
                result  = resp.json()
                result["routed_to"] = url
                result["latency"]   = latency
                results[index] = result

            except Exception as e:
                results[index] = {
                    "status":    "error",
                    "error":     str(e),
                    "routed_to": url,
                }
            finally:
                with self._lock:
                    self._active_connections[url] -= 1
                    if self._active_connections[url] < 0:
                        self._active_connections[url] = 0

        # launch one thread per worker
        for i, url in enumerate(workers):
            payload = dict(base_payload)
            payload["context"] = contexts[i % len(contexts)]

            t = threading.Thread(target=call_worker, args=(i, url, payload))
            threads.append(t)
            t.start()

        # wait for all threads to finish
        for t in threads:
            t.join()

        return results


    #  STATUS  (snapshot of all worker health + stats)                    
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                u: {
                    "alive":       self._alive[u],
                    "processed":   self._session_processed[u],
                    "fail_count":  self._fail_counts[u],
                    "active_conn": self._active_connections[u],
                    "gpu_util":    self._worker_stats[u]["gpu_util"],
                    "gpu_mem":     self._worker_stats[u]["gpu_mem"],
                    "queue":       self._worker_stats[u]["queue"],
                }
                for u in self.worker_urls
            }

    def set_strategy(self, strategy: str):
        assert strategy in (ROUND_ROBIN, LEAST_CONN, LOAD_AWARE), \
            f"Unknown strategy: {strategy}"
        self.strategy = strategy
        print(f"[LB] Strategy switched to: {strategy}")
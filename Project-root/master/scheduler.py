import time
import threading
import logging

log = logging.getLogger("Master")


class Scheduler:

    def __init__(self, load_balancer, strategy="round_robin"):
        self.lb = load_balancer
        self.strategy = strategy

        self._lock = threading.Lock()

        # global stats
        self.total_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0

        # per-worker stats
        self._worker_stats = {
            url: {
                "requests": 0,
                "failed": 0,
                "total_latency": 0.0,
                "last_gpu_util": None,
                "last_gpu_mem": None,
                "last_worker_id": None,
            }
            for url in load_balancer.worker_urls
        }

        log.info(f"Scheduler started | strategy={strategy}")

   
    # GPU STATUS 
    def fetch_worker_gpu(self, url):
        try:
            import requests
            data = requests.get(f"{url}/load", timeout=3).json()
            return {
                "worker_id": data.get("worker_id"),
                "gpu_utilization": data.get("gpu_utilization", 0),
                "gpu_memory_used": data.get("gpu_memory_used", 0),
                "active_tasks": data.get("active_tasks", 0),
            }
        except Exception:
            return None

    def print_gpu_status(self, header="GPU Worker Status"):

        print(f"\n{'─'*55}")
        print(f"  {header}")
        print(f"{'─'*55}")

        alive = self.lb.status()

        for url in self.lb.worker_urls:

            if not alive.get(url, {}).get("alive", False):
                print(f"  ✖ {url}")
                print("      Status: DOWN")
                continue

            gpu = self.fetch_worker_gpu(url)

            if gpu:
                print(f"  ✔ {url}")
                print(f"      Worker ID      : {gpu['worker_id']}")
                print(f"      GPU Utilization: {gpu['gpu_utilization']}%")
                print(f"      GPU Memory Used: {gpu['gpu_memory_used']} MB")
                print(f"      Active Tasks   : {gpu['active_tasks']}")
            else:
                print(f"  ? {url} (no /load response)")

        print(f"{'─'*55}\n")

    # MAIN ENTRY
    def handle_request(self, request):
        return self.assign_task(request)

    def assign_task(self, request):

        start = time.time()

        with self._lock:
            self.total_requests += 1

        # normalize request
        if isinstance(request, dict):
            payload = request
        else:
            payload = {"query": str(request)}

        # ── ONLY ROUND ROBIN DISPATCH ──
        response = self.lb.dispatch(payload)

        latency = time.time() - start
        chosen_url = response.get("routed_to")

        with self._lock:
            self.total_latency += latency

            if response.get("status") != "success":
                self.failed_requests += 1

        # update per-worker stats
        if chosen_url and chosen_url in self._worker_stats:
            gpu = self.fetch_worker_gpu(chosen_url)

            with self._lock:
                ws = self._worker_stats[chosen_url]
                ws["requests"] += 1
                ws["total_latency"] += latency

                if gpu:
                    ws["last_gpu_util"] = gpu["gpu_utilization"]
                    ws["last_gpu_mem"] = gpu["gpu_memory_used"]
                    ws["last_worker_id"] = gpu["worker_id"]

        response["latency"] = latency
        response["routed_to"] = chosen_url

        return response

  
    # STATS
    def get_stats(self):

        with self._lock:

            avg = (
                self.total_latency / self.total_requests
                if self.total_requests else 0
            )

            return {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "avg_latency_ms": round(avg * 1000, 2),
            }

    # WORKER SUMMARY
    def print_worker_summary(self):

        print(f"\n{'═'*55}")
        print("  PER-WORKER SUMMARY")
        print(f"{'═'*55}")

        with self._lock:

            for url, ws in self._worker_stats.items():

                reqs = ws["requests"]

                avg = (
                    ws["total_latency"] / reqs * 1000
                    if reqs else 0
                )

                print(f"\n  Worker : {url}")
                print(f"  ID     : {ws['last_worker_id']}")
                print(f"  Handled: {reqs} requests  |  Failed: {ws['failed']}")
                print(f"  Avg Lat: {avg:.1f} ms")
                print(f"  Last GPU Util : {ws['last_gpu_util']}%")
                print(f"  Last GPU Mem  : {ws['last_gpu_mem']} MB")

        print(f"{'═'*55}\n")
# master/scheduler.py

import time
import threading
import logging
from enum import Enum

log = logging.getLogger("Master")


class Strategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE = "load_aware"


class WorkerStatus:
    def __init__(self, worker_id, total_capacity=10):
        self.worker_id = worker_id
        self.total_capacity = total_capacity
        self.active_tasks = 0
        self.is_alive = True
        self.last_heartbeat = time.time()
        self.lock = threading.Lock()

    @property
    def is_free(self):
        return self.is_alive and self.active_tasks < self.total_capacity

    @property
    def load_percent(self):
        return self.active_tasks / self.total_capacity if self.total_capacity else 1.0


class Scheduler:
    def __init__(self, load_balancer, num_workers=4,
                 strategy=Strategy.LEAST_CONNECTIONS,
                 heartbeat_timeout=15.0):   # FIXED

        self.lb = load_balancer
        self.strategy = strategy
        self.heartbeat_timeout = heartbeat_timeout

        self.worker_table = {
            i: WorkerStatus(i) for i in range(num_workers)
        }

        self._rr_index = 0
        self._rr_lock = threading.Lock()

        self._metrics_lock = threading.Lock()
        self.total_requests = 0
        self.failed_assigns = 0
        self.total_latency = 0.0

        threading.Thread(target=self._heartbeat_monitor, daemon=True).start()

        log.info(f"Scheduler started | strategy={strategy.value}")

    # =========================
    # FIXED WORKER SELECTION
    # =========================
    def select_worker(self):
        alive = [
            ws for ws in self.worker_table.values()
            if ws.is_free and ws.is_alive
        ]

        if not alive:
            log.warning("No available workers — all are busy or failed.")
            return None

        if self.strategy == Strategy.LEAST_CONNECTIONS:
            return min(alive, key=lambda w: w.active_tasks)

        if self.strategy == Strategy.LOAD_AWARE:
            return min(alive, key=lambda w: w.load_percent)

        # ROUND ROBIN fallback
        with self._rr_lock:
            self._rr_index += 1
            return alive[self._rr_index % len(alive)]

    # =========================
    # FIXED HEARTBEAT LOGIC
    # =========================
    def _heartbeat_monitor(self):
        log.info("Heartbeat monitor started")

        while True:
            time.sleep(1)
            now = time.time()

            for ws in self.worker_table.values():
                silence = now - ws.last_heartbeat

                # ONLY fail if idle AND silent
                if ws.is_alive and silence > self.heartbeat_timeout and ws.active_tasks == 0:
                    ws.is_alive = False
                    log.warning(f"Worker {ws.worker_id} FAILED (idle timeout)")

                elif not ws.is_alive and silence < self.heartbeat_timeout:
                    ws.is_alive = True
                    log.info(f"Worker {ws.worker_id} recovered")

    # =========================
    # TASK ASSIGNMENT
    # =========================
    def assign_task(self, request):
        MAX_RETRIES = 3
        start = time.time()

        with self._metrics_lock:
            self.total_requests += 1

        for attempt in range(MAX_RETRIES):

            ws = self.select_worker()

            if ws is None:
                log.error(f"Request {request.id}: no worker available")
                continue

            with ws.lock:
                ws.active_tasks += 1
                ws.last_heartbeat = time.time()   # FIX

            try:
                response = self.lb.dispatch_to_worker(ws.worker_id, request)

                with ws.lock:
                    ws.active_tasks = max(0, ws.active_tasks - 1)
                    ws.last_heartbeat = time.time()

                latency = time.time() - start

                if isinstance(response, dict):
                    response["worker_id"] = ws.worker_id
                    response["latency"] = latency

                return response

            except Exception as e:
                log.error(f"Worker {ws.worker_id} failed: {e}")

                with ws.lock:
                    ws.active_tasks = max(0, ws.active_tasks - 1)
                    ws.is_alive = False

                with self._metrics_lock:
                    self.failed_assigns += 1

                # fallback retry continues

        return {
            "id": request.id,
            "result": "FAILED: all retries exhausted",
            "latency": -1
        }

    def handle_request(self, request):
        return self.assign_task(request)

    def update_heartbeat(self, worker_id):
        if worker_id in self.worker_table:
            self.worker_table[worker_id].last_heartbeat = time.time()
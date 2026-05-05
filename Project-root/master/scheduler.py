import time
import threading
import logging
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Master] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("Master")


class Strategy(Enum):
    ROUND_ROBIN       = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE        = "load_aware"


class WorkerStatus:
    def __init__(self, worker_id, total_capacity=10):
        self.worker_id      = worker_id
        self.total_capacity = total_capacity
        self.active_tasks   = 0
        self.is_alive       = True
        self.last_heartbeat = time.time()
        self.lock           = threading.Lock()

    @property
    def is_free(self):
        return self.is_alive and self.active_tasks < self.total_capacity

    @property
    def load_percent(self):
        if self.total_capacity == 0:
            return 1.0
        return self.active_tasks / self.total_capacity

    def __repr__(self):
        status = "ALIVE" if self.is_alive else "FAILED"
        return (f"Worker-{self.worker_id} [{status}] "
                f"tasks={self.active_tasks}/{self.total_capacity} "
                f"load={self.load_percent:.0%}")


class Scheduler:
    def __init__(self, load_balancer, num_workers=4,
                 strategy=Strategy.LEAST_CONNECTIONS,
                 heartbeat_timeout=5.0):

        self.lb                = load_balancer
        self.strategy          = strategy
        self.heartbeat_timeout = heartbeat_timeout

        self.worker_table = {
            i: WorkerStatus(worker_id=i) for i in range(num_workers)
        }

        self._rr_index = 0
        self._rr_lock  = threading.Lock()

        self.total_requests = 0
        self.failed_assigns = 0
        self.total_latency  = 0.0
        self._metrics_lock  = threading.Lock()

        threading.Thread(
            target=self._heartbeat_monitor, daemon=True
        ).start()

        log.info(f"Scheduler started | strategy={strategy.value} | workers={num_workers}")

    def select_worker(self):
        alive = [ws for ws in self.worker_table.values() if ws.is_free]

        if not alive:
            log.warning("No available workers — all are busy or failed.")
            return None

        if self.strategy == Strategy.ROUND_ROBIN:
            return self._round_robin(alive)
        elif self.strategy == Strategy.LEAST_CONNECTIONS:
            return self._least_connections(alive)
        elif self.strategy == Strategy.LOAD_AWARE:
            return self._load_aware(alive)

        return alive[0]

    def _round_robin(self, alive):
        with self._rr_lock:
            alive_ids = [ws.worker_id for ws in alive]
            for _ in range(len(self.worker_table)):
                wid = self._rr_index % len(self.worker_table)
                self._rr_index += 1
                if wid in alive_ids:
                    return self.worker_table[wid]
        return alive[0]

    def _least_connections(self, alive):
        return min(alive, key=lambda ws: ws.active_tasks)

    def _load_aware(self, alive):
        return min(alive, key=lambda ws: ws.load_percent)

    def assign_task(self, request):
        MAX_RETRIES = 3
        start_time  = time.time()

        with self._metrics_lock:
            self.total_requests += 1

        for attempt in range(1, MAX_RETRIES + 1):
            ws = self.select_worker()

            if ws is None:
                log.error(f"Request {request.id}: no worker available "
                          f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(0.1)
                continue

            with ws.lock:
                ws.active_tasks += 1

            log.info(f"Assigned request {request.id} → Worker {ws.worker_id} "
                     f"[strategy={self.strategy.value}, attempt={attempt}]")

            try:
                response = self.lb.dispatch_to_worker(ws.worker_id, request)

                with ws.lock:
                    ws.active_tasks   = max(0, ws.active_tasks - 1)
                    ws.last_heartbeat = time.time()

                latency = time.time() - start_time
                with self._metrics_lock:
                    self.total_latency += latency

                if isinstance(response, dict):
                    response["worker_id"] = ws.worker_id
                    response["latency"]   = latency

                return response

            except Exception as e:
                log.error(f"Worker {ws.worker_id} FAILED on request {request.id}: {e}")
                with ws.lock:
                    ws.is_alive     = False
                    ws.active_tasks = max(0, ws.active_tasks - 1)
                with self._metrics_lock:
                    self.failed_assigns += 1
                log.warning(f"Reassigning request {request.id} "
                            f"(attempt {attempt}/{MAX_RETRIES})")

        log.error(f"Request {request.id} DROPPED after {MAX_RETRIES} failed attempts.")
        return {"id": request.id, "result": "ERROR: all workers failed", "latency": -1}

    def handle_request(self, request):
        return self.assign_task(request)

    def _heartbeat_monitor(self):
        log.info("Heartbeat monitor started (checking every 1s)")
        while True:
            time.sleep(1)
            now = time.time()
            for ws in self.worker_table.values():
                silence = now - ws.last_heartbeat
                if ws.is_alive and silence > self.heartbeat_timeout:
                    ws.is_alive = False
                    log.warning(f"Worker {ws.worker_id} declared FAILED "
                                f"(silent for {silence:.1f}s)")
                elif not ws.is_alive and silence < self.heartbeat_timeout:
                    ws.is_alive = True
                    log.info(f"Worker {ws.worker_id} recovered → ALIVE")

    def update_heartbeat(self, worker_id):
        if worker_id in self.worker_table:
            self.worker_table[worker_id].last_heartbeat = time.time()

    def mark_worker_failed(self, worker_id):
        if worker_id in self.worker_table:
            self.worker_table[worker_id].is_alive = False
            log.warning(f"Worker {worker_id} manually marked FAILED")

    def mark_worker_alive(self, worker_id):
        if worker_id in self.worker_table:
            ws = self.worker_table[worker_id]
            ws.is_alive, ws.active_tasks = True, 0
            ws.last_heartbeat = time.time()
            log.info(f"Worker {worker_id} manually marked ALIVE")

    def set_strategy(self, strategy):
        log.info(f"Strategy: {self.strategy.value} → {strategy.value}")
        self.strategy = strategy

    def get_metrics(self):
        with self._metrics_lock:
            avg = (self.total_latency / self.total_requests
                   if self.total_requests > 0 else 0.0)
            return {
                "total_requests": self.total_requests,
                "failed_assigns": self.failed_assigns,
                "avg_latency_s" : round(avg, 4),
                "strategy"      : self.strategy.value,
                "worker_states" : [str(ws) for ws in self.worker_table.values()],
            }

    def print_status(self):
        print("\n" + "="*55)
        print(f"  MASTER NODE STATUS  |  Strategy: {self.strategy.value}")
        print("="*55)
        for ws in self.worker_table.values():
            bar   = "█" * ws.active_tasks + "░" * (ws.total_capacity - ws.active_tasks)
            state = "✓ ALIVE " if ws.is_alive else "✗ FAILED"
            print(f"  Worker {ws.worker_id} [{state}]  [{bar}] "
                  f"{ws.active_tasks}/{ws.total_capacity}")
        m = self.get_metrics()
        print("-"*55)
        print(f"  Requests: {m['total_requests']}  |  "
              f"Failures: {m['failed_assigns']}  |  "
              f"Avg latency: {m['avg_latency_s']}s")
        print("="*55 + "\n")
import time
import logging

log = logging.getLogger("GPUWorker")


class GPUWorker:
    def __init__(self, worker_id, capacity=70):
        self.worker_id = worker_id
        self.capacity = capacity
        self.scheduler = None
        self.active = True

        print(f"[GPU-WORKER-{worker_id}] initialized (capacity={capacity})")

    def process(self, request):

        # 🔥 CRITICAL FIX: heartbeat BEFORE work
        if self.scheduler:
            self.scheduler.update_heartbeat(self.worker_id)

        print(f"[GPU-WORKER-{self.worker_id}] started request {request.id}")

        # simulate workload
        time.sleep(2)

        if self.scheduler:
            self.scheduler.update_heartbeat(self.worker_id)

        return {
            "id": request.id,
            "result": f"processed by worker {self.worker_id}",
            "status": "success"
        }
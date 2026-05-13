import threading
import time
import random
from llm.inference_engine import infer


class GPUWorker:

    def __init__(self, worker_id: int):
        self.worker_id = worker_id

        self.active_tasks = 0
        self.total_processed = 0
        self.failed = False

        self.gpu_utilization = 0
        self.gpu_memory_used = 0

        self.lock = threading.Lock()

    # PROCESS REQUEST

    def process(self, request):

        if self.failed:
            raise Exception(f"GPU Worker {self.worker_id} is unavailable")

        # SAFE request parsing (supports dict OR object)
        request_id = getattr(request, "id", None) or (request.get("id") if isinstance(request, dict) else None)
        query = getattr(request, "query", None) or (request.get("query") if isinstance(request, dict) else None)
        context = getattr(request, "context", None) or (request.get("context") if isinstance(request, dict) else "")

        if query is None:
            return {
                "request_id": request_id,
                "worker_id": self.worker_id,
                "error": "Invalid request: missing query",
                "status": "error"
            }

        with self.lock:
            self.active_tasks += 1
            self.gpu_utilization = min(
                100,
                self.gpu_utilization + random.randint(5, 15)
            )
            self.gpu_memory_used = min(
                24,
                self.gpu_memory_used + random.uniform(0.5, 2.0)
            )

        try:

            print(
                f"[GPU-{self.worker_id}] Processing Request {request_id} | "
                f"GPU Utilization: {self.gpu_utilization}%"
            )

            start = time.time()

            # REAL OLLAMA INFERENCE (no fake sleep)
            response = infer(
                prompt=query,
                context=context
            )

            latency = time.time() - start

            # SAFE response handling
            if isinstance(response, dict):
                output_text = response.get("response", "")
                status = response.get("status", "success")
            else:
                output_text = str(response)
                status = "success"

            result = {
                "request_id": request_id,
                "worker_id": self.worker_id,
                "response": output_text,
                "latency": latency,
                "gpu_utilization": self.gpu_utilization,
                "gpu_memory_used": round(self.gpu_memory_used, 2),
                "status": status
            }

            with self.lock:
                self.total_processed += 1

            return result

        except Exception as e:

            return {
                "request_id": request_id,
                "worker_id": self.worker_id,
                "error": str(e),
                "status": "error"
            }

        finally:

            with self.lock:
                self.active_tasks = max(0, self.active_tasks - 1)

                self.gpu_utilization = max(
                    0,
                    self.gpu_utilization - random.randint(3, 10)
                )

                self.gpu_memory_used = max(
                    0,
                    self.gpu_memory_used - random.uniform(0.3, 1.5)
                )

    # LOAD INFO

    def get_load(self):

        with self.lock:
            return {
                "worker_id": self.worker_id,
                "active_tasks": self.active_tasks,
                "gpu_utilization": self.gpu_utilization,
                "gpu_memory_used": round(self.gpu_memory_used, 2),
                "processed": self.total_processed,
                "failed": self.failed
            }

    # FAILURE SIMULATION

    def fail_worker(self):
        self.failed = True
        print(f"[GPU-{self.worker_id}] FAILED")

    def recover_worker(self):
        self.failed = False
        print(f"[GPU-{self.worker_id}] RECOVERED")

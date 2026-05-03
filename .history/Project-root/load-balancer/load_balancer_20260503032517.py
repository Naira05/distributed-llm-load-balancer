import threading
from collections import defaultdict

class LoadBalancer:
    def __init__(self, workers, strategy="round_robin"):
        self.workers = workers
        self.strategy = strategy
        self.lock = threading.Lock()
        self.index = 0
        self.connections = defaultdict(int)

    def forward_request(self, request):
        worker = self.get_worker()
        response = worker.process(request)
        self.release_worker(worker)
        return response

    def get_worker(self, request=None):
        with self.lock:
            if self.strategy == "round_robin":
                return self._round_robin()
            elif self.strategy == "least_connections":
                return self._least_connections()
            elif self.strategy == "load_aware":
                return self._load_aware()
            else:
                raise ValueError("Unknown strategy")

    def _round_robin(self):
        worker = self.workers[self.index]
        self.index = (self.index + 1) % len(self.workers)
        self.connections[worker] += 1
        return worker

    def _least_connections(self):
        worker = min(self.workers, key=lambda w: self.connections[w])
        self.connections[worker] += 1
        return worker

    def _load_aware(self):
        worker = min(self.workers, key=lambda w: w.load)
        self.connections[worker] += 1
        return worker

    def release_worker(self, worker):
        with self.lock:
            if self.connections[worker] > 0:
                self.connections[worker] -= 1
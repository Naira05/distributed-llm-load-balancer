
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test_sync
import logging

logging.getLogger("Master").setLevel(logging.WARNING)

from llm.inference_engine import set_backend
set_backend("simulated")  # or "claude" or "ollama"

def main():
    workers = [GPUWorker(i, capacity=70) for i in range(4)]
    
    lb = LoadBalancer(workers, master=None)
    scheduler = Scheduler(lb)
    lb.master = scheduler
    lb.dispatch_to_worker = lambda worker_id, request: workers[worker_id].process(request)

    print("Running custom heavy load tests...\n")

    results_25 = run_load_test_sync(scheduler, num_users=100, concurrency_limit=50)
    results_100 = run_load_test_sync(scheduler, num_users=300, concurrency_limit=100)
    results_200 = run_load_test_sync(scheduler, num_users=1000, concurrency_limit=200) 

    print("\nSummary:")
    print(f"{'Users':<8} {'Throughput':<12} {'Avg Latency':<12} {'P95 Latency':<12}")
    print("-" * 44)

    for r in [results_25, results_100, results_200]:
        print(f"{r.total_requests:<8} {r.throughput:<12.2f} {r.avg_latency*1000:<12.2f} {r.p95_latency*1000:<12.2f}")
        
if __name__ == "__main__":
    main()
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from workers.gpu_worker import GPUWorker
from client.load_generator import run_load_test_sync
import logging
import requests


import logging
logging.getLogger("Master").setLevel(logging.WARNING)


def main():
    workers = [GPUWorker(i) for i in range(4)]

    worker_urls = [
        "http://localhost:8001",
        "http://localhost:8002",
        "http://localhost:8003",
        "http://localhost:8004",
    ]

    scheduler = Scheduler(None)

    lb = LoadBalancer(worker_urls=worker_urls, master=scheduler)

    scheduler.lb = lb

    lb.dispatch_to_worker = lambda worker_id, request: requests.post(
        worker_urls[worker_id],
        json=request
    ).json()

    print("Running small test (10 users)...\n")

    # SMALL TEST FIRST
    results_10 = run_load_test_sync(
        scheduler,
        num_users=1000,
        concurrency_limit=50
    )

    print("\nSummary:")
    print(f"{'Users':<8} {'Throughput':<12} {'Avg Latency':<12} {'P95 Latency':<12}")
    print("-" * 44)

    print(f"{results_10.total_requests:<8} "
          f"{results_10.throughput:<12.2f} "
          f"{results_10.avg_latency*1000:<12.2f} "
          f"{results_10.p95_latency*1000:<12.2f}")

if __name__ == "__main__":
    main()
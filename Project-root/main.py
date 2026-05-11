# main_master.py
# Run this on the master machine to distribute requests across GPU workers

from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test_sync
import logging

logging.basicConfig(level=logging.INFO)


def main():

    workers = [
        {
            "id": "Mai",
            "url": "https://powwow-platypus-vice.ngrok-free.dev/",
            "capacity": 2,
        },
        {
            "id": "Naira",
            "url": "https://decidable-chubby-muppet.ngrok-free.dev/",
            "capacity": 2,
        },
        {
            "id": "Maryam",
            "url": "https://affront-squint-embolism.ngrok-free.dev/",
            "capacity": 2,
        },
         {
            "id": "Nourhan",
            "url": "https://gullible-anybody-gluten.ngrok-free.dev/",
            "capacity": 2,
        },
  
        
    ]

    workers_url = [w["url"] for w in workers]

    lb = LoadBalancer(worker_urls=workers_url)
    scheduler = Scheduler(lb)

    # GPU snapshot BEFORE
    scheduler.print_gpu_status("GPU Workers — Before Load Test")

    print("\n[Master] Running load test ..\n")
    results = run_load_test_sync(scheduler, num_users=10, concurrency_limit=5)

    # GPU snapshot AFTER
    scheduler.print_gpu_status("GPU Workers — After Load Test")

    # Per-worker breakdown
    scheduler.print_worker_summary()

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Total Requests':<20} {results.total_requests}")
    print(f"{'Throughput':<20} {results.throughput:.2f} req/s")
    print(f"{'Avg Latency':<20} {results.avg_latency * 1000:.2f} ms")
    print(f"{'P95 Latency':<20} {results.p95_latency * 1000:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
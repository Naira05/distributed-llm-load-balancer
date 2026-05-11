from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test_sync
import logging

logging.getLogger("Master").setLevel(logging.WARNING)


def main():

    workers = [
        {
            "id": "Mai",
            "url": "https://powwow-platypus-vice.ngrok-free.dev/",
            "capacity": 4,
        },
        {
            "id": "Naira",
            "url": "https://decidable-chubby-muppet.ngrok-free.dev/",
            "capacity": 4,
        },
        {
            "id" : "Maryam",
            "url": "https://affront-squint-embolism.ngrok-free.dev",
            "Capacity": 2,
        },
        """
         {
            "id" : "Mariam",
            "url": "",
            "Capacity": 4,
        },   
        {
            "id" : "Nourhan",
            "url": "",
            "Capacity": 3,
        },
        """
    ]

    # ✅ extract URLs only
    worker_urls = [w["url"] for w in workers]

    lb = LoadBalancer(worker_urls=worker_urls)
    scheduler = Scheduler(lb)

    print("\n[Main] Worker status before test:")
    for url, info in lb.status().items():
        print(f"  {url} → {'✔ alive' if info['alive'] else '✖ DOWN'}")

    print("\n[Main] Running load test (1000 users, concurrency=50)...\n")

    results = run_load_test_sync(
        scheduler,
        num_users=1000,
        concurrency_limit=50,
    )

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

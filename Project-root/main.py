from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test_sync
import logging

logging.getLogger("Master").setLevel(logging.WARNING)


def main():
    # ── point at whichever worker servers are actually running ────────────────
    # Add or remove URLs to match how many terminals/machines you started
    worker_urls = [
        "http://localhost:8001",
        "http://localhost:8002",
        # "http://localhost:8003",   # uncomment when you start a 3rd worker
        # "http://localhost:8004",   # uncomment when you start a 4th worker
    ]

    # ── build LB (starts background health-checker automatically) ─────────────
    lb = LoadBalancer(worker_urls=worker_urls)

    # ── scheduler dispatches through the LB, not directly to worker objects ───
    scheduler = Scheduler(lb)

    # ── quick sanity check: ping all workers before starting load test ─────────
    print("\n[Main] Worker status before test:")
    for url, info in lb.status().items():
        print(f"  {url}  →  {'✔ alive' if info['alive'] else '✖ DOWN'}")

    print(f"\n[Main] Running load test (1000 users, concurrency=50)...\n")

    results = run_load_test_sync(
        scheduler,
        num_users=1000,
        concurrency_limit=50,
    )

    # ── print summary ─────────────────────────────────────────────────────────
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
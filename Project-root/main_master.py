# main_master.py
# Run this on the master machine to distribute requests across GPU workers

from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test_sync
import logging
import threading
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask

logging.basicConfig(level=logging.INFO)


# HEARTBEAT SERVER

heartbeat_app = Flask(__name__)

@heartbeat_app.route("/health", methods=["GET"])
def health():
    return "alive", 200

def run_heartbeat():
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    heartbeat_app.run(host="127.0.0.1", port=8000)


# METRICS PLOT

def plot_metrics(scheduler, workers):

    urls = [w["url"] for w in workers]

    throughput = []
    latency = []
    gpu_util = []

    for url in urls:
        stats = scheduler._worker_stats.get(url, {})

        reqs = stats.get("requests", 0)
        total_latency = stats.get("total_latency", 0.0)

        avg_latency = (total_latency / reqs * 1000) if reqs else 0

        throughput.append(reqs)
        latency.append(avg_latency)

        gpu = scheduler.fetch_worker_gpu(url)
        gpu_util.append(gpu["gpu_utilization"] if gpu else 0)

    x = np.arange(len(urls))

    plt.figure()
    plt.bar(x, throughput)
    plt.xticks(x, [w["id"] for w in workers])
    plt.title("Throughput per Worker")
    plt.ylabel("Requests handled")

    plt.figure()
    plt.bar(x, latency)
    plt.xticks(x, [w["id"] for w in workers])
    plt.title("Average Latency per Worker (ms)")
    plt.ylabel("Latency (ms)")

    plt.figure()
    plt.bar(x, gpu_util)
    plt.xticks(x, [w["id"] for w in workers])
    plt.title("GPU Utilization per Worker (%)")
    plt.ylabel("GPU %")

    plt.show()



# MAIN
def main():

    workers = [
        {"id": "Mai",     "url": "https://powwow-platypus-vice.ngrok-free.dev/",       "capacity": 2},
        {"id": "Naira",   "url": "https://decidable-chubby-muppet.ngrok-free.dev/",    "capacity": 2},
        {"id": "Maryam",  "url": "https://affront-squint-embolism.ngrok-free.dev/",    "capacity": 2},
        {"id": "Nourhan", "url": "https://gullible-anybody-gluten.ngrok-free.dev/",    "capacity": 2},
    ]

    workers_url = [w["url"] for w in workers]

    lb = LoadBalancer(worker_urls=workers_url)
    scheduler = Scheduler(lb)

    scheduler.print_gpu_status("GPU Workers — Before Load Test")

    print("\n[Master] Running load test ..\n")
    results = run_load_test_sync(scheduler, num_users=100, concurrency_limit=5)

    scheduler.print_gpu_status("GPU Workers — After Load Test")
    scheduler.print_worker_summary()

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Total Requests':<20} {results.total_requests}")
    print(f"{'Throughput':<20} {results.throughput:.2f} req/s")
    print(f"{'Avg Latency':<20} {results.avg_latency * 1000:.2f} ms")
    print(f"{'P95 Latency':<20} {results.p95_latency * 1000:.2f} ms")
    print("=" * 50)

    plot_metrics(scheduler, workers)


if __name__ == "__main__":
    heartbeat_thread = threading.Thread(target=run_heartbeat, daemon=True)
    heartbeat_thread.start()
    print("[Master] Heartbeat server running on localhost:8000")

    main()
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test_sync
from llm.inference_engine import set_backend, infer

import logging
logging.getLogger("Master").setLevel(logging.WARNING)

# choose backend
set_backend("ollama")   # or "simulated" if Ollama is slow

def main(): 
    print("Test LLM response:")
    print(infer("What is load balancing?")["response"])
    print("-" * 50)

    workers = [GPUWorker(i, capacity=70) for i in range(4)]
    
    lb = LoadBalancer(workers, master=None)
    scheduler = Scheduler(lb)
    lb.master = scheduler
    lb.dispatch_to_worker = lambda worker_id, request: workers[worker_id].process(request)

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
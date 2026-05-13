import sys
import os
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lb.load_balancer import LoadBalancer


WORKERS = [
    "https://powwow-platypus-vice.ngrok-free.dev/",
    "https://decidable-chubby-muppet.ngrok-free.dev/",
    "https://affront-squint-embolism.ngrok-free.dev/",
    "https://gullible-anybody-gluten.ngrok-free.dev/",
]


def send_request(lb, i, query):

    payload = {
        "id": i,
        "query": query
    }

    result = lb.dispatch(payload)

    print("\n────────────────────────────")
    print(f"Request {i}")
    print(f"Query   : {query}")
    print(f"Worker  : {result.get('routed_to')}")
    print(f"Status  : {result.get('status')}")
    print("────────────────────────────")

    return result


def main():

    lb = LoadBalancer(worker_urls=WORKERS)

    queries = [
        "hello",
        "what is AI?",
        "write a short poem",
        "explain distributed systems",
        "what is GPU inference?",
        "how does load balancing work?"
    ]

    print("\n[TEST] Starting parallel request simulation...\n")

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

        futures = [
            executor.submit(send_request, lb, i, q)
            for i, q in enumerate(queries, start=1)
        ]

        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    print("\n[TEST] Completed all requests.")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    main()
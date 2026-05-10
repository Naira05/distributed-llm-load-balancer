import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.worker import process_parallel_requests

if __name__ == "__main__":
    queries = [
        "hello",
        "what is AI?",
        "write a short poem"
    ]

    results = process_parallel_requests(queries, num_workers=2)

    for r in results:
        print(r)
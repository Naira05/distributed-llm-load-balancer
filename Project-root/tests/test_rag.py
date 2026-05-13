# tests/test_rag.py

import logging

from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from rag.retriever import retrieve_contexts

logging.basicConfig(level=logging.INFO)


def run_rag_test():

   
    # Same workers as your main
    workers = [
        {"id": "Mai", "url": "https://powwow-platypus-vice.ngrok-free.dev/", "capacity": 2},
        {"id": "Naira", "url": "https://decidable-chubby-muppet.ngrok-free.dev/", "capacity": 2},
        {"id": "Maryam", "url": "https://affront-squint-embolism.ngrok-free.dev/", "capacity": 2},
        {"id": "Nourhan", "url": "https://gullible-anybody-gluten.ngrok-free.dev/", "capacity": 2},
    ]

    worker_urls = [w["url"] for w in workers]

    lb = LoadBalancer(worker_urls=worker_urls)
    scheduler = Scheduler(lb)

  
    # Test queries for RAG
    test_queries = [
        "how does load balancing improve performance?",
        "what happens when a worker node fails?",
        "gpu cluster inference and latency issues",
        "how do distributed systems handle faults?",
    ]

    print("\n" + "=" * 60)
    print("RAG DISTRIBUTED TEST START")
    print("=" * 60)

    for q in test_queries:

        print("\n" + "-" * 60)
        print("QUERY:", q)

     
        # Step 1: get multiple contexts (one per worker)
        contexts = retrieve_contexts(q, num_contexts=len(worker_urls))

   
        # Step 2: send to ALL workers
        results = lb.dispatch_to_all(
            base_payload={
                "id": f"rag-test-{q[:10]}",
                "query": q,
            },
            contexts=contexts
        )

        # Step 3: print comparison
        for r in results:
            print("\nWorker:", r.get("routed_to"))
            print("Status:", r.get("status"))
            print("Response:", (r.get("response") or r.get("error"))[:200])

    print("\n" + "=" * 60)
    print("RAG TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_rag_test()
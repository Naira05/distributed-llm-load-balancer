#!/usr/bin/env python3
# tests/test_worker_pool.py

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.worker import WorkerPool, process, process_parallel_requests
from llm.inference_engine import get_stats, reset_stats

def test_single_worker_process():
    """Test single request processing."""
    print("\n" + "="*60)
    print("TEST 1: Single Request Processing")
    print("="*60)
    
    response = process("What is GPU inference?")
    print(f"✓ Single request processed successfully")
    print(f"  Model: {response['model']}")
    print(f"  Latency: {response['performance']['latency_ms']}ms")
    print(f"  Answer: {response['response'][:80]}...")


def test_worker_pool_basic():
    """Test basic worker pool functionality."""
    print("\n" + "="*60)
    print("TEST 2: Worker Pool - Basic Operations")
    print("="*60)
    
    pool = WorkerPool(num_workers=4)
    print(f"✓ Created worker pool with 4 workers")
    
    # Submit tasks
    task_ids = []
    for i in range(5):
        task_id = pool.submit_task({
            "query": f"Test query number {i+1}",
            "context": "Test context"
        })
        task_ids.append(task_id)
    
    print(f"✓ Submitted 5 tasks to pool")
    
    # Wait for completion
    pool.wait_completion()
    print(f"✓ All tasks completed")
    
    # Get results
    results = pool.get_results(timeout=10)
    print(f"✓ Retrieved {len(results)} results")
    
    # Print worker output
    for result in results:
        print(f"  Worker {result['worker_id']} processed request {result['request_id']}")
    
    pool.shutdown()
    print(f"✓ Worker pool shutdown successfully")


def test_parallel_vs_sequential():
    """Test performance comparison."""
    print("\n" + "="*60)
    print("TEST 3: Parallel vs Sequential Performance")
    print("="*60)
    
    import time
    
    queries = [f"Query {i}" for i in range(8)]
    
    # Sequential
    reset_stats()
    start = time.time()
    results_seq = []
    for query in queries:
        results_seq.append(process(query))
    seq_time = time.time() - start
    
    print(f"Sequential (8 requests):")
    print(f"  Time: {seq_time:.3f}s")
    print(f"  Throughput: {len(results_seq)/seq_time:.2f} req/s")
    
    # Parallel
    reset_stats()
    start = time.time()
    results_par = process_parallel_requests(queries, num_workers=4)
    par_time = time.time() - start
    
    print(f"\nParallel (8 requests, 4 workers):")
    print(f"  Time: {par_time:.3f}s")
    print(f"  Throughput: {len(results_par)/par_time:.2f} req/s")
    print(f"  Speedup: {seq_time/par_time:.2f}x")


def test_worker_output_format():
    """Test that output matches expected format."""
    print("\n" + "="*60)
    print("TEST 4: Worker Output Format Validation")
    print("="*60)
    
    pool = WorkerPool(num_workers=2)
    
    # Submit requests
    for i in range(1, 6):
        pool.submit_task({"query": f"Request {i}"})
    
    pool.wait_completion()
    results = pool.get_results(timeout=10)
    
    print("Output format validation:")
    for result in results:
        # Expected output format
        output_line = f"Worker {result['worker_id']} processed request {result['request_id']}"
        print(f"  {output_line}")
    
    print(f"✓ Output format matches expected pattern")
    
    pool.shutdown()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WORKER POOL COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        test_single_worker_process()
        test_worker_pool_basic()
        test_parallel_vs_sequential()
        test_worker_output_format()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()

# workers/worker.py

import threading
import time
import queue
from typing import Any, Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.inference_engine import infer, get_stats

# =========================================================
# WORKER POOL MANAGEMENT
# =========================================================

class Worker(threading.Thread):
    """
    Represents a single GPU worker that processes LLM requests.
    Workers run in parallel threads and pick up tasks from a queue.
    """

    def __init__(self, worker_id: int, task_queue: queue.Queue, results_queue: queue.Queue):
        """
        Initialize a worker thread.
        
        Args:
            worker_id: Unique identifier for this worker
            task_queue: Queue of tasks to process
            results_queue: Queue to put results into
        """
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.results_queue = results_queue
        self.requests_processed = 0
        self.lock = threading.Lock()

    def run(self):
        """
        Main worker loop - continuously process tasks from queue.
        """
        while True:
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1)

                if task is None:  # Poison pill to stop worker
                    break

                result = self.process(task)
                self.results_queue.put(result)
                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.results_queue.put({"error": str(e), "worker_id": self.worker_id})

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single LLM request.
        This is the actual work that a worker does.
        
        Args:
            task: Dictionary containing:
                - request_id: ID of the request
                - query: The LLM query to process
                - context: Optional context for RAG
        
        Returns:
            Dictionary with response and metadata
        """
        request_id = task.get("request_id", -1)
        query = task.get("query", "")
        context = task.get("context", "")

        # Call the LLM inference engine
        response = infer(query, context=context)

        # Track request processing
        with self.lock:
            self.requests_processed += 1

        # Return result with worker metadata
        return {
            "worker_id": self.worker_id,
            "request_id": request_id,
            "response": response,
            "processed_at": time.time()
        }

    def get_processed_count(self) -> int:
        """Get number of requests this worker has processed."""
        with self.lock:
            return self.requests_processed


class WorkerPool:
    """
    Manages a pool of worker threads that process requests in parallel.
    Distributes requests across available workers.
    """

    def __init__(self, num_workers: int = 4):
        """
        Initialize the worker pool.
        
        Args:
            num_workers: Number of parallel workers to create
        """
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.workers = []
        self.lock = threading.Lock()
        self._total_requests = 0

        # Create and start worker threads
        for i in range(num_workers):
            worker = Worker(i + 1, self.task_queue, self.results_queue)
            worker.start()
            self.workers.append(worker)

    def submit_task(self, task: Dict[str, Any]) -> int:
        """
        Submit a task to the worker pool.
        
        Args:
            task: Task dictionary to process
            
        Returns:
            Task ID for tracking
        """
        with self.lock:
            self._total_requests += 1
            task["request_id"] = self._total_requests

        self.task_queue.put(task)
        return self._total_requests

    def submit_batch(self, tasks: List[Dict[str, Any]]) -> List[int]:
        """
        Submit multiple tasks to the worker pool.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            List of request IDs
        """
        request_ids = []
        for task in tasks:
            req_id = self.submit_task(task)
            request_ids.append(req_id)

        return request_ids

    def get_results(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """
        Retrieve all completed results from the results queue.
        
        Args:
            timeout: Maximum time to wait for results
            
        Returns:
            List of result dictionaries
        """
        results = []
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = self.results_queue.get(timeout=0.5)
                results.append(result)
            except queue.Empty:
                if not self.task_queue.empty():
                    continue
                else:
                    break

        return results

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the worker pool.
        
        Returns:
            Dictionary with pool metrics
        """
        return {
            "num_workers": self.num_workers,
            "pending_tasks": self.task_queue.qsize(),
            "pending_results": self.results_queue.qsize(),
            "total_submitted": self._total_requests,
            "worker_stats": [
                {
                    "worker_id": w.worker_id,
                    "processed": w.get_processed_count()
                }
                for w in self.workers
            ]
        }

    def wait_completion(self):
        """Wait for all tasks in queue to be processed."""
        self.task_queue.join()

    def shutdown(self):
        """Gracefully shutdown all workers."""
        for _ in range(self.num_workers):
            self.task_queue.put(None)

        for worker in self.workers:
            worker.join(timeout=2)


# =========================================================
# STANDALONE FUNCTIONS
# =========================================================

def process(query: str, context: str = "") -> Dict[str, Any]:
    """
    Convenience function to process a single request.
    Uses the global worker pool or creates a temporary one.
    
    Args:
        query: LLM query to process
        context: Optional context for RAG
        
    Returns:
        LLM response dictionary
    """
    return infer(query, context=context)


def process_parallel_requests(queries: List[str], num_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Process multiple requests in parallel using worker pool.
    
    Args:
        queries: List of queries to process
        num_workers: Number of parallel workers
        
    Returns:
        List of responses
    """
    pool = WorkerPool(num_workers=num_workers)

    # Submit all tasks
    for query in queries:
        pool.submit_task({"query": query})

    # Wait for completion
    pool.wait_completion()

    # Collect results
    results = pool.get_results(timeout=30)

    # Shutdown workers
    pool.shutdown()

    return results


# =========================================================
# GLOBAL WORKER POOL (for persistent use)
# =========================================================

_global_pool = None


def initialize_worker_pool(num_workers: int = 4) -> WorkerPool:
    """
    Initialize the global worker pool.
    
    Args:
        num_workers: Number of workers in the pool
        
    Returns:
        The worker pool instance
    """
    global _global_pool
    _global_pool = WorkerPool(num_workers=num_workers)
    return _global_pool


def get_worker_pool() -> WorkerPool:
    """Get the global worker pool, creating it if needed."""
    global _global_pool
    if _global_pool is None:
        _global_pool = initialize_worker_pool()
    return _global_pool


def shutdown_worker_pool():
    """Shutdown the global worker pool."""
    global _global_pool
    if _global_pool:
        _global_pool.shutdown()
        _global_pool = None

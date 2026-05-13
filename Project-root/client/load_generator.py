import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import Request


@dataclass
class ClientMetrics:
    #Single request metrics recorded client-side.
    request_id: int
    latency: float        # seconds
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestResults:
    #Aggregated results from a load test run.
    total_requests: int
    successful: int
    failed: int
    total_time: float                     # wall-clock duration of test
    throughput: float                     # requests/sec
    latencies: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.successful / self.total_requests) * 100 if self.total_requests > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0
    
    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0
    
    @property
    def p50_latency(self) -> float:
        #50th percentile (median).
        return self._percentile(50)
    
    @property
    def p95_latency(self) -> float:
        #95th percentile.
        return self._percentile(95)
    
    @property
    def p99_latency(self) -> float:
        #99th percentile.
        return self._percentile(99)
    
    def _percentile(self, p: float) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        index = int(len(sorted_lat) * p / 100)
        # Clamp to valid range
        index = min(index, len(sorted_lat) - 1)
        return sorted_lat[index]
    
    def print_report(self):
        print(f"Users: {self.total_requests} | "
          f"Success: {self.success_rate:.1f}% | "
          f"Time: {self.total_time:.2f}s | "
          f"Throughput: {self.throughput:.2f} req/s | "
          f"Avg: {self.avg_latency*1000:.1f}ms | "
          f"P50: {self.p50_latency*1000:.1f}ms | "
          f"P95: {self.p95_latency*1000:.1f}ms | "
          f"P99: {self.p99_latency*1000:.1f}ms")


class LoadGenerator:
    #Async load generator that simulates concurrent users.
    #Uses asyncio for efficient concurrency (not threads).
    def __init__(self, scheduler, concurrency_limit: int = 100):
        #Args:
            #scheduler: The Master Scheduler instance (or LB) to send requests to.
            #concurrency_limit: Max number of concurrent in-flight requests.
        self.scheduler = scheduler
        self.concurrency_limit = concurrency_limit
        self.metrics: List[ClientMetrics] = []
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._lock = asyncio.Lock()
    
    async def _send_single_request(self, user_id: int) -> ClientMetrics:
        #Send one request and measure client-side latency.
        #Returns ClientMetrics with independent timing.
        request = Request(id=user_id, query=f"Query {user_id}")
        
        start_time = time.perf_counter()
        try:
            # Run the synchronous scheduler call in a thread to avoid blocking
            response = await asyncio.to_thread(self.scheduler.handle_request, request)
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            
            metric = ClientMetrics(
                request_id=user_id,
                latency=latency,
                success=True
            )
        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            metric = ClientMetrics(
                request_id=user_id,
                latency=latency,
                success=False,
                error=str(e)
            )
        
        # Thread-safe append to metrics list
        async with self._lock:
            self.metrics.append(metric)
        
        return metric
    
    async def _bounded_request(self, user_id: int) -> ClientMetrics:
        #Request wrapped with semaphore for concurrency control.
        async with self._semaphore:
            return await self._send_single_request(user_id)
    
    async def run_load_test(self, num_users: int = 1000) -> LoadTestResults:
        #Run load test with specified number of concurrent users.
        
        print(f"\n[LoadGenerator] Starting load test with {num_users} users...")
        print(f"[LoadGenerator] Concurrency limit: {self.concurrency_limit}")
        
        self.metrics.clear()
        
        start_time = time.perf_counter()
        
        # Create all tasks at once, asyncio handles scheduling efficiently
        tasks = [self._bounded_request(i) for i in range(num_users)]
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Count successes/failures from our stored metrics
        successful = sum(1 for m in self.metrics if m.success)
        failed = sum(1 for m in self.metrics if not m.success)
        latencies = [m.latency for m in self.metrics]
        
        throughput = num_users / total_time if total_time > 0 else 0
        
        test_results = LoadTestResults(
            total_requests=num_users,
            successful=successful,
            failed=failed,
            total_time=total_time,
            throughput=throughput,
            latencies=latencies
        )
        
        test_results.print_report()
        return test_results


def run_load_test_sync(scheduler, num_users: int = 1000, concurrency_limit: int = 100) -> LoadTestResults:
    generator = LoadGenerator(scheduler, concurrency_limit=concurrency_limit)
    return asyncio.run(generator.run_load_test(num_users))



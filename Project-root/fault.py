# fault.py
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [FaultTolerance] %(message)s")
log = logging.getLogger("Fault")

def handle_failure(scheduler, worker_id: int):
    """
    Simulates a catastrophic, immediate failure of a worker.
    Marks the target node offline within the master scheduler.
    """
    print(f"\n[CRASH DETECTED] Worker {worker_id} crashed unexpectedly! ")
    scheduler.mark_worker_failed(worker_id)


def reassign_task(scheduler, request, failed_worker_id: int):
    """
    Pulls a task that was running on a failed node and re-routes it
    to a healthy, active worker in the cluster.
    """
    log.info(f"Initiating failover recovery for request {request.id}...")
    
    # Select another available, active node
    new_worker_status = scheduler.select_worker()
    if new_worker_status is None:
        log.error(f"Cannot complete recovery for request {request.id}. No alternate workers available.")
        return {"id": request.id, "result": "ERROR: Failover failure", "latency": -1, "status": "error"}
        
    # Matches the exact console output pattern requested on your project checklist
    print(f"Worker {failed_worker_id} failed → reassigned to Worker {new_worker_status.worker_id}") #
    
    with new_worker_status.lock:
        new_worker_status.active_tasks += 1
        
    try:
        # Execute immediately on the healthy target node using the LB dispatch hook
        response = scheduler.lb.dispatch_to_worker(new_worker_status.worker_id, request)
        
        with new_worker_status.lock:
            new_worker_status.active_tasks = max(0, new_worker_status.active_tasks - 1)
            
        return response
    except Exception as e:
        log.error(f"Failover routing to Worker {new_worker_status.worker_id} also failed: {e}")
        return {"id": request.id, "result": f"ERROR: Task reassignment failure ({e})", "status": "error"}
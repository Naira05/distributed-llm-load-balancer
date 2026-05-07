import logging

log = logging.getLogger("Fault")


def handle_failure(scheduler, worker_id: int):
    print(f"\n[CRASH DETECTED] Worker {worker_id}")
    scheduler.mark_worker_failed(worker_id)


def reassign_task(scheduler, request, failed_worker_id: int, max_retries: int = 2):

    log.info(f"Failover request {request.id}")

    for _ in range(max_retries):

        worker = scheduler.select_worker()
        if worker is None:
            return {"id": request.id, "status": "error"}

        print(f"Reassigned {request.id}: Worker {failed_worker_id} → {worker.worker_id}")

        try:
            with worker.lock:
                worker.active_tasks += 1

            result = scheduler.lb.dispatch_to_worker(worker.worker_id, request)

            with worker.lock:
                worker.active_tasks = max(0, worker.active_tasks - 1)

            return result

        except Exception:
            with worker.lock:
                worker.active_tasks = max(0, worker.active_tasks - 1)

    return {"id": request.id, "status": "error"}
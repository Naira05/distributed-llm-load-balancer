# fault/fault.py

import logging
import time
from typing import Optional

log = logging.getLogger("Fault")


def handle_failure(lb, failed_url: str):
    """
    Mark a worker URL as dead in the load balancer.
    Called when a worker returns an error or times out.
    """
    log.warning(f"[Fault] Marking worker as failed: {failed_url}")
    with lb._lock:
        lb._alive[failed_url] = False
    print(f"\n[CRASH DETECTED] Worker at {failed_url} marked as DOWN.")


def reassign_task(
    lb,
    payload: dict,
    failed_url: str,
    strategy: str = "load_aware",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """
    Retry a failed request on a different healthy worker.
    Excludes the failed worker from selection for this request.
    """
    log.info(f"[Fault] Reassigning request {payload.get('id')} away from {failed_url}")

    for attempt in range(1, max_retries + 1):
        # temporarily hide the failed worker
        with lb._lock:
            original_state = lb._alive.get(failed_url, False)
            lb._alive[failed_url] = False

        try:
            # pick a different worker
            strategy_fn = {
                "round_robin":       lb.round_robin,
                "least_connections": lb.least_connections,
                "load_aware":        lb.load_aware,
            }.get(strategy, lb.load_aware)

            new_url = strategy_fn()

            if new_url is None or new_url == failed_url:
                print(f"[Fault] Attempt {attempt}: No other workers available.")
                time.sleep(retry_delay)
                continue

            print(f"[Fault] Attempt {attempt}: Reassigning → {new_url}")

            import requests as req
            resp = req.post(f"{new_url}/process", json=payload, timeout=lb.request_timeout)
            result = resp.json()

            print(f"[Fault] Reassignment succeeded on attempt {attempt} → {new_url}")
            log.info(f"[Fault] Success on attempt {attempt} via {new_url}")
            return result

        except Exception as e:
            log.warning(f"[Fault] Attempt {attempt} failed: {e}")
            time.sleep(retry_delay)

        finally:
            # restore the failed worker's original state
            with lb._lock:
                lb._alive[failed_url] = original_state

    log.error(f"[Fault] All {max_retries} retry attempts exhausted for request {payload.get('id')}")
    return {"id": payload.get("id"), "status": "error", "error": "All retry attempts failed"}


def with_fault_tolerance(
    lb,
    payload: dict,
    strategy: str = "load_aware",
    max_retries: int = 3,
) -> dict:
    """
    Drop-in wrapper: dispatch a request with automatic failover.
    Use this instead of lb.dispatch() for fault-tolerant sending.
    """
    strategy_fn = {
        "round_robin":       lb.round_robin,
        "least_connections": lb.least_connections,
        "load_aware":        lb.load_aware,
    }.get(strategy, lb.load_aware)

    url = strategy_fn()

    if url is None:
        return {"status": "error", "error": "No workers available"}

    try:
        import requests as req
        resp = req.post(f"{url}/process", json=payload, timeout=lb.request_timeout)
        result = resp.json()

        if result.get("status") != "success":
            log.warning(f"[Fault] Worker {url} returned non-success, triggering failover.")
            handle_failure(lb, url)
            return reassign_task(lb, payload, url, strategy, max_retries)

        return result

    except Exception as e:
        log.warning(f"[Fault] Worker {url} threw exception: {e}, triggering failover.")
        handle_failure(lb, url)
        return reassign_task(lb, payload, url, strategy, max_retries)
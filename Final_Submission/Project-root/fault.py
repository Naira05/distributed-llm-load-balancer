# fault/fault.py

import logging
import time

log = logging.getLogger("Fault")


def handle_failure(lb, failed_url: str):
    """
    Explicitly mark a worker URL as dead in the load balancer.
    Only call this for hard failures (connection refused, timeout, etc).
    Do NOT call this for soft HTTP failures (500, error status in JSON) —
    the health-check thread will handle recovery automatically.
    """
    log.warning(f"[Fault] Marking worker as failed: {failed_url}")
    with lb._lock:
        lb._alive[failed_url] = False
    print(f"\n[CRASH DETECTED] Worker at {failed_url} marked as DOWN.")


def with_fault_tolerance(
    lb,
    payload: dict,
    strategy: str = "load_aware",
    max_retries: int = 3,
) -> dict:
    """
    Drop-in fault-tolerant dispatcher.

    Delegates entirely to lb.dispatch() which already handles:
      - worker selection via the chosen strategy
      - per-attempt failover with exclude logic
      - marking workers dead only on hard (connection) failures
      - returning routed_to so the scheduler can attribute stats correctly

    Previously this function duplicated dispatch() logic and called
    handle_failure() on soft errors (non-success JSON responses), which
    permanently blacklisted workers 3 & 4 after their first 500 — causing
    all subsequent requests to pile onto workers 1 & 2 only.
    """
    result = lb.dispatch(payload, strategy=strategy)

    if result.get("status") != "success":
        log.warning(
            f"[Fault] Request {payload.get('id')} failed after {max_retries} attempts "
            f"| last worker: {result.get('routed_to')} "
            f"| error: {result.get('error')}"
        )

    return result


def reassign_task(
    lb,
    payload: dict,
    failed_url: str,
    strategy: str = "load_aware",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """
    Retry a request while explicitly excluding a known-bad worker.
    Use this when you already know a specific worker failed and want
    to route away from it — e.g. after a timeout on a specific URL.

    For general fault-tolerant dispatch, prefer with_fault_tolerance().
    """
    log.info(f"[Fault] Reassigning request {payload.get('id')} away from {failed_url}")

    strategy_fn = {
        "round_robin":       lb.round_robin,
        "least_connections": lb.least_connections,
        "load_aware":        lb.load_aware,
    }.get(strategy, lb.load_aware)

    last_failed = failed_url

    for attempt in range(1, max_retries + 1):
        new_url = strategy_fn(exclude=last_failed)

        if new_url is None:
            print(f"[Fault] Attempt {attempt}: No other workers available.")
            time.sleep(retry_delay)
            continue

        print(f"[Fault] Attempt {attempt}: Reassigning → {new_url}")

        try:
            import requests as req
            resp   = req.post(f"{new_url}/process", json=payload, timeout=lb.request_timeout)
            result = resp.json()
            result["routed_to"] = new_url

            if result.get("status") == "success":
                print(f"[Fault] Reassignment succeeded on attempt {attempt} → {new_url}")
                log.info(f"[Fault] Success on attempt {attempt} via {new_url}")
                return result

            # Soft failure — skip this worker next attempt, do NOT mark it dead
            log.warning(f"[Fault] Attempt {attempt}: {new_url} returned non-success, skipping.")
            last_failed = new_url
            time.sleep(retry_delay)

        except Exception as e:
            # Hard failure — safe to mark dead; health check will revive it
            log.warning(f"[Fault] Attempt {attempt}: {new_url} threw exception ({e}), marking dead.")
            handle_failure(lb, new_url)
            last_failed = new_url
            time.sleep(retry_delay)

    log.error(f"[Fault] All {max_retries} retry attempts exhausted for request {payload.get('id')}")
    return {"id": payload.get("id"), "status": "error", "error": "All retry attempts failed", "routed_to": last_failed}
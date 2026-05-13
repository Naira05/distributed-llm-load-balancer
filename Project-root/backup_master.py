

import time
import threading
import requests
import logging
from flask import Flask, jsonify

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

IS_PRIMARY = False
HAS_TAKEN_OVER = False

PRIMARY_HEALTH_URL = "http://127.0.0.1:8000/health"

WORKERS = [
    {"id": "Mai",     "url": "https://powwow-platypus-vice.ngrok-free.dev/"},
    {"id": "Naira",   "url": "https://decidable-chubby-muppet.ngrok-free.dev/"},
    {"id": "Maryam",  "url": "https://affront-squint-embolism.ngrok-free.dev/"},
    {"id": "Nourhan", "url": "https://gullible-anybody-gluten.ngrok-free.dev/"},
]


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "role": "primary" if IS_PRIMARY else "standby",
        "taken_over": HAS_TAKEN_OVER
    })



def start_master_workload():
    global IS_PRIMARY

    print("\n[BACKUP MASTER] Starting workload as new primary...\n")

    i = 0
    req_id = 0

    while IS_PRIMARY:
        worker = WORKERS[i % len(WORKERS)]
        i += 1
        req_id += 1

        try:
            print(f"[BACKUP MASTER] Sending request {req_id} to {worker['id']} ({worker['url']})")
            res = requests.post(
                worker["url"] + "process",
                json={"id": req_id, "query": "failover request"},
                timeout=10
            )
            print(f"[BACKUP MASTER] Response from {worker['id']}: {res.status_code}")

        except Exception as e:
            print(f"[BACKUP MASTER] Error reaching {worker['id']}: {e}")

        time.sleep(1)


def monitor_primary():
    global IS_PRIMARY, HAS_TAKEN_OVER

    print(f"[BACKUP MASTER] Standby - monitoring {PRIMARY_HEALTH_URL}")
    consecutive_failures = 0
    FAILURE_THRESHOLD = 3

    while True:
        try:
            res = requests.get(PRIMARY_HEALTH_URL, timeout=2)
            alive = (res.status_code == 200)
        except Exception:
            alive = False

        if alive:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            print(f"[BACKUP MASTER] Primary not responding ({consecutive_failures}/{FAILURE_THRESHOLD})")

            if consecutive_failures >= FAILURE_THRESHOLD and not HAS_TAKEN_OVER:
                print("\n[BACKUP MASTER] Primary is DOWN. Taking over as PRIMARY.\n")

                IS_PRIMARY = True
                HAS_TAKEN_OVER = True

                threading.Thread(
                    target=start_master_workload,
                    daemon=True
                ).start()

        time.sleep(1)



if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    monitor_thread = threading.Thread(target=monitor_primary, daemon=True)
    monitor_thread.start()

    print("[BACKUP MASTER] Status server running on localhost:9000")
    print("[BACKUP MASTER] Check status: http://localhost:9000/status\n")

    app.run(host="0.0.0.0", port=9000)
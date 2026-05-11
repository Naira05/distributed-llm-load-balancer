# main_worker.py
# Run this on each GPU machine to start the worker server

from workers.gpu_worker_server import app  # adjust import to match your project
import uvicorn
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker-id", type=int, default=1)
    args = parser.parse_args()

    print(f"[Worker] Starting worker {args.worker_id} on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
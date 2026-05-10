import requests


class LoadBalancer:

    def __init__(self, worker_urls, master=None):

        self.worker_urls = worker_urls  # ["http://localhost:8000", ...]
        self.master = master
        self.current = 0

    # ROUND ROBIN

    def round_robin(self):

        url = self.worker_urls[
            self.current % len(self.worker_urls)
        ]

        self.current += 1
        return url

    # LEAST CONNECTIONS (REMOTE VERSION)

    def least_connections(self):

        # would require health API from workers
        best = None
        best_load = float("inf")

        for url in self.worker_urls:

            res = requests.get(f"{url}/load").json()

            load = res["active_tasks"]

            if load < best_load:
                best_load = load
                best = url

        return best

    # LOAD AWARE ROUTING

    def load_aware(self):

        best = None
        best_score = float("inf")

        for url in self.worker_urls:

            res = requests.get(f"{url}/load").json()

            score = (
                res["active_tasks"] +
                res["gpu_utilization"] / 100
            )

            if score < best_score:
                best_score = score
                best = url

        return best

    # DISPATCH REQUEST

    def dispatch(self, request):

        url = self.load_aware()

        response = requests.post(
            f"{url}/process",
            json=request
        )

        return response.json()
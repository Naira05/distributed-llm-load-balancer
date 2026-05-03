import os
import sys
from load_balancer.load_balancer import LoadBalancer
from master.scheduler import MasterNode
from workers.worker import Worker

# create workers (LLM nodes)
workers = [Worker(i) for i in range(3)]

# master controls workers
master = MasterNode(workers)

# load balancer routes to master
lb = LoadBalancer(workers, master)

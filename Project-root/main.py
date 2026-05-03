import os
import sys
from load_balancer import LoadBalancer
from master import MasterNode
from llm import Worker


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# create workers (LLM nodes)
workers = [Worker(i) for i in range(3)]

# master controls workers
master = MasterNode(workers)

# load balancer routes to master
lb = LoadBalancer(workers, master)

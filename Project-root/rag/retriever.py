import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("RAG")

KNOWLEDGE_BASE = [
    "In distributed systems, components located on networked computers communicate and coordinate actions via messages.",
    "Load balancing routes incoming requests across a cluster of nodes to optimize utilization and minimize latency.",
    "Load balancing distributes work across multiple nodes using strategies like round-robin, least-connections, or load-aware routing.",
    "Fault tolerance enables continuity of operations despite partial node failures.",
    "Fault tolerance is achieved through retries, failover to healthy nodes, and health-check monitoring.",
    "GPUs are specialized processors optimized for parallel computation used in LLM inference.",
    "A GPU cluster consists of multiple GPU nodes working together to handle distributed inference workloads.",
    "Large Language Models use deep neural networks to generate human-like text.",
    "LLM models such as GPT and LLaMA process text using transformer-based architectures.",
    "Inference is the process of running a trained LLM model to generate responses from input prompts.",
    "A cluster is a group of networked machines that work together as a unified compute resource.",
    "Requests are incoming user queries routed through the load balancer to available GPU worker nodes.",
    "Workers are individual nodes in the cluster that receive and process inference requests.",
    "The scheduler coordinates task assignment across workers based on load, availability, and strategy.",
    "Latency is the time taken from sending a request to receiving a response in distributed systems.",
    "Throughput measures how many requests a system can handle per second under load.",
    "Retry logic re-attempts failed requests on healthy workers to improve reliability.",
    "Health checks periodically ping workers to detect failures and remove dead nodes.",
    "Round-robin distributes requests evenly across all workers in cyclic order.",
    "Context in RAG refers to retrieved background knowledge injected into the prompt."
]

model = SentenceTransformer("all-mpnet-base-v2")

KB_EMBEDDINGS = model.encode(
    KNOWLEDGE_BASE,
    normalize_embeddings=True
)



def retrieve_contexts(query: str, top_k: int = 2) -> List[str]:
 
    query_vec = model.encode(query, normalize_embeddings=True)
    scores = np.dot(KB_EMBEDDINGS, query_vec)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [KNOWLEDGE_BASE[i] for i in top_indices]


def build_prompt(query: str, contexts: List[str]) -> str:


    context_block = "\n".join(
        [f"[Context {i+1}] {ctx}" for i, ctx in enumerate(contexts)]
    )

    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

{context_block}

Question: {query}

Answer:
"""

    return prompt.strip()


def retrieve_and_build(query: str) -> str:
    contexts = retrieve_contexts(query, top_k=2)
    prompt = build_prompt(query, contexts)

    log.info(f"Query: {query}")
    log.info(f"Retrieved contexts: {contexts}")

    return prompt
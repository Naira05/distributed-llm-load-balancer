# rag/retriever.py

import logging
from typing import List, Tuple

log = logging.getLogger("RAG")

KNOWLEDGE_BASE = {
    "distributed": "In distributed systems, components located on networked computers communicate and coordinate actions via messages.",
    "load": "Load balancing routes incoming requests across a cluster of nodes to optimize utilization and minimize latency.",
    "load balancing": "Load balancing distributes work across multiple nodes using strategies like round-robin, least-connections, or load-aware routing.",
    "fault": "Fault tolerance enables continuity of operations despite partial node failures.",
    "fault tolerance": "Fault tolerance is achieved through retries, failover to healthy nodes, and health-check monitoring.",
    "gpu": "GPUs are specialized processors optimized for parallel computation used in LLM inference.",
    "gpu cluster": "A GPU cluster consists of multiple GPU nodes working together to handle distributed inference workloads.",
    "llm": "Large Language Models use deep neural networks to generate human-like text.",
    "model": "LLM models such as GPT and LLaMA process text using transformer-based architectures.",
    "inference": "Inference is the process of running a trained LLM model to generate responses from input prompts.",
    "cluster": "A cluster is a group of networked machines that work together as a unified compute resource.",
    "request": "Requests are incoming user queries routed through the load balancer to available GPU worker nodes.",
    "worker": "Workers are individual nodes in the cluster that receive and process inference requests.",
    "scheduler": "The scheduler coordinates task assignment across workers based on load, availability, and strategy.",
    "latency": "Latency is the time taken from sending a request to receiving a response, critical in real-time inference.",
    "throughput": "Throughput measures how many requests a system can handle per second under load.",
    "retry": "Retry logic re-attempts failed requests on healthy workers to improve reliability.",
    "health check": "Health checks periodically ping workers to detect failures and remove dead nodes from rotation.",
    "round robin": "Round-robin distributes requests evenly across all workers in a cyclic order.",
    "context": "Context in RAG refers to retrieved background knowledge injected into the prompt to improve response quality.",
}

# synonym / alias map → maps alternate phrasings to KB keys
SYNONYMS = {
    "balance":      "load balancing",
    "balancer":     "load balancing",
    "crash":        "fault tolerance",
    "failure":      "fault tolerance",
    "fail":         "fault",
    "down":         "fault tolerance",
    "node":         "worker",
    "machine":      "worker",
    "server":       "worker",
    "response":     "inference",
    "generate":     "inference",
    "generation":   "inference",
    "speed":        "latency",
    "slow":         "latency",
    "fast":         "throughput",
    "scale":        "cluster",
    "parallel":     "gpu",
    "neural":       "llm",
    "transformer":  "model",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase and split into words + bigrams."""
    words = text.lower().split()
    tokens = words[:]
    # add bigrams
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")
    return tokens


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant KB entries for a query.
    Scoring: exact KB key match (2pts) + synonym match (1pt) + frequency bonus.
    """
    log.info(f"[RAG] Retrieving context for: '{query}'")

    tokens   = _tokenize(query)
    scores: dict[str, float] = {}

    for token in tokens:
        # direct KB key match
        if token in KNOWLEDGE_BASE:
            scores[token] = scores.get(token, 0) + 2

        # synonym → KB key match
        if token in SYNONYMS:
            kb_key = SYNONYMS[token]
            if kb_key in KNOWLEDGE_BASE:
                scores[kb_key] = scores.get(kb_key, 0) + 1

    if not scores:
        log.info("[RAG] No matches found, returning fallback context.")
        return (
            "General distributed system context: a cluster of GPU workers "
            "processes inference requests routed by a load balancer with "
            "fault-tolerant retry logic."
        )

    # sort by score descending, take top_k
    ranked: List[Tuple[float, str]] = sorted(
        ((score, key) for key, score in scores.items()),
        reverse=True,
    )[:top_k]

    contexts = [KNOWLEDGE_BASE[key] for _, key in ranked]
    result   = " ".join(contexts)

    log.info(f"[RAG] Matched keys: {[k for _, k in ranked]}")
    return result
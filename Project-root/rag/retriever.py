import logging

log = logging.getLogger("RAG")

MOCK_KNOWLEDGE_BASE = {
    "distributed": "In distributed systems, components located on networked computers communicate and coordinate actions via messages.",
    "load": "Load balancing routes incoming requests across a cluster of nodes to optimize utilization and minimize latency.",
    "fault": "Fault tolerance enables continuity of operations despite partial node failures.",
    "gpu": "GPUs are specialized processors optimized for parallel computation used in LLM inference.",
    "llm": "Large Language Models use deep neural networks to generate human-like text.",
    "model": "LLM models such as GPT and LLaMA process text using transformer-based architectures.",
    "inference": "Inference is the process of running a trained LLM model to generate responses from input prompts.",
    "cluster": "A GPU cluster consists of multiple GPU nodes working together to handle distributed inference workloads.",
    "request": "Requests are incoming user queries routed through the load balancer to available GPU worker nodes.",
}

def retrieve_context(query: str) -> str:
    log.info(f"Retrieving context for query: '{query}'")

    query_lower = query.lower()
    scored_contexts = []

    for keyword, context in MOCK_KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            score = query_lower.count(keyword)
            scored_contexts.append((score, context))

    if scored_contexts:
        scored_contexts.sort(reverse=True)
        return " ".join([ctx for _, ctx in scored_contexts])

    return "General distributed system context with standard inference pipeline behavior."
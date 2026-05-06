# rag/retriever.py
import random
import logging

##logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [RAG] %(message)s")
log = logging.getLogger("RAG")

# Mock database mapping keywords to contextual knowledge base blocks
MOCK_KNOWLEDGE_BASE = {
    "distributed": "In distributed systems, components located on networked computers communicate and coordinate actions via messages.",
    "load": "Load balancing routes incoming requests across a cluster of nodes to optimize utilization and minimize latency.",
    "fault": "Fault tolerance is the system properties enabling continuity of operations in the event of partial node failures.",
    "gpu": "GPUs are specialized hardware processors designed for heavy parallel computation pathways required by LLMs.",
    "llm": "Large Language Models use highly parameterized neural network architectures to generate human-like text sequence predictions.",
}

def retrieve_context(query: str) -> str:
    """
    Simulates semantic context retrieval from a vector database.
    Scans the client query for keywords and returns corresponding contextual facts.
    """
    log.info(f"Retrieving vector DB context for query: '{query}'")
    
    query_lower = query.lower()
    matched_contexts = []
    
    for keyword, context in MOCK_KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            matched_contexts.append(context)
            
    if matched_contexts:
        # Join matching contexts if multiple keywords are identified
        return " ".join(matched_contexts)
    
    # Fallback contexts if no keywords hit
    fallback_contexts = [
        "Standard high-concurrency computing environment context.",
        "General-purpose inference pipeline parameters applied.",
        "Default contextual system knowledge telemetry."
    ]
    return random.choice(fallback_contexts)
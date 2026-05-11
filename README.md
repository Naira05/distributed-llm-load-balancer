
# Distributed LLM System with Load Balancing and GPU Task Distribution

## Overview
This project implements a distributed computing system capable of handling 1000+ concurrent user requests involving Large Language Model (LLM) inference and Retrieval-Augmented Generation (RAG).

The system focuses on:
- Efficient load balancing
- GPU task distribution
- Fault tolerance
- High scalability

## System Architecture

The system consists of the following components:

- **Client Layer**: Simulates concurrent users
- **Load Balancer**: Distributes incoming requests
- **Master Node**: Schedules tasks
- **Worker Nodes**: Perform LLM inference
- **RAG Module**: Enhances responses with retrieved data

## Workflow

1. Client sends request
2. Load Balancer distributes request
3. Master Node schedules task
4. Worker processes request
5. Response returned to client

## Features

### Load Balancing
- Round Robin
- Least Connections (optional)

### GPU Task Distribution
- Parallel processing
- Efficient utilization

### Fault Tolerance
- Worker failure detection
- Task reassignment

### Scalability
- Supports 1000+ concurrent users

## Testing

- Load testing with 100–1000 users
- Failure simulation (node shutdown)
- Performance metrics:
  - Latency
  - Throughput
  - GPU utilization

## How to Run

```bash
python main.py

```bash 
## windows command to run the gpu worker: 
curl.exe -H "ngrok-skip-browser-warning: true" https://authToken/health

## Macbook
curl -H "ngrok-skip-browser-warning: true" https://authToken/health



import matplotlib.pyplot as plt


results = [
    {"users": 100, "throughput": 7.14, "avg_latency": 20001.26, "p95_latency": 20002.24},
    {"users": 25, "throughput": 4.16, "avg_latency": 2001.07, "p95_latency": 2002.16},
  
]


users = [r["users"] for r in results]
throughput = [r["throughput"] for r in results]
avg_latency = [r["avg_latency"] for r in results]
p95_latency = [r["p95_latency"] for r in results]

plt.figure()
plt.plot(users, throughput, marker='o')
plt.title("Throughput vs Number of Users")
plt.xlabel("Users")
plt.ylabel("Throughput (req/s)")
plt.grid(True)
plt.show()


plt.figure()
plt.plot(users, avg_latency, marker='o')
plt.title("Average Latency vs Number of Users")
plt.xlabel("Users")
plt.ylabel("Latency (ms)")
plt.grid(True)
plt.show()


plt.figure()
plt.plot(users, p95_latency, marker='o')
plt.title("P95 Latency vs Number of Users")
plt.xlabel("Users")
plt.ylabel("Latency (ms)")
plt.grid(True)
plt.show()
import networkx as nx
from node2vec import Node2Vec
import numpy as np
#이건 그냥 임베딩 된값 확인용
# Generate instance
def generate_instance(num_jobs, num_machines, time_min=1, time_max=100, machine_range=(1, 20)):
    np.random.seed(0)  # Seed for reproducibility

    # Initialize arrays
    times = np.zeros((num_jobs, num_machines), dtype=int)
    machines = np.zeros((num_jobs, num_machines), dtype=int)

    for i in range(num_jobs):
        # Generate non-duplicate times for each job
        times[i] = np.random.choice(range(time_min, time_max + 1), size=num_machines, replace=False)
        # Generate non-duplicate machines for each job
        machines[i] = np.random.permutation(range(machine_range[0], machine_range[1] + 1))

    return times, machines

# Generate Disjunctive Graph
def make_graph(times, machines) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for job_index, (time_row, machine_row) in enumerate(zip(times, machines)):
        previous_node = None
        for step_index, (time, machine) in enumerate(zip(time_row, machine_row)):
            node = f"{job_index}-{step_index}"
            G.add_node(node, machine=machine, time=time)
            if previous_node:
                G.add_edge(previous_node, node, type="CONJUNCTIVE")
            previous_node = node

    # Create disjunctive edges
    for machine in range(1, np.max(machines) + 1):
        job_indices, step_indices = np.where(machines == machine)
        for i, (job, step) in enumerate(zip(job_indices, step_indices)):
            node = f"{job}-{step}"
            for other_job, other_step in zip(job_indices[i + 1:], step_indices[i + 1:]):
                other_node = f"{other_job}-{other_step}"
                G.add_edge(node, other_node, type="DISJUNCTIVE")
                G.add_edge(other_node, node, type="DISJUNCTIVE")
    
    return G

# Node2Vec Embedding
def generate_embeddings(G):
    node2vec = Node2Vec(G, dimensions=20, walk_length=30, num_walks=200, workers=4, p=1, q=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return node_embeddings

# Parameters
num_jobs = 20
num_machines = 100

# Generate instance data
times, machines = generate_instance(num_jobs, num_machines, machine_range=(1, num_machines))

# Create disjunctive graph
graph = make_graph(times, machines)

# Generate Node2Vec embeddings
node_embeddings = generate_embeddings(graph)

# Display results
print(f"Node Embeddings shape: {node_embeddings.shape}")

print(node_embeddings[:10])



# -*- coding: cp949 -*-
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
import torch
from torch.utils.data import Dataset, DataLoader
#transformer까지 포함
# 인스턴스 생성
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

# 불연속 그래프 생성
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

    # 불연속 엣지 생성
    for machine in range(1, np.max(machines) + 1):
        job_indices, step_indices = np.where(machines == machine)
        for i, (job, step) in enumerate(zip(job_indices, step_indices)):
            node = f"{job}-{step}"
            for other_job, other_step in zip(job_indices[i + 1:], step_indices[i + 1:]):
                other_node = f"{other_job}-{other_step}"
                G.add_edge(node, other_node, type="DISJUNCTIVE")
                G.add_edge(other_node, node, type="DISJUNCTIVE")
    
    return G

# Node2Vec 임베딩
def generate_embeddings(G):
    node2vec = Node2Vec(G, dimensions=20, walk_length=30, num_walks=200, workers=4, p=1, q=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return node_embeddings

# 포지셔널 인코딩 생성
def generate_positional_encoding(num_nodes, embedding_dim):
    position = np.arange(num_nodes)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    positional_encoding = np.zeros((num_nodes, embedding_dim))
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)
    return positional_encoding

# 학습 데이터셋 생성
class JSSPDataset(Dataset):
    def __init__(self, node_embeddings, positional_encoding, labels):
        self.node_embeddings = node_embeddings
        self.positional_encoding = positional_encoding
        self.labels = labels

    def __len__(self):
        return len(self.node_embeddings)

    def __getitem__(self, idx):
        # 임베딩 차원 수를 유지하기 위해 2차원 배열로 만듭니다.
        combined_embeddings = self.node_embeddings[idx] + self.positional_encoding[idx]
        combined_embeddings = combined_embeddings[np.newaxis, :]  # 차원을 (1, embedding_dim)으로 변경
        return torch.tensor(combined_embeddings, dtype=torch.float), torch.tensor(int(self.labels[idx]), dtype=torch.long)

# 그래프 시각화 함수
def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, font_color='white')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Parameters
num_jobs = 20
num_machines = 100

# Generate instance data
times, machines = generate_instance(num_jobs, num_machines, machine_range=(1, num_machines))

# Create disjunctive graph
graph = make_graph(times, machines)

# Generate Node2Vec embeddings
node_embeddings = generate_embeddings(graph)

# Generate positional encoding
num_nodes = node_embeddings.shape[0]
embedding_dim = node_embeddings.shape[1]
positional_encoding = generate_positional_encoding(num_nodes, embedding_dim)

# 가상의 레이블 생성 (실제 데이터에 맞게 변경해야 함)
labels = np.zeros(len(node_embeddings))

# JSSP 데이터셋 생성
dataset = JSSPDataset(node_embeddings, positional_encoding, labels)

# DataLoader 설정
batch_size = 5
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# BertConfig를 사용해 Transformer 모델 구성
config = BertConfig(vocab_size=30522, hidden_size=20, num_attention_heads=2, intermediate_size=64)
model = BertModel(config)

# Optimizer 설정 및 학습 루프 시작
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Attention Mask 생성 함수
def create_attention_mask(inputs):
    attention_mask = (inputs != 0).long()
    return attention_mask

# 학습 루프
# for epoch in range(3):
#     for batch in dataloader:
#         inputs, targets = batch
#         attention_mask = torch.ones(inputs.size()[:-1], dtype=torch.long)  # 배치 및 시퀀스 차원에만 마스크 생성

#         optimizer.zero_grad()
#         # BERT 모델에 Node2Vec 임베딩과 포지셔널 인코딩을 입력합니다.
#         outputs = model(inputs_embeds=inputs, attention_mask=attention_mask)
#         loss = torch.nn.functional.mse_loss(outputs.last_hidden_state.squeeze(), targets.float())
#         loss.backward()
#         optimizer.step()

for epoch in range(3):
    for batch in dataloader:
        inputs, targets = batch
        attention_mask = torch.ones(inputs.size()[:-1], dtype=torch.long)

        optimizer.zero_grad()
        # BERT 모델에 Node2Vec 임베딩과 포지셔널 인코딩을 입력합니다.
        outputs = model(inputs_embeds=inputs, attention_mask=attention_mask)

        # 배치당 단일 출력으로 축소 (예: 평균)
        output_mean = outputs.last_hidden_state.mean(dim=-1).mean(dim=1)

        # 손실 계산 시 타겟과 크기를 맞춤
        loss = torch.nn.functional.mse_loss(output_mean, targets.float())
        loss.backward()
        optimizer.step()


# Display results
print(f"Node Embeddings shape: {node_embeddings.shape}")

# 그래프 시각화
visualize_graph(graph)

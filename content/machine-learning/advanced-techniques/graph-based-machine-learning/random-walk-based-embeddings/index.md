---
linkTitle: "Random Walk Based Embeddings"
title: "Random Walk Based Embeddings: Generating Node Embeddings Using Random Walks on Graphs"
description: "This article explores the Random Walk Based Embeddings design pattern, detailing how random walks on graphs can be leveraged to generate effective node embeddings. Techniques such as DeepWalk and Node2Vec are thoroughly examined."
categories:
- Advanced Techniques
tags:
- Graph-Based Machine Learning
- Node Embeddings
- Random Walks
- DeepWalk
- Node2Vec
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/graph-based-machine-learning/random-walk-based-embeddings"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Random Walk Based Embeddings: Generating Node Embeddings Using Random Walks on Graphs

### Introduction
Random Walk Based Embeddings is a powerful design pattern within Graph-Based Machine Learning, where random walks on graph structures are employed to create node embeddings. These techniques are pivotal in representing complex network data in continuous vector spaces for downstream machine learning tasks.

### Key Techniques

1. **DeepWalk**
2. **Node2Vec**

Both methods convert nodes in a graph into dense vectors, making them easier to work with in machine learning models.

### DeepWalk

#### Overview
DeepWalk is inspired by word embeddings in natural language processing. It uses truncated random walks to produce sequences of nodes, then applies the Skip-Gram model to learn node embeddings.

#### Mechanism
1. **Random Walks**: Perform multiple truncated random walks from each node.
2. **Skip-Gram Model**: Use these walks to train the Skip-Gram model, treating the sequences as sentences and nodes as words.

#### Detailed Example

```python
import networkx as nx
import random
from gensim.models import Word2Vec

graph = nx.fast_gnp_random_graph(100, 0.1)

def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk

corpus = [random_walk(graph, node, walk_length=10) for node in graph.nodes for _ in range(10)]
model = Word2Vec(corpus, vector_size=64, window=5, min_count=1, sg=1)
node_embedding = model.wv['0']
print(node_embedding)
```

### Node2Vec

#### Overview
Node2Vec enhances DeepWalk by introducing two parameters, \\(p\\) and \\(q\\), controlling the breadth-first search (BFS) and depth-first search (DFS) characteristics in the random walks.

#### Mechanism
1. **Biased Random Walks**: Adjust the probability of the next node in the walk using \\(p\\) and \\(q\\).
2. **Skip-Gram Model**: Similar to DeepWalk, the sequences are used to train a Skip-Gram model.

#### Detailed Example

```python
from node2vec import Node2Vec
import networkx as nx

graph = nx.fast_gnp_random_graph(100, 0.1)

node2vec = Node2Vec(graph, dimensions=64, walk_length=10, num_walks=100, p=0.25, q=0.75, workers=1)
model = node2vec.fit(window=5, min_count=1, batch_words=4)
node_embedding = model.wv['0']
print(node_embedding)
```

### Related Design Patterns

1. **Graph Convolutional Networks (GCNs)**
   - **Description**: GCNs focus on directly processing graph-structured data through layers of convolutions across node features and their neighborhoods.
   - **Example**:
     ```python
     import torch
     import torch.nn.functional as F
     from torch_geometric.nn import GCNConv

     class GCN(torch.nn.Module):
         def __init__(self):
             super(GCN, self).__init__()
             self.conv1 = GCNConv(16, 16)
             self.conv2 = GCNConv(16, 16)

         def forward(self, data):
             x, edge_index = data.x, data.edge_index
             x = self.conv1(x, edge_index)
             x = F.relu(x)
             x = self.conv2(x, edge_index)
             return F.log_softmax(x, dim=1)
     ```

2. **GraphSAGE**
   - **Description**: GraphSAGE generates embeddings by sampling and aggregating features from a node’s local neighborhood.
   - **Example**:
     ```python
     import torch
     import torch.nn as nn
     from torch_geometric.nn import SAGEConv

     class GraphSAGE(nn.Module):
         def __init__(self):
             super(GraphSAGE, self).__init__()
             self.conv1 = SAGEConv(16, 16)
             self.conv2 = SAGEConv(16, 16)

         def forward(self, data):
             x, edge_index = data.x, data.edge_index
             x = self.conv1(x, edge_index)
             x = F.relu(x)
             x = self.conv2(x, edge_index)
             return F.log_softmax(x, dim=1)
     ```

### Additional Resources

1. **DeepWalk Paper**: "DeepWalk: Online Learning of Social Representations" by Perozzi, Al-Rfou, and Skiena. [Link](http://stanford.edu/~jure/pubs/deepwalk-kdd14.pdf)
2. **Node2Vec Paper**: "node2vec: Scalable Feature Learning for Networks" by Grover and Leskovec. [Link](https://arxiv.org/abs/1607.00653)
3. **Graph Embedding Libraries**:
    - [Gensim](https://radimrehurek.com/gensim/)
    - [Node2Vec Python Implementation](https://github.com/eliorc/node2vec)
    - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

### Summary
Random Walk Based Embeddings leverage random paths within graphs to effectively capture structural information, crucial for generating meaningful node embeddings. Techniques like DeepWalk and Node2Vec have shown tremendous utility in various graph-based tasks by combining the simplicity of random walks with the power of Skip-Gram models. These methods represent a cornerstone in graph-based machine learning, providing a solid foundation for further advancements and applications in network analysis and beyond.
{{< katex />}}


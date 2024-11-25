---
linkTitle: "Horizontal Federated Learning"
title: "Horizontal Federated Learning: Federated learning using data with the same feature space across different institutions"
description: "A detailed exploration of Horizontal Federated Learning, a variant of federated learning where multiple institutions share the same feature space but hold data for different sets of entities."
categories:
- Emerging Fields
- Federated Learning Variants
tags:
- federated learning
- horizontal federated learning
- machine learning
- data privacy
- decentralized computation
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/emerging-fields/federated-learning-variants/horizontal-federated-learning"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Horizontal Federated Learning (HFL) refers to a method in which multiple institutions collaboratively train a machine learning model using data that is horizontally partitioned. Institutions share an identical feature space, but data items pertain to different entities. The goal is to leverage collective datasets without exposing or transferring any sensitive information, thus maintaining privacy and compliance with regulations such as GDPR.

## Horizontal Federated Learning Workflow

In Horizontal Federated Learning, the data held by different parties shares the same schema but has distinct records. The workflow includes:

1. **Initialization**: Models are initialized at each participating institution.
2. **Local Training**: Each institution trains its local model using its respective dataset.
3. **Model Aggregation**: The locally trained models are encrypted and sent to a central server for aggregation.
4. **Global Model Update**: The aggregated model is sent back to each institution, updating local models.
5. **Iteration**: Repeat the local training and model aggregation steps until convergence.

## Example Implementation

We will demonstrate a simple example using Python and PySyft, a Python library for secure and private Deep Learning.

### Step-by-Step Implementation

#### Setting Up the Environment

```python
!pip install syft

import syft as sy
import torch
import torch.nn as nn
import torch.optim as optim
hook = sy.TorchHook(torch)
```

#### Define Virtual Workers

```python
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
secure_worker = sy.VirtualWorker(hook, id="secure_worker")
```

#### Sample Data Allocation

```python
data_alice = torch.tensor([[1.0, 2.0], [1.1, 2.1]], requires_grad=True).tag("data").send(alice)
target_alice = torch.tensor([[1.0], [0.0]], requires_grad=True).tag("target").send(alice)

data_bob = torch.tensor([[3.0, 4.0], [4.1, 4.2]], requires_grad=True).tag("data").send(bob)
target_bob = torch.tensor([[1.0], [0.0]], requires_grad=True).tag("target").send(bob)
```

#### Define Model and Train Locally

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def train(data, target, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = nn.BCELoss()(output, target)
    loss.backward()
    optimizer.step()
    return model

model_alice = SimpleModel()
model_bob = SimpleModel()

optimizer_alice = optim.SGD(model_alice.parameters(), lr=0.01)
optimizer_bob = optim.SGD(model_bob.parameters(), lr=0.01)

for epoch in range(5):
    model_alice = train(data_alice, target_alice, model_alice, optimizer_alice)
    model_bob = train(data_bob, target_bob, model_bob, optimizer_bob)
```

#### Model Aggregation

```python
with torch.no_grad():
    with alice:
        model_alice.fc.weight += model_bob.fc.weight
        model_alice.fc.bias += model_bob.fc.bias
        model_alice.fc.weight /= 2
        model_alice.fc.bias /= 2
```

### Example Continuous Workflow

This implementation covers one iteration, implying the cyclic process between local training and global model updating.

## Related Design Patterns

- **Vertical Federated Learning (VFL)**: Federated learning where datasets across institutions share different feature spaces but have common entities.
- **Federated Averaging**: A key algorithm for federated learning where models are locally trained and then averaged to form an improved global model.
- **Secure Multi-Party Computation (SMPC)**: Techniques to perform computations over data which is split across parties to ensure privacy.

## Additional Resources

- *Google AI Blog on Federated Learning*: [Link](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- *Federated Learning: Collaborative Machine Learning without Centralized Training Data* by Brendan McMahan et al.

## Summary

Horizontal Federated Learning is a pioneering technique facilitating collaboration across institutions with homogeneous data features, preserving privacy and complying with data regulations. This design pattern ensures that machine learning benefits are accessible even when data is distributed and sensitive.

This approach is particularly relevant as privacy concerns grow, providing a feasible solution for collaborative, decentralized learning.

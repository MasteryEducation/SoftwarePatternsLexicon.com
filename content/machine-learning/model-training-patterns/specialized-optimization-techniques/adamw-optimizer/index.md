---
linkTitle: "AdamW Optimizer"
title: "AdamW Optimizer: Variant of Adam with weight decay for convergence improvement"
description: "A detailed overview of the AdamW optimizer, a variant of the Adam optimizer that includes weight decay for improved convergence and regularization."
categories:
- Model Training Patterns
tags:
- optimization
- AdamW
- weight decay
- convergence
- regularization
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/specialized-optimization-techniques/adamw-optimizer"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to AdamW Optimizer

The **AdamW Optimizer** is an optimization algorithm that builds upon the original Adam optimizer by incorporating a weight decay term for regularization. This modification is aimed at separating the gradient updates from the weight decay term, which results in improved generalization performance and convergence properties. 

## Theoretical Background

The Adam algorithm combines the advantages of two other extensions of Stochastic Gradient Descent (SGD): Adaptive Gradient Algorithm (AdaGrad) and Root Mean Squared Propagation (RMSProp). It computes adaptive learning rates for each parameter. The updates are maintained using first and second moment estimates of the gradients.

However, Adam's weight decay implementation traditionally entwines with the gradient calculations, causing suboptimal regularization. To counter this, AdamW applies weight decay directly, creating a more effective optimization process.

### Mathematical Formulation

The AdamW update rule can be captured by the following equations:

1. **Gradient Calculation**:
   {{< katex >}}
   g_t = \nabla_{\theta} L(\theta_{t-1})
   {{< /katex >}}
   
2. **First Moment Estimate**:
   {{< katex >}}
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   {{< /katex >}}
   
3. **Second Moment Estimate**:
   {{< katex >}}
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   {{< /katex >}}
   
4. **Bias-Corrected First Moment Estimate**:
   {{< katex >}}
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
   {{< /katex >}}
   
5. **Bias-Corrected Second Moment Estimate**:
   {{< katex >}}
   \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   {{< /katex >}}
   
6. **Parameter Update**:
   {{< katex >}}
   \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
   {{< /katex >}}
   
   where $\theta_t$ denotes the parameters at time step $t$, $\eta$ is the learning rate, $\epsilon$ is a small constant for numerical stability, $\beta_1$ and $\beta_2$ are hyperparameters for controlling the decay rates of the moment estimates, and $\lambda$ is the weight decay coefficient.

## Implementation in Different Programming Languages

### Python and PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 2)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

input = torch.randn(64, 10)
target = torch.randn(64, 2)

criterion = nn.MSELoss()

optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### TensorFlow

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(10,))
])

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
model.compile(optimizer=optimizer, loss='mse')

input_data = tf.random.normal([64, 10])
target_data = tf.random.normal([64, 2])

model.train_on_batch(input_data, target_data)
```

### Julia and Flux

```julia
using Flux

model = Chain(Dense(10, 2))

η = 0.001
optimizer = Flux.Optimise.AdamW(η, 0.01)

input_data = randn(Float32, 10, 64)
target_data = randn(Float32, 2, 64)

loss(x, y) = Flux.mse(model(x), y)

Flux.train!(loss, params(model), [(input_data, target_data)], optimizer)
```

## Related Design Patterns

### 1. **Adam Optimizer**
   - The AdamW optimizer is a direct improvement and extension of the Adam optimizer, which balances the advantages of RMSProp and momentum by maintaining per-parameter learning rates.

### 2. **SGD with Nesterov Momentum**
   - An extension of the basic SGD algorithm, incorporating Nesterov momentum for improved convergence speed and accuracy.

### 3. **RMSProp**
   - Another adaptive learning rate method, where the learning rate adjusts based on a moving window average of recent gradient magnitudes. It helps to tackle diminishing learning rates and slow convergence.

## Additional Resources

1. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): The original paper by Loshchilov and Hutter, introducing AdamW and offering theoretical insights.
2. [PyTorch Documentation](https://pytorch.org/docs/stable/optim.html): Official documentation for implementing AdamW in PyTorch.
3. [TensorFlow Guide](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/AdamW): Details on using AdamW in TensorFlow.
4. [Flux.jl documentation](https://fluxml.ai/Flux.jl/stable/): Guide to using the AdamW optimizer in the Flux machine learning library for Julia.

## Summary

The **AdamW Optimizer** offers a significant improvement over the traditional Adam optimizer by making weight decay more effective and ensuring proper regularization. This results in better convergence characteristics and often improved model generalization. The optimizer is widely supported across popular frameworks like PyTorch, TensorFlow, and Flux, making it a robust choice for diverse machine learning applications.

By decoupling weight decay from the gradient updates, AdamW achieves superior performance in deep learning model training, establishing itself as a valuable tool in the machine learning engineer's toolkit.

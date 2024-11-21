---
linkTitle: "Adam Optimizer"
title: "Adam Optimizer: Combining the Advantages of Two Other Extensions of Stochastic Gradient Descent"
description: "Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent, providing an efficient and robust technique for training machine learning models."
categories:
- Model Training Patterns
tags:
- ML
- Optimization
- StochasticGradientDescent
- DeepLearning
- ModelTraining
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/optimization-techniques/adam-optimizer"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Adam, or Adaptive Moment Estimation, is an efficient optimization algorithm that combines the merits of two popular methods: AdaGrad and RMSProp. It is widely used in deep learning due to its adaptive learning rate and momentum properties.

## Background

Stochastic Gradient Descent (SGD) is a core technique for training machine learning models. However, traditional SGD can suffer from poor convergence and sensitivity to tuning hyperparameters. Overcoming these issues, several extensions have been developed:

1. **AdaGrad**: It adapts the learning rate of each parameter based on the gradients' historical magnitudes, improving convergence for sparse gradients.
2. **RMSProp**: It aims to resolve AdaGrad's diminishing learning rates by using a moving average of squared gradients to normalize the learning rate.

Adam incorporates ideas from both AdaGrad and RMSProp, providing benefits such as adaptive learning rates and momentum, leading to faster and more robust convergence.

## Algorithm

Adam’s optimization technique is based on calculating adaptive learning rates for each parameter by keeping an exponentially decaying average of past squared gradients (RMSProp) and past gradients (momentum).

### Steps

1. **Initialization**: Initialize the parameters \\( \theta_0 \\) and the moments \\( m \\) and \\( v \\) to zero:
   
{{< katex >}}
m_0 = 0,\quad v_0 = 0
{{< /katex >}}

2. **Gradient Calculation**: Calculate the gradient \\( g_t \\) with respect to the parameter \\( \theta_t \\).

3. **Update Biased First and Second Moment Estimates**:

{{< katex >}} 
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t 
{{< /katex >}}

{{< katex >}} 
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 
{{< /katex >}}

4. **Bias Correction**:

{{< katex >}} 
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
{{< /katex >}}

{{< katex >}} 
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} 
{{< /katex >}}

5. **Parameter Update**:

{{< katex >}} 
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} 
{{< /katex >}}

Where:
- \\( \alpha \\) is the learning rate.
- \\( \beta_1, \beta_2 \\) are decay rates for the momentum estimates.
- \\( \epsilon \\) is a small constant to prevent division by zero.

### Example Implementations

#### Python (TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10)
```

#### Python (PyTorch)

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
```

## Related Design Patterns

- **AdaGrad**: Focuses on the adaption of learning rates based on the magnitude of gradients previously encountered.
- **RMSProp**: Similar to AdaGrad but improves on it by using a moving average of squared gradients, thereby mitigating diminishing learning rates.
- **SGD with Momentum**: Introduces an additive momentum term to the gradient, accelerating the convergence, especially in relevant directions.

## Additional Resources

- **Adam Paper**: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) by Diederik P. Kingma and Jimmy Ba.
- **Deep Learning Book**: Chapter on optimization algorithms (Ian Goodfellow, Yoshua Bengio, and Aaron Courville).
- **TensorFlow Documentation**: [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

## Summary

Adam is a powerful and widely-used optimizer in machine learning. It effectively combines the adaptive learning rate of AdaGrad with the momentum concept of RMSProp, making it suitable for a vast range of applications. Its robustness and efficiency have established it as a default choice for training deep learning models in today's research and industry practices.

Understanding Adam and its foundational components enables practitioners to optimize their models better, leading to more efficient and scalable AI solutions.

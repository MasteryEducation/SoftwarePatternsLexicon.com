---
linkTitle: "SGD (Stochastic Gradient Descent)"
title: "SGD (Stochastic Gradient Descent): Using Partial Data to Update Weights in Each Iteration"
description: "An efficient optimization technique using partial data to update weights in each iteration during model training."
categories:
- Model Training Patterns
tags:
- Optimization
- SGD
- Gradient Descent
- Machine Learning
- Model Training
date: 2023-10-12
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/optimization-techniques/sgd-(stochastic-gradient-descent)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
Stochastic Gradient Descent (SGD) is a widely utilized optimization technique in machine learning, particularly for training large-scale models. Unlike traditional gradient descent, which uses the entire dataset to calculate gradients and update weights, SGD employs only a random subset of the data (often just one sample) for each iteration. This technique significantly reduces computation time and often leads to faster convergence.

## Detailed Explanation

### Mathematical Formulation
Gradient Descent's objective is to minimize a given cost function \\(J(\theta)\\), which quantifies the error between predictions and actual values. The traditional update rule for Gradient Descent is:

{{< katex >}}
\theta := \theta - \eta \cdot \nabla J(\theta)
{{< /katex >}}

where:
- \\(\theta\\) represents the model parameters (weights).
- \\(\eta\\) is the learning rate.
- \\(\nabla J(\theta)\\) is the gradient of the cost function with respect to \\(\theta\\).

In the case of SGD, the cost function is approximated with a single training sample or a mini-batch:

{{< katex >}}
\theta := \theta - \eta \cdot \nabla J(\theta; x^{(i)}, y^{(i)})
{{< /katex >}}

where \\( (x^{(i)}, y^{(i)}) \\) is a randomly chosen training sample or a mini-batch from the dataset.

### Algorithm Steps
1. **Initialization**
   - Initialize weights \\( \theta \\).
   - Set the learning rate \\( \eta \\).

2. **Iteration**
   - For each iteration:
     1. Shuffle the training data.
     2. Select a random sample \\( (x^{(i)}, y^{(i)}) \\) from the dataset.
     3. Compute the gradient of the cost function with respect to the selected sample.
     4. Update weights \\( \theta \\) using the computed gradient.

### Advantages
- Efficiency: Faster to compute for large datasets.
- Stochastic Nature: Helps escape local minima and can find more optimal global minima.
- Online Learning: Can be used for real-time model updates.

### Disadvantages
- Noisy Updates: Fluctuations in the objective function due to individual data points.
- Hyperparameter Sensitivity: More sensitive to the learning rate and requires careful tuning.

## Example Implementations

### Python Example using PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X_data = torch.randn(100, 10)  # 100 samples, 10 features
y_data = torch.randn(100, 1)   # 100 target values

train_set = TensorDataset(X_data, y_data)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):  # number of epochs
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
```

### R Example using TensorFlow
```R
library(tensorflow)

set.seed(123)
X_data <- matrix(rnorm(1000), ncol=10)
y_data <- matrix(rnorm(100, 0, 1), ncol=1)

model <- keras_model_sequential() %>%
  layer_dense(units = 1, input_shape = c(10))

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_sgd(lr = 0.01)
)

model %>% fit(X_data, y_data, epochs = 100, batch_size = 1, shuffle = TRUE)

summary(model)
```

## Related Design Patterns

### Mini-Batch Gradient Descent
Instead of using a single sample, mini-batch gradient descent uses a small and fixed number of samples (mini-batch) to update weights, reducing the noise of SGD while maintaining computational efficiency.

### Momentum
Momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging. It's an extension of SGD by means of incorporating the momentum term.

### RMSProp
RMSProp divides the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight. This makes it akin to a learning rate hand-tuner.

## Additional Resources
- [An overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/index.html)
- [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning) by Andrew Ng
- Books: "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville

## Summary
SGD is a powerful optimization technique crucial for training large-scale machine learning models efficiently. Using only partial data, it updates model weights iteratively, resulting in faster convergence. Despite its noisy updates and sensitivity to hyperparameters, its benefits make it a preferred choice for online learning and large datasets.

Understanding and correctly implementing SGD can significantly improve the training process and performance of machine learning models.

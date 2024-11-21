---
linkTitle: "LR Finder"
title: "LR Finder: Finding the Optimal Initial Learning Rate"
description: "A method to determine the optimal initial learning rate by observing how the loss decreases during training."
categories:
- Model Training Patterns
- Specialized Optimization Techniques
tags:
- Machine Learning
- Deep Learning
- Learning Rate
- Model Training
- Optimization
date: 2023-10-10
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/specialized-optimization-techniques/lr-finder"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The **LR Finder** is a specialized machine learning optimization technique designed to identify the most effective initial learning rate for training deep learning models. By systematically increasing the learning rate and tracking the rate of loss reduction, practitioners can select an optimal starting point that accelerates convergence and improves overall model performance.

## Overview

Choosing the right learning rate is critical in the optimization of neural networks. If the learning rate is too low, convergence can be slow and computationally expensive. Conversely, if it's too high, the optimization process may overshoot minima, resulting in poor model performance. The **LR Finder** addresses this by generating a plot (loss vs. learning rate) to help select a learning rate within a range that ensures efficient and effective training.

## Detailed Explanation

### Theory and Methodology

The LR Finder technique involves gradually increasing the learning rate on a logarithmic scale and recording the corresponding training loss. The loss typically follows certain patterns:
- **Plateau**: At very low learning rates, loss decreases very slowly.
- **Steep Decline**: There is a steep decline as the learning rate increases and starts to effectively minimize the loss.
- **Divergence**: At very high learning rates, the loss increases rapidly due to instability in the training process.

From this collected data, a suitable learning rate can be identified from the region just before the divergence. Typically, this can be done as follows:
1. Train the model for a few epochs, starting with a very low learning rate and exponentially scaling it up.
2. Plot the learning rate against the loss.
3. Choose a learning rate from the range before the loss starts to rise abruptly.

### Implementation

Here, we will implement the LR Finder using Python with Keras and Pytorch.

#### Example in Python using Keras
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class LRFinder(Callback):
    def __init__(self, min_lr=1e-5, max_lr=1e-1, steps=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_batches = steps
        self.batch_count = 0
        self.history = {}

    def on_train_begin(self, logs=None):
        self.model.save_weights('tmp_init_weights.h5')
        self.lr_mult = (self.max_lr / self.min_lr) ** (1 / self.total_batches)
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        lr = self.min_lr * (self.lr_mult ** self.batch_count)
        self.learning_rates.append(lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        if self.batch_count >= self.total_batches:
            self.model.stop_training = True

    def plot_lr(self):
        plt.plot(self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([Dense(10, input_shape=(10,), activation='relu'), Dense(1)])
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
model.fit(np.random.rand(1000, 10), np.random.rand(1000), callbacks=[lr_finder], epochs=1)
lr_finder.plot_lr()
```

#### Example in Python using PyTorch
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def range_test(self, dataloader, end_lr, num_iter):
        self.model.train()
        gamma = (end_lr / self.optimizer.param_groups[0]['lr']) ** (1/num_iter)
        lr_list, losses = [], []

        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_iter:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            lr_list.append(self.optimizer.param_groups[0]['lr'])
            losses.append(loss.item())

            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= gamma

        plt.plot(lr_list, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss, Linear

model = Linear(10, 1)
optimizer = SGD(model.parameters(), lr=1e-6)
criterion = MSELoss()

dataloader = DataLoader(TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1)), batch_size=32)
lr_finder = LRFinder(model, optimizer, criterion, device='cpu')
lr_finder.range_test(dataloader, end_lr=0.1, num_iter=100)
```

## Related Design Patterns

1. **Cyclic Learning Rates**: A method that varies the learning rate between a minimum and maximum boundary based on a particular schedule during training. This helps in finding optimum learning rates during different phases of the training process.

2. **Learning Rate Schedulers**: These are predefined schedules or rules that adjust the learning rate during training. Common schedulers include Step Decay, Exponential Decay, and Reduce On Plateaus.

3. **Early Stopping**: This technique complements the notion of LR Finder by halting training once the model’s performance on a validation set starts to degrade, ensuring the model does not overfit.

## Summary

The **LR Finder** is a powerful technique to empirically determine an optimal learning rate for neural network training. By progressively increasing the learning rate and monitoring the loss, practitioners can identify a rate that promises the right balance between speedy convergence and stability. Implementations in Keras and PyTorch showcase its versatility and utility across different frameworks. By leveraging this pattern, in conjunction with related strategies like Cyclic Learning Rates and Learning Rate Schedulers, one can achieve more efficient and effective model training.

## Additional Resources

- Leslie Smith's paper: ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186)
- Keras Callbacks: [Keras Callbacks Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
- PyTorch Learning Rate Schedulers: [PyTorch Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)



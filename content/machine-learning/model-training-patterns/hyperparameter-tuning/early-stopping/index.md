---
linkTitle: "Early Stopping"
title: "Early Stopping: Stopping Training When Performance Stops Improving"
description: "An effective technique for preventing overfitting by halting the training process once the model's performance ceases to improve on the validation set."
categories:
- Model Training Patterns
tags:
- Early Stopping
- Hyperparameter Tuning
- Overfitting
- Model Training
- Performance Monitoring
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/hyperparameter-tuning/early-stopping"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Early Stopping: Stopping Training When Performance Stops Improving

### Overview
Early stopping is a crucial technique in the realm of machine learning that aims to halt the training process of a model when its performance stops improving on a validation set. This is particularly instrumental in preventing overfitting, where a model becomes excessively tailored to the training data, leading to suboptimal performance on unseen data.

### Why Use Early Stopping?

The primary goal of early stopping is to find an optimal balance between underfitting and overfitting:

- **Underfitting:** When the model is too simple and fails to capture the underlying patterns in the data.
- **Overfitting:** When the model is excessively complex, capturing noise in the training data as if it were a legitimate pattern.

By monitoring the performance on a validation set, early stopping allows the training process to stop once improvements begin to plateau, thereby preserving a model that generalizes well to new data.

### How Early Stopping Works

1. Split the training data into training and validation sets.
2. Train the model on the training set while periodically evaluating performance on the validation set.
3. Track a performance metric (e.g., loss, accuracy) from the validation set.
4. Halt training when the performance metric stops improving for a predefined number of epochs (patience).

### Example Implementations

#### TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train / 255.0
x_val = x_val / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x_train, y_train, epochs=50, 
                    validation_data=(x_val, y_val), 
                    callbacks=[early_stopping])
```

In this example, the `EarlyStopping` callback from `tensorflow.keras.callbacks` is used to monitor the validation loss. Training stops if the validation loss does not improve for three consecutive epochs.

#### PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(50):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

In this example, the patience mechanism is manually implemented in PyTorch to halt training when the validation loss does not improve for a specified number of epochs.

### Related Design Patterns

1. **Cross-Validation**: This technique involves dividing the dataset into multiple folds and training the model on different subsets while validating on the remaining data. Early stopping can be applied during the cross-validation process to ensure robust model performance across different folds.
2. **Checkpointing**: This design pattern involves saving the model's state at certain points during training. When used with early stopping, it ensures that the best-performing model (based on validation metrics) is saved.
3. **Learning Rate Scheduling**: In some cases, instead of stopping training, the learning rate can be adjusted when improvements plateau. This can be combined with early stopping to fine-tune the model's performance.

### Additional Resources

- [Early Stopping in Keras](https://keras.io/api/callbacks/early_stopping/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Understanding Early Stopping in Machine Learning](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)

### Summary

Early stopping is an efficient and straightforward method to prevent overfitting by halting the training process when the model's performance on a validation set stops improving. This technique is widely supported across various machine learning frameworks like TensorFlow and PyTorch. By adopting early stopping, along with related patterns like cross-validation, checkpointing, and learning rate scheduling, one can develop robust machine learning models that generalize well to unseen data.



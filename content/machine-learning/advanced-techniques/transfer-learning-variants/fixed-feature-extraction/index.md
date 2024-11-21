---
linkTitle: "Fixed Feature Extraction"
title: "Fixed Feature Extraction: Using Pre-trained Models as Fixed Feature Extractors"
description: "Leveraging pre-trained models to extract features for new tasks without retraining the model."
categories:
- Advanced Techniques
tags:
- Feature Extractors
- Transfer Learning
- Pre-trained Models
- Representation Learning
- Machine Learning
date: 2023-10-15
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/transfer-learning-variants/fixed-feature-extraction"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Fixed Feature Extraction: Using Pre-trained Models as Fixed Feature Extractors

### Introduction

Fixed Feature Extraction is a transfer learning technique where a pre-trained model is used as a fixed feature extractor. This involves leveraging the learned representations of a pre-existing model to extract features for a new task without retraining the original model. Typically, the pre-trained model is trained on a large and diverse dataset, enabling it to learn robust and generalizable features. Utilizing these features can significantly reduce computational costs and improve performance on the new task.

### How it works

The core idea is straightforward:
1. **Select a Pre-trained Model**: Choose an appropriate pre-trained model, typically trained on a large dataset.
2. **Remove Final Layers**: Discard the last few layers of the pre-trained model, usually the classification layers.
3. **Extract Features**: Use the remaining layers to extract features from the input data.
4. **Train New Classifier**: Train a new model (often a simple classifier) using the extracted features for the new task.

### Example: Image Classification with TensorFlow

Below is an example of using a pre-trained VGG16 model as a fixed feature extractor for an image classification task with TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # Assume 10 classes for the new task

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
```

### Examples in Other Frameworks

#### PyTorch

```python
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

vgg16 = models.vgg16(pretrained=True)

for param in vgg16.parameters():
    param.requires_grad = False

vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.classifier.parameters(), lr=0.001)

train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
val_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
val_dataset = datasets.ImageFolder('data/validation', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for epoch in range(10):
    vgg16.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Validation accuracy calculation
    vgg16.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          
    print(f"Epoch [{epoch+1}/10], Validation Accuracy: {100 * correct / total}%")
```

### Related Design Patterns

- **Fine-Tuning**: Unlike Fixed Feature Extraction, Fine-Tuning involves partially or fully updating the pre-trained model's weights on the new task, generally using a smaller learning rate.
- **Model Ensemble**: Combining the outputs of multiple models (potentially including feature extractors and retrained models) to improve accuracy and generalization.
- **Zero-Shot Learning**: Using pre-trained models to recognize or classify data for tasks they were not explicitly trained on, leveraging the model’s ability to generalize from known classes to unseen ones.

### Additional Resources

- [Transfer Learning with TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Transfer Learning & Fine-Tuning with PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet

### Summary

Fixed Feature Extraction is a powerful and efficient technique within the broader framework of transfer learning. By leveraging pre-trained models as fixed feature extractors, developers can significantly reduce training time and computational requirements for new tasks. This approach is particularly advantageous when dealing with limited data or when high-quality pre-trained models are available. This design pattern exemplifies the practicality of reusing advanced models to accelerate machine learning workflows and improve overall model performance on diverse tasks.

---

This article covers the essentials of the Fixed Feature Extraction design pattern, detailing its implementation and providing code examples in both TensorFlow and PyTorch. Related design patterns and additional resources are offered to guide further learning, ending with a succinct summary of its benefits and applications.

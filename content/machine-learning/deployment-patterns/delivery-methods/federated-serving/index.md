---
linkTitle: "Federated Serving"
title: "Federated Serving: Serving Models in a Federated Manner Across Distributed Systems"
description: "Detailed explanation of the Federated Serving design pattern, its examples, related patterns, resources, and final summary."
categories:
- Deployment Patterns
tags:
- machine learning
- federated learning
- deployment
- distributed systems
- model serving
date: 2024-10-19
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/deployment-patterns/delivery-methods/federated-serving"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In modern artificial intelligence (AI) and machine learning (ML) scenarios, the complexity and scale of models often require innovative strategies for training and deployment. Federated Serving leverages the concept of **federated learning** to deploy machine learning models across distributed systems. This strategy enhances scalability, privacy, and resilience by enabling model serving on multiple nodes without necessitating centralization.

### Concept and Benefits

Federated Serving involves deploying models to multiple federated edges (devices or servers) that collaboratively operate to serve predictions. This pattern allows for:
- **Scalability**: Distributed architectures can handle larger volumes of predictions by balancing the load.
- **Privacy**: Since data remains localized, privacy concerns linked to data centralization are mitigated.
- **Resilience**: Loss of a few nodes doesn't incapacitate the system, ensuring higher availability.

## Detailed Example

Let's explore an example using Python and TensorFlow Federated (TFF).

### Prerequisites

Ensure you have TensorFlow Federated installed:
```bash
pip install tensorflow_federated
```

### Model Training and Serving

In a simplified federated setup, we can train a model across distributed nodes and then serve it.

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

def create_federated_data():
    return [tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform([500, 784]), tf.random.uniform([500], maxval=10, dtype=tf.int64))
    ).batch(20) for _ in range(10)]

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=create_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

state = iterative_process.initialize()  # Initialize server state

federated_data = create_federated_data()

for _ in range(10):
    state, metrics = iterative_process.next(state, federated_data)


model_weights = iterative_process.get_model_weights(state)
final_model = create_model()
final_model.set_weights(model_weights.model_weights)
final_model.save('federated_model')



# tensorflow_model_server --rest_api_port=8501 --model_name=federated_model --model_base_path=/path/to/federated_model/ 

import requests

data = {
    "instances": [[0.0]*784]
}
response = requests.post('http://localhost:8501/v1/models/federated_model:predict', json=data)
print(response.json())
```

This Python example outlines federated model training and serving, where TensorFlow Federated (TFF) accumulates knowledge from different nodes and TensorFlow Serving facilitates distributed, scalable predictions.

## Related Design Patterns

### 1. **Federated Learning**
Federated Learning trains a global model by aggregating updates from localized models, enabling ML on decentralized data. Federated Serving extends this concept by focusing on model deployment and serving.

### 2. **Active Learning**
Active Learning strategically selects critical data points to be labeled for training, optimizing model accuracy. Combined with Federated Learning, this ensures efficient use of local data.

### 3. **Model Parallelism**
This pattern partitions a model across multiple devices to facilitate parallel computations, reducing latency. Federated Serving aggregates multiple models across devices, enhancing model distribution.

## Additional Resources

- **TensorFlow Federated Documentation**: [TensorFlow Federated](https://www.tensorflow.org/federated)
- **Federated Learning Research**: "Communication-Efficient Learning of Deep Networks from Decentralized Data" by H.B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Arcas.
- **TensorFlow Serving Documentation**: [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

## Summary

Federated Serving effectively deploys machine learning models across distributed systems, ensuring high scalability, enhanced privacy, and robustness. Integrating federated learning techniques with scalable serving frameworks like TensorFlow Serving, this pattern addresses many challenges associated with centralized model serving infrastructure. By adopting Federated Serving, organizations can manage decentralized user data while ensuring efficient, scalable, and resilient model deployment.

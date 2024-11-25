---
linkTitle: "Cross-Silo Federated Learning"
title: "Cross-Silo Federated Learning: Federated Learning Across Organizations or Institutions"
description: "Federated learning enables collaboration across organizations or institutions by training models on local data without sharing the data itself."
categories:
- Emerging Fields
tags:
- Federated Learning
- Privacy
- Data Security
- Collaboration
- Distributed Computing
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/emerging-fields/federated-learning-variants/cross-silo-federated-learning"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


### Introduction

Cross-Silo Federated Learning (CSFL) allows multiple organizations or institutions to collaboratively train machine learning models without sharing their raw data. This approach enhances privacy, data security, and allows leveraging diverse datasets, which are crucial in sectors like healthcare, finance, and research.

### Concept and Working

Federated learning consists of several clients (organizations) and a central server. Each client trains a local model on its own dataset, and only the trained parameters (such as weights and biases) are shared with the central server. The server aggregates these parameters to form a global model, which is then redistributed among clients.

Here's a step-by-step workflow of CSFL:

1. **Initialization**: A global model is initialized and distributed to all participating organizations.
2. **Local Training**: Each organization trains the global model on its local data for several epochs.
3. **Parameter Update**: The local model parameters are sent back to the central server.
4. **Aggregation**: The central server aggregates the local parameters to update the global model.
5. **Distribution**: The updated global model is sent back to the organizations to begin another round of training.
6. **Iteration**: Steps 2-5 are repeated until the model converges.

### Mathematical Formulation

The general concept of federated averaging (FedAvg), as introduced by McMahan et al., can be formulated mathematically:

Let \\( w \\) represent the model parameters. The objective is to minimize the global loss:

{{< katex >}}
\min_w \sum_{k=1}^K \frac{n_k}{n} L_k(w)
{{< /katex >}}

where:
- \\( K \\) is the number of organizations.
- \\( n_k \\) is the number of data points in organization \\( k \\).
- \\( n = \sum_{k=1}^K n_k \\) is the total number of data points.
- \\( L_k(w) \\) is the loss of the model on the \\( k^\text{th} \\) organization's data.

The updated parameter \\( w \\) can be obtained using:

{{< katex >}}
w_{t+1} \leftarrow \sum_{k=1}^K \frac{n_k}{n} w_t^k
{{< /katex >}}

where \\( w_t^k \\) represents the parameters after training at the \\( t^\text{th} \\) iteration on the \\( k^\text{th} \\) organization's local data.

### Example Using TensorFlow Federated

Below is an example of implementing Cross-Silo Federated Learning using TensorFlow Federated (TFF):

```python
import tensorflow as tf
import tensorflow_federated as tff

def model_fn():
    return tff.learning.from_keras_model(
        keras_model=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ]),
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

federated_averaging = tff.learning.build_federated_averaging_process(model_fn)

state = federated_averaging.initialize()

federated_data = [...]  # Replace with federated datasets from multiple organizations

for round_num in range(num_rounds):
    state, metrics = federated_averaging.next(state, federated_data)
    print(f'Round {round_num + 1}, Metrics={metrics}')
```

### Related Design Patterns

1. **Vertical Federated Learning**:
   - **Description**: Different organizations hold different features of the same user. They collaborate without sharing their feature sets.
   - **Example**: Banks and insurance companies co-train models with financial and demographic data respectively.

2. **Horizontal Federated Learning**:
   - **Description**: Organizations have the same features but on different users.
   - **Example**: Multiple hospitals with patient data collaborate to train a robust healthcare model.

3. **Federated Transfer Learning**:
   - **Description**: Combines federated learning with transfer learning. Useful when datasets are small but essential features are similar.
   - **Example**: Using a pre-trained model on a large public dataset and refining it via federated learning.

### Additional Resources

- **Research Papers**
  - McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). "Communication-efficient learning of deep networks from decentralized data." 
  - Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., ... & Yang, Q. (2021). "Advances and open problems in federated learning."

- **Libraries and Frameworks**
  - [TensorFlow Federated (TFF)](https://www.tensorflow.org/federated)
  - [FedML: Federated Machine Learning Framework](https://fedml.ai/)

### Summary

Cross-Silo Federated Learning is a powerful variant of federated learning that enables collaboration across organizational boundaries, preserving privacy and data security. Through aggregated model training, CSFL allows organizations to build robust models by harnessing collective data without exposing sensitive information. This design pattern holds immense potential in the era of data-driven industries, especially where privacy and security are paramount.

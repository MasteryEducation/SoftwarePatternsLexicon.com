---
linkTitle: "Hybrid Federated Learning"
title: "Hybrid Federated Learning: Combining Horizontal and Vertical Federated Learning Characteristics"
description: "A comprehensive guide on the hybrid federated learning design pattern, which combines advantages of both horizontal and vertical federated learning."
categories:
- Emerging Fields
- Federated Learning Variants
tags:
- Machine Learning
- Federated Learning
- Data Privacy
- Distributed Systems
- Hybrid Models
date: 2023-10-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/emerging-fields/federated-learning-variants/hybrid-federated-learning"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

**Hybrid Federated Learning** is a sophisticated technique that amalgamates the characteristics of both horizontal and vertical federated learning. Horizontal federated learning (HFL) refers to federated learning where datasets are partitioned by samples. Conversely, vertical federated learning (VFL) operates on data that is partitioned by features. Hybrid federated learning aims to combine these two paradigms to leverage the strengths of each and overcome their individual limitations.

### Horizontal Federated Learning (HFL)

In horizontal federated learning, datasets share the same feature space but differ in their samples. Collaboration involves aggregating local models without raw data exchange, preserving data privacy.

### Vertical Federated Learning (VFL)

Vertical federated learning, on the other hand, involves datasets that share the same sample space but differ in feature sets. Different parties contribute their unique feature sets for the same group of samples.

## Hybrid Federated Learning: The Best of Both Worlds

Hybrid federated learning involves scenarios in which datasets are partitioned both horizontally and vertically, combining the sample-based partitioning of HFL and feature-based partitioning of VFL.

### Characteristics

* **Cross-Domain Learning**: Enables learning from multiple domains where both feature spaces and sample spaces are partitioned.
* **Enhanced Privacy**: Higher data security by using aligned encryption techniques across both horizontal and vertical partitions.
* **Increased Model Accuracy**: By integrating more features and samples, hybrid federated learning can potentially yield more accurate models.

## Examples

### Example 1: Collaborative Healthcare Analysis

Consider a scenario involving several hospitals and healthcare providers:

- **Horizontal Partition**: Each hospital has patient records (e.g., Hospital A with patients 1-1000, Hospital B with patients 1001-2000).
- **Vertical Partition**: Each hospital has different features/documents related to patients such as medical records, prescription records, lab tests, etc.

Using hybrid federated learning, hospitals can collaboratively build a comprehensive predictive healthcare model without exchanging any sensitive patient information.

### Example 2: Financial Industry

In the financial industry, multiple banks could collaborate, each holding different client information.

- **Horizontal Partition**: Different banks hold records for different clients.
- **Vertical Partition**: Each bank holds different types of financial data (e.g., transaction records, investment portfolios).

This can lead to more accurate predictions or risk assessments by sharing synthesized insights rather than raw data.

## Implementation: TensorFlow Federated Example

Below is a high-level code example using TensorFlow Federated to illustrate hybrid federated learning:

```python
import tensorflow_federated as tff

def model_fn():
    # Define the model here
    return tff.learning.from_keras_model(
        keras_model=model,
        dummy_batch=X, # Add representative samples
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.Accuracy()])

client_data = tff.simulation.datasets.ClientData(...)
federated_train_data = [client_data.create_tf_dataset_for_client(x) for x in client_ids]

iterative_process = tff.learning.build_federated_averaging_process(model_fn)

state = iterative_process.initialize()

for round_num in range(1, total_rounds):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f'Round {round_num}, Metrics={metrics}')
```

## Related Design Patterns

### 1. **Cross-Silo Federated Learning**

Cross-silo federated learning involves training across organizational or institutional silos. The scope can be restricted to HFL, VFL, or hybrid settings.

### 2. **Cross-Device Federated Learning**

This pattern focuses on using edge devices like smartphones or IoT devices in HFL mode. These devices contribute to the model training without sharing data, emphasizing decentralized learning.

### 3. **Federated Transfer Learning**

Federated transfer learning adapts pre-trained models to different datasets in a federated environment. Useful in scenarios with limited data or computational resources.

## Additional Resources

### Research Papers

1. Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine Learning: Concept and Applications. *ACM Transactions on Intelligent Systems and Technology (TIST)*, 10(2), 1-19.
   
2. Harden, N., Rieke, N., Meynet, J., Klimku, G., Roth, H. (2020). Training and evaluating medical models with federated learning. 2020 *IEEE 17th International Symposium on Biomedical Imaging (ISBI)*.

### Tutorials and Courses

- **Federated Learning with TensorFlow** ([TensorFlow.org](https://www.tensorflow.org/federated))
- **Federated Learning Specialization** on Coursera by *Andrew Ng* and *DeepLearning.AI*

## Summary

Hybrid Federated Learning merges horizontal and vertical federated learning paradigms to produce models that can leverage combined datasets for enhanced accuracy while ensuring data privacy. This pattern is particularly useful in cross-domain applications where both sample and feature spaces are shared among multiple parties. By embracing this method, industries such as healthcare and finance can collaborate on more robust and secure predictive models.

Exploring additional resources and related design patterns provides a comprehensive understanding and further reading for practical implementations.

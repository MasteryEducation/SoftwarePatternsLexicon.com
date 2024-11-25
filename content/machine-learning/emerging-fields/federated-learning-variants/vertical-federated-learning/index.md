---
linkTitle: "Vertical Federated Learning"
title: "Vertical Federated Learning: Federated Learning with Different Feature Sets"
description: "An in-depth analysis of Vertical Federated Learning, a federated learning approach involving data with different features from different parties. Includes examples, related design patterns, additional resources, and a summary."
categories:
- Emerging Fields
tags:
- Federated Learning
- Machine Learning
- Data Privacy
- Distributed Computing
- Secure Computation
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/emerging-fields/federated-learning-variants/vertical-federated-learning"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Vertical Federated Learning: Federated Learning with Different Feature Sets

Vertical Federated Learning (VFL) is a branch of federated learning where multiple parties, each possessing a different subset of features for the same set of users, collaborate to train a unified machine learning model without sharing their raw data. This design pattern maintains data privacy and confidentiality by leveraging techniques such as secure multi-party computation (SMPC) and secure aggregation.

### Core Concept

In traditional centralized machine learning, data from disparate sources is combined into a single dataset that powers the training of a machine learning model. This paradigm presents notable privacy risks, given the necessity of aggregating sensitive data in a central repository. VFL offers a solution by allowing different organizations to collaboratively build models using their own data, which contains different features for an overlapping set of entities (e.g., customers). The key characteristic of VFL is that the datasets held by different organizations have the same entities (or rows) but disjoint sets of features (or columns).

### Example

Consider two companies, a bank and an e-commerce platform, that want to build a credit scoring model:

- **Bank’s Dataset:**
  - Features: Age, Annual Income, Credit History
  - Entities: Customer IDs shared with the e-commerce platform

- **E-commerce Platform’s Dataset:**
  - Features: Monthly Expenditure, Purchase History
  - Entities: Customer IDs shared with the bank

#### Python & PySyft Example

Using PySyft, an open-source federated learning library:

```python
import syft as sy
from syft.frameworks.torch.federated import utils
import torch

bank = sy.VirtualWorker(hook, id="bank")
ecommerce = sy.VirtualWorker(hook, id="ecommerce")

bank_data = torch.tensor([[56, 70000, 1], [33, 80000, 0], [29, 60000, 0],[45, 50000, 1]]) # Age, Annual Income, Credit History
ecommerce_data = torch.tensor([[1000, 25], [2500, 45], [1500, 30], [3000, 50]]) # Monthly Expenditure, Purchase History

bank_data_ptr = bank_data.send(bank)
ecommerce_data_ptr = ecommerce_data.send(ecommerce)

client_ids = [0, 1, 2, 3]
aggregated_data = [bank_data_ptr[idx] + ecommerce_data_ptr[idx] for idx in client_ids]
model_input = utils.secure_aggregation(aggregated_data)

# Note: Simplified Example - missing several steps involving synchronization and model training process
model = ... # Placeholder for model definition
```

### Related Design Patterns

1. **Horizontal Federated Learning (HFL):**
   - **Description:** Involves multiple organizations that have the same types of data (features) for different entities. Each organization holds data covering different users.
   - **Example:** Several hospitals with patient health records with same schema collaborate.

2. **Federated Averaging (FedAvg):**
   - **Description:** Aggregates model parameters (rather than raw data) from different clients participating in VFL or HFL to update a global model iteratively.
   - **Example:** Google’s use case with Android devices updating language models.

3. **Federated Stochastic Gradient Descent (FedSGD):**
   - **Description:** A variant where each client computes gradients on their local data which are then aggregated centrally.
   - **Example:** Training a global model in a cross-silo federated learning setup.

### Additional Resources

- **Research Papers:**
  - Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). "Federated Machine Learning: Concept and Applications."
  - Hardy, S., Berend, Y., Bontemps, Y., Blatt, Altman, T., & Tsourtou, M. (2017). "Private federated learning with anonymization."

- **Libraries:**
  - [PySyft](https://github.com/OpenMined/PySyft): Python library for secure, privacy-preserving machine learning.
  - [TensorFlow Federated (TFF)](https://www.tensorflow.org/federated): A framework for computing machine learning models on decentralized data.

### Summary

Vertical Federated Learning (VFL) circumvents the need to create a unified data repository by enabling multiple parties to collaboratively train models using different features for the same set of users while maintaining privacy. VFL employs advanced cryptographic techniques, such as SMPC, to safeguard data privacy during the model training process. This pattern is particularly beneficial for industries like finance and healthcare, where data privacy is paramount, and can be integrated into broader federated learning architectures such as HFL and FedAvg.

By leveraging VFL, disparate organizations can inform their machine learning models with rich, diverse data features while upholding critical privacy standards. Such approaches underpin the future of collaborative, privacy-preserving artificial intelligence.

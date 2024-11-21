---
linkTitle: "Synthetic Minority Over-sampling (SMOTE)"
title: "Synthetic Minority Over-sampling (SMOTE): Handling Class Imbalance with Synthetic Samples"
description: "A detailed exploration of the SMOTE design pattern for creating synthetic samples to address class imbalance in datasets."
categories:
- Data Management Patterns
- Data Augmentation in Specific Domains
tags:
- SMOTE
- Data Augmentation
- Class Imbalance
- Synthetic Samples
- Machine Learning
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-management-patterns/data-augmentation-in-specific-domains/synthetic-minority-over-sampling-(smote)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Synthetic Minority Over-sampling (SMOTE): Handling Class Imbalance with Synthetic Samples

Class imbalance is a common challenge in machine learning, where one class vastly outnumbers other classes. This imbalance can lead to biased models that perform poorly on minority classes. Synthetic Minority Over-sampling Technique (SMOTE) is a widely used strategy to address this imbalance by generating synthetic samples for the minority classes.

### What is SMOTE?

SMOTE stands for Synthetic Minority Over-sampling Technique. It is a data augmentation method that creates synthetic data points for minority classes to balance the class distribution in a dataset. Unlike random oversampling, which duplicates existing instances, SMOTE generates new, synthetic instances by interpolating between existing minority class instances.

### How Does SMOTE Work?

The SMOTE algorithm operates in the following steps:

1. **Selection of Minority Class Instances:** For each instance in the minority class, identify its *k* nearest neighbors using a specific distance metric, typically Euclidean distance.
2. **Synthetic Sample Generation:** For each minority instance, randomly select one of its *k* nearest neighbors. A new synthetic instance is created as follows:
   
   {{< katex >}} \mathbf{x}_{\text{new}} = \mathbf{x}_{\text{original}} + \lambda (\mathbf{x}_{\text{neighbor}} - \mathbf{x}_{\text{original}}) {{< /katex >}}
   
   where \\( \mathbf{x}_{\text{original}} \\) is the original minority instance, \\( \mathbf{x}_{\text{neighbor}} \\) is one of its nearest neighbors, and \\( \lambda \\) is a random number between 0 and 1.

### Implementing SMOTE

#### Example in Python using scikit-learn and imbalanced-learn

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, 
                           weights=[0.1, 0.9], flip_y=0, random_state=1)

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.5)
plt.legend()
plt.title('Original Dataset')
plt.show()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Class 0', alpha=0.5)
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Class 1', alpha=0.5)
plt.legend()
plt.title('Resampled Dataset with SMOTE')
plt.show()

print(f'Original dataset distribution: {Counter(y)}')
print(f'Resampled dataset distribution: {Counter(y_resampled)}')
```

### Related Design Patterns

1. **Random Under-sampling:** This design pattern involves randomly removing instances from the majority class to achieve a balanced class distribution. It's simpler than SMOTE but may result in the loss of important information.
2. **Adaptive Synthetic Sampling (ADASYN):** An extension of SMOTE, ADASYN not only creates synthetic samples but also pays more attention to those samples which are harder to classify.
3. **Cost-sensitive Learning:** Instead of balancing the data, this design pattern adjusts the learning algorithm to place higher penalties on misclassifications of the minority class.
4. **Ensemble Methods for Imbalance:** Using techniques like boosting, bagging, or stacking, ensemble methods can also help address class imbalance by combining multiple weak models into a strong one.

### Additional Resources

1. [Original SMOTE Paper by N. Chawla et al.](https://arxiv.org/abs/1106.1813)
2. [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [Machine Learning Mastery Guide on SMOTE](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

### Summary

SMOTE is a robust technique for addressing class imbalance in datasets by generating synthetic samples for the minority class. It helps in creating a more balanced dataset, which typically results in more robust and unbiased machine learning models. When implementing SMOTE, it's crucial to tune the nearest neighbors' parameter and evaluate the impact on the model's performance. Alongside other design patterns like random under-sampling and cost-sensitive learning, SMOTE provides a valuable toolset for data scientists dealing with imbalanced data.

By integrating synthetic samples judiciously, practitioners can significantly improve their model's performance on minority classes, thereby achieving fairer and more accurate machine learning solutions.

---
linkTitle: "Normalization"
title: "Normalization: Scaling Individual Samples to Have Zero Mean and Unit Variance"
description: "Normalization is a data preprocessing technique where individual samples are scaled to have zero mean and unit variance."
categories:
- Data Management Patterns
- Data Preprocessing
tags:
- Normalization
- Data Preprocessing
- Standardization
- Machine Learning
- Data Management
date: 2023-10-24
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-management-patterns/data-preprocessing/normalization"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

Normalization is a crucial data preprocessing step in machine learning that ensures the features of the data have zero mean and unit variance. This preprocessing technique helps in faster convergence during the training phase and improves the performance of the algorithm.

### Formula and Concept

Normalization can be mathematically represented as:

{{< katex >}}
X' = \frac{X - \mu}{\sigma}
{{< /katex >}}

where:
- \\( X \\) is the original data,
- \\( \mu \\) is the mean of the data,
- \\( \sigma \\) is the standard deviation of the data,
- \\( X' \\) is the normalized data.

### Importance of Normalization

Normalization ensures that the features contribute equally to the result without being dominated by any single feature due to differences in scale. This is particularly important for algorithms that compute distances between data points, such as K-Nearest Neighbors (KNN) and support vector machines (SVM).

## Implementations Across Languages and Frameworks

### Python with Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[10, 2.5, 6.5], [8, 4.0, 5.0], [9, 3.0, 7.0]])

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

print(normalized_data)
```

### R with `scale`

```r
data <- as.matrix(data.frame(feature1 = c(10, 8, 9), feature2 = c(2.5, 4.0, 3.0), feature3 = c(6.5, 5.0, 7.0)))

normalized_data <- scale(data)

print(normalized_data)
```

### TensorFlow

```python
import tensorflow as tf

data = tf.constant([[10, 2.5, 6.5], [8, 4.0, 5.0], [9, 3.0, 7.0]], dtype=tf.float32)

mean, variance = tf.nn.moments(data, axes=[0])
normalized_data = (data - mean) / tf.sqrt(variance)

print(normalized_data)
```

### SQL with Window Functions

```sql
WITH statistics AS (
    SELECT
        AVG(value) OVER (PARTITION BY feature) AS mean,
        STDDEV(value) OVER (PARTITION BY feature) AS stddev,
        value,
        feature
    FROM
        sample_data
)
SELECT
    (value - mean) / stddev AS normalized_value,
    feature
FROM
    statistics;
```

## Related Design Patterns

- **Standardization:** While normalization scales data to have zero mean and unit variance, standardization scales data to a given range, often [0, 1] or [-1, 1]. It is essential to understand the application context to choose between normalization and standardization.
  
- **Dimensionality Reduction:** Techniques such as Principal Component Analysis (PCA) often require normalized data for optimal performance to reduce data redundancy while retaining important information.

- **Feature Engineering:** Normalization is a fundamental part of feature engineering, essential for improving machine learning model performances.

## Additional Resources

- [Scikit-Learn Documentation on Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [TensorFlow Normalization Methods](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization)
- [Normalization in Data Science](https://towardsdatascience.com/data-normalization-7f52619325d8)

## Summary

Normalization is a vital preprocessing step aimed at scaling individual samples to have zero mean and unit variance, thus ensuring that no particular feature dominates. This aids significantly in the performance and convergence of machine learning models. Employing techniques like normalization, along with other design patterns such as standardization and dimensionality reduction, can profoundly impact the overall efficacy of a machine learning pipeline. Whether using Python, R, TensorFlow, or even SQL, applying normalization correctly is critical.

For developers and data scientists, understanding and utilizing normalization appropriately can streamline the data preparation phase, leading to more robust and reliable models.



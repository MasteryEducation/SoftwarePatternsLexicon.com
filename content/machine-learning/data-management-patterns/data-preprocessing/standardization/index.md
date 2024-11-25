---
linkTitle: "Standardization"
title: "Standardization: Transforming Features to Have Specific Properties"
description: "Standardization is a preprocessing technique in machine learning that transforms feature values of data to have specific statistical properties such as zero mean and unit variance."
categories:
- Data Management Patterns
tags:
- Data Preprocessing
- Scaling
- Normalization
- Feature Engineering
- Standardization
date: 2023-10-08
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-management-patterns/data-preprocessing/standardization"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview

Standardization is a data preprocessing technique used in machine learning to transform the features of a dataset so they align with a particular statistical distribution, typically a normal distribution with zero mean and unit variance. This can improve the performance of many algorithms, particularly those that rely on distance metrics, such as k-nearest neighbors (KNN) and support vector machines (SVM).

## The Need for Standardization
Raw data often has features with vastly different ranges, scales, and units, which can significantly skew the performance of certain machine learning models. Models that compute distances (e.g., KNN) or assume certain distributions of data (e.g., Gaussian Naive Bayes) can especially benefit from standardization.

Consider the following dataset prior to standardization:

| Feature 1 | Feature 2 |
|---|---|
| 1.0 | 2000 |
| 1.5 | 3000 |
| 2.0 | 8000 |

In this example, `Feature 2` ranges from 2000 to 8000, while `Feature 1` ranges from 1.0 to 2.0. This imbalance could cause the model training process to overemphasize the significance of `Feature 2`.

## Mathematical Formulation

Standardization transforms the data as follows:

{{< katex >}}
x' = \frac{x - \mu}{\sigma}
{{< /katex >}}

where:
- \\( x \\) is the original feature value
- \\( \mu \\) is the mean of the feature values
- \\( \sigma \\) is the standard deviation of the feature values
- \\( x' \\) is the standardized feature value

This formula ensures that the mean of the transformed data is 0 and the standard deviation is 1.

## Example Implementation

### Python with Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[1.0, 2000], [1.5, 3000], [2.0, 8000]])

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print(scaled_data)
```

### R with Scale Function

```R
data <- data.frame(Feature1 = c(1.0, 1.5, 2.0), Feature2 = c(2000, 3000, 8000))

scaled_data <- scale(data)

print(scaled_data)
```

### Spark with PySpark

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName("standardization").getOrCreate()

data = [(1.0, 2000), (1.5, 3000), (2.0, 8000)]
df = spark.createDataFrame(data, ["feature_1", "feature_2"])

assembler = VectorAssembler(inputCols=["feature_1", "feature_2"], outputCol="features")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
scaled_data = scaler_model.transform(df)

scaled_data.select("scaled_features").show()
```

## Related Design Patterns

1. **Normalization**: Similar to standardization, normalization aims to rescale feature values but to a different range, generally [0,1]. Both techniques are often used together based on the requirements of a specific model.
2. **Imputation**: Before standardizing, missing data must be addressed. Imputation handles missing values in the data, which is a necessary precursor to many data preprocessing steps, including standardization.
3. **Discretization**: Converts continuous features into categorical values. It can be used once the data is standardized to bin the data into broader categories.

## Additional Resources

1. **Scikit-Learn Documentation**: [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
2. **Spark MLlib**: [StandardScaler](https://spark.apache.org/docs/latest/ml-features.html#standardscaler)
3. **Machine Learning Mastery**: [StandardScaler Tutorial](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

## Summary

Standardization is an essential preprocessing step in machine learning that transforms each feature to have a zero mean and unit variance. This transformation ensures that the data follows a normal distribution, which optimizes the performance of machine learning algorithms that rely on distance measures or gradient descent optimization. By understanding and applying the standardization pattern correctly, you can significantly enhance the effectiveness of your machine learning models.

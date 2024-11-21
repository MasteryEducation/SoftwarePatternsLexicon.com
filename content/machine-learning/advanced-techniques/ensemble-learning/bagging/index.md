---
linkTitle: "Bagging"
title: "Bagging: Training Multiple Models on Different Subsets of Data and Combining Their Predictions"
description: "Bagging is an ensemble learning method that trains multiple models on different subsets of the data and combines their predictions to improve overall model performance."
categories:
- Ensemble Learning
- Advanced Techniques
tags:
- machine learning
- ensemble learning
- bagging
- bootstrap aggregation
- model averaging
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/ensemble-learning/bagging"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Bagging, short for Bootstrap Aggregating, is a powerful ensemble learning technique aimed at improving the stability and accuracy of machine learning algorithms. Bagging primarily focuses on variance reduction by training multiple models on different subsets of the dataset and then combining their predictions.

## How Bagging Works

The core idea behind bagging is to create multiple versions of a predictor and then average them. This generally works because the aggregation reduces the variance of the final model.

### Steps to Implement Bagging

1. **Bootstrap Sampling**: Generate 'n' different subsets of the training data by sampling with replacement.
2. **Training Models**: Train a model on each subset.
3. **Combining Predictions**: Aggregate the predictions from all models by techniques like averaging for regression or majority voting for classification.

### Mathematical Formulation

For each model \\( M_i \\), the training set \\( D_i \\) is a bootstrap sample from the original training set \\( D \\). The final output for a regression problem is an average of individual model predictions:

{{< katex >}}
\hat{f}(x) = \frac{1}{n} \sum_{i=1}^{n} M_i(x)
{{< /katex >}}

For classification, it commonly uses majority voting:

{{< katex >}}
\hat{Y}(x) = \text{mode}\{M_i(x) | i = 1, 2, ..., n\}
{{< /katex >}}

## Example Implementations

### Python with Scikit-Learn

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

base_model = DecisionTreeClassifier()

bagging = BaggingClassifier(base_estimator=base_model, n_estimators=50, random_state=42)

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
```

### R with caret

```R
library(caret)

data(iris)

set.seed(42)
trainIndex <- createDataPartition(iris$Species, p=0.7, list=FALSE)
dataTrain <- iris[trainIndex,]
dataTest <- iris[-trainIndex,]

model <- train(Species ~ ., data=dataTrain, method="treebag")

predictions <- predict(model, newdata=dataTest)
confusionMatrix(predictions, dataTest$Species)
```

## Related Design Patterns

### 1. **Boosting**
Boosting focuses on training weak learners sequentially, each one correcting the errors of its predecessor. Unlike bagging, which reduces variance, boosting reduces bias.

### 2. **Stacking**
Stacking involves training multiple types of models on the same dataset and using a meta-learner to combine their outputs. It aims to leverage the strengths of different types of models.

### 3. **Random Forest**
A specific type of bagging where each model is a decision tree and it introduces additional randomness by selecting a random subset of features for each split in the tree.

## Additional Resources

- [Scikit-Learn Documentation on Bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging)
- [Ensemble Methods: Foundations and Algorithms by Zhi-Hua Zhou](https://www.amazon.com/Ensemble-Methods-Algorithms-Chapman-Statistical/dp/1439830037)
- [Machine Learning Yearning by Andrew Ng](http://www.mlyearning.org/)

## Summary

Bagging is an effective ensemble technique designed to improve the performance of machine learning models by reducing variance through average smoothing. By training multiple base models on randomly sampled subsets of the original dataset, and then combining their outputs (e.g., averaging for regression or majority voting for classification), bagging can significantly enhance predictive accuracy compared to individual models alone. Widely applicable through libraries like Scikit-Learn and caret, bagging remains a cornerstone in the suite of advanced machine learning methods.

---

By following and adapting the bagging design pattern, practitioners can build more robust models that capitalize on the predictive power of ensembling traditionally weak learners into a strong and cohesive predictive system.

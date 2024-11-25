---
linkTitle: "Stratified K-Fold Validation"
title: "Stratified K-Fold Validation: Ensuring Each Fold in Cross-Validation Maintains the Class Proportion"
description: "An advanced cross-validation technique that ensures each fold maintains the class distribution found in the original dataset to provide more reliable validation results."
categories:
- Model Validation and Evaluation Patterns
tags:
- Machine Learning
- Cross-Validation
- Model Evaluation
- Stratified Sampling
- Python
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-cross-validation-techniques/stratified-k-fold-validation"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Overview

In machine learning, cross-validation is a commonly used approach to evaluate the performance of a model. However, when dealing with imbalanced datasets, traditional K-Fold Cross-Validation may not adequately represent all classes in each fold, leading to biased model evaluation results. Stratified K-Fold Validation addresses this issue by ensuring that each fold maintains the same class distribution as the original dataset. This technique is particularly useful for classification problems where preserving the class proportion in training and validation sets is crucial.

## How It Works

Stratified K-Fold Validation splits the dataset into K folds such that each fold maintains the proportion of each class label consistent with the original dataset. The model is trained and validated K times, each time using a different fold as the validation set and the remaining K-1 folds as the training set.

**Mathematical Representation:**

Let \\( D \\) be our dataset with \\( N \\) samples, each labeled with one of \\( C \\) possible classes. Denote \\( Y \\) as the corresponding class labels.

1. Split the dataset into K equally-sized (or nearly equally-sized) folds.
2. Ensure that the proportion of each class \\( c \\) in each fold \\( f \\) matches the proportion of that class in the entire dataset.

{{< katex >}}
\text{Proportion of class } c \text{ in fold } f \approx \frac{\sum_{i=1}^N \mathbb{I}(Y_i = c)}{N}
{{< /katex >}}

where \\( \mathbb{I} \\) is the indicator function.

## Implementation 

### Example in Python using scikit-learn

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

skf = StratifiedKFold(n_splits=5)

accuracies = []
model = LogisticRegression(solver='liblinear', multi_class='auto')

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

print("Accuracy scores for each fold: ", accuracies)
print("Mean accuracy: ", np.mean(accuracies))
```

### Example in R using caret package

```r
library(caret)

data(iris)

train_control <- trainControl(method="cv", number=5, classProbs=TRUE, 
                              summaryFunction=multiClassSummary, savePredictions="final")

model <- train(Species ~ ., data=iris, method="multinom", 
               trControl=train_control, preProcess=c("center", "scale"))

print(model)
```

## Related Design Patterns

1. **K-Fold Cross-Validation**: This is the non-stratified version of cross-validation. The dataset is split into K folds, but without maintaining the class proportions. It is simpler but can lead to biased results, especially with imbalanced datasets.

2. **Leave-One-Out Cross-Validation (LOOCV)**: Each fold consists of a single sample for validation, and the rest for training. It ensures that each data point is used for validation exactly once.

3. **Repeated K-Fold Cross-Validation**: This entails performing K-Fold Cross-Validation multiple times (e.g., with different random seeds for the splits), and taking the average of the results. It can help mitigate the variance that might be observed in a single execution of K-Fold Cross-Validation.

## Additional Resources

- [scikit-learn: Stratified K-Folds cross-validator](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [Cross-Validation Strategies for Imbalanced Datasets](https://www.machinelearningmastery.com/cross-validation-data-imbalanced-classification/)
- [K-Fold Cross-Validation with TensorFlow and Keras](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#k-fold_cross-validation)

## Summary

Stratified K-Fold Validation is a powerful technique in the toolbox of machine learning practitioners, particularly when dealing with imbalanced datasets. By ensuring that the class proportions are maintained in each fold, it provides more reliable and unbiased evaluation of the model's performance. Implementing Stratified K-Fold Validation in popular frameworks like scikit-learn and caret ensures that evaluations are both robust and consistent with real-world data distributions.

Integrating Stratified K-Fold in your model validation workflow can enhance the reliability of your evaluations, ultimately leading to models that generalize better and perform more reliably across different datasets.

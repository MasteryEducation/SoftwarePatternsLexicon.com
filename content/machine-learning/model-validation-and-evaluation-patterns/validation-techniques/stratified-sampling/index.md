---
linkTitle: "Stratified Sampling"
title: "Stratified Sampling: Ensuring that each class is accurately represented in the training and test sets"
description: "Stratified Sampling is a technique used to ensure that each class is proportionally represented in both the training and test sets during data splitting, which helps improve model validation and performance."
categories:
- Model Validation and Evaluation Patterns
tags:
- Stratified Sampling
- Data Splitting
- Model Validation
- Class Representation
- Supervised Learning
date: 2023-11-01
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/validation-techniques/stratified-sampling"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction
Stratified Sampling is a crucial technique in model validation, especially when dealing with imbalanced datasets. The primary aim is to maintain the class distribution in both the training and test sets such that each subset is a representative of the whole dataset. This design pattern helps in mitigating issues that arise due to underrepresented classes during model evaluation.

## Why Stratified Sampling?
When working with classification problems, particularly those with imbalanced classes, random splitting might yield a training or test set that doesn't accurately represent the original class distribution. This can lead to misleading performance metrics. Stratified Sampling addresses this by ensuring that each class in the dataset is proportionally represented in each split.

## Detailed Explanation

### Algorithm

1. **Partition Dataset:** Begin by partitioning the dataset based on the target variable classes.
2. **Calculate Proportions:** Calculate the proportion of each class.
3. **Resample:** Resample from each partition to ensure that each class proportion is maintained in the training and test sets.

### Formal Definition

For a dataset \\( D \\) with \\( n \\) instances, stratified sampling can be approached as follows:
1. Let \\( C \\) be the set of classes in \\( D \\).
2. Let \\( D_c \\) be the subset of \\( D \\) belonging to class \\( c \in C \\).
3. If the train-test split ratio is \\( r \\) (e.g., 0.8 for 80% training and 20% testing),
   * Assign \\( r \times |D_c| \\) samples from \\( D_c \\) to the training set.
   * Assign \\( (1-r) \times |D_c| \\) samples from \\( D_c \\) to the test set.

### Mathematical Formulation
Given:

* \\( N \\): Total number of instances.
* \\( N_i \\): Number of instances in class \\( i \\) where \\(i \in \{1, 2, ..., k\}\\) and \\( k \\) is the number of classes.
* \\( R \\): Split ratio (e.g., 0.8 for 80% training).

Training Samples (\\( T \\)) and Test Samples (\\( S \\)) should satisfy:
{{< katex >}} T_i = R \times N_i {{< /katex >}}
{{< katex >}} S_i = (1 - R) \times N_i {{< /katex >}}

where \\( T_i \\) and \\( S_i \\) are the number of instances from class \\( i \\) in the training and test sets respectively.

## Examples

### Python with Scikit-learn
Here is an example using Scikit-learn’s `train_test_split` function:

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")
```

### R with Caret
In R, stratified sampling can be performed using the `caret` package:

```r
library(caret)
data(iris)

set.seed(42)
trainIndex <- createDataPartition(iris$Species, p = .8, 
                                  list = FALSE, 
                                  times = 1)
irisTrain <- iris[ trainIndex,]
irisTest  <- iris[-trainIndex,]

table(irisTrain$Species)
table(irisTest$Species)
```

### Python with Pandas
Using Pandas to manually create stratified samples:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'feature1': [i for i in range(100)],
    'feature2': [i % 5 for i in range(100)],
    'label': [0 if i < 90 else 1 for i in range(100)]
})

train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print(train['label'].value_counts(normalize=True))
print(test['label'].value_counts(normalize=True))
```

## Related Design Patterns

### K-Fold Cross Validation
K-Fold Cross Validation is another widely-used validation technique that involves splitting the dataset into \\( k \\) subsets (folds) and using each fold as a test set while training on the remaining \\( k-1 \\) folds. Stratified K-Fold Cross Validation is the stratified version ensuring each fold has the class distribution similar to the original dataset.

### Oversampling and Undersampling
While Stratified Sampling ensures proportional class representation in splits, techniques like oversampling (e.g., SMOTE) or undersampling adjust the class distribution to handle imbalanced datasets and might be used before applying stratified split.

## Additional Resources

1. [Scikit-learn Documentation - Stratified Sampling](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
2. [Towards Data Science - Stratified Sampling](https://towardsdatascience.com/stratified-sampling-in-machine-learning-pros-and-cons-ad7eef29ff28)
3. [Kaggle - Hands-On Stratified Sampling](https://www.kaggle.com/code/alexisbcook/stratified-sampling)

## Summary
Stratified Sampling is a fundamental design pattern in model validation for ensuring each class is proportionally represented in both the train and test sets. This is particularly essential when working with imbalanced datasets to avoid misleading metrics and ensure reliable model evaluation. While it works in tandem with many other techniques like K-Fold Cross Validation and sampling methods, its primary objective remains providing a more accurate reflection of the original data distribution, impacting the model's performance evaluation positively.

This method is easy to implement using various data-science tools and programming languages, making it an indispensable part of any data scientist’s toolkit.

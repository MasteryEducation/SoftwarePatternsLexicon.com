---
linkTitle: "Cross-Validation"
title: "Cross-Validation: An Essential Technique for Model Validation"
description: "A comprehensive guide on Cross-Validation, a technique for splitting data into several folds and rotating them as the test set, with examples in various programming languages."
categories:
- Model Validation and Evaluation Patterns
tags:
- Cross-Validation
- Model Evaluation
- Machine Learning
- Data Splits
- Performance Metrics
date: 2023-10-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/validation-techniques/cross-validation"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Cross-validation is a powerful technique used in the context of machine learning (ML) to evaluate the efficacy of a model. By splitting the data into several folds and using each fold as a part of the training and testing process, cross-validation provides a robust measure for model assessment and helps in avoiding overfitting.

## Key Concepts and Definitions

### Cross-Validation

Cross-validation involves dividing your dataset into two segments: one used to train a model and another used to validate it. In k-fold cross-validation, for example, the initial data is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k−1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation.

### Fold
A fold is one of the k partitions that data is divided into in the k-fold cross-validation process. 

### Rotation
Rotation in this context refers to the process of systematically rotating which subset of the data is used for validation and which subsets are used for training in each iteration.

## Step-by-Step Workflow

1. **Divide the dataset into k subsets (folds).**
2. **For each unique fold:**
    - Consider the fold as the test set.
    - Set aside the remaining folds as the training set.
    - Train the model on the training set.
    - Evaluate the model on the test set.
    - Retain the evaluation score and discard the model.
3. **Summarize the skill of the model using the sample of model evaluation scores.**

## Mathematical Representation

- Let \\( D \\) be the entire dataset.
- Split \\( D \\) into \\( k \\) subsets \\( D_1, D_2, \ldots, D_k \\).
- For each subset \\( D_i \\):
  - Let \\( T \\) denote the union of all subsets except \\( D_i \\): \\( T = \bigcup\limits_{j \neq i} D_j \\).
  - Train the model on \\( T \\) and test on \\( D_i \\).

The average performance metric over all folds can be represented mathematically as: 

{{< katex >}}
\text{CV\_Score} = \frac{1}{k} \sum\limits_{i=1}^k M(D_i \| T)
{{< /katex >}}

where \\( M \\) is the performance metric like accuracy, RMSE, etc.

## Example Implementations

### Python with Scikit-Learn

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

results = cross_val_score(model, X, y, cv=kfold)

print(f'Cross-Validation Accuracy Scores: {results}')
print(f'Mean accuracy: {results.mean()}')
```

### R with caret package

```R
library(caret)
library(randomForest)

data(iris)

train_control <- trainControl(method="cv", number=5)

model <- train(Species~., data=iris, method="rf", trControl=train_control)

print(model)
```

### Java with Weka

```java
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

public class CrossValidation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("iris.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        RandomForest rf = new RandomForest();
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(rf, data, 5, new java.util.Random(1));
        System.out.println(eval.toSummaryString());
    }
}
```

## Related Design Patterns

1. **Bootstrap Aggregating (Bagging)**:
   - Combines multiple models (usually of the same type) trained on overlapping subsets of data, created via sampling with replacement to reduce variance.
  
2. **Stratified K-Fold Cross-Validation**:
   - Similar to k-fold cross-validation but it ensures that each fold maintains the class distribution similar to the original dataset. It is particularly useful with imbalanced datasets.
  
3. **Leave-One-Out Cross-Validation (LOOCV)**:
   - A special case of k-fold cross-validation where k is equal to the number of data points. It's computationally expensive but can be useful with very small datasets.

## Additional Resources

- [Scikit-Learn Cross-Validation Documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Carets Training Control Documentation](https://topepo.github.io/caret/using-your-own-model-functions.html)
- [Weka Classifiers Evaluation Documentation](https://weka.sourceforge.io/doc.dev/weka/classifiers/Evaluation.html)

## Summary

Cross-validation is an essential technique in machine learning for evaluating the performance of a model in a robust manner. By distributing the data into multiple folds and rotating them as the test set, it ensures that every data point has a chance to be in the validation set, thus providing a better generalized performance measure. This pattern helps in mitigating overfitting and offers a stable estimate of model skill. Adopting cross-validation in your development pipeline can significantly enhance the reliability of your models.

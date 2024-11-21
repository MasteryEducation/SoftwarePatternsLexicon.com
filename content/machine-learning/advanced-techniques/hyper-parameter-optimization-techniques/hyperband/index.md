---
linkTitle: "Hyperband"
title: "Hyperband: Efficient Hyperparameter Optimization using Adaptive Resource Allocation"
description: "Hyperband optimizes hyperparameters by efficiently allocating resources based on intermediate performance assessments, reducing training time and resource wastage."
categories:
- Advanced Techniques
- Hyper-Parameter Optimization Techniques
tags:
- Hyperband
- Hyperparameter Optimization
- Machine Learning
- Resource Allocation
- Bayesian Optimization
date: 2023-10-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/hyper-parameter-optimization-techniques/hyperband"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

Hyperband is an advanced technique for hyperparameter optimization in machine learning that focuses on efficiently allocating computational resources. This method evaluates multiple configurations rapidly by adaptively allocating resources based on their intermediate performance, thereby significantly reducing the time and computational cost involved in finding the optimal hyperparameters.

## Overview

### What is Hyperband?

Hyperband builds on the principles of multi-armed bandit algorithms and introduces a novel strategy to hyperparameter optimization through adaptive resource allocation. It aims to identify the most promising hyperparameter configurations early in the process, which helps in avoiding extensive evaluations of underperforming models.

### Why Use Hyperband?

Hyperparameter tuning is often expensive, especially for large datasets and complex models. Traditional methods like grid search and random search can be computationally intensive. Hyperband, on the other hand, dynamically allocates resources and intelligently prunes poorly performing trials, leading to faster convergence as well as more efficient use of computational resources.

## The Hyperband Algorithm

### Steps:

1. **Budget Settings**:
   Define a maximum resource budget `R` (e.g., iterations, epochs).
   
2. **Bracket Definition**:
   Divide the search space into different "brackets". Each bracket has different configurations and budgets.

3. **Successive Halving**:
   Run configurations with increasing amounts of resources. Gradually prune poor performers based on their intermediate results.

Mathematically, given a maximum budget `R` and a reduction factor `\eta`, Hyperband splits the resources `R` across different `s_max` brackets:

{{< katex >}} s_{\text{max}} = \left\lfloor \log_\eta (R) \right\rfloor {{< /katex >}}

Within each bracket `s`, starting with $n_i$ configurations:

{{< katex >}} n_i = \left\lceil \frac{s_{\text{max}}}{s+1} \right\rceil \eta^s {{< /katex >}}

For each configuration, the budget $r_i$ (resources) allocated is:

{{< katex >}} r_i = R \eta^{-s} {{< /katex >}}

### Python Example with Sklearn and Hyperopt

The following example demonstrates how to implement Hyperband for optimizing the hyperparameters of a Support Vector Machine (SVM) classifier using `hyperopt`.

```python
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from hyperopt import STATUS_OK
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def objective(params):
    clf = SVC(**params)
    iris = load_iris()
    X, y = iris.data, iris.target
    accuracy = cross_val_score(clf, X, y, scoring='accuracy').mean()

    return {'loss': -accuracy, 'status': STATUS_OK}

space = {
    'C': hp.loguniform('C', -6, 0),
    'gamma': hp.loguniform('gamma', -6, 0),
    'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best parameters found: ", best)
```

### Related Design Patterns

#### Bayesian Optimization
- **Description**: Uses probabilistic models to predict the performance of hyperparameters and choose the next set of parameters to evaluate based on expected improvement or other criteria.
- **Benefit**: Bayesian Optimization is more sample-efficient compared to random or grid search as it leverages the performance predictions to focus on the most promising hyperparameters.

#### Random Search
- **Description**: Samples hyperparameters from a defined space stochastic way.
- **Benefit**: Simple to implement and can be surprisingly effective, especially when only a few hyperparameters need tuning.

### Additional Resources

- [Original Hyperband Paper](https://arxiv.org/abs/1603.06560)
- [Hyperopt Documentation](https://github.com/hyperopt/hyperopt)
- [Practical Guide to Hyperparameter Optimization in Python](https://www.kdnuggets.com/2020/07/hyperparameter-tuning-machine-learning-models-scikit-learn.html)
- [Bayesian Optimization vs Hyperband (blog)](https://www.optimalflow.com/blog/bayesian-optimization-vs-hyperband/)

## Summary

Hyperband stands out as an efficient hyperparameter optimization method by effectively leveraging adaptive resource allocation. It improves upon traditional techniques like grid and random search by intelligently focusing computational efforts on the most promising hyperparameter configurations detected early in the training process. By incorporating Hyperband, data scientists can significantly reduce training time and computational costs while still achieving high model performance.



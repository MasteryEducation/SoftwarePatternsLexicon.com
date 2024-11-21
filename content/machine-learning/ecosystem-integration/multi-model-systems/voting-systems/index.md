---
linkTitle: "Voting Systems"
title: "Voting Systems: Using multiple models and aggregating their votes for final prediction"
description: "Leveraging an ensemble of models to improve prediction accuracy by combining their individual predictions into a final decision."
categories:
- Ecosystem Integration
tags:
- Multi-Model Systems
- Ensemble Learning
- Model Aggregation
- Machine Learning
- Classification
date: 2024-01-01
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/ecosystem-integration/multi-model-systems/voting-systems"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

Voting Systems, also known as ensemble methods, involve using multiple machine learning models and aggregating their predictions to produce a final output. This approach typically results in improved prediction accuracy, robustness, and generalizability as compared to relying on a single model.

## Explanation

Ensemble methods harness the power of diversity among multiple learning models to improve predictive performance. In a voting system, each model gets to "vote" on the final prediction:

1. **Hard Voting**: Each model in the ensemble casts a vote for a particular class, and the class that receives the majority of votes wins.
2. **Soft Voting**: The predicted probabilities (or confidence levels) of each class from all models are averaged, and the class with the highest average probability is selected.

Statistically, combining multiple imperfect models can often produce a resilient system that performs consistently well on a variety of data.

## KaTeX Formulation

For a classification problem with $k$ classes and an ensemble of $n$ models, let $P_i(c)$ be the probability of class $c$ predicted by the $i$-th model.

In soft voting, the final class prediction $\hat{C}$ is given by:
{{< katex >}}
\hat{C} = \arg\max_{c \in \{1,\ldots,k\}} \left( \frac{1}{n} \sum_{i=1}^{n} P_i(c) \right)
{{< /katex >}}

In hard voting, the final class prediction $\hat{C'}$ is given by:
{{< katex >}}
\hat{C'} = \arg\max_{c \in \{1,\ldots,k\}} \left( \sum_{i=1}^{n} I[M_i = c] \right)
{{< /katex >}}
where $I$ is the indicator function that returns 1 when $M_i$ predicts class $c$ and 0 otherwise.

## Example Implementations

### Python with Scikit-learn

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
svc_clf = SVC(probability=True)

voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf), ('dt', tree_clf), ('svc', svc_clf)], voting='soft')

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

### R with mlr3 package

```r
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)

task = tsk("iris")
split = partition(task, ratio = 0.7)

lrn_logreg = lrn("classif.log_reg")
lrn_tree = lrn("classif.rpart")
lrn_svm = lrn("classif.svm", predict_type = "prob")

graph = po("classifavg") %>>%
  gunion(list(lrn_logreg, lrn_tree, lrn_svm))

voting_learner = GraphLearner$new(graph)

voting_learner$train(task, row_ids = split$train)
prediction = voting_learner$predict(task, row_ids = split$test)

accuracy = prediction$score(msr("classif.acc"))
cat(sprintf("Accuracy: %.2f", accuracy))
```

## Related Design Patterns

### Bagging
* **Description**: Bootstrap Aggregating (Bagging) reduces variance by training multiple instances of a model on various bootstrap samples and aggregating their results.
* **Example**: Random Forest, which builds multiple decision trees and aggregates their votes.

### Boosting
* **Description**: Boosting models are sequentially trained where each new model corrects the errors of the prior ones.
* **Example**: AdaBoost, Gradient Boosting.

### Stacking
* **Description**: Combines multiple base learners with a meta-learner that learns how to best combine the outputs of base learners.
* **Example**: Combining various strong models (like SVMs, Neural Networks) with a meta-model such as a logistic regression.

### Hybrid Ensemble
* **Description**: Combines the strengths of both Bagging and Boosting or other ensemble techniques to further enhance performance.
* **Example**: Blending Bagging with Boosting methods.

## Additional Resources

1. **Books**:
   - *Ensemble Methods: Foundations and Algorithms* by Zhi-Hua Zhou

2. **Research Papers**:
   - Dietterich, T.G., *Ensemble Methods in Machine Learning*, Multiple Classifier Systems (MCS), 2000.

3. **Online Tutorials**:
   - [Sklearn Voting Classifier Documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)

## Summary

The Voting Systems design pattern enhances model accuracy and generalizability by aggregating the outputs of multiple models. Through either hard or soft voting, ensemble methods help mitigate the weaknesses of individual models, often resulting in robust predictions. Understanding and applying this pattern can significantly improve the performance of machine learning tasks, especially in classification problems. By combining various models, we leverage their unique strengths, leading to a powerful cumulative effect.

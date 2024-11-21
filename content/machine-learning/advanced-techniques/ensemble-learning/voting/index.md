---
linkTitle: "Voting"
title: "Voting: Aggregating Predictions by Majority Vote"
description: "A comprehensive guide to the Voting design pattern in machine learning, used for aggregating predictions of multiple models by majority vote."
categories:
- Advanced Techniques
tags:
- Ensemble Learning
- Voting
- Model Aggregation
- Prediction
- Machine Learning
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/ensemble-learning/voting"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


**Voting** is an ensemble learning technique in machine learning where predictions from multiple models are aggregated to make a final decision. The primary idea is that by combining the strengths of different models, one can achieve more robust and accurate predictions. This pattern falls under the category of *Advanced Techniques* and is a critical subcategory of ensemble learning.

## How Voting Works

Voting can be done in various ways, but the most common types are:

1. **Hard Voting**: Each model casts a vote for a predicted class, and the class with the majority votes is chosen as the final prediction.
2. **Soft Voting**: Each model predicts a probability for each class, and the predictions are averaged. The class with the highest average probability is selected.

### Mathematical Formulation

For *Hard Voting*:
{{< katex >}} \hat{y} = \text{mode}\{ \hat{y_1}, \hat{y_2}, \ldots, \hat{y_M}\} {{< /katex >}}

For *Soft Voting*:
{{< katex >}} \hat{y} = \arg\max_c \left( \frac{1}{M} \sum_{i=1}^{M} P(y=c|x; \theta_i) \right) {{< /katex >}}

Where \\( M \\) is the number of models, \\( \hat{y_i} \\) is the prediction of the \\( i \\)-th model, \\( P(y=c|x; \theta_i) \\) is the predicted probability of class \\( c \\) given \\( x \\) by the \\( i \\)-th model.

## Example Implementation

### Python Example using Scikit-learn

Let's consider a dataset with features \\( X \\) and corresponding labels \\( y \\). We will use three different classifiers and combine their predictions using the Voting technique.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()
clf3 = SVC(probability=True)  # SVC with probability=True for soft voting

voting_clf_hard = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)], voting='hard')
voting_clf_hard.fit(X_train, y_train)
print("Hard Voting Accuracy:", voting_clf_hard.score(X_test, y_test))

voting_clf_soft = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)], voting='soft')
voting_clf_soft.fit(X_train, y_train)
print("Soft Voting Accuracy:", voting_clf_soft.score(X_test, y_test))
```

### R Example

In R, the VotingClassifier is part of the `caret` package.

```R
library(caret)
library(randomForest)
library(e1071)

data(iris)
set.seed(42)
trainIndex <- createDataPartition(iris$Species, p = .8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

model1 <- train(Species ~., data = trainData, method = "glm", family = binomial)
model2 <- train(Species ~., data = trainData, method = "rpart")
model3 <- train(Species ~., data = trainData, method = "svmRadial")

pred1 <- predict(model1, newdata = testData)
pred2 <- predict(model2, newdata = testData)
pred3 <- predict(model3, newdata = testData)

final_pred <- ifelse(pred1 == pred2, pred1, pred3)

accuracy <- sum(final_pred == testData$Species) / nrow(testData)
print(paste("Hard Voting Accuracy:", accuracy))
```

## Related Design Patterns

- **Bagging**: This is a method where multiple versions of the same model are trained on different subsets of the data. The final output is the average (regression) or majority vote (classification) of all models.

- **Boosting**: Involves training a sequence of models, each trying to correct the errors of the previous ones. The final prediction is a weighted sum or vote of all models.

- **Stacking**: Different from Voting, stacking leverages another model (meta-learner) that learns how to best combine the base models' outputs.

## Additional Resources

- [Ensemble Methods - A powerful tool for improved predictive performance](https://scikit-learn.org/stable/modules/ensemble.html)
- [Bagging and Boosting - Simplifying Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

## Summary

The Voting design pattern is a powerful method in ensemble learning, allowing the aggregation of multiple models to enhance the robustness and accuracy of predictions. By leveraging both Hard and Soft voting mechanisms, one can combine the predictive strengths of different models, thereby achieving superior performance. Understanding related design patterns such as Bagging, Boosting, and Stacking provides a more comprehensive view of ensemble techniques, each with its unique advantages and applications.


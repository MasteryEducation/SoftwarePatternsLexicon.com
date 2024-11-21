---
linkTitle: "Stacking"
title: "Stacking: Combining Multiple Models Using a Meta-Model"
description: "An advanced ensemble learning technique that combines multiple models using a meta-model to improve predictive performance and reduce overfitting."
categories:
- Advanced Techniques
tags:
- Machine Learning
- Ensemble Learning
- Meta-Learning
- Model Stacking
- Performance Optimization
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/ensemble-learning/stacking"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Stacking: Combining Multiple Models Using a Meta-Model

Stacking, or stacked generalization, is an advanced ensemble learning technique where multiple base models (level-0 models) are combined using a meta-model (level-1 model) to improve predictive performance and reduce overfitting. By leveraging the strengths of different models, stacking aims to exploit their predictive power more effectively than any single model alone.

### The Concept of Stacking

The core idea behind stacking is to train multiple diverse models on the same dataset and subsequently train a meta-model using the predictions of these individual models as features. This meta-model, also known as the combiner or blender, learns to weight and combine the predictions of the base models to form a final prediction.

### Algorithmic Steps

1. **Split the dataset:** Divide the dataset into a training set and a validation set (or k-fold cross-validation sets).
2. **Train base models:** Train multiple base models on the training set.
3. **Predict base model outputs:** Use the base models to generate predictions on the validation set.
4. **Create new dataset:** Construct a new dataset where the features are the predictions of the base models and the target remains the original target variable.
5. **Train the meta-model:** Use this new dataset to train the meta-model.
6. **Make final predictions:** Use the base models and meta-model in conjunction to produce the final predictions on unseen data.

### Example Implementation

Below is an implementation of stacking using the Python programming language with the Scikit-learn library:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models=None, meta_model=None, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])
        
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        
        return self.meta_model_.predict(meta_features)

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [LogisticRegression(), DecisionTreeClassifier(), SVC(probability=True)]
meta_model = RandomForestClassifier()

stacking_clf = StackingClassifier(base_models=base_models, meta_model=meta_model)
stacking_clf.fit(X_train, y_train)
predictions = stacking_clf.predict(X_test)

print("Stacking Classifier Accuracy:", accuracy_score(y_test, predictions))
```

In this example, we built a stacking classifier combining logistic regression, decision tree, and support vector machine (base models) with a random forest (meta-model).

### Related Design Patterns

1. **Bagging:** Aggregates multiple models using the same algorithm on different subsets of the data to reduce variance and prevent overfitting. An example is the Random Forest algorithm.
   
2. **Boosting:** Sequentially trains models, where each new model attempts to correct the errors of the previous ones to reduce bias and variance. Examples include AdaBoost and Gradient Boosting Machines (GBM).
   
3. **Blending:** Similar to stacking, but the meta-model is fitted on a holdout validation dataset rather than on cross-validated out-of-fold predictions.

### Additional Resources

1. [Wikipedia: Stacked Generalization](https://en.wikipedia.org/wiki/Stacked_generalization)
2. [Scikit-learn: Model Stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
3. [KDnuggets: Ensemble Methods](https://www.kdnuggets.com/2016/03/ensemble-methods-super-models-machine-learning.html)
4. [Towards Data Science: Stacking Ensembles](https://towardsdatascience.com/stacking-classifiers-in-python-43091118a349)

### Summary

Stacking is a powerful ensemble learning technique that combines the strengths of multiple base models using a meta-model. By transforming the predictions of the base models into new features, the meta-model can learn to make more accurate final predictions. This technique is widely applicable across various domains and can significantly enhance the performance of machine learning models. As with all ensemble techniques, careful configuration and validation are essential to ensuring optimal performance.



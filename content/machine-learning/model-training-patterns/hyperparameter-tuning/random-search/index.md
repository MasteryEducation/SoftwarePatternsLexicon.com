---
linkTitle: "Random Search"
title: "Random Search: Hyperparameter Tuning through Random Sampling"
description: "An exploration of Random Search, a method of optimizing hyperparameters by randomly sampling the hyperparameter space."
categories:
- Model Training Patterns
tags:
- Hyperparameter Tuning
- Model Training
- Random Search
- Optimization
- Machine Learning
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/hyperparameter-tuning/random-search"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Random Search: Hyperparameter Tuning through Random Sampling

Hyperparameter tuning is crucial in machine learning as it significantly impacts model performance. One commonly used method for tuning hyperparameters is Random Search. This approach involves randomly sampling the hyperparameter space and evaluating model performance based on these samples. Unlike Grid Search, which exhaustively evaluates predefined combinations, Random Search explores more diverse configurations, making it a more efficient and often more effective method in practical scenarios.

### Basic Concept

Random Search involves defining a range or a set of possible values for each hyperparameter and then randomly picking combinations to evaluate. Each chosen set of hyperparameters is used to train and validate the model. The process continues for a predefined number of iterations or until other stopping criteria are met. The best performing hyperparameter configuration based on a chosen metric is then selected.

### KaTeX Representation

If we define the hyperparameter space as \\( \mathcal{H} \\), Random Search aims to find the optimal configuration \\( \mathbf{h}^{*} \\):
{{< katex >}}
\mathbf{h}^{*} = \underset{\mathbf{h} \in \mathcal{H}}{\arg\max} \, \mathcal{F}(\mathbf{h})
{{< /katex >}}
where \\( \mathcal{F}(\mathbf{h}) \\) is the performance metric (e.g., accuracy, F1-score) as a function of the hyperparameters.

### Algorithm

The algorithm for Random Search can be described in the following steps:

1. **Define the hyperparameter space**: Specify the possible values or distributions for each hyperparameter.
2. **Randomly sample**: Randomly choose a set of hyperparameters from the defined hyperparameter space.
3. **Train and evaluate**: Train the model using the randomly sampled hyperparameters and evaluate its performance on a validation set.
4. **Store results**: Record the performance metrics for each set of hyperparameters.
5. **Repeat**: Continue the process for a predefined number of iterations.
6. **Select best configuration**: Identify the hyperparameter set with the highest performance metric.

### Example: Using Scikit-Learn in Python

Below is an example of using Random Search for hyperparameter tuning with Scikit-Learn's RandomizedSearchCV.

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

iris = load_iris()
X, y = iris.data, iris.target

model = SVC()

param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'degree': randint(1, 5),
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)

print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")
```

### Example: Using Keras with TensorFlow in Python

For deep learning models, you might use Random Search in conjunction with Keras.

```python
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

def create_model(units=32, activation='relu'):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=10, verbose=0)

param_dist = {
    'units': [16, 32, 64, 128],
    'activation': ['relu', 'tanh'],
    'batch_size': [10, 20, 40],
    'epochs': [5, 10]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
random_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")
```

### Related Design Patterns

1. **Grid Search**: Systematically evaluates all possible combinations of hyperparameter values within a predefined grid. Though more exhaustive, it is computationally intensive compared to Random Search.
2. **Bayesian Optimization**: An advanced method for hyperparameter tuning that builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters, aiming to balance exploration and exploitation.
3. **Hyperband**: A resource allocation strategy that balances exploration and exploitation efficiently by dynamically allocating resources to the most promising hyperparameter configurations.

### Additional Resources

1. [Original Paper](https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) by Bergstra and Bengio detailing the efficiency of Random Search.
2. [Scikit-Learn documentation](https://scikit-learn.org/stable/modules/grid_search.html) on hyperparameter search strategies.
3. [TensorFlow documentation](https://www.tensorflow.org/tfx/guide/keras) for using Keras and tuning hyperparameters.

### Summary

Random Search offers an effective and computationally less expensive alternative to Grid Search for hyperparameter tuning by sampling hyperparameter values randomly. It strikes a balance between exhaustive search and heuristic methods. By tuning hyperparameters efficiently, Random Search can lead to better model performance in a feasible amount of time. Its simplicity and practical efficiency make it a staple in the toolbox of machine learning practitioners.

Incorporating Random Search into your workflow can help optimize models for a wide range of tasks, providing a straightforward yet powerful approach to hyperparameter tuning.

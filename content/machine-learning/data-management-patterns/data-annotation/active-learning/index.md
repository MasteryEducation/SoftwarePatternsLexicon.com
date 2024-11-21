---
linkTitle: "Active Learning"
title: "Active Learning: Choosing the Most Informative Samples to Label"
description: "Active Learning (AL) is a strategy designed to improve the efficiency and effectiveness of the data annotation process by selecting the most informative examples to label."
categories:
- Data Management Patterns
tags:
- Active Learning
- Data Annotation
- Machine Learning
- Data Management
- Data Efficiency
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-management-patterns/data-annotation/active-learning"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Active Learning (AL) is a machine learning paradigm where the learning algorithm is given the capability to query a user (or some other mechanism) to label new data points. This is particularly effective when labeled data is scarce or expensive to obtain. By prioritizing and actively selecting the most informative samples for annotation, AL aims to maximize the accuracy of the model while minimizing the cost and effort of labeling.

---

## Key Concepts

### Informativeness in Active Learning

Informativeness refers to how much new information a sample can provide to the model. Samples that lie near the decision boundary or exhibit high uncertainty are often the most informative, as labeling them can greatly reduce ambiguity in the model's predictions.

### Query Strategies

Several querying strategies are commonly used to identify the most informative samples:

- **Uncertainty Sampling**: Selects samples for which the model is least confident about the output.
- **Query-by-Committee (QbC)**: Uses multiple models (a committee) and selects samples on which the models disagree the most.
- **Expected Model Change**: Chooses samples that are expected to most significantly alter the current model.
- **Density-weighted Methods**: Weighs samples according to both their informativeness and their representativeness of the underlying data distribution.

### Stopping Criteria

Stopping criteria are important in active learning, as they determine when the iterative process should halt. Common strategies include:
- **Performance Plateau**: Stop when additions no longer significantly enhance model performance.
- **Fixed Budget**: Halt when you reach a predetermined number of samples or total labeling cost.
- **Confidence Threshold**: Terminate when the model achieves a certain level of confidence.

---

## Implementations in Different Programming Frameworks

### Python with Scikit-Learn

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

X, y = make_classification(n_classes=2, n_informative=4, n_samples=1000, random_state=0)
X_train, X_pool, y_train, y_pool = train_test_split(X, y, train_size=0.1, stratify=y, random_state=0)

learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_train, y_training=y_train,
    query_strategy=uncertainty_sampling
)

n_queries = 10
for i in range(n_queries):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx)

print("Final model accuracy:", learner.score(X_pool, y_pool))
```

### R with the `activeLearning` Package

```r
library(activeLearning)
library(caret)

set.seed(0)
data <- twoClassSim(1000)
trainIndex <- createDataPartition(data$Class, p = 0.1, list = FALSE)
trainData <- data[trainIndex, ]
poolData <- data[-trainIndex, ]

learner <- createLearner(poolData, trainData$Class, method = "rf", queryStrategy = "uncertainty")

for (i in 1:10) {
    learner <- activeLearn(learner)
}

finalModel <- learner@model
poolPred <- predict(finalModel, poolData[-ncol(poolData)])
confusionMatrix(poolPred, poolData$Class)
```

### JavaScript with TensorFlow.js

```javascript
const tf = require('@tensorflow/tfjs');
const { ActiveLearner, uncertaintySampling } = require('tfjs-active-learning');

// Create synthetic data
const data = tf.randomNormal([1000, 20]);
const labels = tf.randomUniform([1000, 1], 0, 2, 'int32');

// Split into train and pool
const trainSize = 100;
const X_train = data.slice([0, 0], [trainSize, -1]);
const y_train = labels.slice([0, 0], [trainSize, 1]);
const X_pool = data.slice([trainSize, 0], [-1, -1]);
const y_pool = labels.slice([trainSize, 0], [-1, -1]);

// Define model and active learner
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [20] }));
model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
model.compile({ optimizer: 'adam', loss: 'sparse_categorical_crossentropy', metrics: ['accuracy'] });

const learner = new ActiveLearner({ model, X_train, y_train, queryStrategy: uncertaintySampling });

// Active learning loop
for (let i = 0; i < 10; i++) {
    learner.query().then(([X_query, y_query]) => {
        learner.teach(X_query, y_query).then(() => {
          // Proceed with the next iteration or stop based on criteria  
        });
    });
}
```

---

## Related Design Patterns

### Data Augmentation
**Description**: Data Augmentation involves generating additional training samples by applying transformations such as rotations, flips, or shifts on existing data. This enhances the diversity of the training set and reduces overfitting.

### Semi-Supervised Learning
**Description**: In Semi-Supervised Learning, the model is trained using both labeled and unlabeled data. This approach is beneficial when unlabeled data is abundant and labeled data is scarce.

### Transfer Learning
**Description**: Transfer Learning involves leveraging a pre-trained model on a different but related task. The pre-trained knowledge is then fine-tuned with task-specific data, greatly reducing the need for extensive labeled datasets.

---

## Additional Resources

- **Papers**:
  - "A Survey of Active Learning" by Burr Settles
  - "Active Learning Literature Survey" by Burr Settles (Technical Report)
  
- **Books**:
  - "Machine Learning Yearning" by Andrew Ng
  - "Pattern Recognition and Machine Learning" by Christopher Bishop

- **Online Tutorials**:
  - Official documentation for the `modAL` library in Python.
  - TensorFlow.js examples and tutorials on active learning.

---

## Summary

Active Learning is a powerful data management strategy aimed at selecting the most informative data samples to label, thus improving the efficiency of the annotation process and the performance of the resulting model. By utilizing various querying strategies such as uncertainty sampling, query-by-committee, and density-weighted methods, and incorporating stopping criteria, active learning efficiently directs labeling efforts towards the most beneficial data points. Whether implemented in Python, R, or JavaScript, active learning can significantly help in situations where data labeling is an expensive or time-consuming process. By combining active learning with other patterns such as data augmentation, semi-supervised learning, or transfer learning, even more effective machine learning solutions can be developed.

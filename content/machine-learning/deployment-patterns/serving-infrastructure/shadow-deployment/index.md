---
linkTitle: "Shadow Deployment"
title: "Shadow Deployment: Running new models alongside existing models to compare performance"
description: "Shadow Deployment is a design pattern where new machine learning models are deployed alongside existing models in production to compare their performance without impacting the end-users. This comparison helps in validating the new models before fully rolling them out."
categories:
- Serving Infrastructure
tags:
- Machine Learning
- Deployment
- Model Validation
- Performance Comparison
- Production
date: 2023-10-05
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/deployment-patterns/serving-infrastructure/shadow-deployment"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
Shadow Deployment is a machine learning design pattern belonging to the subcategory of Serving Infrastructure and fitting within Deployment Patterns. This pattern involves deploying new machine learning models alongside existing (or current) models in a live production environment. The purpose is to run the new model in "shadow mode," where it receives live traffic and makes inferences, but its outputs do not affect the end users. Instead, its predictions and performance are logged and compared against the existing model. This allows data scientists and ML engineers to evaluate the new model's performance in a real-world environment without any user-facing risks.

## Objectives and Advantages
- **Safe Validation**: Validate new models in a production setting without affecting user experience.
- **Performance Comparison**: Compare key metrics (e.g., accuracy, latency, resource usage) between the existing and new models.
- **Risk Mitigation**: Identify potential issues with new models before fully rolling them out.
- **Incremental Deployment**: Enables gradual improvements to the model serving infrastructure.

## Example Implementation

### Python with TensorFlow Serving

Below, we describe an example of implementing a shadow deployment using Python and TensorFlow Serving. For simplicity, we use Flask for creating REST endpoints.

1. **Existing Model Deployment**:
   ```python
   import tensorflow as tf
   from flask import Flask, request, jsonify

   app = Flask(__name__)

   # Load the existing TensorFlow model
   existing_model = tf.keras.models.load_model('path/to/existing/model')

   @app.route('/predict', methods=['POST'])
   def predict_existing():
       data = request.json
       prediction = existing_model.predict(data['input'])
       return jsonify({'prediction': prediction.tolist()})

   ```

2. **New Model Deployment (Shadow Mode)**:
   ```python
   new_model = tf.keras.models.load_model('path/to/new/model')

   @app.route('/shadow_predict', methods=['POST'])
   def predict_shadow():
       data = request.json
       # Predictions from the shadow model
       shadow_prediction = new_model.predict(data['input'])
       # Here, log shadow prediction for analysis without returning to user
       log_shadow_prediction(data, shadow_prediction)
       return jsonify({'status': 'shadow prediction recorded'})

   # Function to log shadow predictions
   def log_shadow_prediction(input_data, prediction):
       with open('shadow_predictions.log', 'a') as f:
           f.write(f"Input: {input_data}, Prediction: {prediction}\n")
   ```

3. **Running Both Models**:
   ```python
   if __name__ == "__main__":
       app.run(host='0.0.0.0', port=5000)
   ```

### JavaScript with Node.js and Express

For JavaScript enthusiasts, here's an example using Node.js and Express to achieve shadow deployment:

```javascript
const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');

const app = express();

app.use(bodyParser.json());

const existingModel = await tf.loadLayersModel('file://path/to/existing/model/model.json');
const newModel = await tf.loadLayersModel('file://path/to/new/model/model.json');

app.post('/predict', async (req, res) => {
    const inputData = req.body.input;
    const existingPrediction = existingModel.predict(tf.tensor(inputData));
    res.json({ prediction: existingPrediction.arraySync() });
});

app.post('/shadow_predict', async (req, res) => {
    const inputData = req.body.input;
    const shadowPrediction = newModel.predict(tf.tensor(inputData));
    logShadowPrediction(inputData, shadowPrediction);
    res.json({ status: "shadow prediction recorded" });
});

function logShadowPrediction(inputData, prediction) {
    const fs = require('fs');
    fs.appendFile('shadow_predictions.log', `Input: ${JSON.stringify(inputData)}, Prediction: ${prediction.arraySync()}\n`, err => {
        if (err) {
            console.error('Error saving shadow prediction:', err);
        }
    });
}

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
```

## Related Design Patterns

1. **A/B Testing**: A/B Testing involves splitting live traffic into two groups, where each group is served by a different version of the model. Unlike Shadow Deployment, users might be exposed to different model outputs. This pattern is used for direct comparison of user engagement and satisfaction.

2. **Canary Deployment**: In Canary Deployment, the new model is gradually rolled out to a small subset of users. Monitoring their experience allows identification and mitigation of issues before the full roll-out.

3. **Blue-Green Deployment**: This involves running two identical production environments - Blue (current production) and Green (new version). Traffic is gradually switched from Blue to Green. While this is often used for software deployment, it can also apply to model deployment.

## Additional Resources
1. **TensorFlow Extended (TFX)** - A scalable, high-performance ML platform: https://www.tensorflow.org/tfx
2. **Kubeflow** - Dedicated to making deployments of ML workflows on Kubernetes simple, portable, and scalable: https://www.kubeflow.org
3. **Flask** - Micro web framework written in Python: https://flask.palletsprojects.com/
4. **Express.js** - Fast, unopinionated, minimalist web framework for Node.js: https://expressjs.com/

## Summary
Shadow Deployment is an essential design pattern for safely validating new ML models without impacting the end-users. By running the new model in parallel to the existing one and comparing their performance metrics, organizations can ensure that new models are reliable, accurate, and resource-efficient before full deployment. This pattern helps in risk mitigation and enables careful planning for model upgrades in production environments. It complements and contrasts with other deployment strategies like A/B Testing, Canary Deployment, and Blue-Green Deployment, each designed for specific aspects of upgrading machine learning systems.

By leveraging frameworks like TensorFlow Serving, Flask, and Express, one can efficiently implement shadow deployment, ensuring robust and risk-free deployment cycles.

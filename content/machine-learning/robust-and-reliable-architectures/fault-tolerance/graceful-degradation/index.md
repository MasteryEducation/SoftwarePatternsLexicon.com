---
linkTitle: "Graceful Degradation"
title: "Graceful Degradation: Ensuring Continuity with Reduced Functionality"
description: "Ensuring the system continues to operate, even at reduced functionality, when parts fail."
categories:
- Robust and Reliable Architectures
- Fault Tolerance
tags:
- Fault Tolerance
- Robust Architecture
- Reliability
- Machine Learning
- Graceful Degradation
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/robust-and-reliable-architectures/fault-tolerance/graceful-degradation"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Graceful Degradation is a design pattern in machine learning and system architecture that focuses on ensuring a system continues to operate under partial failure conditions, albeit with reduced functionality. This design pattern significantly enhances the robustness and reliability of machine learning systems by addressing potential points of failure proactively.

## Fault Tolerance and Graceful Degradation

In machine learning systems, fault tolerance is crucial, given the complex interplay of various components like data processing pipelines, model training, inference engines, and service endpoints. Graceful Degradation ensures that even if some of these components fail, the overall system remains functional at some level.

## Practical Examples

Here, we discuss two practical implementations of Graceful Degradation in machine learning systems using Python with TensorFlow and JavaScript with TensorFlow.js.

### Example 1: Graceful Degradation in Python with TensorFlow

Imagine we have a system that performs real-time image classification and recommendation. If the recommendation service fails, the system should still provide degraded functionality by performing image classification alone.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

model = MobileNetV2(weights='imagenet')

def load_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def classify_image(img_path):
    image = load_image(img_path)
    preds = model.predict(image)
    return decode_predictions(preds, top=3)[0]

def recommend_alternate_objects(object_name):
    # Simulate recommendation system, which may fail
    recommendations = {
        'dog': ['ball', 'bone', 'leash'],
        'cat': ['scratching post', 'toy mouse', 'collar']
    }
    return recommendations.get(object_name.lower(), ['no recommendations available'])

def main(img_path):
    try:
        # Attempt both classification and recommendation
        predictions = classify_image(img_path)
        print("Image Classification Results:")
        for (_, obj_name, _) in predictions:
            print(f" - {obj_name}")
        
        print("\nRecommendations:")
        for (_, obj_name, _) in predictions:
            recommendations = recommend_alternate_objects(obj_name)
            print(f" - If you like {obj_name}, you might also like {recommendations}")
    except:
        # Fall back to classification only with graceful degradation
        print("Recommendation system failed. Displaying classification results only.")
        predictions = classify_image(img_path)
        for (_, obj_name, _) in predictions:
            print(f" - {obj_name}")

if __name__ == "__main__":
    img_path = 'example_image.jpg'  # Replace with your image path
    main(img_path)
```

### Example 2: Graceful Degradation in JavaScript with TensorFlow.js

In a web-based application, if a model cannot be loaded, a basic pre-processing task can continue to ensure the application does not completely fail.

```javascript
<!DOCTYPE html>
<html>
<head>
    <title>Graceful Degradation with TensorFlow.js</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
</head>
<body>
    <h1>Graceful Degradation Example</h1>
    <input type="file" id="upload-image" />
    <button onclick="processImage()">Process Image</button>
    <div id="output"></div>

    <script>
      async function loadModel() {
          try {
              const model = await tf.loadLayersModel('path_to_model/model.json');
              return model;
          } catch (error) {
              console.error('Model loading failed:', error);
              return null;
          }
      }

      async function preprocessImage(file) {
          const img = new Image();
          img.src = URL.createObjectURL(file);
          await new Promise((resolve) => (img.onload = resolve));
          const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat();
          const offset = tf.scalar(127.5);
          const normalized = tensor.sub(offset).div(offset);
          const batched = normalized.expandDims(0);
          return batched;
      }

      async function processImage() {
          const uploadImage = document.getElementById('upload-image');
          const file = uploadImage.files[0];
          const model = await loadModel();

          if (model) {
              const tensor = await preprocessImage(file);
              const predictions = model.predict(tensor).dataSync();
              // Logic to handle predictions
              document.getElementById('output').textContent = `Predictions: ${predictions}`;
          } else {
              // Graceful degradation: Perform only preprocessing or display a message
              console.log('Model could not be loaded. Displaying fallback response.');
              document.getElementById('output').textContent = 'Unable to load the model, but preprocessing is completed.';
          }
      }
    </script>
</body>
</html>
```

## Related Design Patterns

### 1. **Retry Pattern**
The Retry Pattern involves attempting a failed operation a specified number of times before considering it a total failure. It is often combined with exponential backoff to avoid overwhelming the system.

### 2. **Fallback Pattern**
The Fallback Pattern involves defining alternative processing or data retrieval strategies in case the primary method fails. E.g., caching results locally if the database is down.

### 3. **Circuit Breaker Pattern**
A Circuit Breaker Pattern monitors for failures and suspends requests once a failure threshold is reached, preventing constant retries and allowing time for system recovery.

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow.js Documentation](https://js.tensorflow.org/api/latest/)
- [Resilient System Design](https://martinfowler.com/articles/masstransit-in-the-enterprise.html)
- [Designing Data-Intensive Applications](https://dataintensive.net/)

## Summary

Graceful Degradation is essential for building robust and reliable machine learning systems. By ensuring a system continues to provide partial functionality even during component failures, we can design solutions that are resilient and provide a better user experience. Through strategic coding practices and fallback mechanisms, developers can create applications that handle faults gracefully while maintaining essential services.

By leveraging this design pattern in combination with other fault-tolerance strategies such as Retry, Fallback, and Circuit Breaker Patterns, one can significantly enhance the robustness and reliability of machine learning solutions.

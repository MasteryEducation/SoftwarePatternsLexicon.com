---
linkTitle: "Edge Computing"
title: "Edge Computing: Processing Data and Running Models on Distributed Edge Devices to Decrease Latency"
description: "An exploration of the Edge Computing design pattern in machine learning, highlighting its implementation, benefits, and related design patterns."
categories:
- Infrastructure and Scalability
- Distributed Systems
tags:
- Edge Computing
- Machine Learning
- Distributed Systems
- Scalability
- Low Latency
date: 2023-10-05
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/infrastructure-and-scalability/distributed-systems/edge-computing"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Edge Computing: Processing Data and Running Models on Distributed Edge Devices to Decrease Latency

### Introduction

Edge Computing is a design pattern in machine learning where data processing and model inference are performed on distributed edge devices closer to the data source. This paradigm contrasts with traditional cloud computing, where data is sent to centralized servers for processing. Edge computing aims to minimize latency, reduce bandwidth demand, and enhance real-time processing capabilities.

### Benefits of Edge Computing

1. **Reduced Latency**: By processing data locally, edge computing minimizes latency, enabling faster decision-making and real-time analytics.
2. **Bandwidth Efficiency**: Reduces the necessity to transmit large volumes of data to centralized data centers, saving bandwidth.
3. **Enhanced Privacy**: Sensitive data can be processed locally, reducing the risk of data breaches and enhancing data privacy.
4. **Reliability**: Local processing remains unaffected by internet connectivity issues, providing consistent performance.

### Implementation

#### Architecture

The architecture of edge computing typically includes the following components:

1. **Edge Devices**: IoT devices, smartphones, or local servers that directly interact with data sources and perform computations.
2. **Edge Servers**: Local servers handling more complex tasks and providing additional resources to edge devices.
3. **Cloud**: Centralized data storage and advanced analytics, where data from edge devices can be aggregated and further analyzed.

#### Example Scenario: Smart Home System

A smart home system comprising several IoT devices (like smart thermostats, cameras, and lighting systems) can leverage edge computing to enhance performance. Key functionalities like motion detection, facial recognition, and temperature control are processed on edge devices close to the data source.

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

### Related Design Patterns

1. **Model Compression**:
    - **Description**: Techniques to reduce the size of machine learning models enabling them to run efficiently on resource-constrained edge devices.
    - **Example**: Pruning, quantization, and knowledge distillation.

2. **Federated Learning**:
    - **Description**: Training machine learning models across decentralized devices holding local data samples without exchanging them.
    - **Example**: Google's Gboard, which learns from user interactions across many mobile devices.

3. **Data Decoupling**:
    - **Description**: Separating the data ingestion layer from the data processing layer allowing edge devices to serve as data ingestors while processing can happen either at the edge, fog, or cloud.
    - **Example**: Edge devices collecting raw sensor data, which is then processed either locally or sent to a central node for processing based on the requirement.

### Additional Resources

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Apache MyNewt](https://mynewt.apache.org/)
- [AWS IoT Greengrass](https://aws.amazon.com/greengrass/)
- [Azure IoT Edge](https://azure.microsoft.com/en-us/services/iot-edge/)
- [Edge Computing: A Primer](https://www.acm.org/articles/edge-computing-primer)

### Summary

Edge Computing represents a transformative design pattern in machine learning, optimizing data processing workflows for lower latency, enhanced privacy, and greater reliability. By distributing computations closer to data sources across edge devices, systems can now manage real-time applications more effectively while conserving bandwidth and safeguarding data privacy. This pattern synergizes with other practices such as model compression and federated learning to achieve a robust and scalable machine learning infrastructure.

Implementing edge computing requires understanding the ecosystem of edge devices and utilizing suitable frameworks. With advancements in technologies supporting this paradigm, edge computing is set to play a pivotal role in the future of distributed, intelligent systems.

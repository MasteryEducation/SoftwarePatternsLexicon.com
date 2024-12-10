---
canonical: "https://softwarepatternslexicon.com/kafka/7/1/4/2"
title: "Creating Custom Single Message Transforms (SMTs) for Kafka Connect"
description: "Learn how to create custom Single Message Transforms (SMTs) in Kafka Connect to address specialized data transformation needs, with step-by-step instructions, code examples, and best practices."
linkTitle: "7.1.4.2 Writing Custom SMTs"
tags:
- "Apache Kafka"
- "Kafka Connect"
- "Single Message Transforms"
- "Data Transformation"
- "Java"
- "Scala"
- "Kotlin"
- "Clojure"
date: 2024-11-25
type: docs
nav_weight: 71420
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 7.1.4.2 Writing Custom SMTs

Single Message Transforms (SMTs) in Kafka Connect are a powerful feature that allows you to modify messages as they flow through the pipeline. While Kafka Connect provides a variety of built-in SMTs, there are scenarios where custom transformations are necessary to meet specific business requirements. This guide will walk you through the process of writing custom SMTs, from understanding when they are needed to implementing, packaging, and deploying them.

### When to Use Custom SMTs

Custom SMTs are particularly useful in the following scenarios:

- **Complex Data Transformations**: When the built-in SMTs do not support the specific transformation logic required, such as complex field manipulations or conditional transformations.
- **Integration with Legacy Systems**: When integrating with legacy systems that require specific data formats or structures.
- **Data Enrichment**: When additional data needs to be added to the message from external sources or databases.
- **Custom Validation**: When messages need to be validated against complex business rules before being processed further.

### Implementing the Transformation Interface

To create a custom SMT, you need to implement the `Transformation` interface provided by Kafka Connect. This interface requires you to define the transformation logic that will be applied to each message.

#### Steps to Implement a Custom SMT

1. **Define the Transformation Class**: Create a new class that implements the `Transformation` interface.

2. **Implement the Required Methods**: The `Transformation` interface requires the implementation of several methods, including `configure`, `apply`, and `close`.

3. **Handle Configuration**: Use the `configure` method to handle any configuration parameters your SMT may need.

4. **Apply the Transformation**: Implement the `apply` method to define the transformation logic.

5. **Clean Up Resources**: Use the `close` method to release any resources when the SMT is no longer needed.

#### Java Example

Here is an example of a custom SMT in Java that adds a timestamp to each message:

```java
package com.example.kafka.transforms;

import org.apache.kafka.connect.connector.ConnectRecord;
import org.apache.kafka.connect.transforms.Transformation;
import org.apache.kafka.connect.transforms.util.SimpleConfig;

import java.util.Map;

public class AddTimestamp<R extends ConnectRecord<R>> implements Transformation<R> {

    @Override
    public R apply(R record) {
        // Add a timestamp to the record
        Map<String, Object> updatedValue = new HashMap<>(record.value());
        updatedValue.put("timestamp", System.currentTimeMillis());
        return record.newRecord(
            record.topic(),
            record.kafkaPartition(),
            record.keySchema(),
            record.key(),
            record.valueSchema(),
            updatedValue,
            record.timestamp()
        );
    }

    @Override
    public void configure(Map<String, ?> configs) {
        // Handle any configuration here
    }

    @Override
    public void close() {
        // Clean up resources if needed
    }

    @Override
    public ConfigDef config() {
        return new ConfigDef();
    }
}
```

#### Scala Example

```scala
package com.example.kafka.transforms

import org.apache.kafka.connect.connector.ConnectRecord
import org.apache.kafka.connect.transforms.Transformation
import org.apache.kafka.connect.transforms.util.SimpleConfig

import scala.collection.JavaConverters._

class AddTimestamp[R <: ConnectRecord[R]] extends Transformation[R] {

  override def apply(record: R): R = {
    // Add a timestamp to the record
    val updatedValue = record.value().asInstanceOf[Map[String, Any]].asJava
    updatedValue.put("timestamp", System.currentTimeMillis())
    record.newRecord(
      record.topic(),
      record.kafkaPartition(),
      record.keySchema(),
      record.key(),
      record.valueSchema(),
      updatedValue,
      record.timestamp()
    )
  }

  override def configure(configs: java.util.Map[String, _]): Unit = {
    // Handle any configuration here
  }

  override def close(): Unit = {
    // Clean up resources if needed
  }

  override def config(): ConfigDef = new ConfigDef()
}
```

#### Kotlin Example

```kotlin
package com.example.kafka.transforms

import org.apache.kafka.connect.connector.ConnectRecord
import org.apache.kafka.connect.transforms.Transformation
import org.apache.kafka.connect.transforms.util.SimpleConfig

class AddTimestamp<R : ConnectRecord<R>> : Transformation<R> {

    override fun apply(record: R): R {
        // Add a timestamp to the record
        val updatedValue = record.value() as MutableMap<String, Any>
        updatedValue["timestamp"] = System.currentTimeMillis()
        return record.newRecord(
            record.topic(),
            record.kafkaPartition(),
            record.keySchema(),
            record.key(),
            record.valueSchema(),
            updatedValue,
            record.timestamp()
        )
    }

    override fun configure(configs: Map<String, *>?) {
        // Handle any configuration here
    }

    override fun close() {
        // Clean up resources if needed
    }

    override fun config(): ConfigDef = ConfigDef()
}
```

#### Clojure Example

```clojure
(ns com.example.kafka.transforms.AddTimestamp
  (:import [org.apache.kafka.connect.connector ConnectRecord]
           [org.apache.kafka.connect.transforms Transformation]
           [org.apache.kafka.connect.transforms.util SimpleConfig]))

(defn add-timestamp [record]
  (let [updated-value (assoc (into {} (.value record)) "timestamp" (System/currentTimeMillis))]
    (.newRecord record
                (.topic record)
                (.kafkaPartition record)
                (.keySchema record)
                (.key record)
                (.valueSchema record)
                updated-value
                (.timestamp record))))

(deftype AddTimestamp []
  Transformation
  (apply [this record]
    (add-timestamp record))
  (configure [this configs]
    ;; Handle any configuration here
    )
  (close [this]
    ;; Clean up resources if needed
    )
  (config [this]
    (org.apache.kafka.common.config.ConfigDef.)))
```

### Packaging and Deploying Custom SMTs

Once you have implemented your custom SMT, the next step is to package and deploy it so that it can be used in your Kafka Connect environment.

#### Packaging the SMT

1. **Build the JAR**: Use a build tool like Maven or Gradle to compile your SMT and package it into a JAR file.

2. **Include Dependencies**: Ensure that all necessary dependencies are included in the JAR or are available in the classpath of your Kafka Connect workers.

#### Deploying the SMT

1. **Copy the JAR**: Place the JAR file in the `plugins` directory of your Kafka Connect installation.

2. **Update the Worker Configuration**: Modify the `connect-distributed.properties` or `connect-standalone.properties` file to include the path to your custom SMT.

3. **Restart Kafka Connect**: Restart the Kafka Connect workers to load the new SMT.

### Testing and Performance Best Practices

Testing custom SMTs is crucial to ensure they perform as expected and do not introduce bottlenecks in your data pipeline.

#### Testing Strategies

- **Unit Testing**: Write unit tests to verify the transformation logic. Use mock objects to simulate Kafka Connect records.
- **Integration Testing**: Deploy the SMT in a test environment and verify its behavior with real data.
- **Performance Testing**: Measure the performance impact of the SMT on your data pipeline. Use tools like Apache JMeter or Gatling for load testing.

#### Performance Considerations

- **Optimize Transformation Logic**: Ensure that the transformation logic is efficient and does not perform unnecessary computations.
- **Minimize External Calls**: Avoid making external calls (e.g., database lookups) within the SMT, as they can significantly impact performance.
- **Monitor Resource Usage**: Use monitoring tools to track CPU and memory usage of your Kafka Connect workers.

### Best Practices for Custom SMTs

- **Keep It Simple**: Focus on a single responsibility for each SMT to make it easier to maintain and test.
- **Document Configuration Options**: Clearly document any configuration options your SMT supports.
- **Handle Errors Gracefully**: Implement error handling to manage unexpected input or transformation failures.
- **Ensure Compatibility**: Test your SMT with different versions of Kafka Connect to ensure compatibility.

### Conclusion

Creating custom SMTs in Kafka Connect allows you to tailor data transformations to meet specific business needs. By following the steps outlined in this guide, you can implement, package, and deploy custom SMTs effectively. Remember to test thoroughly and monitor performance to ensure your SMTs contribute positively to your data pipeline.

### Related Patterns

- **[7.1.4.1 Built-in SMTs]({{< ref "/kafka/7/1/4/1" >}} "Built-in SMTs")**: Explore the built-in SMTs provided by Kafka Connect.
- **[7.2.1 Change Data Capture with Debezium]({{< ref "/kafka/7/2/1" >}} "Change Data Capture with Debezium")**: Learn about integrating Kafka Connect with databases for change data capture.

### Further Reading

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Documentation](https://docs.confluent.io/)
- [Kafka Connect SMTs GitHub Repository](https://github.com/apache/kafka/tree/trunk/connect/transforms)

## Test Your Knowledge: Custom SMTs in Kafka Connect Quiz

{{< quizdown >}}

### What is the primary purpose of a custom SMT in Kafka Connect?

- [x] To handle specialized data transformation requirements not covered by built-in SMTs.
- [ ] To manage Kafka Connect worker configurations.
- [ ] To optimize Kafka broker performance.
- [ ] To monitor Kafka Connect metrics.

> **Explanation:** Custom SMTs are used to implement specialized data transformations that are not supported by the built-in SMTs.

### Which method in the Transformation interface is responsible for applying the transformation logic?

- [x] apply
- [ ] configure
- [ ] close
- [ ] config

> **Explanation:** The `apply` method is where the transformation logic is implemented for each message.

### What is a recommended practice when deploying a custom SMT?

- [x] Place the JAR file in the plugins directory of the Kafka Connect installation.
- [ ] Modify the Kafka broker configuration.
- [ ] Deploy the SMT directly to the Kafka topic.
- [ ] Use the Kafka CLI to install the SMT.

> **Explanation:** The JAR file containing the custom SMT should be placed in the plugins directory of the Kafka Connect installation.

### What is a key consideration when implementing the apply method in a custom SMT?

- [x] Ensure the transformation logic is efficient and does not perform unnecessary computations.
- [ ] Use external API calls for every message.
- [ ] Store transformation results in a database.
- [ ] Ignore message keys during transformation.

> **Explanation:** It is important to ensure that the transformation logic is efficient to avoid introducing bottlenecks in the data pipeline.

### Which testing strategy is NOT recommended for custom SMTs?

- [ ] Unit Testing
- [ ] Integration Testing
- [x] Ignoring Performance Testing
- [ ] Load Testing

> **Explanation:** Ignoring performance testing is not recommended as it is crucial to understand the impact of the SMT on the data pipeline.

### What should be included in the JAR file when packaging a custom SMT?

- [x] All necessary dependencies or ensure they are available in the classpath.
- [ ] Only the source code files.
- [ ] Kafka broker configuration files.
- [ ] Kafka Connect worker logs.

> **Explanation:** The JAR file should include all necessary dependencies or ensure they are available in the classpath.

### What is a potential drawback of making external calls within an SMT?

- [x] It can significantly impact performance.
- [ ] It simplifies the transformation logic.
- [ ] It enhances data security.
- [ ] It reduces resource usage.

> **Explanation:** Making external calls within an SMT can significantly impact performance due to increased latency.

### What is the role of the configure method in a custom SMT?

- [x] To handle any configuration parameters the SMT may need.
- [ ] To apply the transformation logic.
- [ ] To release resources when the SMT is no longer needed.
- [ ] To monitor Kafka Connect metrics.

> **Explanation:** The `configure` method is used to handle any configuration parameters the SMT may need.

### What is a best practice for error handling in custom SMTs?

- [x] Implement error handling to manage unexpected input or transformation failures.
- [ ] Ignore errors and continue processing.
- [ ] Log errors to a separate Kafka topic.
- [ ] Use exceptions to stop the Kafka Connect worker.

> **Explanation:** Implementing error handling is a best practice to manage unexpected input or transformation failures.

### True or False: Custom SMTs should focus on a single responsibility to make them easier to maintain and test.

- [x] True
- [ ] False

> **Explanation:** Focusing on a single responsibility makes custom SMTs easier to maintain and test.

{{< /quizdown >}}

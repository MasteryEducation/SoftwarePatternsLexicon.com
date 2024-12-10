---
canonical: "https://softwarepatternslexicon.com/kafka/14/2/1"
title: "Setting Up Embedded Kafka Brokers for Integration Testing"
description: "Learn how to configure embedded Kafka brokers for testing, enabling realistic interaction with Kafka APIs using libraries like Kafka Embedded and Testcontainers."
linkTitle: "14.2.1 Setting Up Embedded Kafka Brokers"
tags:
- "Apache Kafka"
- "Integration Testing"
- "Embedded Kafka"
- "Testcontainers"
- "Java"
- "Scala"
- "Kotlin"
- "Clojure"
date: 2024-11-25
type: docs
nav_weight: 142100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.2.1 Setting Up Embedded Kafka Brokers

### Introduction

Setting up embedded Kafka brokers is a crucial step in integration testing, allowing developers to simulate a real Kafka environment without the overhead of managing external Kafka clusters. This approach facilitates testing Kafka-based applications in a controlled, isolated environment, ensuring that the application logic interacts correctly with Kafka APIs. In this section, we will explore the use of libraries such as Kafka Embedded and Testcontainers to set up embedded Kafka brokers, provide detailed code examples, and discuss configuration considerations for testing various scenarios.

### Understanding Embedded Kafka

Embedded Kafka refers to running a Kafka broker within the same process as your application, typically for testing purposes. This setup is beneficial for integration tests where you need to verify the interaction between your application and Kafka without deploying a full Kafka cluster.

#### Benefits of Using Embedded Kafka

- **Isolation**: Tests run in isolation, ensuring no interference from other processes or network issues.
- **Speed**: Faster setup and teardown compared to deploying a full Kafka cluster.
- **Simplicity**: Simplifies the testing environment by reducing external dependencies.
- **Flexibility**: Allows testing of various Kafka configurations and scenarios.

### Libraries for Embedded Kafka

Several libraries facilitate the setup of embedded Kafka brokers. Two popular choices are Kafka Embedded and Testcontainers.

#### Kafka Embedded

Kafka Embedded is a lightweight library that allows you to run a Kafka broker within your application. It is particularly useful for unit and integration tests.

#### Testcontainers

Testcontainers is a Java library that provides lightweight, throwaway instances of common databases, Selenium web browsers, or anything else that can run in a Docker container. It supports Kafka and is ideal for integration testing in a more realistic environment.

### Setting Up Embedded Kafka with Kafka Embedded

#### Step-by-Step Guide

1. **Add Dependencies**: Include the necessary dependencies in your project. For Maven, add the following to your `pom.xml`:

    ```xml
    <dependency>
        <groupId>org.apache.kafka</groupId>
        <artifactId>kafka_2.13</artifactId>
        <version>3.0.0</version>
        <scope>test</scope>
    </dependency>
    <dependency>
        <groupId>org.apache.kafka</groupId>
        <artifactId>kafka-clients</artifactId>
        <version>3.0.0</version>
        <scope>test</scope>
    </dependency>
    ```

2. **Initialize Embedded Kafka**: Create a class to initialize and manage the embedded Kafka broker.

    - **Java Example**:

        ```java
        import org.apache.kafka.clients.producer.ProducerConfig;
        import org.apache.kafka.clients.producer.KafkaProducer;
        import org.apache.kafka.clients.producer.ProducerRecord;
        import org.apache.kafka.clients.producer.RecordMetadata;
        import org.apache.kafka.common.serialization.StringSerializer;
        import org.apache.kafka.streams.StreamsConfig;
        import org.apache.kafka.streams.TopologyTestDriver;
        import org.apache.kafka.streams.test.ConsumerRecordFactory;
        import org.apache.kafka.streams.test.OutputVerifier;
        import org.apache.kafka.streams.KeyValue;

        import java.util.Properties;

        public class EmbeddedKafkaExample {
            private KafkaProducer<String, String> producer;

            public void setup() {
                Properties props = new Properties();
                props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
                props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
                props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

                producer = new KafkaProducer<>(props);
            }

            public void produceMessage(String topic, String key, String value) {
                ProducerRecord<String, String> record = new ProducerRecord<>(topic, key, value);
                producer.send(record, (RecordMetadata metadata, Exception exception) -> {
                    if (exception != null) {
                        exception.printStackTrace();
                    } else {
                        System.out.printf("Sent message to topic %s partition %d offset %d%n",
                                metadata.topic(), metadata.partition(), metadata.offset());
                    }
                });
            }

            public void teardown() {
                producer.close();
            }
        }
        ```

    - **Scala Example**:

        ```scala
        import org.apache.kafka.clients.producer.{KafkaProducer, ProducerConfig, ProducerRecord}
        import org.apache.kafka.common.serialization.StringSerializer

        import java.util.Properties

        object EmbeddedKafkaExample {
          def main(args: Array[String]): Unit = {
            val props = new Properties()
            props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092")
            props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, classOf[StringSerializer].getName)
            props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, classOf[StringSerializer].getName)

            val producer = new KafkaProducer[String, String](props)

            val record = new ProducerRecord[String, String]("test-topic", "key", "value")
            producer.send(record, (metadata, exception) => {
              if (exception != null) exception.printStackTrace()
              else println(s"Sent message to topic ${metadata.topic()} partition ${metadata.partition()} offset ${metadata.offset()}")
            })

            producer.close()
          }
        }
        ```

    - **Kotlin Example**:

        ```kotlin
        import org.apache.kafka.clients.producer.KafkaProducer
        import org.apache.kafka.clients.producer.ProducerConfig
        import org.apache.kafka.clients.producer.ProducerRecord
        import org.apache.kafka.common.serialization.StringSerializer
        import java.util.Properties

        fun main() {
            val props = Properties().apply {
                put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092")
                put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer::class.java.name)
                put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer::class.java.name)
            }

            val producer = KafkaProducer<String, String>(props)
            val record = ProducerRecord("test-topic", "key", "value")

            producer.send(record) { metadata, exception ->
                if (exception != null) exception.printStackTrace()
                else println("Sent message to topic ${metadata.topic()} partition ${metadata.partition()} offset ${metadata.offset()}")
            }

            producer.close()
        }
        ```

    - **Clojure Example**:

        ```clojure
        (ns embedded-kafka-example
          (:import [org.apache.kafka.clients.producer KafkaProducer ProducerConfig ProducerRecord]
                   [org.apache.kafka.common.serialization StringSerializer]))

        (defn create-producer []
          (let [props (doto (java.util.Properties.)
                        (.put ProducerConfig/BOOTSTRAP_SERVERS_CONFIG "localhost:9092")
                        (.put ProducerConfig/KEY_SERIALIZER_CLASS_CONFIG StringSerializer)
                        (.put ProducerConfig/VALUE_SERIALIZER_CLASS_CONFIG StringSerializer))]
            (KafkaProducer. props)))

        (defn produce-message [producer topic key value]
          (let [record (ProducerRecord. topic key value)]
            (.send producer record
                   (reify org.apache.kafka.clients.producer.Callback
                     (onCompletion [_ metadata exception]
                       (if exception
                         (.printStackTrace exception)
                         (println (format "Sent message to topic %s partition %d offset %d"
                                          (.topic metadata) (.partition metadata) (.offset metadata)))))))))

        (defn -main []
          (let [producer (create-producer)]
            (produce-message producer "test-topic" "key" "value")
            (.close producer)))
        ```

3. **Configure Broker Settings**: Customize the broker settings to match your testing requirements. This includes setting up topics, partitions, and replication factors.

4. **Run Tests**: Execute your tests, ensuring that the embedded Kafka broker is running and accessible.

### Setting Up Embedded Kafka with Testcontainers

Testcontainers provides a more realistic testing environment by running Kafka in a Docker container. This approach is beneficial for integration tests that require a more production-like setup.

#### Step-by-Step Guide

1. **Add Dependencies**: Include Testcontainers in your project. For Maven, add the following to your `pom.xml`:

    ```xml
    <dependency>
        <groupId>org.testcontainers</groupId>
        <artifactId>testcontainers</artifactId>
        <version>1.16.0</version>
        <scope>test</scope>
    </dependency>
    <dependency>
        <groupId>org.testcontainers</groupId>
        <artifactId>kafka</artifactId>
        <version>1.16.0</version>
        <scope>test</scope>
    </dependency>
    ```

2. **Initialize Kafka Container**: Create a class to manage the Kafka container lifecycle.

    - **Java Example**:

        ```java
        import org.testcontainers.containers.KafkaContainer;
        import org.testcontainers.utility.DockerImageName;

        public class TestcontainersKafkaExample {
            private KafkaContainer kafkaContainer;

            public void setup() {
                kafkaContainer = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:latest"));
                kafkaContainer.start();
            }

            public String getBootstrapServers() {
                return kafkaContainer.getBootstrapServers();
            }

            public void teardown() {
                kafkaContainer.stop();
            }
        }
        ```

    - **Scala Example**:

        ```scala
        import org.testcontainers.containers.KafkaContainer
        import org.testcontainers.utility.DockerImageName

        object TestcontainersKafkaExample {
          def main(args: Array[String]): Unit = {
            val kafkaContainer = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:latest"))
            kafkaContainer.start()

            println(s"Kafka bootstrap servers: ${kafkaContainer.getBootstrapServers}")

            kafkaContainer.stop()
          }
        }
        ```

    - **Kotlin Example**:

        ```kotlin
        import org.testcontainers.containers.KafkaContainer
        import org.testcontainers.utility.DockerImageName

        fun main() {
            val kafkaContainer = KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:latest"))
            kafkaContainer.start()

            println("Kafka bootstrap servers: ${kafkaContainer.bootstrapServers}")

            kafkaContainer.stop()
        }
        ```

    - **Clojure Example**:

        ```clojure
        (ns testcontainers-kafka-example
          (:import [org.testcontainers.containers KafkaContainer]
                   [org.testcontainers.utility DockerImageName]))

        (defn -main []
          (let [kafka-container (KafkaContainer. (DockerImageName/parse "confluentinc/cp-kafka:latest"))]
            (.start kafka-container)
            (println (str "Kafka bootstrap servers: " (.getBootstrapServers kafka-container)))
            (.stop kafka-container)))
        ```

3. **Configure Container Settings**: Adjust the container settings as needed, such as environment variables, network settings, and exposed ports.

4. **Run Tests**: Execute your tests, ensuring that the Kafka container is running and accessible.

### Configuring Brokers for Testing

When setting up embedded Kafka brokers, consider the following configurations to ensure comprehensive testing:

- **Topics and Partitions**: Define the topics and partitions required for your tests. Use different partitioning strategies to test load balancing and fault tolerance.
- **Replication Factor**: Set the replication factor to test data redundancy and failover scenarios.
- **Security Settings**: If your application uses security features like SSL or SASL, configure the embedded broker to support these protocols.
- **Data Retention**: Adjust data retention settings to simulate different data lifecycle scenarios.
- **Logging and Monitoring**: Enable logging and monitoring to capture test results and diagnose issues.

### Testing Multiple Scenarios

Embedded Kafka brokers allow you to test a variety of scenarios, including:

- **Message Production and Consumption**: Verify that messages are produced and consumed correctly.
- **Consumer Group Rebalancing**: Test how consumer groups handle rebalancing when consumers join or leave.
- **Error Handling**: Simulate errors and verify that your application handles them gracefully.
- **Performance Testing**: Measure the performance of your application under different load conditions.

### Best Practices

- **Isolation**: Ensure that each test runs in isolation to prevent interference from other tests.
- **Resource Management**: Monitor resource usage to avoid excessive consumption that could impact test performance.
- **Cleanup**: Always clean up resources after tests to prevent resource leaks and ensure a clean environment for subsequent tests.
- **Version Control**: Keep track of Kafka and library versions to ensure compatibility and reproducibility.

### Conclusion

Setting up embedded Kafka brokers is an essential part of integration testing for Kafka-based applications. By using libraries like Kafka Embedded and Testcontainers, you can create a realistic testing environment that simulates a full Kafka cluster. This approach allows you to test various scenarios, ensuring that your application interacts correctly with Kafka APIs. By following the guidelines and best practices outlined in this section, you can effectively test your Kafka-based applications and ensure their reliability and performance.

## Test Your Knowledge: Embedded Kafka Brokers for Integration Testing

{{< quizdown >}}

### What is the primary benefit of using embedded Kafka brokers for testing?

- [x] They provide a controlled and isolated testing environment.
- [ ] They are faster than production Kafka clusters.
- [ ] They offer better performance than external brokers.
- [ ] They are easier to configure than external brokers.

> **Explanation:** Embedded Kafka brokers provide a controlled and isolated environment, which is ideal for testing purposes.

### Which library is commonly used for running Kafka in a Docker container for testing?

- [x] Testcontainers
- [ ] Kafka Embedded
- [ ] Docker Compose
- [ ] Kubernetes

> **Explanation:** Testcontainers is a popular library for running Kafka in a Docker container for testing purposes.

### What is a key advantage of using Testcontainers for Kafka testing?

- [x] It provides a more realistic testing environment.
- [ ] It is faster to set up than Kafka Embedded.
- [ ] It requires less configuration than Kafka Embedded.
- [ ] It is more lightweight than Kafka Embedded.

> **Explanation:** Testcontainers provides a more realistic testing environment by running Kafka in a Docker container, simulating a production-like setup.

### Which of the following is a consideration when configuring embedded Kafka brokers?

- [x] Topics and partitions
- [ ] Database connections
- [ ] User interface design
- [ ] Network latency

> **Explanation:** Configuring topics and partitions is an important consideration when setting up embedded Kafka brokers for testing.

### What should be done after running tests with embedded Kafka brokers?

- [x] Clean up resources to prevent leaks.
- [ ] Increase the number of partitions.
- [ ] Deploy the application to production.
- [ ] Reduce the replication factor.

> **Explanation:** Cleaning up resources after tests is crucial to prevent resource leaks and ensure a clean environment for subsequent tests.

### Which language is NOT shown in the code examples for setting up embedded Kafka brokers?

- [ ] Java
- [ ] Scala
- [ ] Kotlin
- [x] Python

> **Explanation:** The code examples provided are in Java, Scala, Kotlin, and Clojure, but not Python.

### What is a common use case for embedded Kafka brokers in testing?

- [x] Verifying message production and consumption
- [ ] Designing user interfaces
- [ ] Managing database schemas
- [ ] Optimizing network protocols

> **Explanation:** Embedded Kafka brokers are commonly used to verify message production and consumption in testing scenarios.

### Which configuration is important for testing consumer group rebalancing?

- [x] Partitioning strategies
- [ ] Database schemas
- [ ] User interface design
- [ ] Network latency

> **Explanation:** Partitioning strategies are important for testing how consumer groups handle rebalancing when consumers join or leave.

### What is a benefit of using Kafka Embedded for testing?

- [x] It is lightweight and fast to set up.
- [ ] It provides a production-like environment.
- [ ] It requires no configuration.
- [ ] It offers better performance than Testcontainers.

> **Explanation:** Kafka Embedded is lightweight and fast to set up, making it ideal for unit and integration tests.

### True or False: Testcontainers can be used to run Kafka in a Docker container for integration testing.

- [x] True
- [ ] False

> **Explanation:** Testcontainers is a library that can be used to run Kafka in a Docker container for integration testing.

{{< /quizdown >}}

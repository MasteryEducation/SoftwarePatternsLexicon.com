---
canonical: "https://softwarepatternslexicon.com/kafka/14/3/1"
title: "Mastering Kafka Streams Testing with TopologyTestDriver"
description: "Explore the comprehensive guide to using TopologyTestDriver for testing Kafka Streams applications, including setup, input/output handling, and state store testing."
linkTitle: "14.3.1 TopologyTestDriver"
tags:
- "Apache Kafka"
- "Kafka Streams"
- "TopologyTestDriver"
- "Stream Processing"
- "Testing"
- "Java"
- "Scala"
- "Kotlin"
date: 2024-11-25
type: docs
nav_weight: 143100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.3.1 TopologyTestDriver

### Introduction to TopologyTestDriver

The `TopologyTestDriver` is a powerful tool provided by Kafka Streams for testing stream processing applications. It allows developers to test Kafka Streams topologies in isolation, without the need for a running Kafka cluster. This capability is crucial for unit testing, enabling developers to verify the correctness of their stream processing logic efficiently and effectively.

#### Benefits of Using TopologyTestDriver

- **Isolation**: Test your Kafka Streams logic without a running Kafka cluster, reducing dependencies and setup complexity.
- **Speed**: Execute tests quickly as there is no need to start and manage a Kafka cluster.
- **Determinism**: Achieve consistent test results by controlling the input and output of your streams.
- **Debugging**: Simplify debugging by focusing on the stream processing logic without external system noise.

### Setting Up TopologyTestDriver

To begin using the `TopologyTestDriver`, you need to set up your Kafka Streams topology and configure the driver. Below are examples in Java, Scala, Kotlin, and Clojure to illustrate the setup process.

#### Java Example

```java
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.Topology;
import org.apache.kafka.streams.TopologyTestDriver;
import org.apache.kafka.streams.test.ConsumerRecordFactory;
import org.apache.kafka.streams.test.OutputVerifier;
import org.apache.kafka.common.serialization.StringSerializer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.streams.test.TestRecord;

import java.util.Properties;

public class TopologyTestDriverExample {
    public static void main(String[] args) {
        // Define the properties for the test driver
        Properties props = new Properties();
        props.put("application.id", "test-app");
        props.put("bootstrap.servers", "dummy:1234");

        // Build the topology
        StreamsBuilder builder = new StreamsBuilder();
        builder.stream("input-topic").to("output-topic");
        Topology topology = builder.build();

        // Create the test driver
        TopologyTestDriver testDriver = new TopologyTestDriver(topology, props);

        // Create a factory for input records
        ConsumerRecordFactory<String, String> factory = new ConsumerRecordFactory<>(new StringSerializer(), new StringSerializer());

        // Pipe an input record into the topology
        testDriver.pipeInput(factory.create("input-topic", "key", "value"));

        // Verify the output record
        TestRecord<String, String> outputRecord = testDriver.readOutput("output-topic", new StringDeserializer(), new StringDeserializer());
        OutputVerifier.compareKeyValue(outputRecord, "key", "value");

        // Close the test driver
        testDriver.close();
    }
}
```

#### Scala Example

```scala
import org.apache.kafka.streams.scala._
import org.apache.kafka.streams.scala.kstream._
import org.apache.kafka.streams.{StreamsConfig, TopologyTestDriver}
import org.apache.kafka.streams.test.{ConsumerRecordFactory, OutputVerifier}
import org.apache.kafka.common.serialization.{Serdes, StringSerializer, StringDeserializer}

object TopologyTestDriverExample extends App {
  val builder = new StreamsBuilder()
  val source: KStream[String, String] = builder.stream[String, String]("input-topic")
  source.to("output-topic")

  val topology = builder.build()
  val props = new java.util.Properties()
  props.put(StreamsConfig.APPLICATION_ID_CONFIG, "test-app")
  props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "dummy:1234")

  val testDriver = new TopologyTestDriver(topology, props)
  val factory = new ConsumerRecordFactory[String, String](new StringSerializer(), new StringSerializer())

  testDriver.pipeInput(factory.create("input-topic", "key", "value"))

  val outputRecord = testDriver.readOutput("output-topic", new StringDeserializer(), new StringDeserializer())
  OutputVerifier.compareKeyValue(outputRecord, "key", "value")

  testDriver.close()
}
```

#### Kotlin Example

```kotlin
import org.apache.kafka.streams.StreamsBuilder
import org.apache.kafka.streams.TopologyTestDriver
import org.apache.kafka.streams.test.ConsumerRecordFactory
import org.apache.kafka.streams.test.OutputVerifier
import org.apache.kafka.common.serialization.StringSerializer
import org.apache.kafka.common.serialization.StringDeserializer
import java.util.Properties

fun main() {
    val builder = StreamsBuilder()
    builder.stream<String, String>("input-topic").to("output-topic")

    val topology = builder.build()
    val props = Properties().apply {
        put("application.id", "test-app")
        put("bootstrap.servers", "dummy:1234")
    }

    val testDriver = TopologyTestDriver(topology, props)
    val factory = ConsumerRecordFactory<String, String>(StringSerializer(), StringSerializer())

    testDriver.pipeInput(factory.create("input-topic", "key", "value"))

    val outputRecord = testDriver.readOutput("output-topic", StringDeserializer(), StringDeserializer())
    OutputVerifier.compareKeyValue(outputRecord, "key", "value")

    testDriver.close()
}
```

#### Clojure Example

```clojure
(ns topology-test-driver-example
  (:import [org.apache.kafka.streams StreamsBuilder TopologyTestDriver]
           [org.apache.kafka.streams.test ConsumerRecordFactory OutputVerifier]
           [org.apache.kafka.common.serialization StringSerializer StringDeserializer]
           [java.util Properties]))

(defn -main []
  (let [builder (StreamsBuilder.)
        _ (.stream builder "input-topic")
        _ (.to (.stream builder "input-topic") "output-topic")
        topology (.build builder)
        props (doto (Properties.)
                (.put "application.id" "test-app")
                (.put "bootstrap.servers" "dummy:1234"))
        test-driver (TopologyTestDriver. topology props)
        factory (ConsumerRecordFactory. (StringSerializer.) (StringSerializer.))]

    (.pipeInput test-driver (.create factory "input-topic" "key" "value"))

    (let [output-record (.readOutput test-driver "output-topic" (StringDeserializer.) (StringDeserializer.))]
      (OutputVerifier/compareKeyValue output-record "key" "value"))

    (.close test-driver)))
```

### Supplying Input Records

The `TopologyTestDriver` allows you to simulate input records using the `ConsumerRecordFactory`. This factory creates records that can be piped into the topology for testing.

#### Key Steps

1. **Create a ConsumerRecordFactory**: Use appropriate serializers for your key and value types.
2. **Generate Input Records**: Use the factory to create records with specified keys and values.
3. **Pipe Input Records**: Use the `pipeInput` method of `TopologyTestDriver` to feed records into the topology.

### Retrieving Output Records

After processing input records, you can retrieve the output from the topology using the `readOutput` method.

#### Key Steps

1. **Specify Output Topic**: Indicate the topic from which to read the output.
2. **Use Deserializers**: Apply appropriate deserializers for the key and value.
3. **Verify Output**: Use `OutputVerifier` to compare expected and actual output.

### Testing State Stores and Punctuators

State stores and punctuators are critical components in Kafka Streams applications. Testing them ensures that your application maintains correct state and processes time-based events accurately.

#### State Store Testing

- **Access State Stores**: Use `TopologyTestDriver` to access and verify the contents of state stores.
- **Verify State**: Ensure that the state store contains the expected data after processing input records.

#### Punctuator Testing

- **Simulate Time**: Use `TopologyTestDriver` to advance the stream time and trigger punctuators.
- **Verify Punctuator Actions**: Check that the punctuator performs the expected actions when triggered.

### Considerations and Best Practices

- **Isolation**: Keep tests isolated to ensure they do not depend on external systems.
- **Determinism**: Ensure tests are deterministic by controlling input and output.
- **Coverage**: Aim for comprehensive test coverage, including edge cases and error handling.
- **Performance**: While `TopologyTestDriver` is fast, ensure that your tests do not become a bottleneck.

### Conclusion

The `TopologyTestDriver` is an essential tool for testing Kafka Streams applications. By providing a controlled environment for testing, it allows developers to focus on the correctness of their stream processing logic. By following the examples and best practices outlined in this guide, you can effectively test your Kafka Streams applications and ensure they meet the desired requirements.

### References and Further Reading

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Documentation](https://docs.confluent.io/)
- [Kafka Streams Testing Guide](https://kafka.apache.org/23/documentation/streams/developer-guide/testing.html)

## Test Your Knowledge: Mastering Kafka Streams Testing with TopologyTestDriver

{{< quizdown >}}

### What is the primary benefit of using TopologyTestDriver?

- [x] It allows testing Kafka Streams topologies without a running Kafka cluster.
- [ ] It improves the performance of Kafka Streams applications.
- [ ] It provides real-time monitoring of Kafka Streams.
- [ ] It simplifies the deployment of Kafka Streams applications.

> **Explanation:** The primary benefit of using `TopologyTestDriver` is that it allows testing Kafka Streams topologies without the need for a running Kafka cluster, enabling isolated and efficient unit testing.

### Which method is used to feed input records into the topology in TopologyTestDriver?

- [x] pipeInput
- [ ] readOutput
- [ ] createInput
- [ ] addInput

> **Explanation:** The `pipeInput` method is used to feed input records into the topology in `TopologyTestDriver`.

### How can you verify the output of a Kafka Streams topology using TopologyTestDriver?

- [x] Use the OutputVerifier class.
- [ ] Use the OutputComparator class.
- [ ] Use the OutputChecker class.
- [ ] Use the OutputAnalyzer class.

> **Explanation:** The `OutputVerifier` class is used to verify the output of a Kafka Streams topology by comparing expected and actual output records.

### What is the role of ConsumerRecordFactory in TopologyTestDriver?

- [x] It creates input records for testing.
- [ ] It verifies output records.
- [ ] It manages state stores.
- [ ] It configures the test driver.

> **Explanation:** The `ConsumerRecordFactory` is used to create input records that can be piped into the topology for testing.

### Which of the following is a key consideration when testing state stores with TopologyTestDriver?

- [x] Access and verify the contents of state stores.
- [ ] Ensure state stores are empty before testing.
- [ ] Use state stores only for output verification.
- [ ] Avoid using state stores in tests.

> **Explanation:** When testing state stores with `TopologyTestDriver`, it is important to access and verify the contents of the state stores to ensure they contain the expected data after processing input records.

### How can you simulate time to test punctuators in TopologyTestDriver?

- [x] Advance the stream time using TopologyTestDriver.
- [ ] Use a time simulator tool.
- [ ] Manually adjust the system clock.
- [ ] Use a time-based trigger.

> **Explanation:** You can simulate time to test punctuators by advancing the stream time using `TopologyTestDriver`, which allows you to trigger punctuators and verify their actions.

### What is a best practice for ensuring test determinism with TopologyTestDriver?

- [x] Control input and output records.
- [ ] Use random input data.
- [ ] Rely on external systems for input.
- [ ] Test with varying configurations.

> **Explanation:** To ensure test determinism with `TopologyTestDriver`, it is best to control input and output records, providing consistent and predictable test results.

### Which programming languages are demonstrated in the TopologyTestDriver examples?

- [x] Java, Scala, Kotlin, Clojure
- [ ] Python, Ruby, JavaScript, PHP
- [ ] C++, C#, Go, Rust
- [ ] Swift, Objective-C, Perl, Haskell

> **Explanation:** The `TopologyTestDriver` examples in the guide are demonstrated in Java, Scala, Kotlin, and Clojure.

### What is the purpose of the readOutput method in TopologyTestDriver?

- [x] To retrieve output records from the topology.
- [ ] To feed input records into the topology.
- [ ] To configure the test driver.
- [ ] To verify state store contents.

> **Explanation:** The `readOutput` method in `TopologyTestDriver` is used to retrieve output records from the topology for verification.

### True or False: TopologyTestDriver can be used for integration testing with a running Kafka cluster.

- [ ] True
- [x] False

> **Explanation:** False. `TopologyTestDriver` is designed for unit testing Kafka Streams topologies without the need for a running Kafka cluster, making it unsuitable for integration testing with a live cluster.

{{< /quizdown >}}

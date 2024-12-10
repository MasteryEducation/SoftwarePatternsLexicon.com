---
canonical: "https://softwarepatternslexicon.com/kafka/4/4/1"
title: "Kafka Message Delivery Semantics: At-Most-Once, At-Least-Once, and Exactly-Once"
description: "Explore the intricacies of Kafka's message delivery semantics, including at-most-once, at-least-once, and exactly-once guarantees. Learn how to implement each level, understand their trade-offs, and discover practical use cases."
linkTitle: "4.4.1 At-Most-Once, At-Least-Once, and Exactly-Once Semantics"
tags:
- "Apache Kafka"
- "Message Delivery"
- "At-Most-Once"
- "At-Least-Once"
- "Exactly-Once"
- "Idempotent Producers"
- "Transactions"
- "Data Processing"
date: 2024-11-25
type: docs
nav_weight: 44100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 4.4.1 At-Most-Once, At-Least-Once, and Exactly-Once Semantics

In the realm of distributed systems and stream processing, ensuring reliable data delivery is paramount. Apache Kafka, a leading distributed event streaming platform, offers three distinct levels of message delivery semantics: at-most-once, at-least-once, and exactly-once. Each of these semantics provides different guarantees and trade-offs, influencing how data is processed and delivered across systems. This section delves into these semantics, exploring their implementations, configurations, and suitable use cases.

### At-Most-Once Delivery

#### Definition

At-most-once delivery semantics ensure that a message is delivered zero or one time. This means that there is no guarantee that a message will be delivered, but if it is, it will be delivered only once. This semantic is the simplest in terms of implementation but comes with the risk of data loss.

#### Implementation

In Kafka, at-most-once delivery can be achieved by not retrying message delivery in case of failure. This approach is typically used when the application can tolerate data loss, or when the cost of reprocessing is higher than the cost of losing some messages.

- **Producer Configuration**: Set `acks=0` to ensure that the producer does not wait for any acknowledgment from the broker.
- **Consumer Configuration**: Commit offsets immediately after receiving messages, without waiting for processing to complete.

#### Code Example

**Java**

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("acks", "0"); // No acknowledgment
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
ProducerRecord<String, String> record = new ProducerRecord<>("topic", "key", "value");
producer.send(record);
producer.close();
```

**Scala**

```scala
val props = new Properties()
props.put("bootstrap.servers", "localhost:9092")
props.put("acks", "0") // No acknowledgment
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

val producer = new KafkaProducer[String, String](props)
val record = new ProducerRecord[String, String]("topic", "key", "value")
producer.send(record)
producer.close()
```

**Kotlin**

```kotlin
val props = Properties().apply {
    put("bootstrap.servers", "localhost:9092")
    put("acks", "0") // No acknowledgment
    put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
}

val producer = KafkaProducer<String, String>(props)
val record = ProducerRecord("topic", "key", "value")
producer.send(record)
producer.close()
```

**Clojure**

```clojure
(def props
  {"bootstrap.servers" "localhost:9092"
   "acks" "0" ; No acknowledgment
   "key.serializer" "org.apache.kafka.common.serialization.StringSerializer"
   "value.serializer" "org.apache.kafka.common.serialization.StringSerializer"})

(def producer (KafkaProducer. props))
(def record (ProducerRecord. "topic" "key" "value"))
(.send producer record)
(.close producer)
```

#### Use Cases

- **Logging Systems**: Where occasional data loss is acceptable.
- **Telemetry Data**: In scenarios where data can be sampled, and missing data points do not significantly impact the overall analysis.

### At-Least-Once Delivery

#### Definition

At-least-once delivery semantics ensure that a message is delivered one or more times. This means that while there is a guarantee that a message will be delivered, it may be delivered more than once, leading to potential duplicates.

#### Default Behavior in Kafka

Kafka's default behavior is at-least-once delivery. This is achieved by retrying message delivery until an acknowledgment is received. Consumers must handle potential duplicates by implementing idempotent processing logic.

- **Producer Configuration**: Set `acks=all` to ensure that the producer waits for the acknowledgment from all in-sync replicas.
- **Consumer Configuration**: Commit offsets after processing messages to ensure that messages are not lost.

#### Code Example

**Java**

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("acks", "all"); // Wait for all replicas
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
ProducerRecord<String, String> record = new ProducerRecord<>("topic", "key", "value");
producer.send(record, (metadata, exception) -> {
    if (exception != null) {
        // Handle exception
    }
});
producer.close();
```

**Scala**

```scala
val props = new Properties()
props.put("bootstrap.servers", "localhost:9092")
props.put("acks", "all") // Wait for all replicas
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

val producer = new KafkaProducer[String, String](props)
val record = new ProducerRecord[String, String]("topic", "key", "value")
producer.send(record, (metadata, exception) => {
  if (exception != null) {
    // Handle exception
  }
})
producer.close()
```

**Kotlin**

```kotlin
val props = Properties().apply {
    put("bootstrap.servers", "localhost:9092")
    put("acks", "all") // Wait for all replicas
    put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
}

val producer = KafkaProducer<String, String>(props)
val record = ProducerRecord("topic", "key", "value")
producer.send(record) { metadata, exception ->
    if (exception != null) {
        // Handle exception
    }
}
producer.close()
```

**Clojure**

```clojure
(def props
  {"bootstrap.servers" "localhost:9092"
   "acks" "all" ; Wait for all replicas
   "key.serializer" "org.apache.kafka.common.serialization.StringSerializer"
   "value.serializer" "org.apache.kafka.common.serialization.StringSerializer"})

(def producer (KafkaProducer. props))
(def record (ProducerRecord. "topic" "key" "value"))
(.send producer record
  (reify Callback
    (onCompletion [this metadata exception]
      (when exception
        ;; Handle exception
        ))))
(.close producer)
```

#### Use Cases

- **Financial Transactions**: Where data loss is unacceptable, but duplicates can be managed.
- **Order Processing Systems**: Where ensuring data delivery is critical, and duplicates can be filtered out.

### Exactly-Once Semantics

#### Definition

Exactly-once delivery semantics ensure that a message is delivered exactly one time, with no duplicates and no data loss. This is the most complex semantic to achieve, as it requires coordination between producers, brokers, and consumers.

#### Kafka's Support for Exactly-Once Semantics

Kafka supports exactly-once semantics through a combination of idempotent producers and transactions. This ensures that messages are not duplicated and are processed exactly once.

- **Idempotent Producers**: Ensure that duplicate messages are not produced.
- **Transactions**: Ensure that a group of operations (produce and consume) are atomic.

#### Code Example

**Java**

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("acks", "all");
props.put("enable.idempotence", "true"); // Enable idempotence
props.put("transactional.id", "my-transactional-id"); // Set transactional ID
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();

try {
    producer.beginTransaction();
    ProducerRecord<String, String> record = new ProducerRecord<>("topic", "key", "value");
    producer.send(record);
    producer.commitTransaction();
} catch (ProducerFencedException | OutOfOrderSequenceException | AuthorizationException e) {
    // Fatal errors, cannot recover
    producer.close();
} catch (KafkaException e) {
    // Abort transaction and retry
    producer.abortTransaction();
}
producer.close();
```

**Scala**

```scala
val props = new Properties()
props.put("bootstrap.servers", "localhost:9092")
props.put("acks", "all")
props.put("enable.idempotence", "true") // Enable idempotence
props.put("transactional.id", "my-transactional-id") // Set transactional ID
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

val producer = new KafkaProducer[String, String](props)
producer.initTransactions()

try {
  producer.beginTransaction()
  val record = new ProducerRecord[String, String]("topic", "key", "value")
  producer.send(record)
  producer.commitTransaction()
} catch {
  case e: ProducerFencedException | OutOfOrderSequenceException | AuthorizationException =>
    // Fatal errors, cannot recover
    producer.close()
  case e: KafkaException =>
    // Abort transaction and retry
    producer.abortTransaction()
}
producer.close()
```

**Kotlin**

```kotlin
val props = Properties().apply {
    put("bootstrap.servers", "localhost:9092")
    put("acks", "all")
    put("enable.idempotence", "true") // Enable idempotence
    put("transactional.id", "my-transactional-id") // Set transactional ID
    put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
}

val producer = KafkaProducer<String, String>(props)
producer.initTransactions()

try {
    producer.beginTransaction()
    val record = ProducerRecord("topic", "key", "value")
    producer.send(record)
    producer.commitTransaction()
} catch (e: ProducerFencedException) {
    // Fatal errors, cannot recover
    producer.close()
} catch (e: KafkaException) {
    // Abort transaction and retry
    producer.abortTransaction()
}
producer.close()
```

**Clojure**

```clojure
(def props
  {"bootstrap.servers" "localhost:9092"
   "acks" "all"
   "enable.idempotence" "true" ; Enable idempotence
   "transactional.id" "my-transactional-id" ; Set transactional ID
   "key.serializer" "org.apache.kafka.common.serialization.StringSerializer"
   "value.serializer" "org.apache.kafka.common.serialization.StringSerializer"})

(def producer (KafkaProducer. props))
(.initTransactions producer)

(try
  (.beginTransaction producer)
  (let [record (ProducerRecord. "topic" "key" "value")]
    (.send producer record))
  (.commitTransaction producer)
  (catch ProducerFencedException e
    ;; Fatal errors, cannot recover
    (.close producer))
  (catch KafkaException e
    ;; Abort transaction and retry
    (.abortTransaction producer)))
(.close producer)
```

#### Use Cases

- **Financial Systems**: Where duplicate transactions can lead to significant issues.
- **Inventory Management**: Where accurate counts are crucial, and duplicates can cause discrepancies.

### Performance and Complexity Implications

- **At-Most-Once**: Offers the best performance due to minimal overhead but risks data loss.
- **At-Least-Once**: Balances performance and reliability, with potential duplicate processing.
- **Exactly-Once**: Provides the highest reliability but at the cost of increased complexity and resource usage.

### Configuration Parameters

- **acks**: Controls the acknowledgment level (`0`, `1`, `all`).
- **enable.idempotence**: Enables idempotent producer mode.
- **transactional.id**: Sets a unique transactional ID for exactly-once semantics.

### Conclusion

Understanding and implementing the appropriate message delivery semantics in Kafka is crucial for building robust and reliable systems. By carefully selecting the right semantic based on the application's requirements, developers can optimize for performance, reliability, or a balance of both.

## Test Your Knowledge: Kafka Message Delivery Semantics Quiz

{{< quizdown >}}

### What is the primary characteristic of at-most-once delivery semantics?

- [x] Messages may be lost but are never duplicated.
- [ ] Messages are always delivered at least once.
- [ ] Messages are delivered exactly once.
- [ ] Messages are delivered multiple times.

> **Explanation:** At-most-once semantics ensure that messages may be lost but are never duplicated, prioritizing performance over reliability.

### Which Kafka configuration is used to achieve at-least-once delivery?

- [x] `acks=all`
- [ ] `acks=0`
- [ ] `enable.idempotence=true`
- [ ] `transactional.id=my-transactional-id`

> **Explanation:** At-least-once delivery is achieved by setting `acks=all`, ensuring that messages are acknowledged by all in-sync replicas.

### What is the role of idempotent producers in Kafka?

- [x] To prevent duplicate message production.
- [ ] To ensure messages are delivered at most once.
- [ ] To manage consumer offsets.
- [ ] To handle message serialization.

> **Explanation:** Idempotent producers prevent duplicate message production, ensuring exactly-once semantics.

### How does Kafka achieve exactly-once semantics?

- [x] Through idempotent producers and transactions.
- [ ] By setting `acks=0`.
- [ ] By using consumer groups.
- [ ] By enabling SSL encryption.

> **Explanation:** Kafka achieves exactly-once semantics through idempotent producers and transactions, ensuring atomic message processing.

### In which scenario is at-most-once delivery most appropriate?

- [x] Logging systems where occasional data loss is acceptable.
- [ ] Financial transactions where data loss is unacceptable.
- [x] Telemetry data where missing data points are tolerable.
- [ ] Inventory management where accurate counts are crucial.

> **Explanation:** At-most-once delivery is suitable for systems like logging and telemetry, where occasional data loss is acceptable.

### What is the default delivery semantic in Kafka?

- [x] At-least-once
- [ ] At-most-once
- [ ] Exactly-once
- [ ] None of the above

> **Explanation:** Kafka's default delivery semantic is at-least-once, ensuring messages are delivered at least once.

### Which configuration parameter enables idempotent producer mode?

- [x] `enable.idempotence=true`
- [ ] `acks=all`
- [x] `transactional.id=my-transactional-id`
- [ ] `acks=0`

> **Explanation:** The `enable.idempotence=true` configuration enables idempotent producer mode, preventing duplicate message production.

### What is a potential drawback of exactly-once semantics?

- [x] Increased complexity and resource usage.
- [ ] Data loss.
- [ ] Duplicate message delivery.
- [ ] Reduced reliability.

> **Explanation:** Exactly-once semantics increase complexity and resource usage, as they require coordination between producers, brokers, and consumers.

### Which semantic is best for financial systems where duplicate transactions are problematic?

- [x] Exactly-once
- [ ] At-most-once
- [ ] At-least-once
- [ ] None of the above

> **Explanation:** Exactly-once semantics are best for financial systems where duplicate transactions are problematic, ensuring messages are processed exactly once.

### True or False: At-least-once semantics guarantee no data loss but may result in duplicate messages.

- [x] True
- [ ] False

> **Explanation:** At-least-once semantics guarantee no data loss but may result in duplicate messages, requiring consumers to handle duplicates.

{{< /quizdown >}}

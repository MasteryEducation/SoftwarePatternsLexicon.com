---
canonical: "https://softwarepatternslexicon.com/kafka/13/1/2"
title: "Mastering Kafka Delivery Guarantees: At-Most-Once, At-Least-Once, and Exactly-Once"
description: "Explore Kafka's delivery guarantees from the producer's perspective, including configuration settings and trade-offs between reliability and performance."
linkTitle: "13.1.2 Delivery Guarantees"
tags:
- "Apache Kafka"
- "Delivery Guarantees"
- "Producer Configuration"
- "Fault Tolerance"
- "Reliability Patterns"
- "At-Most-Once"
- "At-Least-Once"
- "Exactly-Once"
date: 2024-11-25
type: docs
nav_weight: 131200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 13.1.2 Delivery Guarantees

In the world of distributed systems, ensuring reliable message delivery is paramount. Apache Kafka, a cornerstone of modern data architectures, offers robust delivery guarantees that cater to various application needs. This section delves into Kafka's delivery guarantees—at-most-once, at-least-once, and exactly-once—from the producer's perspective. We will explore how to configure producers to meet the desired guarantee level, discuss the trade-offs between reliability and performance, and provide best practices for configuring producers based on application requirements.

### Understanding Delivery Guarantees

#### At-Most-Once Delivery

**Definition**: At-most-once delivery ensures that messages are delivered zero or one time. This guarantee prioritizes performance over reliability, as it does not attempt to resend messages that may have been lost during transmission.

**Implications**: 
- **Data Loss**: There is a risk of data loss if a message fails to be delivered.
- **Performance**: This approach minimizes latency and resource usage, making it suitable for applications where occasional data loss is acceptable.

#### At-Least-Once Delivery

**Definition**: At-least-once delivery guarantees that every message is delivered at least once. This is achieved by retrying message delivery until an acknowledgment is received.

**Implications**:
- **Duplication**: Messages may be delivered more than once, requiring consumers to handle duplicates.
- **Reliability**: This guarantee is suitable for applications where data loss is unacceptable, but duplicates can be managed.

#### Exactly-Once Delivery

**Definition**: Exactly-once delivery ensures that each message is delivered exactly once, with no duplicates or losses. This is the most stringent guarantee and is crucial for applications requiring high data integrity.

**Implications**:
- **Complexity**: Achieving exactly-once semantics involves additional overhead and complexity, particularly in distributed systems.
- **Performance**: This guarantee may impact performance due to the need for additional coordination and state management.

### Configuring Producers for Delivery Guarantees

Kafka producers can be configured to achieve the desired delivery guarantee level through various settings. Key configuration parameters include `acks`, `retries`, and `retries.backoff.ms`.

#### Producer Configuration Parameters

- **`acks`**: Determines the number of acknowledgments the producer requires before considering a request complete.
  - **`acks=0`**: The producer does not wait for any acknowledgment. This setting is used for at-most-once delivery.
  - **`acks=1`**: The producer waits for the leader to acknowledge the record. This setting is commonly used for at-least-once delivery.
  - **`acks=all`**: The producer waits for the full set of in-sync replicas to acknowledge the record. This setting is used for exactly-once delivery.

- **`retries`**: Specifies the number of retry attempts if a request fails.
  - **Higher retry values** increase the likelihood of successful delivery, supporting at-least-once and exactly-once semantics.

- **`retries.backoff.ms`**: Sets the time to wait before retrying a failed request.
  - **Longer backoff times** can help reduce the load on the Kafka cluster during retries.

#### Achieving At-Most-Once Delivery

To configure a producer for at-most-once delivery, prioritize performance and minimize resource usage:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("acks", "0"); // No acknowledgment
props.put("retries", "0"); // No retries

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
```

**Trade-offs**: While this configuration minimizes latency, it risks data loss, making it suitable for non-critical data.

#### Achieving At-Least-Once Delivery

To configure a producer for at-least-once delivery, ensure retries and acknowledgments are enabled:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("acks", "1"); // Leader acknowledgment
props.put("retries", "3"); // Retry three times
props.put("retries.backoff.ms", "100"); // 100ms backoff

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
```

**Trade-offs**: This configuration improves reliability but may result in duplicate messages, requiring consumer-side deduplication.

#### Achieving Exactly-Once Delivery

To configure a producer for exactly-once delivery, use idempotent producers and transactional APIs:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("acks", "all"); // All replicas acknowledgment
props.put("enable.idempotence", "true"); // Enable idempotence
props.put("transactional.id", "my-transactional-id"); // Set transactional ID

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();
```

**Trade-offs**: This configuration ensures data integrity but may introduce latency and complexity due to transactional overhead.

### Best Practices for Configuring Producers

1. **Understand Application Requirements**: Determine the appropriate delivery guarantee based on the criticality of data and tolerance for duplicates or losses.

2. **Optimize for Performance**: Balance reliability with performance by tuning `acks`, `retries`, and `retries.backoff.ms` settings.

3. **Use Idempotent Producers**: For exactly-once semantics, enable idempotence to prevent duplicate message production.

4. **Leverage Transactions**: Use Kafka's transactional API for atomic writes across multiple partitions or topics.

5. **Monitor and Adjust**: Continuously monitor producer performance and adjust configurations as needed to meet evolving application demands.

### Real-World Scenarios

- **Financial Transactions**: Require exactly-once delivery to ensure data integrity and prevent duplicate transactions.
- **Log Aggregation**: Can tolerate at-least-once delivery, as duplicate logs can be filtered during processing.
- **Telemetry Data**: May use at-most-once delivery for non-critical metrics where occasional data loss is acceptable.

### Conclusion

Kafka's delivery guarantees provide flexibility to meet diverse application needs, from high-performance scenarios to those requiring stringent data integrity. By understanding and configuring producer settings appropriately, developers can optimize Kafka's reliability and performance to suit their specific use cases.

## Test Your Knowledge: Kafka Delivery Guarantees Quiz

{{< quizdown >}}

### What is the primary benefit of at-most-once delivery?

- [x] Minimizes latency and resource usage
- [ ] Ensures no data loss
- [ ] Guarantees exactly-once delivery
- [ ] Provides duplicate message handling

> **Explanation:** At-most-once delivery prioritizes performance by minimizing latency and resource usage, but it risks data loss.

### Which `acks` setting is used for exactly-once delivery?

- [ ] `acks=0`
- [ ] `acks=1`
- [x] `acks=all`
- [ ] `acks=2`

> **Explanation:** `acks=all` ensures that the full set of in-sync replicas acknowledges the record, supporting exactly-once delivery.

### What is a potential drawback of at-least-once delivery?

- [ ] Data loss
- [x] Duplicate messages
- [ ] Increased latency
- [ ] No retries

> **Explanation:** At-least-once delivery may result in duplicate messages, requiring consumer-side deduplication.

### How can exactly-once delivery be achieved in Kafka?

- [ ] By setting `acks=0`
- [ ] By disabling retries
- [x] By enabling idempotence and using transactions
- [ ] By using `acks=1`

> **Explanation:** Exactly-once delivery is achieved by enabling idempotence and using Kafka's transactional API.

### What is the role of `retries` in producer configuration?

- [x] To specify the number of retry attempts for failed requests
- [ ] To determine the number of acknowledgments required
- [ ] To set the time to wait before retrying
- [ ] To enable idempotence

> **Explanation:** The `retries` parameter specifies how many times the producer will attempt to resend a failed request.

### Which delivery guarantee is most suitable for financial transactions?

- [ ] At-most-once
- [ ] At-least-once
- [x] Exactly-once
- [ ] None of the above

> **Explanation:** Financial transactions require exactly-once delivery to ensure data integrity and prevent duplicates.

### What does `retries.backoff.ms` control in producer settings?

- [ ] The number of retry attempts
- [x] The time to wait before retrying a failed request
- [ ] The acknowledgment level
- [ ] The transactional ID

> **Explanation:** `retries.backoff.ms` sets the time interval to wait before retrying a failed request.

### Why is idempotence important for exactly-once delivery?

- [x] It prevents duplicate message production
- [ ] It reduces latency
- [ ] It increases throughput
- [ ] It simplifies configuration

> **Explanation:** Idempotence ensures that duplicate messages are not produced, which is crucial for exactly-once delivery.

### What is a trade-off of using exactly-once delivery?

- [ ] Increased data loss
- [x] Additional latency and complexity
- [ ] Reduced reliability
- [ ] No duplicate handling

> **Explanation:** Exactly-once delivery introduces additional latency and complexity due to transactional overhead.

### True or False: At-most-once delivery guarantees no data loss.

- [ ] True
- [x] False

> **Explanation:** At-most-once delivery does not guarantee no data loss; it prioritizes performance over reliability.

{{< /quizdown >}}

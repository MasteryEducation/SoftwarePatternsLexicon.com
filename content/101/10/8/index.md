---
linkTitle: "Message Sequencing"
title: "Message Sequencing: Maintaining Order of Messages"
category: "Delivery Semantics"
series: "Stream Processing Design Patterns"
description: "Maintaining the order of messages to ensure they are processed in the correct sequence."
categories:
- cloud computing
- stream processing
- design patterns
tags:
- message sequencing
- delivery semantics
- streaming
- distributed systems
- workflow management
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/10/8"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In distributed systems, preserving the order of messages is crucial for several applications that demand sequential data processing. This design pattern, known as **Message Sequencing**, addresses the challenge of maintaining the order of messages for accurate processing, especially in environments where messages can arrive out of sequence due to network delays, system failures, or parallel processing.

## Problem Statement

When dealing with distributed systems or stream processing, ensuring that messages arrive and are processed in a specific order is crucial for applications like transaction processing, event sourcing, and workflow orchestration. The main challenge is that messages can arrive out of order due to the inherently unreliable nature of networks and asynchronous processing.

## Applicability

This pattern is suitable for:

- Systems requiring ordered processing of messages, such as transactional systems.
- Applications where the sequence of operations dictates the correctness of processing, like workflows and stateful applications.

## Solution

### Architectural Approach

To correctly implement Message Sequencing:

1. **Sequence Numbers**: Attach sequence numbers to each message. This allows the receiver to reconstruct the original order of messages.

2. **Deterministic Ordering**: Use deterministic algorithms to sort incoming messages based on sequence numbers before processing them.

3. **Buffering and Reordering**: Implement buffers to hold out-of-order messages until missing or earlier messages arrive, ensuring ordered processing.

4. **Processing Guarantees**: Leverage techniques like idempotent message processing to handle retries without affecting system integrity.

### Implementation Strategies

1. **Single-Threaded Queue**: Use a central queue to buffer messages and ensure sequential processing. Ideal for low-throughput systems.

2. **Partitioned Logs**: Utilize partitioned logs like those in Apache Kafka, where each partition is an ordered, immutable sequence of messages.

3. **Sorting Buffers**: For systems with distributed consumers, sorting buffers can hold messages in memory until the sequence is ensured.

4. **Use of Message Brokers**: Configure message brokers that guarantee delivery in order, like RabbitMQ with FIFO queues.

### Example Code

Here is an illustrative example using Apache Kafka in Java:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "example-group");
props.put("key.deserializer", 
           "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", 
           "org.apache.kafka.common.serialization.StringDeserializer");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("example-topic"));

try {
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        List<ConsumerRecord<String, String>> sortedRecords = new ArrayList<>(records);
        sortedRecords.sort(Comparator.comparing(ConsumerRecord::offset)); // Sorting records by offset
        for (ConsumerRecord<String, String> record : sortedRecords) {
            System.out.printf("offset = %d, key = %s, value = %s%n", record.offset(), record.key(), record.value());
        }
    }
} finally {
    consumer.close();
}
```

## Advantages

- **Consistency**: Ensures that messages are processed in the intended order, preserving data integrity and application state consistency.
- **Error Handling**: Simplifies handling of repeated processing attempts using idempotency.

## Challenges

- **Latency**: Introducing sequence-based buffering can add latency to message processing.
- **Scalability**: Requires careful design to ensure buffers and resources don't become bottlenecks.

## Related Patterns

- **Idempotent Receiver Pattern**: Used to manage effects of duplicate messages.
- **Retry Pattern**: Handles transient errors by retrying message delivery.

## Additional Resources

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [RabbitMQ Tutorials and Examples](https://www.rabbitmq.com/getstarted.html)

## Summary

Message Sequencing is essential in systems where processing order impacts application correctness. By utilizing sequence numbers, sorting algorithms, and buffering strategies, developers and architects can ensure that messages are processed accurately and efficiently, even in a distributed environment. This pattern plays a crucial role in maintaining system resilience and operational integrity in complex workflows.



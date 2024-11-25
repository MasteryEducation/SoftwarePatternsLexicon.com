---
linkTitle: "Event Sourcing"
title: "Event Sourcing: Managing State with Complete Event Logs"
category: "Late Arrival Handling"
series: "Stream Processing Design Patterns"
description: "Exploring the Event Sourcing pattern which involves storing a complete log of events, enabling systems to rebuild state or recompute results efficiently for handling late arrivals."
categories:
- late-arrival-handling
- event-driven-architecture
- distributed-systems
tags:
- event-sourcing
- stream-processing
- kafka
- immutability
- state-reconstruction
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/12/11"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview

**Event Sourcing** is a design pattern used primarily in event-driven architectures to maintain a complete log of all events that change the state of a system. This approach allows developers to replay events to reconstruct past states or recover from failures, making it particularly useful for handling late-arriving events in streaming contexts.

## Detailed Explanation

Event Sourcing moves away from traditional persistence models by persisting the changes that lead to a certain state rather than the state itself. When using this pattern, every change to an application’s state is captured as a sequence of events. By storing these events in an append-only log:

- Systems can rebuild the current state by replaying the sequence of events from the start.
- Systems can handle late-arriving (out-of-order) events by reapplying these events to achieve eventual consistency.
- Systems can perform auditing and temporal queries on the full history of changes.

### Components

1. **Event Store**: A storage mechanism for persisting the sequence of events. This can be implemented using systems like Apache Kafka or Amazon Kinesis.
   
2. **Event Processor**: Consumes events and applies them to update the state stored in materialized views or other storage services.

3. **Replayer**: Component responsible for replaying events to recreate state or process late events.

## Example Code

Consider an event logging implementation using Kafka:

```java
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;

public class EventLogProducer {
    private final Producer<String, String> producer;

    public EventLogProducer(String bootstrapServers) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        this.producer = new KafkaProducer<>(props);
    }

    public void logEvent(String topic, String eventKey, String eventValue) {
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, eventKey, eventValue);
        producer.send(record);
    }

    public void close() {
        producer.close();
    }
}
```

This simple Kafka producer logs events by serializing them as strings. Events are pushed to a Kafka topic, preserving order and immutability. For processing and replaying these events, Kafka consumers would separately retrieve and apply the events to state stores.

## Architectural Approaches

1. **Immutable Event Logs**: All state changes are captured as a historic sequence of append-only events, ensuring immutability and traceability.
   
2. **Event Replay**: The capability to replay events from the log to rebuild state or recompute its effects, effectively making the system capable of recalculating current application state from its history.

3. **CQRS (Command Query Responsibility Segregation)**: Often used alongside Event Sourcing, CQRS separates the command operations (state changes) from the query paths (state reads), allowing different models for updates and reads.

## Best Practices

- **Schema Evolution**: Plan for changes to event structure over time to ensure backward compatibility.
- **Idempotency**: Ensure that event handlers or processors are idempotent to safely handle duplicate event deliveries.
- **Snapshotting**: Optimize performance by periodically capturing snapshots of the state to avoid replaying the entire event history.

## Related Patterns

- **CQRS**: Often used with Event Sourcing to handle querying and command responsibilities differently.
- **Saga Pattern**: A pattern for managing complex transactions involving multiple microservices ensuring eventual consistency across them.
- **Change Data Capture (CDC)**: Captures and routes changes in data stores, similar in purpose to replicate state changes but from a data-centric approach.

## Additional Resources

- [Martin Fowler's Blog on Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936160/)

## Summary

Event Sourcing is a powerful pattern for managing state changes, allowing systems to maintain a complete, immutable log of all state changes. This enables rebuilding state optimally and recovers like handling late-arriving events in distributed systems. Adopting this pattern requires considering how events are stored, how they are replayed, and ensuring idempotency to manage duplicates efficiently.

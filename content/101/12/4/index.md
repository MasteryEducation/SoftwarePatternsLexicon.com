---
linkTitle: "Out-of-Order Processing"
title: "Out-of-Order Processing: Handling Late Arrivals in Stream Processing"
category: "Late Arrival Handling"
series: "Stream Processing Design Patterns"
description: "Out-of-Order Processing is a design pattern in stream processing that allows systems to accurately process events that arrive out-of-sequence, by implementing mechanisms such as reordering or buffering the events with the help of timestamps and sequence numbers."
categories:
- cloud-computing
- stream-processing
- data-management
tags:
- out-of-order-processing
- stream
- events
- sequence
- late-arrival
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/12/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

Out-of-order processing in stream processing environments refers to the design pattern employed to handle events that arrive at a system in a non-chronological sequence. This scenario is prevalent in distributed systems where network latency, cross-regional data transfer, and fluctuating system loads can cause delays in event delivery.

### Key Concepts

#### Event Time vs. Processing Time
In an out-of-order system, it is vital to comprehend the difference between event time and processing time:
- **Event Time**: The timestamp when an event actually occurred, usually embedded within the event data.
- **Processing Time**: The timestamp when an event enters the system for processing.

Handling out-of-order events necessitates viewing the event timeline based on the embedded event time rather than the system’s processing timeline.

### Core Components and Strategies

#### 1. Timestamps and Watermarks
- **Timestamps**: Essential to track the original sequence of events. They can be extracted directly from the event payload or appended upon system entry.
- **Watermarks**: Indicators of time progression in a streamed data flow, helping to track how far the system has caught up with processing. Late-arriving events falling beyond this watermark may be processed differently or discarded based on the business rules.

#### 2. Buffering and State Management
- Implement buffers and state tracking to hold onto events until they can be processed in the correct order.
  
#### 3. Out-of-Order Handling Policies
- Define strategies for late data treatment, such as:
  - Discarding late events.
  - Logging late events for auditing.
  - Retrospective adjustments or compensating actions.

### Example Code

Here is a conceptual example in Java using the Apache Flink framework, which naturally supports out-of-order processing with watermarks.

```java
DataStream<Event> input = source.assignTimestampsAndWatermarks(
    WatermarkStrategy
        .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(10))
        .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
);

DataStream<ProcessedEvent> processed = input
    .keyBy(event -> event.getKey())
    .process(new EventReorderingProcessFunction());

```

### Related Patterns and Concepts

- **Event Sourcing**: Maintaining a sequence of events which can be reprocessed or reordered to maintain consistency.
- **CQRS (Command Query Responsibility Segregation)**: Separating the handling of command and query operations can aid in consistent event ordering.
- **Windowing Operations**: Allow computations over bounded sets of events that can account for late arrivals through flexible window assignments.

### Additional Resources

- **Apache Flink Documentation on Watermarks**: [Apache Flink's guide to handling out-of-order events](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/dev/datastream/event-time/#watermarks)
- **Google Cloud Pub/Sub**: Effective techniques for handling message delivery with Pub/Sub including ordering keys.
- **AWS Kinesis**: Strategies outlined by AWS for managing record ordering in data streams.

### Final Summary

Out-of-order processing is crucial for distributed streaming applications that require precise control over event timing and sequencing. By incorporating timestamps, watermarks, and effective handling policies, systems can maintain accurate event states despite disorderly arrivals. This pattern ensures data integrity and real-time analytics accuracy across diverse applications in cloud computing, finance, IoT, and more.

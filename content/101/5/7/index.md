---
linkTitle: "Watermark Generation"
title: "Watermark Generation: Ensuring Timely Event Processing"
category: "Event Time vs. Processing Time Patterns"
series: "Stream Processing Design Patterns"
description: "Creating watermarks to progress event time, indicating that no events with timestamps older than the watermark are expected. Watermarks help manage out-of-order events and trigger computations."
categories:
- Stream Processing
- Event Time
- Real-time Analytics
tags:
- Watermark
- Event Time Processing
- Out-of-Order Events
- Stream Computing
- Apache Flink
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/5/7"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In distributed stream processing systems, handling out-of-order events and processing based on event time is crucial for accurate and timely computations. Watermark generation is a technique used to advance event time processing, thereby triggering operations like windowing and aggregation when processing streams of timestamped events.

## What is Watermark Generation?

Watermark generation involves creating markers (watermarks) in the data stream that signal the progression of event time. A watermark indicates the notion of time until which all preceding events have been observed and processed. They guide systems in managing out-of-order arrivals and maintain temporal coherence in data streams.

### Key Features

- **Event Time Progression**: Watermarks help stream processors manage event time by indicating that all events with timestamps earlier than the watermark are expected to have arrived.
- **Out-of-Order Handling**: They provide a mechanism for dealing with events that arrive out of order by waiting for a specific amount of tolerated lateness before advancing.
- **Triggering Computation**: Watermarks serve as signals for computations like window closings or event state checkpoints in stream processing.

## Architectural Approach

Implementing watermark generation requires setting a strategy for watermark creation, which might depend on business use cases and the nature of the data stream. Here are some common approaches:

### Periodic Watermarks

- **Time-Based**: Generate watermarks periodically, using the maximum observed event time minus a tolerable delay for processing out-of-order events.
- **Count-Based**: Emit watermarks after a certain number of events or records have been processed.

### Application-Level Watermarks

- Some applications leverage domain-specific knowledge to introduce watermarks directly into the data. This approach is particularly useful when precise control over event timing is needed, possibly due to strict latency requirements.

### Timely Watermarks

- Employ a hybrid approach, combining periodic watermark generation with adaptive logic to refine the watermark generation strategy in context.

## Example in Apache Flink

Apache Flink is a popular stream processing engine that provides built-in support for watermark generation. Here's an example demonstrating periodic watermark generation in Flink:

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import java.time.Duration;

WatermarkStrategy<Event> watermarkStrategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(30))
    .withTimestampAssigner(new SerializableTimestampAssigner<Event>() {
        @Override
        public long extractTimestamp(Event element, long recordTimestamp) {
            return element.getTimestamp();
        }
    });

DataStream<Event> stream = source.assignTimestampsAndWatermarks(watermarkStrategy);
```

In this example, the watermark strategy waits for a 30-second delay to accommodate out-of-order events before progressing the watermark.

## Related Patterns

- **Tumbling and Sliding Windows**: Watermarks trigger computations for windowed aggregations based on event time.
- **Out-of-order Event Handling**: Design pattern focusing on the reordering and buffering of delayed or out-of-sequence data.

## Best Practices

- **Lateness Tolerance**: Define an acceptable lateness threshold to balance between handling out-of-order events and minimizing computation delay.
- **Monitor and Adapt**: Regularly monitor system performance and adapt the watermarking strategy as needed, to reflect real-world data behavior and processing requirements.
- **Testing and Simulation**: Employ simulations to test watermark strategies under various conditions, ensuring robustness in production.

## Additional Resources

- **Apache Flink Documentation**: Provides comprehensive guidelines on watermarking strategies and windowing concepts.
- **Real-time Stream Processing**: Industry whitepapers and case studies on effective real-time data processing architectures.
- **Event-driven Architecture Best Practices**: A deeper dive into event-driven system optimization, focusing on time-based systems.

## Summary

Watermark generation is a critical enabler for real-time stream processing systems to handle out-of-order events, ensuring computations are timely and accurate. By establishing temporal boundaries and indicating expected event time progress, watermarks are vital for managing windows and states in stream applications. Strategic watermark generation assists in minimizing latencies and enhancing the system's ability to deliver accurate and timely insights from continuous data streams.

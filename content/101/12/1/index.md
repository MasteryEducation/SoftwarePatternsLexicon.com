---
linkTitle: "Watermarks"
title: "Watermarks: Managing Event Time and Late Arrival"
category: "Late Arrival Handling"
series: "Stream Processing Design Patterns"
description: "A mechanism to track the progress of event time and manage out-of-order or late-arriving events by indicating when a system can safely assume all events up to a certain timestamp have been accounted for."
categories:
- stream-processing
- event-time
- real-time-data
tags:
- watermarks
- apache-flink
- event-streaming
- data-latency
- window-computation
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/12/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In real-time stream processing, managing event time can be crucial, especially when dealing with out-of-order or late-arriving events. The watermark design pattern is essential for providing a systematic solution to these challenges. By encoding the notion of time progression, watermarks help you maintain accuracy in time-based operations, such as aggregations over windows in streams.

## Detailed Explanation

### Concept of Watermarks

A **watermark** is a data structure or signal in a stream processing framework that denotes a specific time threshold. It serves as an indicator that all events up to that point in time have been received by the system, and any further data pertaining to that time interval can be safely processed or considered late. This mechanism is particularly useful when processing streams of events that might be out of order or delayed.

- **Event-Time vs. Processing-Time**: Event time is the time when an event was originally generated, whereas processing time is the time when an event is processed by the system. Stream processing systems need to handle discrepancies between these times effectively.
  
- **Late Arrivals**: In distributed systems, events can arrive late due to latency in network transmission or variations in data fetching and processing times. By using watermarks, systems can wait for late arrivals within defined boundaries before taking final actions on streams, such as computing windowed aggregates.

### Architectural Approach

1. **Watermark Generation**: Watermarks are generated based on the event timestamps. Various strategies for watermark generation include:
   - Periodic watermarking: Watermarks are generated at regular processing intervals.
   - Emit-on-event: Watermarks are emitted with each event based on its timestamp.

2. **Watermark Propagation**: Generated watermarks propagate down the stream towards operators like window aggregators.

3. **Window Triggers**: Watermarks trigger computations when all events within certain windows are expected to have been received.

### Example Code: Apache Flink

Apache Flink is a popular stream processing framework that provides native support for watermarks.

```java
// Example of watermark strategy in Flink
DataStream<MyEvent> eventStream = ...;

DataStream<MyEvent> withWatermarks = eventStream
  .assignTimestampsAndWatermarks(
    WatermarkStrategy
      .<MyEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
      .withTimestampAssigner((event, timestamp) -> event.getEventTime())
  );

// Using watermarked stream to define a window
KeyedStream<MyEvent, String> keyedStream = withWatermarks
  .keyBy(MyEvent::getEventId);

DataStream<WindowedResult> windowedResult = keyedStream
  .window(TumblingEventTimeWindows.of(Time.minutes(1)))
  .sum("value");
```

The above code defines a Flink data stream with a watermark strategy that assumes events can arrive up to 5 seconds late. This strategy ensures that windows will wait until the watermark surpasses their end before emitting results, thus accounting for delayed events.

## Related Patterns

- **Windowed Processing**: A natural complement to watermarks, windowed processing allows slicing the data stream into manageable time-based partitions, facilitating batch-like aggregation and analysis.

- **Time Triggered Processing**: Triggers computations based on time passage, often leveraging watermarks to know when time windows are complete.

## Best Practices

- **Configure Watermarks with Care**: Ensure your watermark generation mechanism balances between too conservative and too aggressive to prevent premature or delayed processing of events.
  
- **Monitor Late Events**: Use tools to track the occurrence of late arrivals and adjust watermark strategies according to system behavior.

- **Consider Event Order in Design**: Design your event schema and system to accommodate adjustments for potential out-of-order event arrivals.

## Additional Resources

- [Apache Flink Documentation on Watermarks](https://ci.apache.org/projects/flink/flink-docs-stable/dev/event_timestamps_watermarks.html)
- [Google Cloud Dataflow Watermark Management](https://cloud.google.com/dataflow/model/watermarks)

## Summary

Watermarks are a fundamental element in handling streams with variable latencies and event time dissonances. Using watermarks allows stream processing frameworks to effectively manage late data, support accurate window computations, and leverage real-time insights. As real-time data processing becomes a cornerstone of modern digital ecosystems, mastering watermark strategies can significantly enhance system robustness and operational intelligence.

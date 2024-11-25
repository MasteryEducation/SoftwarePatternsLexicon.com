---
linkTitle: "Aggregation with Watermarks"
title: "Aggregation with Watermarks: Handling Out-of-Order Events in Stream Processing"
category: "Data Transformation Patterns"
series: "Stream Processing Design Patterns"
description: "Using watermarks to manage and process out-of-order events within windowed aggregations, enabling effective handling of late-arriving data in streaming systems."
categories:
- stream-processing
- data-transformation
- real-time-data
tags:
- stream-processing
- watermarks
- windowed-aggregations
- real-time-analytics
- Apache-Flink
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/2/28"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

**Aggregating with Watermarks**

Stream processing is a crucial element in building real-time data processing applications. One challenge in streaming systems is dealing with out-of-order events while performing aggregations over event-time windows. Watermarks provide a solution to this problem by marking specific points in the data stream that indicate the event-time progress, allowing late-arriving events to be processed appropriately.

### Detailed Explanation

#### Watermarks in Stream Processing

In a distributed stream processing framework, such as Apache Flink or Apache Kafka Streams, data often arrives out-of-order due to network delays, varying processing speeds, etc. This presents a challenge for performant and accurate event-time aggregations. 

Watermarks are a mechanism used to manage these out-of-order events, acting as a flow controller that advances the event-time in the processing pipeline. A watermark can be thought of as a special timestamp that signals to the system that no events with timestamps older than this watermark are expected to arrive. This essentially tells the system when it can safely trigger the end of a window.

Here is a simple conceptual example in Apache Flink:

```java
DataStream<Event> events = // Original data stream

DataStream<Event> watermarkedStream = events
    .assignTimestampsAndWatermarks(
        WatermarkStrategy
            .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(15))
            .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    );

watermarkedStream
    .keyBy(event -> event.getKey())
    .window(TumblingEventTimeWindows.of(Time.seconds(30)))
    .aggregate(new MyAggregationFunction());
```

In the example above, the `WatermarkStrategy` is set to allow events arriving as late as 15 seconds past their event time. The aggregation is done using 30-second tumbling windows. The aggregation function will consider these late events within the window boundaries before emitting results.

#### Architectural Considerations

- **Window Types**: Choose appropriate window types—tumbling, sliding, session windows—based on the pattern of data arrival and application requirements.
- **Opt for Late Arrival Handling**: Decide the watermark computation strategy—bounded out-of-orderness, periodic watermarks, or idle partitions.
- **Scalability**: Distributed systems can efficiently manage large-scale data streams, aided by watermarks for maintaining computational efficiency.
- **Accuracy vs Latency Trade-off**: Determine the correct watermark delay based on the acceptable lateness of events to balance between the freshness of processed results and the completeness of data.
  
### Related Patterns

- **Session Windows**: Unlike fixed-length windows, session windows are defined by periods of activity separated by gaps of inactivity, ideal for user interaction tracking.
- **Event Time Processing**: Processing operations should use event-time semantics rather than processing-time semantics to ensure accuracy in the face of network delays.
  
### Additional Resources

- [Apache Flink: Watermarks and Event Time](https://ci.apache.org/projects/flink/flink-docs-stable/dev/event_timestamps_watermarks.html)
- [Streaming 101: The world beyond batch — Jay Kreps](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/)
- [Time and Order in Streams: Flink's Approach](https://www.ververica.com/blog/how-apache-flink-handles-late-events)

### Summary

The Aggregation with Watermarks pattern is central to handling out-of-order events efficiently in streaming applications. By choosing appropriate watermark strategies and window types, systems can ensure data completeness while managing the trade-off between latency and accuracy. This pattern is instrumental in enabling stream processing applications to function reliably in real-world scenarios where data arrival is unpredictable, such as IoT systems, financial tickers, or user activity tracking.

---
linkTitle: "Hopping Windows"
title: "Hopping Windows: Overlapping Stream Processing Windows"
category: "Windowing Patterns"
series: "Stream Processing Design Patterns"
description: "Hopping Windows are a type of windowing pattern in stream processing where the window size is greater than the slide interval, resulting in overlapping windows. This allows for the capture of events that span multiple windows, making it ideal for trend analysis and anomaly detection in real-time data streams."
categories:
- Stream Processing
- Real-time Analytics
- Windowing Patterns
tags:
- Stream Processing
- Hopping Windows
- Real-time Data
- Analytics
- Windowing
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/4/6"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Hopping Windows: Overlapping Stream Processing Windows

### Introduction

Hopping Windows are a critical pattern in stream processing where windows overlap due to a larger window size than the sliding interval. This is beneficial for capturing continuous data flows, allowing for more comprehensive analyses such as trend detection over time. Hopping windows span across multiple intervals, capturing overlapping sets of data points which are essential in calculations involving rolling averages, cumulative sums, and trend analyses.

### Detailed Explanation

Hopping Windows are used to create fixed-size time windows, separated by a fixed-time interval. This windowing pattern results in windows that overlap, meaning a single event can belong to multiple windows, thus contributing to more than one aggregation computation. 

Consider a real-world example, such as monitoring the temperature every minute using sensors distributed across different geographical locations, and we use hopping windows to average temperature readings every 10 minutes with a hop of 5 minutes:

- A window from 00:00 to 00:10
- A window from 00:05 to 00:15
- A window from 00:10 to 00:20

This would mean the temperature at minute 05:00 contributes to the calculations of both the window [00:00, 00:10] and the window [00:05, 00:15].

### Example Code

Here's a conceptual example using Apache Kafka Streams:

```java
KStream<String, Long> temperatureStream = builder.stream("temperature-readings");

KStream<Windowed<String>, Double> averagedTemperature = 
    temperatureStream.groupByKey()
                     .windowedBy(TimeWindows.of(Duration.ofMinutes(10))
                                            .advanceBy(Duration.ofMinutes(5)))
                     .aggregate(
                         () -> 0.0,
                         (key, value, aggregate) -> (aggregate + value) / 2,
                         Materialized.with(Serdes.String(), Serdes.Double())
                     )
                     .toStream();
```

In this code snippet, `temperatureStream` represents a stream of temperature sensor readings partitioned by key (e.g., sensor ID). A hopping window aggregates these readings into overlapping windows, specifically using a hop size of 5 minutes within a window of 10 minutes. The aggregation function calculates an average temperature per window.

### Diagram

Here's a diagram illustrating Hopping Windows:

```mermaid
graph TD
    A[event 00:04] --> B[Window[00:00-00:10]]
    B --> C[event 00:06]
    C --> D[event 00:09]
    A --> E[Window[00:05-00:15]]
    E --> C
    E --> D
    E --> F[event 00:12]
    F --> G[Window[00:10-00:20]]
    G --> F
```

### Related Patterns

- **Sliding Windows**: Unlike hopping windows, sliding windows typically have a smaller slide interval, often equal to one, leading to more frequent but smaller updates.
- **Tumbling Windows**: Non-overlapping windows that reset once a window's time span completes, unlike hopping windows where windows can overlap.

### Best Practices

- Choose an appropriate hop size to balance overlap and data redundancy.
- Use aggregation functions that are optimized for overlapping data, such as incremental updates.

### Additional Resources

- [Apache Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
- [Redis Streams for Time Series Processing](https://redis.io/docs/manual/streams/)
- [Azure Stream Analytics Windowing Concepts](https://docs.microsoft.com/en-us/azure/stream-analytics/stream-analytics-window-functions)

### Summary

Hopping Windows provide an effective solution for analyzing time-based data across overlapping intervals. This pattern allows real-time systems to handle complex analysis like rolling averages and trend predictions efficiently, by ensuring no data is missed between intervals. By segmenting and overlapping data, hopping windows empower real-time analytical frameworks to gain deeper insights into ongoing streaming data.

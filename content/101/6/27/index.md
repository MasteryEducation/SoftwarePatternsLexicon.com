---
linkTitle: "Hierarchical Time-Based Aggregation"
title: "Hierarchical Time-Based Aggregation: Streamlining Multi-Level Aggregation"
category: "Aggregation Patterns"
series: "Stream Processing Design Patterns"
description: "Discover how Hierarchical Time-Based Aggregation facilitates the aggregation of data over different time granularities within a unified processing pipeline."
categories:
- Stream Processing
- Aggregation
- Data Engineering
tags:
- Hierarchical Aggregation
- Stream Processing
- Time Series Data
- Data Aggregation
- Apache Kafka
date: 2024-01-01
type: docs
canonical: "https://softwarepatternslexicon.com/101/6/27"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Hierarchical Time-Based Aggregation is a powerful pattern used in stream processing that supports aggregating data across multiple time granularities, such as minutes, hours, and days, within a single processing pipeline. This pattern is particularly beneficial for applications requiring insights at various temporal scales, like web traffic analysis, sensor data analytics, financial market aggregation, etc.

## Design Pattern Explained

### Key Concepts:

1. **Granularity Levels**: Data is aggregated at multiple, progressively coarser time intervals. For example, finer granularity might be per-minute, while coarser granularity might be per-hour or per-day.

2. **Hierarchical Structure**: A hierarchical structure is followed where aggregations at one level feed into the next level. This ensures that each higher level can use pre-computed aggregates from the level below, reducing computational overhead.

3. **Reusability**: Using this hierarchical approach, intermediate results are reused, reducing redundant calculations and improving resource efficiency.

4. **Scalability**: This method supports the scalable processing of large data sets by concurrently aggregating data at different time resolutions.

### Architectural Approach

1. **Data Ingestion**: Data is captured from sources such as web logs, IoT devices, transactional systems, etc., and ingested into a stream processing framework.

2. **Aggregation Pipeline**:
    - **Minute-Level Aggregation**: Aggregate raw data into minute-level aggregates. This stage is critical for generating detailed insights.
    - **Hour-Level Aggregation**: Use minute-level aggregates to compute hourly aggregates.
    - **Day-Level Aggregation**: Calculate daily aggregates using the hourly aggregated data.

3. **Incremental Updates**: For real-time processing, updates to aggregates are handled incrementally by maintaining state within each aggregation window.

4. **Storage**: Store final aggregated results in a suitable database or data warehouse for querying and analytical purposes.

### Example Code

Below is a simple Scala example using Apache Kafka and Kafka Streams API for implementing Hierarchical Time-Based Aggregation:

```scala
import org.apache.kafka.streams.KafkaStreams
import org.apache.kafka.streams.StreamsBuilder
import org.apache.kafka.streams.kstream.{Consumed, Grouped, Materialized, TimeWindows, Windowed}

val builder = new StreamsBuilder()
val sourceStream = builder.stream[String, Long]("raw-data")(Consumed.`with`(Serdes.String, Serdes.Long))

// Minute-level aggregation
val minuteAggregates = sourceStream
  .groupByKey(Grouped.`with`(Serdes.String, Serdes.Long))
  .windowedBy(TimeWindows.of(java.time.Duration.ofMinutes(1)))
  .reduce((aggValue, newValue) => aggValue + newValue)(Materialized.as("minute-aggregates"))

// Hour-level aggregation
val hourAggregates = minuteAggregates
  .groupBy((windowedKey: Windowed[String], _) => windowedKey.key(), Grouped.`with`(Serdes.String, Serdes.Long))
  .windowedBy(TimeWindows.of(java.time.Duration.ofHours(1)))
  .reduce((aggValue, newValue) => aggValue + newValue)(Materialized.as("hour-aggregates"))

// Day-level aggregation
val dayAggregates = hourAggregates
  .groupBy((windowedKey: Windowed[String], _) => windowedKey.key(), Grouped.`with`(Serdes.String, Serdes.Long))
  .windowedBy(TimeWindows.of(java.time.Duration.ofDays(1)))
  .reduce((aggValue, newValue) => aggValue + newValue)(Materialized.as("day-aggregates"))

val streams = new KafkaStreams(builder.build(), new java.util.Properties())
streams.start()
```

## Related Patterns

- **Windowed Aggregation**: This is a related pattern where data is aggregated over fixed-time windows but not necessarily in a hierarchical manner.
- **Lambda Architecture**: A data processing architecture that supports both real-time and batch-processing paths which can complement hierarchical aggregations.
- **Kappa Architecture**: A simplified version of Lambda Architecture suitable for streaming where data is processed only in a real-time fashion but can integrate with hierarchical aggregation for different time-scale analysis.

## Additional Resources

- [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
- [Designing Data-Intensive Applications](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) by Martin Kleppmann
- [Google Cloud Bigtable: Designing Hierarchical Time-Based Aggregation Layer](https://cloud.google.com/bigtable/docs/profiles)

## Summary

Hierarchical Time-Based Aggregation is essential for efficient multi-resolution data analysis in real-time streaming environments. By leveraging incremental processing and pre-computed aggregates at finer granularities, it scales as data volumes and velocities increase, providing robust support for diverse analytical needs. Using frameworks like Kafka Streams, engineers can implement this pattern seamlessly, ensuring that each level naturally integrates within the subsequent layer, optimizing both computational efficiency and latency.

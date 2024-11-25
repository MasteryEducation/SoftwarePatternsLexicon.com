---
linkTitle: "Late Data Storage"
title: "Late Data Storage: Handling Late-Arriving Data in Stream Processing"
category: "Late Arrival Handling"
series: "Stream Processing Design Patterns"
description: "A pattern for handling late-arriving data by storing it separately for offline processing or future analysis."
categories:
- Stream Processing
- Data Management
- Big Data
tags:
- Late Data
- Data Lake
- Stream Processing
- Batch Processing
- Event-driven Architecture
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/12/29"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Late Data Storage

**Late Data Storage** is a design pattern used in stream processing to handle events that arrive late, beyond the expected processing time window. This can be critical in scenarios where data holds value even if it arrives late, as these events still contain insights that are relevant to the whole data processing objective.

### Problem

In a stream processing system, data is often expected to arrive within a certain time window to be processed reliably. However, various reasons like network latency, batching, or upstream processing delays may lead to late arrivals of some data events. Processing these late events using the same latency-focused stream processor can lead to incorrect aggregations, mismatched windows, or unexpected state mutations. 

### Solution

The **Late Data Storage** pattern suggests isolating incoming late data into separate storage, such as a Data Lake, where it can be batch processed offline. This separation allows the primary stream processing path to remain unaltered for real-time insights, while still preserving the value from late data, which can be processed using batch analytics or re-integrated with the historical data for further analysis.

### Architectural Approach

1. **Stream Processing with Watermarks**: 
   Implement watermarks in your stream processing pipeline to define the threshold for late data. Watermarks serve as markers that indicate events arriving after the checkpoint are considered late.

2. **Forking Late Data**:
   Detect late data and direct it to an alternate storage path such as a Data Lake or dedicated late data repository. Tools like Apache Kafka, Kinesis, or Google Pub/Sub can be used for buffering these late events.

3. **Batch Processing**:
   Use big data processing frameworks such as Apache Spark, AWS Glue Jobs, or Google Cloud Dataflow to process the accumulated late data offline. This allows more complex issues pertinent to late arrivals to be addressed without affecting the real-time pipeline.

### Example Code

Below is a simple illustration of how you might handle late-arriving data in Apache Flink:

```scala
val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

// Define a watermarked stream
val watermarkStrategy = WatermarkStrategy
  .forMonotonousTimestamps[Int]()

val dataStream: DataStream[Int] = env
  .fromElements(1, 2, 3, 4, 5)
  .assignTimestampsAndWatermarks(watermarkStrategy)

dataStream
  .process(new ProcessFunction[Int, Int] {
    override def processElement(value: Int, ctx: ProcessFunction[Int, Int]#Context, out: Collector[Int]): Unit = {
      if (value.isLate) {
        storeLateData(value)
      } else {
        out.collect(value)
      }
    }
  })

def storeLateData(value: Int): Unit = {
  // Logic to store late data in a data lake
}
```

### Related Patterns

- **CQRS (Command Query Responsibility Segregation)**: Separates the logic that reads data from the logic that updates data. In the context of late arrival, the command model can separately handle late events.
  
- **Lambda Architecture**: A data-processing architecture designed to handle massive quantities of data by taking advantage of both batch and stream processing methods. Late data storage is similar to the batch layer handling late events.

### Additional Resources

- [Streaming 101](https://www.oreilly.com/library/view/streaming-101/9781491983874/) - An introduction to stream processing.
- [Designing Data-Intensive Applications](https://dataintensive.net/) by Martin Kleppmann - Includes discussions on handling late data as part of data processing patterns.

### Summary

The **Late Data Storage** design pattern offers an effective strategy for managing late-arriving events that could disrupt the main data stream processing by transferring them to specialized storage for later analysis. It ensures that real-time insights remain timely and accurate while still allowing late data to add value through offline processing. This approach leverages both real-time and batch processing paradigms, offering a comprehensive data processing solution.

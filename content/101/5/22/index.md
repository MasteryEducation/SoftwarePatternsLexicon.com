---
linkTitle: "Processing Time as a Fallback"
title: "Processing Time as a Fallback: Stream Processing Fallbacks"
category: "Event Time vs. Processing Time Patterns"
series: "Stream Processing Design Patterns"
description: "Using processing time as an alternative when event time is either absent or inaccurate, providing a reliable method to continue data processing."
categories:
- event-time
- processing-time
- stream-processing
tags:
- data-streams
- timestamps
- fallback-strategy
- real-time-processing
- time-handling
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/5/22"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview

In stream processing, events often carry timestamps intended to reflect the time of their occurrence, known as event time. However, in environments where events lack such valid timestamps due to errors, latency, or data integrity issues, relying solely on event time can lead to inaccuracies. The "Processing Time as a Fallback" pattern offers a solution by using the actual processing time of data as an alternative timestamp. This ensures data streams are handled timely and correctly, even when event time is unreliable.

## Detailed Explanation

### Architectural Approach

This pattern switches from event time to processing time when:

- Event data lacks a timestamp.
- Event timestamps are unreliable or delayed.
- System operations cannot afford to wait for correct event times due to processing deadlines.

**Event Time vs. Processing Time:**
- **Event Time**: Ideal, as it represents when the event actually occurred.
- **Processing Time**: Practical, reflecting when the system processes the event.

By adopting processing time selectively, systems can maintain a streamlined flow without significant disruptions, even if they temporarily compromise on the precision of event sequencing.

### Implementation

1. **Timestamp Assignment**: As events are ingested, check their timestamps.
   - If the timestamp is missing or invalid, assign the current processing time.
   - Flag such events for special processing or downstream handling.

2. **Data Handling**: Incorporate logic to treat events flagged with processing time differently, if needed. This can affect subsequent analytics or event-driven triggers.

3. **Monitoring and Alerting**: Implement monitoring to flag when high volumes of events require processing time fallback, indicating potential upstream issues.

### Example Code

Here's a simplified example in Scala, illustrating a stream processing pipeline utilizing the Apache Flink framework:

```scala
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.time.Time

// Define an event with an optional timestamp
case class Event(id: String, data: String, eventTime: Option[Long])

val env = StreamExecutionEnvironment.getExecutionEnvironment

val eventStream = env.fromElements(
  Event("1", "data1", Some(1633036800000L)),
  Event("2", "data2", None),  // Missing timestamp
  Event("3", "data3", Some(1633036800500L))
)

val processedStream = eventStream
  .map(event =>
    event.eventTime match {
      case Some(time) => (event.id, time)
      case None => (event.id, System.currentTimeMillis) // Fallback to processing time
    }
  )

processedStream
  .keyBy(_._1)
  .timeWindow(Time.minutes(1))
  .sum(1)
  .print()

env.execute("Processing Time as Fallback Example")
```

### Use Cases

- Ad hoc processing systems where events are ingested from diverse sources, some of which may not stamp events correctly.
- Systems processing user interactions where network delays lead to incomplete data arrival.
- IoT scenarios where devices may lose temporal precision due to connectivity disruptions.

## Related Patterns

- **Late Data Handling**: Strategies for dealing with data that arrives after the assigned window has closed.
- **Event Time Out-of-Order Handling**: Techniques to reorder events based on timestamps even when they arrive out of sequence.

## Additional Resources

- [Apache Flink Documentation](https://flink.apache.org/doc)
- Book: *Stream Processing with Apache Flink* by Fabian Hueske and Vasiliki Kalavri
- [Event Time vs Processing Time Blog Post](https://example.com/event-vs-processing-time)

## Summary

The "Processing Time as a Fallback" pattern is crucial in ensuring robust stream processing when timestamps cannot be relied upon. By employing processing time as a flexible substitute for event time, developers can ensure that their systems continue to function effectively, mitigating delays and inaccuracies. This adaptability is essential for real-time applications that must operate over noisy or incomplete input data.

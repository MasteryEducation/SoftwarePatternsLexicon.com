---
linkTitle: "Distinct Count Aggregation"
title: "Distinct Count Aggregation: Counting Unique Elements"
category: "Aggregation Patterns"
series: "Stream Processing Design Patterns"
description: "This pattern is about counting the number of unique elements or values in a data stream, with applications such as counting the number of unique users who logged in during a specific time period."
categories:
- Stream Processing
- Real-time Analytics
- Big Data Processing
tags:
- Stream Processing
- Distinct Count
- Data Aggregation
- Real-time Analytics
- Flink
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/6/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Distinct Count Aggregation

### Description

In modern data processing, especially within real-time analytics, it is often necessary to compute the number of unique elements in a dataset (or stream). The **Distinct Count Aggregation** pattern helps in identifying unique entries, which is a common requirement in applications like user activity tracking, concentration of resources, and fraud detection. This pattern is especially useful for counting interactions such as the number of unique users logging into a service over a day, determining distinct product views, or unique page hits in an application.

### Use Case Scenario

**Example**: A web application may need to track how many unique users have logged in over the past week. Utilizing the Distinct Count Aggregation pattern, it is possible to process each login event in real-time and maintain a count of distinct user IDs.

### Architectural Approach

The implementation of a Distinct Count Aggregation often relies on mechanisms that efficiently handle potentially large volumes of data where the entire set cannot fit into memory. Two prominent methods are:

1. **Hash Sets**: Used for exact counting, maintaining a set of elements encountered in the stream. This approach works well for datasets that can comfortably fit in memory.
2. **HyperLogLog**: An algorithm that provides an approximate count of distinct elements, significantly reducing memory footprint with minimal accuracy trade-offs.

#### Example Code: Using HyperLogLog in Java

```java
import com.clearspring.analytics.stream.cardinality.HyperLogLog;

public class DistinctUserCount {

    private HyperLogLog hll;

    public DistinctUserCount(double rsd) {
        // rsd: relative standard deviation
        this.hll = new HyperLogLog(rsd);
    }

    public void addUser(String userId) {
        hll.offer(userId);
    }

    public long getDistinctUserCount() {
        return hll.cardinality();
    }
    
    public static void main(String[] args) {
        DistinctUserCount distinctUserCount = new DistinctUserCount(0.05);
        distinctUserCount.addUser("User1");
        distinctUserCount.addUser("User2");
        distinctUserCount.addUser("User1");
        
        System.out.println("Estimated number of distinct users: " + distinctUserCount.getDistinctUserCount());
    }
}
```

### Best Practices

- **Choose the Right Tool**: For exact counts in smaller streams, simple in-memory data structures may be sufficient. For large datasets or in environments with memory constraints, HyperLogLog or similar probabilistic data structures are recommended.
- **Accuracy vs. Memory Trade-off**: When using approximate algorithms like HyperLogLog, consider the trade-off between memory usage and accuracy of the results.
- **Data Partitioning**: In distributed systems, partitioning data can improve computation performance by distributing load, especially pertinent when exact counting is necessary.

### Related Patterns

- **Windowed Aggregation**: This aggregates data based on a time window, commonly used with distinct counting to aggregate counts over specific periods.
- **Count-min Sketch**: An alternative probabilistic data structure used for approximating the frequencies of elements in a stream, particularly useful when tracking large quantities of distinct counts alongside their frequencies.

### Additional Resources

- [Apache Flink Documentation on Unique Count](https://nightlies.apache.org/flink/flink-docs-release-1.14/docs/dev/libs/streaming_analytics/#distinct-count)
- [Understanding HyperLogLog](https://research.google.com/pubs/archive/40671.pdf)

### Summary

The Distinct Count Aggregation pattern is a crucial concept in streaming data processing that helps in determining unique elements efficiently. Whether employing exact methods or approximate algorithms such as HyperLogLog, this pattern simplifies solving challenges in real-time analytics, reducing the complexity of counting unique entries in vast data streams. By understanding and implementing this pattern correctly, businesses can gain deep insights while optimizing their resource utilization.

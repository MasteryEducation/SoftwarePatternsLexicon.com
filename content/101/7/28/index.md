---
linkTitle: "Hash Join"
title: "Hash Join: Efficient Stream Joining"
category: "Join Patterns"
series: "Stream Processing Design Patterns"
description: "Building a hash table on join keys from one stream and probing it with events from the other stream."
categories:
- stream-processing
- data-integration
- join-patterns
tags:
- hash-join
- stream
- data-processing
- stream-join
- data-streaming
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/7/28"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

### Hash Join: Efficient Stream Joining

In data stream processing, combining two data streams efficiently where each data event is correlated by specific attributes (keys) is a common requirement. The Hash Join pattern is a highly effective design pattern for such operations. It constructs a hash table on the join key from events of one stream (the build side) and probes this hash table with events from the other stream (the probe side).

#### Architectural Overview

The Hash Join approach is an efficient way to perform equi-joins and is particularly suited for scenarios where one of the streams can be held in memory as a hash table. This pattern minimizes the computational complexity typically involved in stream joining operations.

##### Components
- **Stream A**: The build stream where events are stored in a hash table as they arrive.
- **Stream B**: The probe stream is evaluated against the hash table to find matches.
- **Hash Table**: A structure holding keys from Stream A with corresponding values.
- **Join Processor**: The logic that performs the hash table construction and lookup for each incoming event from Stream B.

##### How It Works

1. **Build Phase**: As events appear in Stream A, they are used to populate the hash table. Each entry in the hash table represents a key-value pair where the key is derived from the event.
2. **Probe Phase**: For each event in Stream B, the join logic checks if this event's corresponding key exists in the hash table, and if so, combines them.

```java
// Java example demonstrating a simple in-memory hash join
Map<String, Product> productHashTable = new HashMap<>();

// Populating the hash table with Stream A (Products)
streamA.forEach(event -> {
    Product product = event.getProduct();
    productHashTable.put(product.getId(), product);
});

// Processing Stream B (Categories) and performing the join
streamB.forEach(event -> {
    Category category = event.getCategory();
    Product product = productHashTable.get(category.getProductId());
    if (product != null) {
        // Combine product and category information
        processJoinedData(product, category);
    }
});

public void processJoinedData(Product product, Category category) {
    // Implement data-combination logic here
}
```

#### Best Practices

1. **Memory Management**: Ensure that the hash table's memory footprint is manageable, possibly using eviction policies for high-cardinality datasets.
2. **Key Design**: Optimize the key structure to reduce hash collisions and improve lookup efficiency.
3. **Skew Handling**: Prepare strategies for dealing with uneven distribution of data (key skew) that may affect one stream more than the other.

#### Related Patterns

- **Nested Loops Join**: A different join pattern suited for small datasets or scenarios where simplicity is preferred over performance.
- **Sort-Merge Join**: Prefer when streams can be pre-sorted or inherently ordered, providing more performant operations under these conditions.

#### Additional Resources

- [Stream Processing with Apache Flink: Performant Join Operations](https://nightlies.apache.org/flink/)
- [Design Patterns for Streaming: A Guide to Real-Time Processing](https://learn.microsoft.com/en-us/azure/architecture/patterns/)

### Summary

The Hash Join pattern enables efficient equi-join operations on data streams by utilizing in-memory hash tables built and probed dynamically as streams flow through the system. By optimizing key management, memory usage, and data distribution, it delivers streamlined join operations suitable for high-throughput environments. This digest provides the foundational aspects and best application scenarios for deploying a successful Hash Join approach in stream processing systems.

---
linkTitle: "Inner Join"
title: "Inner Join: Stream-to-Stream Data Combination"
category: "Join Patterns"
series: "Stream Processing Design Patterns"
description: "The Inner Join design pattern combines records from two streams based on matching keys. It includes only those records where the join condition is satisfied."
categories:
- Join Patterns
- Stream Processing
- Data Integration
tags:
- Inner Join
- Stream Processing
- Data Integration
- Apache Kafka
- Apache Flink
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/101/7/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Inner Join

The Inner Join pattern in stream processing is fundamental for combining data from two different streams into a new stream of combined and meaningful information. It only retains records where there is a match in specified keys across both streams. This pattern is widely used in data processing tasks that involve merging datasets to find intersecting data points.

### Description

In the context of real-time data processing, the Inner Join design pattern is used to merge records coming from two separate streaming datasets. The join operation is based on a specified condition, typically the equality of keys from both streams. Only those records that have matching keys in both streams will be emitted in the output stream. 

**Use Cases:**
- **Order and Shipment Tracking:** Match records from an order stream with records from a shipment stream to determine which orders have been shipped.
- **Sensor Data Correlation:** Combining time-series data from different sensors to evaluate the correlation between environmental factors.

### Architectural Approaches

- **Stateful Processing:** Inner Joins in stream processing require maintaining state for unmatched records that arrive first, held until their matching counterparts arrive from the other stream.
- **Windowed Joins:** Window functions are often used to limit the time during which records can be joined, controlling memory consumption and allowing for timely record matching.

### Best Practices

- **Time Window Management:** Use appropriate time windows to ensure that the data being processed is relevant and to prevent memory overflow.
- **Key Selection:** Carefully choose the join key based on business logic to ensure correct record combinations and to minimize unnecessary data retention in state storage.
- **State Management:** Optimize the state store size by configuring appropriate data retention policies.

### Example Code

Here's an implementation example using Apache Kafka Streams in Java, where orders are joined with shipments based on their order ID:

```java
StreamsBuilder builder = new StreamsBuilder();

KStream<String, Order> ordersStream = builder.stream("orders");
KStream<String, Shipment> shipmentsStream = builder.stream("shipments");

KStream<String, OrderShipment> orderShipments = ordersStream.join(
    shipmentsStream,
    (order, shipment) -> new OrderShipment(order.getOrderId(), order, shipment),
    JoinWindows.ofTimeDifferenceWithNoGrace(Duration.ofMinutes(5)),
    StreamJoined.with(Serdes.String(), orderSerde, shipmentSerde)
);

orderShipments.to("order-shipments");
```

### Diagrams

#### Inner Join Data Flow Diagram

```mermaid
sequenceDiagram
    participant OrdersStream
    participant ShipmentsStream
    participant JoinedStream

    OrdersStream->>+JoinedStream: Order {ID: 1}
    ShipmentsStream->>+JoinedStream: Shipment {Order ID: 1}
    Note right of JoinedStream: Emit combined record
    OrdersStream-->>-JoinedStream: Order {ID: 2}
    ShipmentsStream-->>-JoinedStream: Shipment {Order ID: 4}
    Note over JoinedStream: No emission, unmatched records
```

### Related Patterns

- **Outer Join**: Returns all records when there is a match. It includes records even if there is no matching record in the second stream.
- **Left Join**: Similar to Inner Join but ensures all records from the left stream are included in the output.

### Additional Resources

- [Apache Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
- [Stream Processing with Apache Flink: Join Operations](https://nightlies.apache.org/flink/)
- [Confluent Blog: Handling Stream-to-Stream Joins](https://www.confluent.io/blog/)

### Summary

The Inner Join Design Pattern is essential for stream processing tasks where intersecting data from two datasets is necessary to create value. Mastering this pattern enables developers to efficiently merge real-time data streams, opening possibilities for sophisticated, real-time data applications. Proper implementation involves understanding stateful processing, window management, and key selection, all of which play pivotal roles in achieving accurate and scalable data integration.

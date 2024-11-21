---

linkTitle: "Bi-Temporal Data Merging"
title: "Bi-Temporal Data Merging"
category: "Bitemporal Tables"
series: "Data Modeling Design Patterns"
description: "Combining temporal data from multiple sources, crucial for aligning historical records, especially in use cases like merging customer records from acquisitions."
categories:
- Data Modeling
- Temporal Data
- Integration
tags:
- Bitemporal
- Data Merging
- Temporal Databases
- Data Integration
- History Management
date: 2024-07-07
type: docs

canonical: "https://softwarepatternslexicon.com/103/2/26"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

Bi-temporal data refers to any dataset in which each record has two dimensions of time. Typically, these include the valid time (when a fact is true in reality) and the transaction time (when a fact was stored in the database). Bi-temporal data merging is essential for organizations dealing with data from multiple sources that need historical alignment.

Merging customer records, especially in scenarios like acquisitions, involves aligning distinct historical timelines which involve both the valid and transaction times. This design pattern enables accurate historical querying and reporting.

## Key Concepts

### Valid Time and Transaction Time

- **Valid Time**: When the data is true in the real world.
- **Transaction Time**: When the data is stored in the database.

Bi-temporal tables store both of these times, allowing for sophisticated temporal queries.

### Data Sources and Merging

With multiple data sources, discrepancies often arise in the interpretation of temporal data. The merging process involves reconciliating these differences, maintaining integrity across both valid and transaction dimensions.

## Architectural Approach

### Data Normalization

Utilize data normalization techniques to ensure consistent data formats and schema. This ensures that all data follows a uniform temporal schema for effective querying and merging.

### Concurrency and Consistency

Align the concurrent data updates from different sources by maintaining a queue or using a locking mechanism. This prevents conflicts and ensures consistency in transaction timelines.

### Change Data Capture (CDC)

Implement CDC to track changes in data across systems. Tools such as Debezium or Apache Kafka's specialized connectors can be useful for real-time data synchronization across systems.

## Example Code

Illustrating a simplified SQL schema for a bi-temporal customer table:

```sql
CREATE TABLE customer_bitemporal (
  customer_id INT,
  name VARCHAR(255),
  valid_start TIMESTAMP,
  valid_end TIMESTAMP,
  transaction_start TIMESTAMP DEFAULT current_timestamp,
  transaction_end TIMESTAMP DEFAULT '9999-12-31 23:59:59',
  PRIMARY KEY (customer_id, transaction_start)
);
```

Here, `valid_start` and `valid_end` capture the period for which the data is valid in the real world, whereas `transaction_start` and `transaction_end` record when the data was stored in the database system.

## Diagrams

### Data Flow Diagram

```
graph LR
  A[Source A] -->|Data Extraction| B[Temp Storage]
  B --> |Transformation & Alignment| C[Bi-Temporal Table]
  D[Source B] -->|Data Extraction| B
```

## Best Practices

- **Audit Trail**: Maintain an audit trail to handle the historical aspect of data changes.
- **Data Quality**: Regularly validate data quality from different sources to ensure accurate merges.
- **Time-zone Consideration**: Handle time zone differences with care, ensuring consistent valid time comparisons.

## Related Patterns

- **Event Sourcing**: Captures state changes as a sequence of events and allows reconstruction of past states.
- **Snapshot Isolation**: Useful in conjunction with bi-temporal systems for maintaining transaction consistency.

## Additional Resources

- [Temporal Tables in Systems](https://temporal.io/)
- [Temporal Design Patterns](https://www.thoughtworks.com/insights/blog/temporal-database-patterns)

## Summary

Bi-temporal data merging is crucial for managing and aligning datasets across different timelines and sources. By understanding and implementing this pattern, organizations can better manage customer data, improve decision-making, and ensure historical data integrity, especially during mergers and acquisitions.

---

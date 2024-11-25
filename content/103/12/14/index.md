---
linkTitle: "Bi-Temporal Change Data Capture"
title: "Bi-Temporal Change Data Capture"
category: "Bi-Temporal Data Warehouses"
series: "Data Modeling Design Patterns"
description: "Capturing changes in source systems along with their valid times to update the data warehouse."
categories:
- data-modeling
- change-data-capture
- temporal-data
tags:
- bi-temporal
- data-warehousing
- cdc
- data-model
- time-series
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/12/14"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

**Bi-Temporal Change Data Capture**

## Overview

Bi-temporal Change Data Capture (CDC) is a data warehousing design pattern specifically used to manage and capture changes in data from various source systems where both transaction time and valid time are relevant. This pattern is critical for scenarios that require auditing, versioning, and compliance, as it captures when changes actually occurred in the source system and when they became effective in the real world.

## Architectural Approach

Bi-temporal storage involves two dimensions of time:

- **Transaction Time**: The time when the data was stored in the database. It's the system-time, usually set by the system itself.
- **Valid Time**: The time period during which a piece of data is effective in the real world. This is usually defined by business processes.

The bi-temporal CDC pattern helps in tracking not only changes in the data but also when these changes were made effective.

### Components

1. **CDC Tool**: A tool or process able to detect changes in the source systems. This can be log-based (e.g., Debezium for Apache Kafka) or trigger-based.
2. **Temp Tables and Schema**: Temporary storage areas to buffer the incoming change events before they are processed.
3. **Data Warehouse**: The place where the bi-temporal data is finally stored, e.g., a bi-temporal table in a data warehouse like Snowflake or Redshift.

## Implementation Process

1. **Capture Phase**: Use the CDC tool to capture change events from the source system, including both transaction time and valid time metadata.
2. **Transformation Phase**: Ensure that changes respect the bi-temporal nature. Map these changes into the data warehouse schema.
3. **Loading Phase**: Store the transformed data in the bi-temporal tables in the data warehouse.

### Example Code

Here's how you can define a bi-temporal table in an SQL-based data warehouse:

```sql
CREATE TABLE CustomerHistory (
    CustomerID INT,
    CustomerName VARCHAR(100),
    TransactionTimeStart TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    TransactionTimeEnd TIMESTAMP NOT NULL DEFAULT '9999-12-31 23:59:59',
    ValidTimeStart TIMESTAMP,
    ValidTimeEnd TIMESTAMP NOT NULL DEFAULT '9999-12-31 23:59:59'
);
```

This SQL table schema captures transactional validity as well as the real-world valid time for the customer data.

## Related Patterns

- **Slowly Changing Dimension (SCD)**: This pattern deals with how to change and manage dimension data, particularly in the realm of data warehousing.
- **Audit Logging**: Captures all changes along with metadata about those changes, usually for regulatory compliance.
- **Data Versioning**: Management of different versions of the same data entity.

## Best Practices

- **Consistent Time Zone**: Ensure time consistency across systems by setting a universal time zone (UTC) for capturing time-based data.
- **Efficient Indexing**: Use indexes on the time fields to optimize query performance.
- **Granularity**: Decide the appropriate level of change granularity that needs to be captured to balance between performance and storage cost.
  
## Additional Resources

- [Apache Kafka and Debezium for CDC](https://debezium.io/)
- [Snowflake Time Travel](https://docs.snowflake.com/en/user-guide/time-travel.html)
- [Temporal Database Management](https://link.springer.com/book/10.1007/978-0-387-34866-1)

## Summary

Bi-Temporal Change Data Capture enables maintaining a comprehensive record for both the reality of when data changes occurred and when they were valid in the realm of business operations. It is especially vital in industries where audit trails, compliance, and accurate history tracking are critical. Through the efficient capture, storage, and querying of bi-temporal data, enterprises can maintain accurate historical insights, uphold transparency, and support complex analytical queries.

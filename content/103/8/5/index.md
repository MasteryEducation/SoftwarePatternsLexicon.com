---
linkTitle: "Temporal Referential Integrity"
title: "Temporal Referential Integrity"
category: "Bi-Temporal Consistency Patterns"
series: "Data Modeling Design Patterns"
description: "Enforcing referential integrity in temporal data to ensure that foreign keys are valid within the same time periods."
categories:
- Data Modeling
- Consistency
- Temporal Patterns
tags:
- Referential Integrity
- Temporal Data
- Data Consistency
- Bi-Temporal
- Data Modeling
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/8/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Temporal Referential Integrity

In traditional database systems, referential integrity ensures that foreign key relationships between tables maintain consistent data. However, when working with temporal data, ensuring referential integrity extends beyond simple existence checks; it involves maintaining validity within the same time periods. Temporal Referential Integrity (TRI) is a pattern where relationships between entities are not only conceptually linked but are also temporally aligned. 

## Importance of Temporal Referential Integrity

Temporal Referential Integrity becomes crucial in systems where historical data tracking and querying of data states at specific points in time are necessary. For example, in a financial system, a transaction must reference the account status and configurations that were valid when the transaction occurred, not necessarily the current state.

## Architectural Approach

1. **Bi-Temporal Data Model**: Adopt a bi-temporal model where each record has valid-time and transaction-time attributes. This ensures that you can track the validity of a record as well as the period it's committed to the database.

2. **Foreign Key with Temporal Validation**: Implement foreign key constraints that validate references based on overlapping validity periods. This ensures that all dependent records point to a valid parent record within the same timeframe.

3. **Pattern of Queries**:
    - Temporal JOINs: Extend SQL capabilities with temporal JOIN predicates to ensure that referred entities are within the correct time period.
    - Time-Effective DML Operations: Any modifications need to respect temporal constraints, using effective start and end times for updates and deletions.

## Example in SQL

Consider two tables, `Customer` and `Order`, where each entry includes validity periods:

```sql
CREATE TABLE Customer (
    customer_id INT PRIMARY KEY,
    name VARCHAR(255),
    valid_from DATE,
    valid_to DATE
);

CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    FOREIGN KEY (customer_id, order_date) REFERENCES Customer(customer_id, valid_from, valid_to)
    WHERE order_date >= valid_from AND order_date <= valid_to
);

-- Example query
SELECT o.*
FROM Orders o
JOIN Customer c ON o.customer_id = c.customer_id
WHERE o.order_date BETWEEN c.valid_from AND c.valid_to;
```

## Related Patterns

- **Bi-Temporal Pattern**: Maintains records with both valid and transaction times, allowing queries on different temporal dimensions.
- **Slowly Changing Dimensions**: Used in data warehousing, dealing with changes over time while preserving historical data.

## Best Practices

- Ensure all updates and deletes adhere to temporal constraints.
- Use automated scripts or database triggers to maintain TRI consistently across large datasets.
- Leverage ORMs or graph databases with built-in temporal support for more abstract TRI modeling.

## Additional Resources

- **Book**: "Temporal Data & the Relational Model" by C.J. Date, Hugh Darwen, and Nikos A. Lorentzos.
- **Articles**: Explore SQL extension proposals such as TSQL2 for handling temporal data natively.

## Summary

Temporal Referential Integrity ensures that temporal data relationships are maintained over time, guaranteeing that foreign keys are valid within prescribed time periods. By adopting suitable data modeling techniques and database constraints, we can maintain robust data integrity in systems that require historical accuracy and temporal query capabilities. Understanding and applying TRI effectively enables better designs of temporal databases, facilitating accurate historical reporting and auditing.

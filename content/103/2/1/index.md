---
linkTitle: "Schema Design with Temporal Columns"
title: "Schema Design with Temporal Columns"
category: "Bitemporal Tables"
series: "Data Modeling Design Patterns"
description: "Designing table schemas to include both valid time and transaction time attributes."
categories:
- Data Modeling
- Database Design
- Temporal Data
tags:
- SQL
- Bitemporality
- Data Management
- Time Travel
- Schema Design
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/2/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Schema Design with Temporal Columns

In modern data architecture, it's often critical to track not only what data is current now but also how it has changed over time. The schema design with temporal columns provides a robust approach by incorporating both **valid time** and **transaction time** attributes. This pattern is an essential consideration for systems where historical data traceability and auditability are fundamental requirements.

### Design Pattern Explanation

#### Key Concepts

1. **Valid Time:** Represents the period during which a fact is true in the real world. Key columns associated with valid time are:
    - `ValidFrom`: Marks when the data starts being applicable.
    - `ValidTo`: Marks when the data ceases to be applicable.
   
2. **Transaction Time:** Reflects the period during which a fact is stored in the database, enabling version control and historical tracking. Key columns include:
   - `TransactionStart`: Indicates when the transaction was executed.
   - `TransactionEnd`: Marks when the transaction was closed or superseded.

### Key Architectural Approaches

In a typical implementation for schema design with temporal columns, we adopt bi-temporal tables which manage historical and contemporary records seamlessly. Implementing this involves creating a well-structured database schema with integrated time-tracking functionalities.

#### Schema Example

```sql
CREATE TABLE CustomerHistory (
    CustomerID INT,
    Name VARCHAR(100),
    Address VARCHAR(255),
    ValidFrom DATE,
    ValidTo DATE,
    TransactionStart TIMESTAMP,
    TransactionEnd TIMESTAMP,
    PRIMARY KEY (CustomerID, ValidFrom, TransactionStart)
);
```

### Best Practices

- **Data Integrity**: Ensure overlapping of valid times does not occur for the same primary key to maintain data integrity.
- **Indexing**: Create composite indexes on temporal columns to improve query performance, especially during time-based searches.
- **Changes History**: When updating data, instead of overwriting existing records, update the transaction end and insert a new entry representing the new state with the current transaction start.

### Related Patterns

- **Audit Log**: Captures fine-grained operations on data over time for compliance tracking.
- **Time Travel Query**: Allows retroactive queries to reconstruct past database states.

### Additional Resources

- [Temporal Tables in SQL Server](https://docs.microsoft.com/en-us/sql/relational-databases/tables/temporal-tables)
- [PostgreSQL Temporal Support](https://www.postgresql.org/docs/current/functions-datetime.html)
- [Bitemporal Data in Oracle](https://docs.oracle.com/en/database/oracle/oracle-database/19/adjsn/bitemporal_tables.html)

### Summary

Incorporating temporal columns within database schema design is paramount for applications that require comprehensive historical data analysis and traceability. By storing valid and transaction time attributes, organizations can manage bi-temporal data effectively, addressing both reality-based time and system-based changes. This meticulous approach to data management enhances insight capabilities and compliance adherence across various business functions.

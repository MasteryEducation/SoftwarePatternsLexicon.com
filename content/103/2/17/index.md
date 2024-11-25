---
linkTitle: "Bi-Temporal Constraints"
title: "Bi-Temporal Constraints"
category: "Bitemporal Tables"
series: "Data Modeling Design Patterns"
description: "Bi-temporal constraints ensure data integrity by managing both valid and transaction times, essential for maintaining accurate time-sensitive records."
categories:
- Data Management
- Temporal Design
- Data Integrity
tags:
- Bi-Temporal
- Constraints
- Data Modeling
- Temporal Tables
- Database Design
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/2/17"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview

In many data systems, accurately reflecting real-world temporal events involves managing multiple timelines. Bi-temporal constraints play a critical role in ensuring temporal data is stored and maintained accurately. These constraints focus on both valid times and transaction times, ensuring the integrity of time-dependent data within databases that utilize bitemporal tables.

## Understanding Bi-Temporal Tables

Bi-temporal tables incorporate two separate time intervals:

1. **Valid Time**: Represents the time period when a fact is true in the real world.
2. **Transaction Time**: Represents the time period during which a fact is stored in the database, reflecting entries, updates, or deletions.

Bi-temporal constraints ensure that the valid time and transaction time work together harmoniously without resulting in any logical contradictions or data integrity issues.

## Architectural Approaches

### 1. Temporal Consistency

Ensuring consistency means no overlapping intervals exist unless intended, and no records have gaps or overlaps between their validity or transaction times when not designed otherwise.

### 2. Version Management

Implementing bi-temporal constraints aids in managing versions of records in the database. Each change in data could result in a new version, marked by unique valid and transaction times.

### 3. Concurrency Control

Concurrency controls are applied to prevent conflicting operations based upon transaction times, ensuring multiple simultaneous transactions do not corrupt bitemporal data integrity.

## Example Code

Here's an example of a SQL table design for a bi-temporal table with constraints ensuring no gaps between the valid periods of successive records:

```sql
CREATE TABLE BiTemporalRecords (
    RecordID INT PRIMARY KEY,
    Data VARCHAR(255),
    ValidFrom DATE NOT NULL,
    ValidTo DATE NOT NULL,
    TransactionFrom TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    TransactionTo TIMESTAMP,
    CONSTRAINT CHK_NoOverlap CHECK (ValidFrom < ValidTo),
    UNIQUE(RecordID, ValidFrom, ValidTo, TransactionFrom, TransactionTo)
);

-- SQL to ensure no gaps exist
ALTER TABLE BiTemporalRecords ADD CONSTRAINT CK_ValidGap 
CHECK (
    NOT EXISTS (
        SELECT *
        FROM BiTemporalRecords AS t1
        JOIN BiTemporalRecords AS t2
            ON t1.RecordID = t2.RecordID
            AND t1.ValidTo = t2.ValidFrom - INTERVAL '1 day'
        WHERE t1.TransactionTo IS NULL
            AND t2.TransactionTo IS NULL
    )
);
```

## Design Diagrams

Here is a UML class diagram illustrating the structure and connections within a bi-temporal system:

```mermaid
classDiagram
    class BiTemporalRecord {
        +int RecordID
        +String Data
        +Date ValidFrom
        +Date ValidTo
        +Timestamp TransactionFrom
        +Timestamp TransactionTo
    }

    BitemporalRecord --> {"1"} RecordValidator
    BitemporalRecord --> {"1"} TransactionManager

    class RecordValidator {
        +validateNoOverlap(): void
        +validateGaps(): void
    }

    class TransactionManager {
        +initiateTransaction(): void
        +recordTransaction(): void
    }
```

## Related Patterns and Descriptions

- **Temporal Tables**: An approach that deals with a single timeline, either valid or transaction time.
- **Slowly Changing Dimensions (SCD)**: Often used in data warehousing to track changes over time, similar to valid time in bi-temporal constraints.
- **Audit Logging**: A pattern focusing on capturing changes over transactional time explicitly.

## Additional Resources

- Janez e-learning's video on [Temporal Design Patterns](https://example.com/temporal-design) covers the theory behind bitemporal design.
- "Temporal Data & the Relational Model" (Book) by C. J. Date and Hugh Darwen for a deep dive into temporal databases.

## Summary

Bi-temporal constraints are crucial in a world where accurate historical data and evolutionary data models need to coexist. By employing them alongside regular database constraints, you ensure high integrity, provide better historical audit trails, and support more complex analytical use cases. This pattern is essential for industries that require accurate time spans, like finance, healthcare, and compliance-focused domains.

---
linkTitle: "Bi-Temporal SCD"
title: "Bi-Temporal SCD"
category: "Slowly Changing Dimensions (SCD)"
series: "Data Modeling Design Patterns"
description: "A design pattern for tracking both valid time and transaction time in the context of Slowly Changing Dimensions."
categories:
- Data Modeling
- Data Warehousing
- SCD
tags:
- bi-temporal
- slowly-changing-dimension
- data-modeling
- data-warehousing
- time-tracking
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/3/23"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
Bi-Temporal Slowly Changing Dimension (SCD) is a design pattern used in data warehousing to manage dimensions with two independent time dimensions: valid time and transaction time. Valid time refers to the actual time period an entry is considered valid or effective in the real world, while transaction time is the time period during which the information about the entry is stored in the database.

## Key Concepts
- **Valid Time**: This is the period during which a fact is true concerning the lifeline of a concept or event. It ranges from a start valid time to an end valid time.
- **Transaction Time**: This is the period during which a database entity knows a fact. It ranges from when a fact is added to when it is logically deleted or superseded.

## Design Pattern Explanation
Bi-Temporal SCD aims to provide a historical view of dimension changes, ensuring that both user-perceived changes and database-recorded changes are trackable:

- **Temporal Consistency**: The pattern allows reconstructing historical states of both reality (valid times) and system knowledge (transaction times).
- **Comprehensive Tracking**: Ideal for systems that require historical accuracy due to auditing or compliance needs.

## Example Scenario

Imagine a `Promotion` dimension table in a retail data warehouse:

| PromotionID | Description   | ValidStartDate | ValidEndDate | TransactionStartDate | TransactionEndDate |
|-------------|---------------|----------------|--------------|----------------------|--------------------|
| 1           | Summer Sale   | 2024-06-01     | 2024-08-31   | 2024-06-01           | 9999-12-31         |
| 2           | Winter Sale   | 2024-12-01     | 2025-02-28   | 2024-12-01           | 9999-12-31         |

In this example:
- The `Summer Sale` promotion is valid from June 1, 2024, to August 31, 2024. It was recorded in the system on June 1, 2024, and is assumed to be current (transaction end date as `9999-12-31`).
- Any updates, like changing the `ValidEndDate`, can be tracked separately with additional rows indicating the transition of belief in the system.

## Design Techniques

### Tables and Schemas
Utilize separate columns for valid and transaction times. Constraints and triggers or application logic should ensure time periods don’t overlap unless explicitly required.

### Database Support
Many modern RDBMS support temporal tables natively. Utilize features provided by platforms like IBM Db2, Oracle, and SQL Server to enforce time-based logic at the database level.

### Query Example
To query for promotions valid at a specific point in time (say July 4, 2024), use SQL like:
```sql
SELECT * 
FROM PromotionDimension 
WHERE ValidStartDate <= '2024-07-04' 
AND ValidEndDate >= '2024-07-04' 
AND TransactionEndDate > NOW();
```

## Related Patterns
- **Type 1 SCD**: Overwrite with new data without history.
- **Type 2 SCD**: Tracks historical data by creating a new record for changes.
- **Temporal Tables**: Database feature for managing historical data and changes more seamlessly.

## Additional Resources
- [Temporal Data & Snodgrass, R. (1999). Developing Time-Oriented Database Applications in SQL](https://www.cs.arizona.edu/~rts/tdbbook.pdf)
- Online resources for Temporal Data Management using SQL

## Conclusion
The Bi-Temporal SCD design pattern is essential for systems needing thorough tracking of changes over time. With the growing complexity of modern applications and the necessity for detailed audit trails, implementing Bi-Temporal dimensions ensures compliance and robust data tracking for current and historical views, providing a richer, dual-faceted temporal perspective in data warehousing.

---
linkTitle: "Slowly Changing Dimension Type 2 (SCD2)"
title: "Slowly Changing Dimension Type 2 (SCD2)"
category: "Dimensional Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Slowly Changing Dimension Type 2 (SCD2) is a design pattern that allows you to preserve historical data in dimensional modeling by creating a new record for every change with associated effective dates. This enables robust trend analysis and historical data tracking."
categories:
- Data Modeling
- Dimensional Modeling
- Historical Data Preservation
tags:
- SCD2
- Dimensional Modeling
- Data Warehousing
- Historical Data
- ETL
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/2/10"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Slowly Changing Dimension Type 2 (SCD2)

### Introduction
In the realm of data warehousing, Slowly Changing Dimensions (SCD) represent a critical design pattern for maintaining historical accuracy and supporting effective analytical queries. The Slowly Changing Dimension Type 2 (SCD2) pattern specifically addresses the need to track changes over time by creating new records instead of overwriting existing ones.

### Description
The SCD2 pattern preserves history within a dimension table by creating a new record each time an attribute value changes. This approach involves implementing additional fields to manage versioning - commonly a "StartDate" and "EndDate" - to signify the active period of a record.

#### Key Features:
- **Preservation of history**: Each change results in a new record, ensuring a complete historical audit trail.
- **Version control**: Implementing "StartDate" and "EndDate" fields delineates the lifespan of each record version.
- **Query flexibility**: Historical queries can retrieve data valid at any given point in time.

### Example
Consider a "Customer" dimension table where customer addresses might change over time. The implementation of SCD2 involves adding a new record each time an address is updated:

```plaintext
| CustomerID | Name     | Address       | StartDate  | EndDate    |
|------------|----------|---------------|------------|------------|
| 1          | John Doe | 123 Main St   | 2022-01-01 | 2022-06-01 |
| 1          | John Doe | 456 Elm St    | 2022-06-02 | NULL       |
```

In this table:
- The original record with "123 Main St" is closed with an "EndDate".
- A new record with "456 Elm St" reflects the latest state.

### Best Practices
- **Implement surrogate keys**: Use surrogate keys as primary keys for dimension tables to differentiate records uniquely.
- **Maintain time attributes**: Ensure consistent use of "StartDate" and "EndDate" for clear version management.
- **Design measures for performance**: Utilize indexing strategies and keep the "EndDate" as NULL for current records to facilitate easy querying of active records.

### Example Code
An example ETL logic using SQL for managing SCD2:

```sql
MERGE INTO Customer_Dimension AS target
USING (SELECT * FROM Staging_Customer) AS source
ON target.CustomerID = source.CustomerID AND target.EndDate IS NULL
WHEN MATCHED AND target.Address <> source.Address THEN
    UPDATE SET EndDate = CURRENT_DATE
WHEN NOT MATCHED BY TARGET THEN
    INSERT (CustomerID, Name, Address, StartDate, EndDate)
    VALUES (source.CustomerID, source.Name, source.Address, CURRENT_DATE, NULL);
```

### Related Patterns
- **Slowly Changing Dimension Type 1 (SCD1)**: Overwrites data directly without preserving history.
- **Slowly Changing Dimension Type 6 (SCD6)**: Combines multiple SCD types for hybrid and comprehensive change history tracking.

### Additional Resources
- *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling* by Ralph Kimball and Margy Ross.
- Online tutorials and resources on implementing SCD2 in various ETL tools such as Informatica, Talend, and Apache Nifi.

### Summary
Slowly Changing Dimension Type 2 (SCD2) plays a pivotal role in data warehousing by ensuring that historical data is preserved across changes in dimension attributes. By implementing this pattern, organizations gain the ability to conduct impactful trend analyses and derive insights from historical data with precision. This design pattern is indispensable for systems requiring historical data visibility and audit capabilities.

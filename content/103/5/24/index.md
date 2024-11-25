---
linkTitle: "Temporal Hierarchical Query"
title: "Temporal Hierarchical Query"
category: "Time Travel Queries"
series: "Data Modeling Design Patterns"
description: "A design pattern for retrieving hierarchical data with temporal aspects, useful for scenarios like tracking organizational changes over time."
categories:
- data-modeling
- temporal-data
- hierarchical-data
tags:
- hierarchical-queries
- temporal-queries
- data-modeling-patterns
- SQL
- time-travel
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/5/24"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Overview

A **Temporal Hierarchical Query** is a design pattern used to retrieve hierarchical data while accounting for changes over time. This pattern is particularly beneficial in scenarios where data structures, like organization charts, evolve, and there is a need to query the state of the hierarchy at a particular point in time.

## Use Cases

- **Organizational Charts**: To generate the structure of an organization as it existed on a specific date, which is critical for historical reporting and auditing purposes.
- **Product Categories**: Understanding how product categorizations/offers have changed over time within a retail system.
- **Versioned Document Management**: Viewing the hierarchical relations of documents and their versions over time.

## Design Considerations

- **Time Dimension**: Integrate a temporal aspect into the data model, typically using effective and expiration dates to represent the validity period of each hierarchy node.
- **Hierarchy Levels**: Define the depth and breadth of the hierarchy to efficiently traverse and query the structure.

## Example Code

```sql
WITH HierarchyCTE AS (
  SELECT 
    id, 
    parent_id, 
    position_name, 
    effective_date, 
    CASE 
      WHEN expiration_date IS NULL THEN '9999-12-31' 
      ELSE expiration_date 
    END AS expiration_date
  FROM employee_hierarchy
  WHERE effective_date <= @query_date 
    AND (expiration_date IS NULL OR expiration_date > @query_date)

  UNION ALL

  SELECT 
    eh.id, 
    eh.parent_id, 
    eh.position_name, 
    eh.effective_date, 
    eh.expiration_date
  FROM employee_hierarchy eh
  INNER JOIN HierarchyCTE hcte ON eh.parent_id = hcte.id
  WHERE eh.effective_date <= @query_date 
    AND (eh.expiration_date IS NULL OR eh.expiration_date > @query_date)
)

SELECT * FROM HierarchyCTE;
```

In this example, we define a Common Table Expression (CTE) to recursively traverse the hierarchy of an organization. The CTE ensures that each node and its children are valid on the query date specified by `@query_date`.

## Architectural Approaches 

- **Temporal Table**: Use database features like temporal tables in SQL Server or Oracle Flashback to keep track of row history. These features can facilitate easy querying of historical data.
- **Dimensional Modeling**: Implement Slowly Changing Dimensions (SCD) to manage the temporal aspects of the hierarchy.
- **Valid-Time Temporal Database**: Employ a database system that natively supports time-variant data to alleviate the need for manual implementation.

## Related Patterns

- **Slowly Changing Dimensions (SCDs)**: Useful for tracking changes over time in dimensional models.
- **Time-Based Snapshot**: Apply when a snapshot of the data at specific intervals is required.
- **Audit Logging**: For maintaining detailed temporal records of data modifications.

## Additional Resources

- ["Temporal Data and the Relational Model" by C.J. Date](https://example-link.com)
- [Temporal Data Processing in SQL](https://example-link.com)
- [Hierarchical Data Management using SQL](https://example-link.com)

## Summary

The **Temporal Hierarchical Query** pattern is essential for any application requiring temporal navigation of hierarchical information. By integrating time-focused querying capabilities, you ensure that your systems can provide insights into historical data structures. Leveraging database features specific to temporal and hierarchical data can significantly enhance both performance and the richness of data insights.

---
linkTitle: "Type 3 SCD - Add New Column"
title: "Type 3 SCD - Add New Column"
category: "Slowly Changing Dimensions (SCD)"
series: "Data Modeling Design Patterns"
description: "A design pattern in data warehousing and modeling where new columns are added to dimension tables to track limited history of data. This pattern is known for its simplicity and utility in keeping track of attributes' state changes over time."
categories:
- Data Warehousing
- Data Modeling
- Slowly Changing Dimensions
tags:
- SCD
- Type 3
- Data History
- Data Modeling
- Database Design
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/3/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

In the world of data warehousing, handling changes to data over time is a crucial aspect of maintaining reliable and meaningful analytics. Slowly Changing Dimension (SCD) patterns address this need, and among them, Type 3 SCD stands out for its simplicity and utility when you want to capture some history without additional complexity.

## What is Type 3 SCD?

Type 3 SCD - Add New Column is a design pattern used in data warehousing to maintain historical information about certain attributes within a dimension table by adding new columns. This method primarily focuses on tracking a limited amount of history, such as keeping both previous and current values of an attribute.

## Architectural Approach

The Type 3 SCD pattern involves the addition of extra columns to the dimension table. Each column corresponds to specific historical data the organization decides to keep. This pattern is particularly well-suited to scenarios where changes are less frequent and only a single old version of an attribute's value needs to be tracked alongside the current value.

## Design Pattern Implementation

1. **Identify Attributes**: Determine which attributes' changes need tracking. For example, tracking changes to a customer's marital status.

2. **Add Columns**: For each historical version you want to track, add separate columns to your dimension table. Typically, this includes a "previous" and "current" column. 
   - Example: `Marital_Status_Current`, `Marital_Status_Previous`.

3. **Update Logic**: Modify your ETL (Extract, Transform, Load) process such that whenever a change is detected in the monitored attribute:
   - The current value is moved to the "previous" column.
   - The new value populates the "current" column.
  
## Example Code

Assume a `Customer` table with attributes such as `Customer_ID`, `Name`, and `Marital_Status`.

```sql
ALTER TABLE Customer ADD COLUMN Marital_Status_Current VARCHAR(50);
ALTER TABLE Customer ADD COLUMN Marital_Status_Previous VARCHAR(50);

-- Initial population of these columns
UPDATE Customer
SET Marital_Status_Current = Marital_Status;

-- ETL Script for updating changes
CREATE OR REPLACE PROCEDURE UpdateCustomerMaritalStatus(new_status VARCHAR(50), id INT) AS
{{< katex >}}
BEGIN
   -- Move current status to previous
   UPDATE Customer
   SET Marital_Status_Previous = Marital_Status_Current
   WHERE Customer_ID = id;
   
   -- Update to the new current status
   UPDATE Customer
   SET Marital_Status_Current = new_status
   WHERE Customer_ID = id;
END;
{{< /katex >}}
```

## Related Patterns

- **Type 1 SCD**: Overwrites the old data with new data, keeping only current values.
- **Type 2 SCD**: Handles historical changes by creating new rows for each change and using surrogate keys for identity.
- **Type 6 SCD**: Combines aspects of types 1, 2, and 3 to offer complete flexibility in managing historical data.

## Additional Resources

- Kimball Group: [Basics of Dimension Design](https://www.kimballgroup.com)
- Books: "The Data Warehouse Toolkit" by Ralph Kimball

## Summary

Type 3 SCD's simplicity makes it a perfect choice for cases requiring only the immediate history alongside current values. While it doesn't offer comprehensive historical data management like Type 2, its strategic use of columns ensures low complexity with effective data traceability for certain business scenarios. This pattern is a valuable part of any data modeler's toolkit for tracking trajectory changes of critical attributes without overwhelming the system with excessive historical data.

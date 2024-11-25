---
linkTitle: "Type 6 SCD - Hybrid"
title: "Type 6 SCD - Hybrid"
category: "Slowly Changing Dimensions (SCD)"
series: "Data Modeling Design Patterns"
description: "Combining Type 1, 2, and 3 SCDs to capture multiple aspects of change in a data warehouse, maintaining historical records while allowing easy access to previous and current values."
categories:
- Data Modeling
- Data Warehousing
- Slowly Changing Dimensions
tags:
- Type 6 SCD
- Hybrid SCD
- Data Patterns
- Data Warehousing
- Historical Data
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/3/6"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

The Type 6 Slowly Changing Dimension (SCD) - Hybrid pattern is a sophisticated design pattern combining elements from Type 1, Type 2, and Type 3 SCDs. This hybrid approach leverages the simplicity of overwriting in Type 1, the historical tracking capability of Type 2, and the capability to track change with additional columns as seen in Type 3. It provides a complete dimension record view over time in a data warehouse by maintaining previous and current data effectively.

## Design Pattern Overview

### Characteristics

- **Type 1 (Overwrite):** Updates the record with the new data without maintaining historical data.
- **Type 2 (Add):** Adds a new record for each change with effective date columns, thus keeping a history of changes.
- **Type 3 (Add Column):** Adds new columns for storing the previous values of changing attributes.

### Hybrid Approach

The Type 6 SCD maintains all the capabilities by adding multiple columns to retain both current and previous states while keeping a full history of changes. This makes it a powerful pattern for contexts needing comprehensive audit trails and changes over time.

## Implementation

### Schema Example

To illustrate, let's consider a `Customer` dimension with columns such as:

- `CustomerID` (Primary Key)
- `CustomerName_ Current` (Type 1)
- `CustomerName_Previous` (Type 3)
- `CustomerName_StartDate`
- `CustomerName_EndDate`
- `IsCurrent` (Type 2)
- `Version`

```sql
CREATE TABLE CustomerDimension (
    CustomerID INT PRIMARY KEY,
    CustomerName_Current VARCHAR(100),
    CustomerName_Previous VARCHAR(100),
    CustomerName_StartDate DATE,
    CustomerName_EndDate DATE,
    IsCurrent BOOLEAN,
    Version INT
);
```

### Procedure

1. **Insertion of Initial Record**  
   New records insert with the current column values in place.

2. **Updating Existing Records**  
   - **Type 1: Overwrite the current value**:
     Update the `CustomerName_Current` directly for non-prioritized changes.
   - **Type 2: Preserve Historical Changes**:
     When significant changes are introduced:
     - Set `IsCurrent` to false for existing rows (closing them by setting `EndDate`).
     - Insert a new row capturing current values with `IsCurrent` set to true and updated `StartDate`.
   - **Type 3: Preserve Significant Value**:
     Move the `CustomerName_Current` to `CustomerName_Previous` when retaining prior value relevance.

### Example SQL Update

```sql
UPDATE CustomerDimension
SET 
    CustomerName_Previous = CustomerName_Current,
    CustomerName_Current = 'New Name',
    CustomerName_EndDate = CURDATE(),
    IsCurrent = FALSE 
WHERE CustomerID = 1 AND IsCurrent = TRUE;

INSERT INTO CustomerDimension (CustomerID, CustomerName_Current, CustomerName_StartDate, IsCurrent, Version)
VALUES (1, 'New Name', CURDATE(), TRUE, 1);
```

## Best Practices

- Carefully select which attributes warrant Type 3 management for previous values.
- Robustly synchronize system processes to ensure typologies of change are handled consistently.
- Design database indices considering the `IsCurrent`, `Version`, and `StartDate` for faster query execution.

## Related Patterns

- **Type 1 SCD**: Overwrites the existing record with the current data.
- **Type 2 SCD**: Maintains complete history with new records for changes.
- **Type 3 SCD**: Limited preservation of history with additional columns for prior values.

## Additional Resources

- [Kimball Group's Data Warehousing Toolkit](https://www.kimballgroup.com/)
- [Slowly Changing Dimensions Overview](https://www.dataversity.net/slowly-changing-dimensions-understanding-types-1-2-3/)

## Summary

The Type 6 SCD - Hybrid pattern stands out by combining the simple overwrite strategy of Type 1 with the historical tracking of Type 2 and the dimensional changes documented by Type 3. Its comprehensive approach makes it suitable for scenarios demanding detailed historical audits, allowing business users transparent views of changes over time while maintaining a manageable structure.

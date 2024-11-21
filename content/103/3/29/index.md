---
linkTitle: "Slowly Changing Facts"
title: "Slowly Changing Facts"
category: "Slowly Changing Dimensions (SCD)"
series: "Data Modeling Design Patterns"
description: "A design pattern that applies Slowly Changing Dimensions (SCD) techniques to fact tables when measures change, allowing for accurate historical data tracking and reconciliation in data warehousing."
categories:
- Data Modeling
- Data Warehousing
- Slowly Changing Dimensions
tags:
- SCD
- Data Modeling
- Fact Table
- Data Warehousing
- Historical Data
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/3/29"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In the realm of data warehousing and business intelligence, accurately tracking changes over time is vital for generating meaningful insights. While Slowly Changing Dimensions (SCD) are commonly used to handle changes in dimension tables, it is sometimes necessary to apply similar techniques to fact tables—this is known as the Slowly Changing Facts (SCF) pattern. SCF allows for the management of evolving business measures in fact tables, such as when transactions are updated with information like returns or corrections after their initial recording.

## Problem Statement

Traditional fact tables often assume that facts (measures) like sales, revenue, or expenses are static after they are initially logged. However, business realities dictate that these facts might need adjustments due to returns, corrections, or updated calculations. The challenge is to preserve the original data while also enabling the comparison of updated metrics over time without losing historical accuracy.

## Solution

The Slowly Changing Facts pattern involves extending fact tables to accommodate changes to facts by either overwriting or appending new rows. The approach taken can vary based on business requirements:

1. **Overwrite (Type 1 SCF)**:
   - Directly update the existing facts in place.
   - Simple to implement, but results in loss of historical data.

2. **Record Versioning (Type 2 SCF)**:
   - Introduce new rows with different versions of facts.
   - Include a versioning mechanism or validity period to distinguish between different states of the same fact entry.
   - Retains historical data allowing full traceability.

3. **Hybrid Approach (Type 3 SCF)**:
   - Maintain limited history directly with original records, such as adding new columns for updated versus original measures.
   - Balance between data accuracy and space efficiency.

Below is an example of Type 2 SCF implemented using SQL:

```sql
CREATE TABLE SalesFact (
    SalesID INT,
    ProductID INT,
    SalesDate DATE,
    Amount DECIMAL(12, 2),
    Version INT,
    EffectiveDate DATE,
    ExpirationDate DATE DEFAULT NULL,
    PRIMARY KEY (SalesID, Version)
);

/* Inserting original fact */
INSERT INTO SalesFact (SalesID, ProductID, SalesDate, Amount, Version, EffectiveDate) 
VALUES (1, 101, '2024-01-15', 150.00, 1, '2024-01-15');

/* Inserting an updated fact due to a return */
INSERT INTO SalesFact (SalesID, ProductID, SalesDate, Amount, Version, EffectiveDate) 
VALUES (1, 101, '2024-01-15', 130.00, 2, '2024-01-20');
```

## Related Patterns

- **Change Data Capture (CDC)**: Techniques for detecting and capturing changes in data, which can be utilized alongside SCF to propagate fact changes effectively.
- **Temporal Tables**: Managing changes in tables, often used to retain full historicity within both facts and dimensions.

## Best Practices

1. **Choose Appropriate SCF Type**: Understanding business tolerance for historical data versus accuracy will guide whether Type 1, Type 2, or Type 3 is appropriate.
2. **Auditing and Lineage Tracking**: Maintain thorough records of every change for traceability and regulatory compliance.
3. **Performance Tuning**: As copying entire records for every change can lead to large datasets, adopt optimization strategies (indexes, partitioning).

## Additional Resources

- Kimball's Dimensional Modeling Techniques documentation
- Snowflake's documentation on managing semi-structured data
- Data Vault modeling for storing historical data

## Summary

The Slowly Changing Facts design pattern applies SCD concepts to fact tables, allowing businesses to deal with evolving facts efficiently. By choosing between overwritting, versioning, or hybrid type changes, organizations can ensure both the integrity of historical data and accommodate the fluidity of real-world data environments. By doing so, analytics and business reporting can reflect true historical trends and corrective events accurately.

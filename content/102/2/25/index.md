---
linkTitle: "Slowly Changing Measures"
title: "Slowly Changing Measures"
category: "Dimensional Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Managing changes in fact table measures that need to be updated over time in a data warehouse environment."
categories:
- data_modeling
- dimensional_modeling
- cloud_computing
tags:
- slowly_changing_measures
- fact_table_patterns
- data_warehousing
- ETL_processes
- data_modeling
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/2/25"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In the landscape of data warehousing, particularly within the realm of dimensional modeling, measures in fact tables often need to dynamically adjust as underlying operational data evolves. This evolving characteristic of measure attributes necessitates strategies to keep fact tables up-to-date without compromising historical accuracy or performance. The Slowly Changing Measures (SCM) pattern offers methodologies for managing these changes effectively.

## Problem Statement

The primary challenge is maintaining the integrity of measure data that changes over time in fact tables. Consider a scenario where a measure like "EstimatedDeliveryDate" in an e-commerce system may need updates based on real-time operational changes. Mismanagement of such evolving measures can lead to reporting inaccuracies and flawed analytics.

## Solution

The Slowly Changing Measures pattern provides robust methods to handle such changes. Here are the approaches typically used:

### 1. Overwriting Measures

The simplest method involves updating the measure directly within the fact table whenever there's a change. This approach provides real-time accuracy but at the loss of historical data.

### 2. Versioning Measures

This involves adding a new record/version of the measure in the fact table with a timestamp or version number, allowing the capture of changes over time while preserving historic records.

### 3. Creating Additional Measures

Instead of overwriting, additional columns (e.g., ‘Original Estimated Delivery Date’ and ‘Current Estimated Delivery Date’) are used to store both original and updated measures. This helps maintain a clear historical trail.

## Implementing Slowly Changing Measures

Let's explore a simple example with a sales fact table:

```sql
CREATE TABLE SalesFact (
    OrderID INT PRIMARY KEY,
    ProductID INT,
    CustomerID INT,
    SaleDate DATE,
    EstimatedDeliveryDate DATE,
    CurrentEstimatedDeliveryDate DATE  -- Column added to hold the updated value
);
```
  
### Example: Adjusting "EstimatedDeliveryDate"

```sql
UPDATE SalesFact
SET CurrentEstimatedDeliveryDate = '2024-08-01'
WHERE OrderID = 12345;
```

## Considerations

- **Performance Impact**: Frequent updates to measures can lead to increased ETL complexity and resource demands.
- **Storage Requirements**: Versioning or adding new columns leads to higher storage utilization.
- **Analytical Complexity**: Complexity arises in query logic when differentiating between original and current measures.

## Related Patterns

- **Slowly Changing Dimension (SCD) Type 2**: Deals with changes in dimension tables similar to SCM for fact tables but more focused on categorical attributes.
- **Temporal Tables**: Another pattern for capturing historical changes over time, useful in slowly changing scenarios.

## Additional Resources

- Kimball, R. & Ross, M. (2013). "The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling."
- Inmon, W.H. (2005). "Building the Data Warehouse."

## Summary

The Slowly Changing Measures pattern is instrumental in analyzing data that reflects temporal changes in measures. By choosing the appropriate strategy such as overwriting, versioning, or adding measures, organizations can achieve accurate and meaningful insights. Balancing performance with analytical demands is crucial, as is understanding storage implications. As data warehousing continues to evolve with higher velocity data flows, SCM becomes an essential pattern in the architect's toolkit.

---
linkTitle: Periodic Snapshot Fact Table
title: "Periodic Snapshot Fact Table"
category: "Dimensional Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Captures data at regular intervals (e.g., daily, weekly), summarizing the state at each point in time."
categories:
- Data Modeling
- Dimensional Modeling
- Data Warehousing
tags:
- Fact Table
- Snapshot
- Data Modeling
- Data Warehousing
- Periodic Data
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/2/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Periodic Snapshot Fact Table

A **Periodic Snapshot Fact Table** is a type of dimensional modeling pattern employed to capture and store data at regular intervals, allowing analysts to observe how business processes change over time. Unlike transaction or accumulating snapshot fact tables, which are event-driven, periodic snapshots capture a broader snapshot of the business at a consistent frequency, such as daily, weekly, or monthly. This approach is particularly useful for trend analysis, performance measurement, and auditing across a defined period.

## Why Use a Periodic Snapshot Fact Table?

Periodic Snapshot Fact Tables are advantageous in scenarios where the goal is to track and analyze the states over a period:

- **Trend Analysis**: They allow stakeholders to see the state of the process or resource at consistent intervals.
- **Simplified Reporting**: By capturing aggregates at intervals, avoid the complexities inherent in transaction-level detail.
- **Performance Metrics**: Enables consistent tracking of metrics like inventory levels, daily sales, and account balances.

## Architectural Approach

The architectural design for creating a Periodic Snapshot Fact Table involves several critical steps:

1. **Identify Key Metrics**: Determine what performance metrics or state information need to be captured.
2. **Choose the Snapshot Interval**: Decide on a uniform snapshot interval that aligns with business needs, whether it be daily, weekly, or another time period.
3. **Design the Schema**: Include key metrics along with dimensions such as date/time, location, or product to contextualize the snapshots.
4. **ETL Process**: Implement an ETL (Extract, Transform, Load) pipeline to extract data from source systems, transform for consistency and load it into the fact table at each interval.

## Example: Daily Inventory Fact Table

Consider a company monitoring its inventory levels at the end of each day. A sample schema might look like this:

```sql
CREATE TABLE DailyInventory (
    SnapshotDate DATE,
    ProductID INT,
    LocationID INT,
    EndingInventory INT,
    AllocatedQuantity INT,
    OnOrderQuantity INT
);
```

Here, `DailyInventory` captures the state of product stock levels at each end-of-day. It includes dimensions for date, product, and location.

## Best Practices

- **Accurate Time Dimension**: Ensure a robust and precise time dimension to prevent latency or future discrepancies.
- **Granular Metrics**: Choose metrics that are both significant and align with business objectives.
- **Consistent Intervals**: Stick to the agreed interval while facilitating accommodations for special events or adjustments.
- **Regular Data Refreshing**: Employ processes that maintain the timeliness of data refreshes to uphold analysis quality.

## Related Patterns

- **Transaction Fact Table**: Records transactional events as they happen.
- **Accumulating Snapshot Fact Table**: Captures lifecycle processes by updating facts as events occur over time.

## Additional Resources

- [Kimball Group's Dimensional Modeling](https://www.kimballgroup.com)
- ["The Data Warehouse Toolkit" by Ralph Kimball](https://www.amazon.com/Data-Warehouse-Toolkit-Definitive-Dimensional/dp/1118530802)

## Summary

The Periodic Snapshot Fact Table facilitates monitoring trends over time by capturing data at pre-determined regular intervals. Used judiciously alongside other fact tables, it grants a valuable perspective for longitudinal business analysis. Ensure to follow best practices when choosing intervals and metrics to maximize insights derived from cumulative data points consistently amassed over your specified period.

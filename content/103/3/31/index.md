---

linkTitle: "Snapshot Fact Tables"
title: "Snapshot Fact Tables"
category: "Slowly Changing Dimensions (SCD)"
series: "Data Modeling Design Patterns"
description: "Capturing the state of a process at regular intervals, such as taking a daily snapshot of account balances for data analysis."
categories:
- Data Modeling
- Data Warehousing
- Business Intelligence
tags:
- Snapshot
- Fact Tables
- Data Warehousing
- Slowly Changing Dimensions
- ETL
date: 2024-07-07
type: docs

canonical: "https://softwarepatternslexicon.com/103/3/31"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview

Snapshot Fact Tables represent a design pattern in data warehousing where the periodic capture of data records the state of certain metrics or processes at consistent points in time. This pattern is primarily used to store regular snapshots of data that are not expected to accumulate in size rapidly, unlike transactional fact tables which grow with every new transaction.

## Detailed Explanation

Snapshot Fact Tables are critical for understanding the periodic behavior of processes or for performing trend analysis. They are essential when you need to observe how values progress over time or if you need to revisit historical states for analysis.

### Characteristics
- **Fixed Intervals**: Snapshots are typically taken at regular time intervals—daily, monthly, quarterly, or any interval that suits the business requirement.
- **Data Granularity**: Each entry in a Snapshot Fact Table generally corresponds to a level of granularity appropriate for easy comparison over time. Daily snapshots typically correspond to close-of-day states.
- **Dimensional Role**: While primarily a fact table, snapshots play a dimensional role by making time displacement analysis easier.
- **Stability**: Since the values are only updated when a snapshot is taken, the data remains stable and immutable for any given period snapshot.

## Example Use Case

Consider a financial institution needing to analyze the end-of-day balance of customer accounts. The snapshot fact table would include fields such as `Date`, `Account ID`, `End-Of-Day Balance`, and potentially additional metrics such as `Total Transactions`.

```sql
CREATE TABLE AccountBalanceSnapshot (
    SnapshotDate DATE,
    AccountID INT,
    EndOfDayBalance DECIMAL,
    TotalTransactions INT,
    PRIMARY KEY (SnapshotDate, AccountID)
);
```

In this table:
- `SnapshotDate` is the date the snapshot was taken.
- `AccountID` references the specific account.
- `EndOfDayBalance` shows the account balance at the day's end.
- `TotalTransactions` highlights the number of transactions during the day.

## Architectural Approaches

### ETL Strategy

The ETL (Extract, Transform, Load) process for Snapshot Fact Tables involves setting up regular jobs that pull data from operational systems, transforming the data into a snapshot format, and subsequently loading it into the data warehouse.

- **Extract**: Data is pulled from relevant source systems.
- **Transform**: Data undergoes transformation logging changes across multiple dimensions.
- **Load**: The transformed data is appended to the Snapshot Fact Tables.

### Scheduling and Automation

Scheduling tools like Apache Airflow or cron jobs are often used to ensure snapshots are taken consistently and in line with business cycle requirements.

## Best Practices

- **Automate the Snapshot Process**: Ensure the snapshot capture is automated and scheduled regularly.
- **Index Fields Wisely**: Given frequent reads, optimized indexing for commonly used attributes is crucial.
- **Monitor Snapshots**: Implement monitoring to ensure that snapshot processes are running correctly.
- **Consider Retention Policies**: Decide on retention periods based on business needs versus storage costs.

## Related Patterns

### Slowly Changing Dimensions (SCD)

Snapshot Fact Tables often integrate with slowly changing dimensions which track the changing state of other dataset dimensions over time, providing more context in analysis.

### Accumulating Snapshot Fact Tables

These capture data at major process milestones and are updated over time rather than at fixed intervals, offering alternative ways to understand long-term trends.

## Additional Resources

1. "The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling" by Ralph Kimball and Margy Ross.
2. Online Tutorials: Various online platforms offer tutorials on ETL processes for Snapshot Fact Tables.

## Summary

Snapshot Fact Tables are invaluable for capturing periodic states of data processes, facilitating historical trend analysis and enabling data-driven decision-making in complex enterprise environments. Properly implementing, managing, and utilizing these tables can yield significant insights about business processes over time. By supporting time-based analysis and supplementing slowly changing dimensions, they offer a comprehensive view into the data warehouse's temporal dimension.

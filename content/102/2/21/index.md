---

linkTitle: "Snapshot Fact Table"
title: "Snapshot Fact Table"
category: "Dimensional Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Capturing the state of a process at a point in time, often used for inventory or balances such as daily account balances in a bank."
categories:
- Dimensional Modeling
- Data Warehousing
- Business Intelligence
tags:
- Snapshot Fact Table
- Dimensional Modeling
- Data Warehousing
- Business Intelligence
- Fact Table Patterns
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/2/21"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
The Snapshot Fact Table is a vital design pattern within the realm of data warehousing and dimensional modeling. This pattern involves capturing the state of a certain process at specific and consistent points in time. The snapshot can represent data like inventory levels, daily account balances, or monthly budget expenses. This periodic capturing facilitates trend analysis over time, financial forecasting, and strategic decision-making. 

## Concept and Use Cases

### Concept
A Snapshot Fact Table is designed to track the status of inventories, account balances, or any periodic occurrences of properties over standardized time intervals. Unlike transactional fact tables, which record measurements at the time of the event, snapshot fact tables deal with maintaining the status at regular time intervals (e.g., daily, weekly, monthly).

### Use Cases
- **Financial Accounting**: Record daily account balances to track the daily financial position of clients.
- **Inventory Management**: Monitor daily or weekly stock levels to ensure optimal inventory control and analysis.
- **Project Management**: Track the status of ongoing projects at monthly intervals to gauge progress and predict delays.

## Architecture

### Designing a Snapshot Fact Table
1. **Grain/Granularity Definition**: Choose the level of detail to capture: daily, weekly, monthly, or annually based on the reporting requirement.
   
2. **Time Dimension**: The time dimension becomes critical in a snapshot fact table. Usually, the primary key combines the snapshot date with other dimensions.

3. **Dimensions Included**: Conventional dimensions—such as Product, Location, and Customer—are linked along with the Time dimension.

4. **Facts/Measures**: Snapshot facts may include static aggregated measures like "Daily Account Balance", "Monthly Inventory Level", or projected data for forecast purposes.

### Schema Example
Let's consider a "DailyAccountBalance" snapshot fact table design:

| Time_ID  | Account_ID | Customer_ID | Balance_Date | Daily_Balance |
|----------|------------|-------------|--------------|---------------|
| 20230701 | ACC123     | CUST456     | 2023-07-01   | 15,000.00     |

- **Time_ID**: Links to the Time Dimension for date-related attributes.
- **Account_ID**: Links to Account Dimension to provide details of applicable accounts.
- **Customer_ID**: Links to Customer Dimension for customer-related insights.
- **Balance_Date**: The actual date of the snapshot.
- **Daily_Balance**: Numerical measure capturing the state as of the snapshot date.

## Example Code

Below is a simple SQL statement to create a snapshot of daily account balances:

```sql
CREATE TABLE DailyAccountBalance (
  Time_ID INT,
  Account_ID VARCHAR(20),
  Customer_ID VARCHAR(20),
  Balance_Date DATE,
  Daily_Balance DECIMAL(15, 2),
  PRIMARY KEY (Time_ID, Account_ID)
);
```

## Related Patterns

- **Transactional Fact Table**: Records individual transactions, capturing facts at the event time.
- **Accumulating Snapshot Fact Table**: Persists data over the lifecycle of an event, updated as the event progresses.
- **Periodic Snapshot Pattern**: Regularly captures the status at fixed intervals like months or quarters beyond daily snapshots.

## Additional Resources
- "The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling" by Ralph Kimball
- Online tutorials on Dimensional Modeling
- Data Warehousing conference seminars and workshops

## Summary
The Snapshot Fact Table is an essential pattern for data analysts and business intelligence architects focusing on time-series analysis and longitudinal trend tracking. By capturing the state of affairs at regular intervals, businesses can make informed decisions based on historical data. Emphasizing appropriate granularity and linking relevant dimensions make this pattern a cornerstone of effective data warehousing solutions.



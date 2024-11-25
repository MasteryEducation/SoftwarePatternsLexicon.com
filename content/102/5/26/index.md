---
linkTitle: "Accumulating Snapshot Fact Tables"
title: "Accumulating Snapshot Fact Tables"
category: "5. Data Warehouse Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Fact tables that track processes with multiple steps, updated as events occur."
categories:
- Data Warehouse
- Data Modeling
- Design Patterns
tags:
- Data Warehouse
- Fact Tables
- ETL
- BI
- Analytics
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/5/26"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Overview

In data warehouse modeling, Accumulating Snapshot Fact Tables are designed to track processes that have a defined start and end, often involving multiple steps. These tables are updated periodically as key events within a process occur, capturing the lifecycle of finalized or ongoing processes. They are particularly useful for measuring the process over time, providing insight into the time taken to transition through different states.

## Key Characteristics

- **Multi-step Process Tracking**: Accumulating Snapshot Fact Tables are designed to inherently support business processes that include multiple, clearly defined steps.
  
- **Periodic Updates**: Unlike transaction or periodic snapshot fact tables, which tend to store immutable records for each transaction or snapshot, accumulating snapshot tables are updated as new data becomes available, reflecting the latest state of a business process.

- **Comprehensive Time Stamps**: These tables typically include multiple date fields indicating various stages of the process, such as order date, payment date, shipment date, and delivery date in an order processing scenario.

- **Convergence on Process Outcome**: Ultimately tracked is the final outcome or completion of the process sequencing through its lifecycle.

## Architectural Approaches

### Design and Implementation

1. **Identify the Business Process**: Clearly define the process to be modeled and identify all the critical milestones or steps involved.

2. **Estimate Expected Process Duration**: Understand typical duration to completion, helping to determine the appropriate frequency for data refresh or updates to this fact table.

3. **Define Staging and Updates**: Establish ETL processes to handle the initial loading of facts and events, as well as incremental updates as additional data or events occur.

4. **Normalization of Dimensions**: Ensure dimensions related to the process steps are well normalized to avoid redundancy and enhance flexibility for query operations.

5. **Insert and Update Operations**: Setup logic to handle both insertions of new processes and updates to existing records as new data becomes available.

### Example Implementation

Below is a simplified example in SQL to illustrate how an accumulating snapshot fact table might be designed and implemented for an order process.

```sql
CREATE TABLE OrderProcessFact (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    ProductID INT,
    OrderDate DATE,
    PaymentDate DATE,
    ShipmentDate DATE,
    DeliveryDate DATE,
    TotalAmount DECIMAL(10, 2),
    CONSTRAINT fk_Customer FOREIGN KEY (CustomerID)
        REFERENCES Customer(CustomerID),
    CONSTRAINT fk_Product FOREIGN KEY (ProductID)
        REFERENCES Product(ProductID)
);

-- ETL pseudo-code for updating an existing process
UPDATE OrderProcessFact
SET PaymentDate = COALESCE(source.PaymentDate, target.PaymentDate),
    ShipmentDate = COALESCE(source.ShipmentDate, target.ShipmentDate),
    DeliveryDate = COALESCE(source.DeliveryDate, target.DeliveryDate),
    TotalAmount = COALESCE(source.TotalAmount, target.TotalAmount)
FROM SourceTable source
WHERE OrderProcessFact.OrderID = source.OrderID;
```

## Best Practices

- **Regularly Update Accumulating Snapshots**: Schedule ETL processes to intelligently update these records, reflecting the business process as it changes.
  
- **Use Efficient Indexing**: To optimize performance, especially in the update phase, ensure proper indexing strategies aligned with query patterns.

- **Historical Changes Management**: Consider strategies like versioning within the facts to retain historical changes while maintaining up-to-date process states.

- **Balance Between Freshness and Performance**: Accumulating snapshots inherently represent compromise; strive to balance them by tuning update frequency and table design.

## Related Patterns

- **Periodic Snapshot Fact Table**: Suitable when each event or state change is to be recorded, regardless of its association with process milestones.

- **Transaction Fact Table**: Optimal for atomic business processes lacking extensive tiered transactions.

## Additional Resources

- Ralph Kimball's "The Data Warehouse Toolkit": Comprehensive resources on dimensional modeling including fact tables.
- ETL Best Practices: Whitepaper detailing efficient ETL design and operations relevant to updating fact tables.
  
## Summary

Accumulating Snapshot Fact Tables serve as a potent tool in the arsenal of data warehouse designs, catering specifically to the vivid representation of progress throughout multi-step processes. Their concise, state-focused architecture promotes a business's ability to analyze the efficiency and timelines associated with vital operational workflows, rendering them indispensable for comprehensive strategic insights.

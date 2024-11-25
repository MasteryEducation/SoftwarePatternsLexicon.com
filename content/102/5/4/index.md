---

linkTitle: "Satellite Tables"
title: "Satellite Tables"
category: "5. Data Warehouse Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Tables holding descriptive attributes and history for hubs and links in Data Vault."
categories:
- Data Vault
- Data Warehouse
- Data Modeling
tags:
- data-vault
- satellite-tables
- data-warehouse
- history-tracking
- data-modeling
date: 2024-07-07
type: docs

canonical: "https://softwarepatternslexicon.com/102/5/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Overview

Satellite Tables play a crucial role in the Data Vault modeling architecture, primarily aimed at tracking the descriptive attributes and historical changes related to core business entities represented by Hub and Link tables.

### Key Characteristics
- **History Tracking**: Satellites are responsible for capturing the historical data related to Hubs and Links. Each row represents a particular state of an entity at a specific point in time.
- **Descriptive Attribute Storage**: These tables store non-identifying, descriptive attributes that do not change the intent or primary key of a Hub or a Link but provide additional context.
- **Multiple Satellites**: A Hub or a Link can have multiple Satellites associated with it to separate different types of attributes or historical tracking requirements, such as operational data and financial attributes.

## Design Pattern Explanation

Satellite Tables offer a flexible way to manage history and descriptive data without polluting the core structure of business keys (Hubs) and relationships (Links) within the Data Vault.

### Example Code

Below is an example SQL snippet for creating a Satellite Table called `ProductSatellite` for a Hub representing products:

```sql
CREATE TABLE ProductSatellite (
    ProductHubKey VARCHAR(255),
    LoadDateTime TIMESTAMP,
    Source varchar(255),
    ProductName VARCHAR(255),
    ProductDescription TEXT,
    RecordHash VARCHAR(64),
    PRIMARY KEY (ProductHubKey, LoadDateTime)
);
```

### Important Fields
- `ProductHubKey`: Foreign key reference to the Hub table.
- `LoadDateTime`: Timestamp indicating when the record was loaded/changed.
- `Source`: The source system identifier from which the data was extracted.
- `RecordHash`: Unique hash field to identify unique records.

## Benefits
- **Historical Data Maintenance**: Ensures full historical data capture without complexity in the Hub or Link tables.
- **Attribute Separation**: Encourages separation of concerns by allowing diverse types of attributes to reside in dedicated Satellites.
- **Scalability**: Supports scalability by enabling more Satellites if additional historical data needs to be tracked.

## Related Patterns

- **Hub Tables**: Central tables holding unique business keys in Data Vault.
- **Link Tables**: Tables representing transactional or logical relationships between Hubs.
- **Data Mart Patterns**: Customized, purpose-built views emanating from Source System structures.

## Best Practices
- **Avoid Data Duplication**: Ensure each Satellite only contains one version of truth for the attributes it holds.
- **Track Source**: Maintain a source system identifier to support traceability and data lineage.

## Additional Resources

1. *The Data Vault Guru*: A comprehensive guide on adapting the Data Vault methodology for cloud environments. 
2. *Data Warehousing in the Age of Big Data* by Krish Krishnan.
3. Online courses and webinars by Data Vault Alliance offering deeper insights into implementation strategies.

## Summary

Satellite Tables provide the foundational structure for capturing and storing descriptive attributes and historical data for business entities within a Data Vault model. They separate concerns by housing attributes away from the primary keys, thereby creating a system that is both robust and flexible, well-suited to handle dynamic changes and detailed historical tracking. Incorporating Satellite Tables effectively is integral to leveraging the full potential of a Data Vault architecture, especially in cloud and big data environments.



---
linkTitle: "Temporal Foreign Keys"
title: "Temporal Foreign Keys"
category: "Bitemporal Tables"
series: "Data Modeling Design Patterns"
description: "Referential integrity constraints that consider temporal validity, ensuring that references are valid over time."
categories:
- Data Modeling
- Temporal Databases
- Referential Integrity
tags:
- Temporal Databases
- Foreign Keys
- Bitemporal
- Data Integrity
- SQL
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/2/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In traditional relational databases, foreign keys are used to enforce referential integrity constraints between tables. However, temporal databases introduce an additional dimension: time. The Temporal Foreign Keys design pattern addresses the challenge of maintaining referential integrity when both the referenced and referencing entities evolve over time.

## Problem Addressed

In scenarios where both entities in a relationship have temporal attributes (e.g., valid time periods), traditional foreign keys fall short. They don't check whether a referenced entity was valid at the same time as the referencing entity, leading to potential integrity issues in temporal datasets.

## Pattern Explanation

### Temporal Foreign Keys

Temporal Foreign Keys ensure that a reference between two entities is valid for a specified time period. This is crucial in scenarios such as financial transactions, where an asset must reference its value as it was during the transaction date.

### Key Aspects

1. **Time Dimensions**: Each table involved has temporal attributes (e.g., valid_start, valid_end).
2. **Integrity Constraint**: The foreign key constraint checks that the overlapping valid period exists between the entities.

### Example

Consider a simplified model of an order system:

- **Orders Table**: Contains `order_id`, `product_id`, `order_date`, `valid_start`, and `valid_end`.
- **Products Table**: Contains `product_id`, `description`, `valid_start`, and `valid_end`.

In this example, `order_date` must overlap with the `valid_start` and `valid_end` of the corresponding product.

```sql
ALTER TABLE Orders
ADD CONSTRAINT fk_product_validity
FOREIGN KEY (product_id, order_date)
REFERENCES Products (product_id, temporal_valid(product_id, order_date));

-- This constraint ensures that the product referenced in an order was valid at the time of the order.
```

### Implementation Note

To implement Temporal Foreign Keys, databases must support bitemporal tables and the ability to define constraints based on temporal attributes. Current DBMS technologies such as those facilitating SQL:2011 or custom procedures can be employed.

## Related Patterns

- **Bitemporal Tables**: Use bi-temporal tables to maintain both business and system time, enhancing comprehensive temporal data management.
- **Slowly Changing Dimensions (SCD)**: Applied for handling historical AVL data and maintaining historical changes.

## Additional Resources

- *Snodgrass, R. T. (1999). Developing Time-Oriented Database Applications in SQL*.
- Blog: [Temporal Databases: What, Why, and How](https://example.com/temporal-databases)
- Research Paper: "Temporal Databases" by Tansel, Clifford et al.

## Summary

The Temporal Foreign Keys pattern extends the traditional concept to ensure temporal consistency in databases. By considering the time dimension, it provides a mechanism to enforce referential integrity over time, ensuring that all references point to entities valid during the desired temporal period.

By employing this pattern, you enhance the integrity and reliability of your data in systems where temporal accuracy is crucial, such as financial transactions, asset management, and legal history databases.

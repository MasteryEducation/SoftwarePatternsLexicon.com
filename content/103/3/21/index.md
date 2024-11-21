---
linkTitle: "SCD with Multi-Active Records"
title: "SCD with Multi-Active Records Design Pattern"
category: "Slowly Changing Dimensions (SCD)"
series: "Data Modeling Design Patterns"
description: "Detailed explanation and implementation guide for the Slowly Changing Dimensions (SCD) pattern with support for multiple active records, ensuring accurate historical data retention while catering to scenarios with multiple concurrent states for an entity."
categories:
- Data Modeling
- ETL Design
- Database Management
tags:
- SCD
- Data Warehousing
- Multi-Active
- Data Modeling
- Historical Data
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/3/21"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

The "Slowly Changing Dimensions with Multi-Active Records" pattern extends traditional slowly changing dimensions by allowing multiple active records per entity. This is particularly useful in scenarios where entities, such as customers or products, may have multiple current statuses or attributes that change over time without deprecating prior states.

### Problem Statement

In many data warehousing scenarios, capturing historical data about dimension changes is crucial. Traditional SCD types either overwrite changes (Type 1), create a new record with validity dates mark previous records as inactive (Type 2), or track historical changes via new columns (Type 3). However, these approaches fall short in situations where an entity can have multiple valid statuses concurrently.

For instance, consider a customer who subscribes to several service plans simultaneously. Without support for multi-active records, capturing all active states can become cumbersome and inaccurate.

### Solution

The SCD with Multi-Active Records pattern addresses this by allowing simultaneous active records for the same entity. Each record represents a specific state, with attributes for status periods and metadata to distinguish overlapping states.

## Key Components

1. **Entity Identifier**: A unique key representing the main entity (e.g., `customer_id`).
2. **Surrogate Key**: A unique identifier for each state record (e.g., `plan_subscription_id`).
3. **Validity Dates**: Attributes marking the start and end (possibly null for current records) of each record's validity.
4. **Active Flag**: Indicates whether a record is active or historical.
5. **Concurrent Status Metadata**: Additional attributes or flags indicating specific concurrently active states.

## Implementation Approach

1. **Schema Design**:
   - Extend the dimensional table to include fields allowing overlapping periods.
   - Add a status identifier to differentiate concurrent records.

2. **ETL Processing**:
   - Modify ETL processes to allow concurrent state data entry.
   - Handle updates intelligently to reflect changes in active records.

3. **Query Patterns**:
   - Construct queries to retrieve an entity's current and historical states efficiently.
   - Optimize queries using indices on validity dates and/or active flags.

4. **Handling Changes**:
   - Implement logic to deactivate records outside the relevant period gracefully without losing concurrent history.

### Example Schema

```sql
CREATE TABLE customer_plans (
    plan_subscription_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    service_plan_id INT,
    start_date DATE,
    end_date DATE DEFAULT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50),
    CONSTRAINT unique_active_plan UNIQUE (customer_id, service_plan_id, start_date)
);
```

### Example Code

Here is a simplified example in Java illustrating the creation of a new active record for a customer who already has another active record, utilizing JDBC for database interaction.

```java
public void addActivePlan(int customerId, int newPlanId, LocalDate startDate) throws SQLException {
    String insertSql = "INSERT INTO customer_plans (customer_id, service_plan_id, start_date, is_active, status) " +
                       "VALUES (?, ?, ?, ?, ?)";
    try (Connection connection = dataSource.getConnection();
         PreparedStatement ps = connection.prepareStatement(insertSql)) {
        ps.setInt(1, customerId);
        ps.setInt(2, newPlanId);
        ps.setDate(3, Date.valueOf(startDate));
        ps.setBoolean(4, true);
        ps.setString(5, "ACTIVE");
        ps.executeUpdate();
    }
}
```

## Related Patterns

- **SCD Type 2**: Maintains historical changes but may not support concurrent states.
- **SCD Type 3**: Tracks limited history with new fields but less suitable for extensive concurrent records.
- **Temporal Pattern**: General concept handling time-variant data across databases.

## Additional Resources

- [Kimball Group's Data Warehousing Toolkit](https://www.kimballgroup.com/) - Covers in-depth SCD concepts.
- [AWS Data Warehousing Solutions](https://aws.amazon.com/big-data/datalakes-and-analytics/) - For cloud-based handling of data warehousing patterns.

## Conclusion

The "SCD with Multi-Active Records" design pattern enriches traditional SCD models by allowing multiple active records, effectively supporting scenarios with inherent concurrency in states. Properly implemented, it ensures comprehensive historical and current data representation, invaluable for accurate data analysis and reporting.



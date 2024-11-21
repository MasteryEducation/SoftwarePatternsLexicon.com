---
linkTitle: "Temporal Referential Integrity"
title: "Temporal Referential Integrity"
category: "Bitemporal Tables"
series: "Data Modeling Design Patterns"
description: "Ensuring foreign keys reference data valid at the same time, exemplified by an insurance policy referencing the agent who sold it during the policy's valid period."
categories:
- Data Modeling
- Database Design
- Integrity Constraints
tags:
- Bitemporal Data
- Referential Integrity
- Data Consistency
- Temporal Databases
- SQL
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/2/13"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Temporal Referential Integrity is a crucial design pattern used in databases that manage temporal data, specifically bitemporal tables. Traditional referential integrity maintains consistency within a single time dimension, whereas temporal referential integrity ensures that the referencing data and the referenced data are valid during overlapping time periods. This pattern is especially useful in scenarios requiring historical tracking and future projections, such as in financial and insurance systems.

## Context and Problem

When managing data that changes over time, such as customer contracts, medical records, or any event-driven data, it is essential to maintain integrity across time dimensions. For example, consider an insurance application where policies are associated with agents, and both entities have effective and expiration dates. Misalignments between these entities can lead to inaccurate data interpretations, like associating a policy with an agent at a time when the agent wasn't active.

## Solution

Temporal Referential Integrity involves enhancing referential integrity with temporal constraints that ensure time validity across referenced relationships. This means implementing checks to ensure:

- The referenced record is valid simultaneously within the timelines of the referencing record.
- Both past (historical) and future (planned) states are consistently managed.

### Implementation Steps

1. **Define Bitemporal Tables**:
   - Use start and end dates to manage the validity range (effective_start, effective_end) for both the referencing and referenced tables.

2. **Ensure Correspondence**:
   - Implement database constraints and triggers to ensure that the validity range of the foreign key reference coincides with or is contained within the validity range of the referenced table.

3. **Query Adjustments**:
   - Modify SQL queries to consider temporal dimensions:
   ```sql
   SELECT *
   FROM policies p
   JOIN agents a ON p.agent_id = a.agent_id
   WHERE p.start_date >= a.start_date
     AND p.end_date <= a.end_date
     AND p.current_flag = true;
   ```

4. **Schema Example**:
   ```sql
   CREATE TABLE agents (
       agent_id SERIAL PRIMARY KEY,
       name VARCHAR(100),
       effective_start DATE,
       effective_end DATE
   );

   CREATE TABLE policies (
       policy_id SERIAL PRIMARY KEY,
       agent_id INT,
       policy_number VARCHAR(50),
       effective_start DATE,
       effective_end DATE,
       FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
           ON UPDATE CASCADE
           ON DELETE RESTRICT
   );
   ```

5. **Add Temporal Constraints**:
   - Implement constraints to validate temporal alignment:
   ```sql
   ALTER TABLE policies
   ADD CONSTRAINT constraint_name
   CHECK (effective_start >= (SELECT MIN(a.effective_start) FROM agents a WHERE a.agent_id = agent_id)
          AND effective_end <= (SELECT MAX(a.effective_end) FROM agents a WHERE a.agent_id = agent_id));
   ```

## Related Patterns

- **Bitemporal Modeling**: Extends the concept to support valid time and transaction time.
- **Audit Trail**: Maintaining a full history of all records change.
- **Slowly Changing Dimension (Type 2)**: Capturing and managing changes over time in dimensional data.

## Additional Resources

- *Temporal Data & the Relational Model* by C.J. Date
- [Temporal Patterns in Databases](https://edbt.edu) (Online course)
- SQL Server Temporal Tables Documentation

## Example Use Case

Consider a healthcare patient management system where patients’ treatments are logged with healthcare providers. Temporal Referential Integrity ensures that treatments are only linked with providers who were valid during the time of treatment.

## Summary

Temporal Referential Integrity ensures robust data consistency across multiple time streams. By aligning foreign key constraints with temporal aspects, it strengthens the integrity of data that evolves or overlaps in time, which is fundamental in temporal and bitemporal data architectures. Understanding and implementing this design pattern are essential for systems requiring precise temporal data management, especially in domains where the time context of data is critical.

---
linkTitle: "Fourth Normal Form (4NF)"
title: "Fourth Normal Form (4NF)"
category: "1. Relational Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "The Fourth Normal Form (4NF) is a database normal form that addresses the problem of multi-valued dependencies. It eliminates redundancy and inconsistencies caused by having multiple independent multi-valued facts about an entity in the same table."
categories:
- normalization
- database design
- relational modeling
tags:
- 4NF
- data normalization
- database optimization
- multi-valued dependency
- relational database
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/1/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Fourth Normal Form (4NF)

The Fourth Normal Form (4NF) is an extension of normalization processes that refine the database design by addressing multi-valued dependencies that exist within a relational database schema. Building on the foundational principles of the Third Normal Form (3NF), 4NF further ensures that records within a database do not contain two or more independent multi-valued facts, thus improving data integrity, reducing redundancy, and enhancing overall efficiency.

## Multi-Valued Dependencies and 4NF

In relational database design, a multi-valued dependency occurs when one attribute in a table is associated with multiple values of another attribute, independent of other attributes. The Fourth Normal Form demands the removal of such multi-valued dependencies to ensure that all data can be systematically categorized without redundancy or loss of information.

### Example

Consider a table recording `Employee`, `Skill`, and `Certification`:

| Employee | Skill       | Certification |
|----------|-------------|---------------|
| John     | Java        | AWS Certified |
| John     | Scala       | AWS Certified |
| John     | Java        | Oracle Cert   |
| ...      | ...         | ...           |

In the table above, an employee like John may have multiple skills and multiple certifications, which often lead to unnecessary redundancy.

**Solution in 4NF**:

Separate the facts into two tables, addressing skills and certifications independently:

**Employee_Skills Table:**

| Employee | Skill |
|----------|-------|
| John     | Java  |
| John     | Scala |

**Employee_Certifications Table:**

| Employee | Certification |
|----------|---------------|
| John     | AWS Certified |
| John     | Oracle Cert   |

This separation ensures the elimination of multi-valued dependencies, reducing redundancy and enhancing integrity.

## Applying 4NF

To transform a table into 4NF, follow these steps:
1. Identify any table that displays multi-valued dependencies by analyzing existing data relationships.
2. Creatively decompose the table into multiple related tables where each table addresses a single multi-valued aspect.
3. Ensure that all resulting tables conform to the standard rules of the lower normal forms, i.e., 1NF, 2NF, and 3NF.

## Related Patterns

- **First Normal Form (1NF):** Focuses on ensuring that columns contain atomic values.
- **Second Normal Form (2NF):** Deals with eliminating partial dependencies of non-prime attributes on candidate keys.
- **Third Normal Form (3NF):** Ensures that there are no transitive dependencies on the primary key.

## Additional Resources

- [Database Normalization Basics](https://en.wikipedia.org/wiki/Database_normalization)
- [Advanced Database Design by Olaf de Smedt](#)
- [Normalization and its Advantages: A Practitioner’s Guide](#)

## Summary

Fourth Normal Form (4NF) is an essential part of advanced database design, enhancing the integrity and efficiency of data structures by removing multi-valued dependencies. By transforming records into 4NF, systems can achieve high levels of optimization, reducing redundancy and ensuring that datasets remain consistent and sensible.

Incorporating 4NF within database schemas is a best practice that aligns perfectly with principles of normalization, and it plays a crucial role in designing robust, scalable databases suited for both modern applications and complex enterprise environments.

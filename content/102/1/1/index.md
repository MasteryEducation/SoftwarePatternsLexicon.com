---
linkTitle: "First Normal Form (1NF)"
title: "First Normal Form (1NF)"
category: "1. Relational Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "First Normal Form (1NF) ensures that all columns are atomic, with no repeating groups or arrays, and that each record in a database table is unique and well-defined."
categories:
- Relational Modeling
- Data Normalization
- Databases
tags:
- 1NF
- Normalization
- Relational Databases
- Atomicity
- Unique Records
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/1/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Relational databases have been foundational in organizing and retrieving data efficiently through structured data representations. The normalization process involves restructuring a relational database in accordance with certain normal forms to reduce data redundancy and improve data integrity. First Normal Form (1NF) is the baseline level of database normalization that must be satisfied by a database schema.

## Description

First Normal Form (1NF) establishes the foundational requirements for a table in a relational database. The fundamental principles of 1NF include:

- **Atomicity**: All columns should contain atomic, indivisible values. Multi-valued attributes, composite values, and nested structures are disallowed.
- **Uniqueness**: Each record (row) in the table must be unique.
- **Tabular Format**: Data should be organized in a table with clear rows and columns, where each column typically describes a different attribute, and each row acts as a unique record.

Achieving 1NF is crucial for ensuring that the database can efficiently execute queries and perform CRUD (Create, Read, Update, Delete) operations without anomalies.

## Design Pattern

1NF is a design pattern that guides the initial and fundamental step in organizing a relational database. Implementing this pattern effectively prepares a table to be further normalized to higher normal forms.

### Principles
- Enforce atomic values.
- Eliminate repeating groups using separate tables or multiple columns.
- Ensure there are primary keys.

### Example

Consider a table capturing contact information:

**Initial Table Structure (Not in 1NF)**

| ID | Name    | PhoneNumbers         |
|----|---------|----------------------|
| 1  | Alice   | 123-4567, 234-5678   |
| 2  | Bob     | 345-6789             |

**Transformed Table Structure (In 1NF)**

| ID | Name  | PhoneNumber |
|----|-------|-------------|
| 1  | Alice | 123-4567    |
| 1  | Alice | 234-5678    |
| 2  | Bob   | 345-6789    |

In this transformation, the `PhoneNumbers` column, which held multiple phone numbers, is split into individual records for each phone number. Now, each number stands independently and satisfies atomicity.

## Related Patterns

- **Second Normal Form (2NF)**: Further normalizes 1NF tables by addressing partial dependencies.
- **Third Normal Form (3NF)**: Eliminates transitive dependencies that exist in 2NF.

## Best Practices

- Define primary keys to ensure every record is unique.
- Minimize the use of null values as placeholders for nonexistent entries.
- Split complex fields containing multiple pieces of data into separate atomic columns.

## Additional Resources

- [Database System Concepts by Silberschatz, Korth, Sudarshan](https://www.db-book.com/)
- [SQL and Relational Databases - Towards Data Normalization](https://www.w3schools.com/sql/sql_normalization.asp)

## Summary

First Normal Form (1NF) sets the groundwork for designing robust and efficient relational database structures. By requiring atomic, indivisible values and ensuring each table entry is unique, 1NF helps prevent potential anomalies, optimizes storage efficiency, and facilitates seamless data retrieval and manipulation in a database system. Adhering to 1NF is pivotal in progressing to higher-order normalization forms, further refining the database's role as an effective data management tool.

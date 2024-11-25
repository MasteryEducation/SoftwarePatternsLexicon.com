---

linkTitle: "Temporal Third Normal Form (3NFt)"
title: "Temporal Third Normal Form (3NFt)"
category: "Temporal Normalization"
series: "Data Modeling Design Patterns"
description: "A design pattern aimed at eliminating transitive dependencies in temporal data, ensuring that non-key temporal attributes depend solely on the temporal primary key."
categories:
- database
- normalization
- data-modeling
tags:
- temporal-data
- normalization
- transitive-dependency
- data-integrity
- database-design
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/103/10/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Temporal Third Normal Form (3NFt)

### Description

The Temporal Third Normal Form (3NFt) is an extension of the traditional Third Normal Form that applies to temporal databases. It aims to eliminate transitive dependencies in temporal data. In a temporal database, data is tracked over time, which introduces additional complexities when ensuring data integrity and normalization.

3NFt ensures that all non-key temporal attributes in a relation depend only on the temporal primary key, not on other non-key attributes. This principle helps maintain data integrity across temporal data stores by preventing anomalies that may arise when non-key temporal attributes depend on other non-key temporal attributes.

### Architectural Approach

1. **Identification of Temporal Attributes**: Recognize which attributes in your database are temporal. Temporal attributes are those which have significance concerning time, i.e., they may change over time, like employee salary over different periods.

2. **Primary Key Definition**: Define a composite primary key that includes both the original primary key and a temporal component (such as a datetime field).

3. **Transitive Dependency Analysis**: Analyze your current database schema to detect any transitive dependencies among temporal attributes.

4. **Schema Redesign**: If temporal attributes have transitive dependencies, move them to separate tables where they depend only on the temporal primary key. This may involve creating tables specifically to handle these dependent temporal attributes.

5. **Maintain Data Consistency**: Utilize triggers or stored procedures to ensure changes in the temporal data are consistently propagated across your schema when primary data changes.

### Example Implementation

Consider a simplified database of employee salary data:

**Initial Schema:**

| EmployeeID | Salary   | StartDate:ValidToDate | TaxCode |
|------------|----------|-----------------------|---------|
| 101        | 60000    | 2024-01-01:2024-12-31 | TX001   |
| 102        | 50000    | 2024-01-01:2024-06-30 | TX002   |

In this case, suppose `TaxCode` is functionally dependent on `Salary`, which violates 3NFt because it introduces a transitive dependency between `TaxCode` and `EmployeeID`.

**Redesigned Schema:**

1. **EmployeeSalary Table**:

   | EmployeeID | Salary   | StartDate:ValidToDate |
   |------------|----------|-----------------------|
   | 101        | 60000    | 2024-01-01:2024-12-31 |
   | 102        | 50000    | 2024-01-01:2024-06-30 |

2. **SalaryTax Table**:

   | Salary   | TaxCode | StartDate:ValidToDate |
   |----------|---------|-----------------------|
   | 60000    | TX001   | 2024-01-01:2024-12-31 |
   | 50000    | TX002   | 2024-01-01:2024-06-30 |

### Related Patterns

- **Temporal First Normal Form (1NFt)**: Ensures that each column in a table contains only atomic, indivisible values, while respecting temporal aspect.
  
- **Temporal Second Normal Form (2NFt)**: Eliminates partial dependencies on non-key attributes in temporal relations.

### Additional Resources

- [Normalization in Temporal Databases](https://en.wikipedia.org/wiki/Temporal_database#Normal_forms)
- [Temporal Database Handbook](https://www.researchgate.net/publication/Temporal_Database_Handbook)

### Summary

Temporal Third Normal Form (3NFt) is a database normalization pattern focused on eliminating transitive dependencies in temporal data. This pattern helps maintain data integrity and prevents anomalies by ensuring that each non-key temporal attribute is solely dependent on the temporal primary key. By adopting 3NFt, data can be accurately tracked and managed over time without introducing redundant or inconsistent data relations.

---

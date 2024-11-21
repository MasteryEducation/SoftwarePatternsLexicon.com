---

linkTitle: "Star Schema"
title: "Star Schema"
category: "Dimensional Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "The Star Schema is a data warehousing model used to organize data into a central fact table connected to multiple denormalized dimension tables, resembling a star structure. It is ideal for supporting complex queries and data analytics in business intelligence applications."
categories:
- Data Modeling
- Data Warehousing
- Schema Design
tags:
- Star Schema
- Data Warehouse
- Business Intelligence
- OLAP
- Dimensional Modeling
date: 2024-07-07
type: docs

canonical: "https://softwarepatternslexicon.com/102/2/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

The Star Schema design pattern is a fundamental paradigm in data warehousing and business intelligence. It organizes data into fact and dimension tables to simplify complex queries and enhance data retrieval performance.

## Architectural Overview

The star schema consists of the following core components:

- **Fact Table**: The central table containing quantitative data for analysis, such as sales, revenue, or transactions. It often holds keys to associated dimension tables and measurable facts (metrics).

- **Dimension Tables**: Surrounding tables that store descriptive attributes related to the facts, such as time periods, product details, customer information, and store data. These tables are denormalized and connected directly to the fact table.

Below is a typical representation of a Star Schema:

```mermaid
erDiagram
    FACT_TABLE {
        PK fact_id
        FK1 time_id
        FK2 product_id
        FK3 customer_id
        FK4 store_id
        measure1 INT
        measure2 INT
    }
    TIME {
        PK time_id
        date DATE
        year INT
        month INT
        day INT
    }
    PRODUCT {
        PK product_id
        product_name STRING
        category STRING
        brand STRING
    }
    CUSTOMER {
        PK customer_id
        customer_name STRING
        email STRING
        region STRING
    }
    STORE {
        PK store_id
        store_name STRING
        location STRING
        manager STRING
    }
    FACT_TABLE }|..|{ TIME : "based on"
    FACT_TABLE }|..|{ PRODUCT : "relates to"
    FACT_TABLE }|..|{ CUSTOMER : "ordered by"
    FACT_TABLE }|..|{ STORE : "sold at"
```

## Design Considerations

### Benefits

- **Simplified Queries**: The denormalized structure allows for simpler and faster SQL queries, optimizing performance for read-heavy operations like OLAP (Online Analytical Processing) tasks.
  
- **Improved Performance**: Since dimension tables are typically small, Joins with a central fact table are efficient, enhancing query speed.

- **User-Friendly**: The design is intuitive and easy for end-users to navigate, making it ideal for reporting tools compared to normalized schemas.

### Drawbacks

- **Data Redundancy**: Denormalization can lead to data redundancy within dimension tables, which might increase storage requirements.

- **Limited Scalability**: This schema might become less manageable as the number of dimensions grows exceedingly, possibly complicating data maintenance.

## Best Practices

- **Stable Dimensions**: Dimension tables should contain relatively static data to avoid frequent updates and ensure query efficiency.
  
- **Consistent Granularity**: Maintain the same level of granularity across all facts and dimensions for consistency in querying and data analysis.

- **Indexing**: Use appropriate indexing strategies on foreign keys and other frequently queried columns to improve access speed.

## Related Patterns

- **Snowflake Schema**: This is an extension of the star schema wherein dimension tables are further normalized into sub-dimensional tables, making it a bit more complex than the star schema but reducing data redundancy.

- **Galaxy Schema (Fact Constellation)**: Combines multiple star schemas by sharing dimension tables among different fact tables.

## Example Code

Example SQL for creating a star schema:

```sql
CREATE TABLE Fact_Sales (
    fact_id SERIAL PRIMARY KEY,
    time_id INT,
    product_id INT,
    customer_id INT,
    store_id INT,
    sales_amount DECIMAL(10, 2),
    units_sold INT
);

CREATE TABLE Dim_Time (
    time_id SERIAL PRIMARY KEY,
    date DATE,
    year INT,
    month INT,
    day INT
);

CREATE TABLE Dim_Product (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    brand VARCHAR(50)
);

CREATE TABLE Dim_Customer (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100),
    email VARCHAR(100),
    region VARCHAR(50)
);

CREATE TABLE Dim_Store (
    store_id SERIAL PRIMARY KEY,
    store_name VARCHAR(100),
    location VARCHAR(100),
    manager VARCHAR(100)
);
```

## Additional Resources

- Kimball, Ralph. *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*.
- Inmon, Bill. *Building the Data Warehouse*.

## Summary

The Star Schema is a widely adopted data modeling pattern due to its simplicity, performance advantages, and ease of use, especially in the context of data warehousing and analytical scenarios. While it may face scaling challenges with excessive dimensions, its benefits in terms of performance and user-friendliness make it a fundamental design choice in business intelligence.

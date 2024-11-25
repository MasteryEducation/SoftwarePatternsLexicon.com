---

linkTitle: "Secondary Indexes"
title: "Secondary Indexes"
category: "NoSQL Data Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Enhance NoSQL database query performance by creating indexes on non-primary key attributes, exemplified by indexing fields such as 'email' in a MongoDB user collection."
categories:
- NoSQL
- DataModeling
- IndexingPatterns
tags:
- SecondaryIndexes
- NoSQL
- DataModeling
- MongoDB
- PerformanceOptimization
date: 2024-07-07
type: docs

canonical: "https://softwarepatternslexicon.com/102/3/17"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Secondary Indexes

Secondary indexes are an essential NoSQL data modeling pattern that allows developers to index non-primary key attributes to enhance query performance. Unlike primary indexes that are automatically generated based on a document's primary key or unique identifier, secondary indexes enable faster retrieval for queries that filter or sort data using other attributes.

## Design Pattern Details

### Motivation

In NoSQL databases, data retrieval often involves accessing documents or records using their primary keys. However, when queries involve attributes other than the primary key, performance can degrade as the system scans each document to filter records. Secondary indexes provide a mechanism to index other attributes that are frequently queried, reducing the time complexity by allowing direct access to these indexed attributes.

### Applicability

Secondary indexes are useful in scenarios where:
- Non-primary key attributes are frequently queried.
- Sorting or filtering operations are required on non-primary attributes.
- Query performance optimization is a priority.

### Example

Consider a MongoDB collection `users` containing documents with fields such as `_id`, `name`, `email`, and `age`. By default, MongoDB creates a primary index on the `_id` field. To improve query performance when searching by `email`, a secondary index can be created:

```json
{
  "_id": ObjectId("507f191e810c19729de860ea"),
  "name": "Alice",
  "email": "alice@example.com",
  "age": 31
}
```

Creating a secondary index in MongoDB on the `email` field can be accomplished with the following command:

```shell
db.users.createIndex({ "email": 1 })
```

This command creates an index on the `email` field, optimizing queries like `db.users.find({ "email": "alice@example.com" })`.

## Architectural Approaches

### Implementation

In a NoSQL database context, implementing secondary indexes typically involves:
1. Identifying the fields that are queried frequently as filters or sort keys.
2. Creating indexes using either the database's CLI, graphical interface, or through programmatic interface.
3. Monitoring index performance and assessing maintenance tasks like re-indexing or index maintenance during data updates.

### Considerations

- **Trade-offs**: While secondary indexing can substantially improve read performance, it involves trade-offs with write performance as the index needs updating whenever a document is inserted or updated.
- **Storage Overhead**: Indexes consume additional storage space, a factor that is crucial when designing for systems with large datasets.
- **Index Complexity**: Introducing secondary indexes should be balanced with application needs, avoiding over-indexing which may lead to unnecessary complexity.

## Paradigms and Best Practices

- Use indexes on fields that have a high rate of cardinality that can benefit from indexed searches.
- Regularly monitor and analyze index performance and adjust as necessary.
- Avoid over-indexing; only create indexes that are justified by query patterns.
- Evaluate the impact of indexes on write performance during benchmarking.

## Related Patterns

- **Compound Indexes**: Combination of multiple attributes to optimize queries on multiple fields simultaneously.
- **Sharding**: Splitting a database into smaller mutable devices (shards) to further enhance scalability, often used together with indexing.
- **Materialized Views**: Precomputed views that similarly enhance performance for read-heavy workloads.

## Additional Resources

- [MongoDB Indexing Documentation](https://docs.mongodb.com/manual/indexes/)
- [Apache Cassandra Secondary Indexes](https://cassandra.apache.org/doc/latest/cql/indexes.html)
- [A Comprehensive Guide to Indexing in NoSQL](https://dzone.com/articles/a-comprehensive-guide-to-indexing-in-nosql-databas)

## Summary

Secondary indexes in NoSQL databases are a vital technique for improving query performance by indexing non-primary key attributes. When properly applied, they reduce query execution time without needing to alter the application logic that interacts with the database. However, the addition of indexes should be reasoned with careful evaluation against their cost on write performance and storage requirements. Understanding and implementing secondary indexes is a key skill in optimizing NoSQL data modeling and retrieval strategies.

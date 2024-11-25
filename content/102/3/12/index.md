---
linkTitle: "Nested Set Model"
title: "Nested Set Model"
category: "NoSQL Data Modeling Patterns"
series: "Data Modeling Design Patterns"
description: "Using left and right values to represent hierarchical relationships without recursion."
categories:
- NoSQL
- Hierarchical Modeling
- Data Modeling
tags:
- Nested Set Model
- Hierarchical Data
- Non-Recursive Hierarchy
- NoSQL Patterns
- Data Modeling Best Practices
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/3/12"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

### Introduction

The Nested Set Model is a design pattern used for representing hierarchical data structures in a database without the need for recursion. This model is especially useful when dealing with read-heavy operations such as querying a hierarchy. The model utilizes two special columns, typically referred to as left and right values, to identify the position of nodes within the hierarchy. 

### Pattern Description

The key idea behind the Nested Set Model is to store each node with two numbers that represent its position in a preorder traversal of the tree. These numbers enable the efficient storage and retrieval of hierarchical data. Each tree or hierarchy is represented by a single table and a set of records. The table records contain the left and right values, making it easy to fetch the descendant nodes of any node without needing to traverse the data recursively.

#### Advantages

- **Efficient Read Operations**: The most significant benefit of the Nested Set Model is the efficiency of read operations. Querying all nodes in a hierarchy or subtree is simple and quick.
- **No Recursion Needed**: Unlike the adjacency list model, which may require recursive queries, the Nested Set Model enables hierarchical data retrieval using simple SQL selects.

#### Disadvantages

- **Complex Writes**: Insertions and deletions can be complex since they involve updating left and right values across potentially many nodes.
- **Overhead for Updates**: Modifications, especially on large trees, can be costly in terms of performance, as they may require recalibration of many nodes' left and right values.

### Example Use Case

Consider managing a file system hierarchy in a NoSQL or SQL database:

- `Documents` folder has `left=1` and `right=14`
- `Images` folder has `left=2` and `right=5`
- File `Sunset.jpg` in `Images` has `left=3` and `right=4`

With this arrangement:
- To retrieve all siblings or children of `Images`, query where `left` is between `2` and `5`.

### Example SQL

To retrieve all nodes under a particular node:
```sql
SELECT * FROM categories WHERE left_value BETWEEN :parent_left AND :parent_right ORDER BY left_value;
```

### Related Patterns

- **Adjacency List Model**: Another common pattern for representing hierarchical data, which uses parent-child relationships explicitly in each record.
- **Closure Table**: A more complex model that creates explicit paths for all node relationships, making both read and write operations comprehensive but complex.

### Additional Resources

- **Books**: "SQL for Smarties: Advanced SQL Programming" by Joe Celko.
- **Online Tutorials**: TutorialsPoint for in-depth Nested Set tutorials.

### Summary

The Nested Set Model offers a clean and efficient way to handle hierarchy traversal in read-heavy operations by way of left and right node identifiers. While it complicates data maintenance tasks slightly due to its need for recalculating indices upon insertions or deletions, the benefits of speed and efficiency in querying hierarchical data often outweigh these concerns. This pattern is ideal for applications where the hierarchy doesn’t change often but requires frequent traversals or reporting on subtree structures.

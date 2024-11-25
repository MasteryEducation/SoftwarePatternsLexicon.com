---
linkTitle: "Nested Set Model"
title: "Nested Set Model"
category: "Hierarchical and Network Modeling"
series: "Data Modeling Design Patterns"
description: "Uses left and right numeric values to represent hierarchical relationships, enabling efficient subtree queries."
categories:
- Hierarchical Modeling
- Data Modeling
- Database Design
tags:
- data-modeling
- hierarchical-structures
- database-optimization
- e-commerce
- sql
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/8/2"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

### Overview

The Nested Set Model is an advanced data modeling pattern used to represent hierarchical data structures in database systems. It assigns two numeric values, "Left" and "Right," to each node (or entity) in a hierarchy, making it efficient to query entire subtrees without recursive SQL queries.

### Purpose

The primary goal of the Nested Set Model is to optimize the retrieval of complex hierarchical data that involve multiple levels, making it an attractive choice for scenarios where read-heavy operations are prevalent.

### Architecture

In the Nested Set Model, each node is represented with the following attributes:

- **ID**: A unique identifier for the node.
- **Left**: An integer representing the left boundary of the node.
- **Right**: An integer indicating the right boundary of the entity.

When traversing from any node, all descendant nodes have values greater than the node's Left and less than the node's Right value.

### Example Use Case: E-Commerce Categories

Imagine a category tree in an e-commerce platform:

```
Electronics
├── Computers
│   ├── Laptops
│   └── Desktops
└── Televisions
```

This hierarchy can be represented in a database table like this:

| ID  | Name        | Left | Right |
|-----|-------------|------|-------|
| 1   | Electronics | 1    | 10    |
| 2   | Computers   | 2    | 5     |
| 3   | Laptops     | 3    | 4     |
| 4   | Desktops    | 5    | 6     |
| 5   | Televisions | 7    | 8     |

### Benefits

- **Efficient Retrieval**: Allows complex hierarchical queries to be executed quickly, without recursive operations.
- **Self-Contained**: Since each node knows its left and right boundary, hierarchy depth is managed effortlessly.

### Challenges

- **Complex Updates**: Inserting, deleting, or moving nodes require complex operations to adjust left and right values.
- **Overhead**: Initial planning and setup are more involved compared to flat models.

### Implementation Example

Here's an example of how you can retrieve all subcategories of "Electronics":

```sql
SELECT child.*
FROM categories AS parent, categories AS child
WHERE child.left BETWEEN parent.left AND parent.right
  AND parent.name = 'Electronics';
```

This query will list all nodes under "Electronics" using the Left and Right boundaries.

### Related Patterns

- **Adjacency List Model**: Simpler model using a parent reference; easier updates but complex sub-tree queries.
- **Closure Table**: Uses an explicit table to represent all ancestor/descendant relationships, allowing efficient traversal.

### Additional Resources

- [Hierarchical Data in SQL](https://www.sqlteam.com/articles/hierarchical-data-in-sql)
- [Managing Hierarchical Data in MySQL](https://mikehillyer.com/articles/managing-hierarchical-data-in-mysql/)

### Summary

The Nested Set Model provides a sophisticated method to model and query hierarchical data effectively, making it suitable for applications with frequent entire subtree read operations, such as e-commerce category mappings or organizational charts. Although update operations are more involved, the advantages in retrieval efficiency make it worthwhile in high-read, low-write scenarios.

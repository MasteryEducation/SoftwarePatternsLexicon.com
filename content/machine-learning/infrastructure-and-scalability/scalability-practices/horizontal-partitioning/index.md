---
linkTitle: "Horizontal Partitioning"
title: "Horizontal Partitioning: Splitting Data and Workloads Horizontally Across Different Databases"
description: "Horizontal Partitioning is a design pattern that involves splitting data and workloads horizontally across different databases to improve scalability, performance, and manageability of large datasets in machine learning pipelines."
categories:
- Infrastructure and Scalability
tags:
- Horizontal Partitioning
- Data Sharding
- Scalability
- Distributed Systems
- Big Data
date: 2023-10-29
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/infrastructure-and-scalability/scalability-practices/horizontal-partitioning"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
Horizontal Partitioning, also known as data sharding, refers to the practice of distributing data and workloads across multiple databases. This design pattern addresses challenges related to data scalability, performance bottlenecks, and the management of large datasets commonly encountered in machine learning infrastructures. By partitioning data horizontally, systems can scale out to handle more significant loads efficiently, enabling more reliable and performant machine learning workflows.

## Detailed Description
Horizontal partitioning involves dividing a database table into smaller, more manageable pieces called "shards." Each shard is an independent database segment that stores a subset of the total data. The process of retrieval and storage is then distributed across these shards, reducing the load on any single database instance.

### Implementation Strategies
The implementation of horizontal partitioning includes:

1. **Range-Based Sharding**:
   - Data is divided into shards based on ranges of a particular key, such as user ID or timestamp. Each shard contains a contiguous block of the data range.
   - Pros: Simple and easy to implement.
   - Cons: Can cause uneven data distribution if the data is not uniformly distributed.

2. **Hash-Based Sharding**:
   - Data is divided based on a hash function applied to a sharding key (e.g., hashed user ID). Each shard receives data as determined by the hash function.
   - Pros: Provides a more even distribution of data.
   - Cons: More complex and can be difficult to rebalance.

3. **Directory-Based Sharding**:
   - Utilizes a lookup table (directory) to route each piece of data to its designated shard.
   - Pros: Flexible and allows dynamic shard reallocation.
   - Cons: Directory management overhead.

### Example

Here is an example in Python using SQLAlchemy to demonstrate range-based sharding:

```python
from sqlalchemy import create_engine, Column, Integer, String, Base
from sqlalchemy.orm import sessionmaker

shard_1 = create_engine('sqlite:///shard_1.db')
shard_2 = create_engine('sqlite:///shard_2.db')

SessionShard1 = sessionmaker(bind=shard_1)
SessionShard2 = sessionmaker(bind=shard_2)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

Base.metadata.create_all(shard_1)
Base.metadata.create_all(shard_2)

def insert_user(user_id, name, email):
    if user_id < 1000:
        session = SessionShard1()
    else:
        session = SessionShard2()
    
    new_user = User(id=user_id, name=name, email=email)
    session.add(new_user)
    session.commit()
    session.close()

insert_user(500, 'Alice', 'alice@example.com')
insert_user(1500, 'Bob', 'bob@example.com')
```

## Related Design Patterns
- **Vertical Partitioning**: Involves splitting tables into columns which are stored in separate databases. It complements horizontal partitioning by categorizing data vertically and distributing workloads accordingly.
- **Data Replication**: Copied datasets are stored across multiple databases to enhance data availability and fault tolerance.
- **Load Balancing**: Used alongside horizontal partitioning to distribute incoming data and computational workloads evenly across multiple shards or machines.

## Additional Resources
- [Google Cloud - Horizontal Partitioning Documentation](https://cloud.google.com/databases/docs/sharding)
- [AWS DynamoDB Sharding Strategy](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Partitions.html)
- [Database Sharding by Martin Kleppmann](https://martin.kleppmann.com/)

## Summary
Horizontal Partitioning is a crucial design pattern that plays a significant role in building scalable, performant, and manageable machine learning systems. By distributing data and workloads across multiple databases, machine learning infrastructures can handle increased loads and complex queries more efficiently. This pattern, when combined with others such as vertical partitioning, data replication, and load balancing, forms a robust foundation for large-scale machine learning applications.

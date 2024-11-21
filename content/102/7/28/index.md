---
linkTitle: "Cloud-Native Data Services"
title: "Cloud-Native Data Services"
category: "7. Polyglot Persistence Patterns"
series: "Data Modeling Design Patterns"
description: "Leveraging cloud-managed database services to handle different types of data workloads, facilitating scalability, availability, and ease of management by utilizing diverse database types offered by cloud providers."
categories:
- Cloud Computing
- Database
- Data Services
tags:
- Cloud-Native
- Polyglot Persistence
- Data Modeling
- AWS
- Azure
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/7/28"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


In the rapidly evolving cloud computing landscape, effective data management across diverse types of workloads is paramount for modern applications. Cloud-Native Data Services embrace this by employing a polyglot persistence strategy, entrusting cloud-managed services to handle varied data storage needs.

## Overview

The Cloud-Native Data Services pattern emphasizes using cloud platforms' off-the-shelf database services to tackle the storage and processing of diverse data forms efficiently. By capitalizing on these services, organizations can dynamically allocate resources, ensure high availability, and reduce the intricacies involved in maintaining and scaling data infrastructures.

## Architectural Approach

### Key Principles

1. **Elastic Scalability**: Extend capacity effortlessly to accommodate fluctuating workloads without downtime or over-provisioning concerns.
2. **Managed Service Model**: Offload database administrative tasks to cloud providers, optimizing operational costs and focusing more on strategic innovations.
3. **Data Diversity**: Leverage multiple database technologies tailored to specific data types and usage patterns, promoting flexible data architecture.

### Implementation Strategy

#### Example Configuration

**Platform: Amazon Web Services (AWS)**

- **Relational Data**: Utilize **AWS RDS (Relational Database Service)** for structured schema-based data, ensuring ACID compliance.
- **Key-Value Store**: Deploy **DynamoDB** for fast, scalable key-value data management, crucial for high-throughput applications.
- **Graph Data**: Leverage **Amazon Neptune** for applications that require contextual data connections, e.g., social networks.

**Platform: Microsoft Azure**

- **Relational Data**: Use **Azure SQL Database** for a fully managed relational database service.
- **NoSQL**: Implement **Azure Cosmos DB** for high-performance, multi-model NoSQL capabilities.
- **Big Data Analytics**: Utilize **Azure Synapse Analytics** for integrated analytics over big data and data warehousing.

### Example Code


#### DynamoDB Table Creation (AWS CLI)

```bash
aws dynamodb create-table \
    --table-name Movies \
    --attribute-definitions AttributeName=Year,AttributeType=N AttributeName=Title,AttributeType=S \
    --key-schema AttributeName=Year,KeyType=HASH AttributeName=Title,KeyType=RANGE \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

#### AWS RDS Instance Launch (Boto3 - Python)

```python
import boto3

rds_client = boto3.client('rds')

response = rds_client.create_db_instance(
    DBInstanceIdentifier='RDSInstance',
    MasterUsername='admin',
    MasterUserPassword='securepassword',
    Engine='mysql',
    DBInstanceClass='db.m4.large',
    AllocatedStorage=20
)
```

## Related Patterns

- **Polyglot Persistence**: Optimizing different database technologies to manage specific data requirements effectively.
- **Event Sourcing**: Employing an event-driven model and utilizing diverse data stores for different aspects of event processing.
- **CQRS (Command Query Responsibility Segregation)**: Decoupling read and write operations using tailored databases for command and query functionality.

## Best Practices

- Ensure seamless integration between data services to avoid data silos.
- Implement robust security measures to protect data custody across services.
- Optimize cost by ensuring resources are adequately aligned with application demands.
- Use monitoring tools provided by your cloud vendor for insights and performance tuning.

## Additional Resources

- [AWS Data Services Overview](https://aws.amazon.com/products/databases/)
- [Azure Database Services](https://azure.microsoft.com/en-us/products/databases/)
- [Google Cloud Databases](https://cloud.google.com/products/databases)

## Summary

By leveraging cloud-native data services, organizations gain the flexibility and power of cloud data management frameworks with minimal operational overheads. Whether dealing with relational databases, key-value pairs, or graph-style data stores, leveraging the right service for the job ensures both performance and efficiency.

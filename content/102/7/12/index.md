---
linkTitle: Database per Service
title: Database per Service
category: 7. Polyglot Persistence Patterns
series: Data Modeling Design Patterns
description: "Each service or microservice has its own database, promoting autonomy and loose coupling."
categories:
- Data Management
- Microservices
- Architectural Patterns
tags:
- microservices
- database
- polyglot persistence
- loose coupling
- data management
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/7/12"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Database per Service

### Introduction

The Database per Service pattern is an architectural approach utilized within microservices architectures to promote autonomy and loose coupling by assigning an independent database to each service. This design pattern is pivotal in supporting diverse data storage requirements and fostering polyglot persistence, whereby different database technologies can be employed according to each service's specific needs.

### Benefits

1. **Autonomy and Independence**: Each microservice can be developed, deployed, and scaled independently since it owns its database.
2. **Encapsulation**: Data is encapsulated within a service boundary, adhering strictly to the microservices paradigm.
3. **Optimized Storage**: Each service can use the most appropriate database type (SQL, NoSQL, in-memory, etc.) for its use case.
4. **Performance Improvements**: Services can choose specialized databases that enhance performance for specific operations.

### Design Considerations

- **Data Consistency**: Implement eventual consistency models for cross-service operations as data is distributed.
- **Transactional Boundaries**: Microservices do not share transactional boundaries, hence careful design of compensation strategies for transaction failure.
- **Synchronous vs. Asynchronous Communication**: Use messaging patterns for inter-service communication rather than synchronous HTTP calls to decouple services.
- **Data Replication and Redundancy**: Account for data duplication across services and establish strategies to manage consistency.

### Example Implementation

Consider an e-commerce platform where multiple microservices perform distinct functions: 

```plaintext
Authentication Service: Utilizes Redis for session storage to enable fast data access and quick user authentication workflows.

Product Catalog Service: Uses Elasticsearch for full-text search capabilities, enabling fast and comprehensive searches over large volumes of product data.

Order Service: Implements PostgreSQL for relational data persistence to efficiently handle transactional workloads and complex queries related to order processing.

Customer Feedback Service: Employs MongoDB, suitable for storing unstructured or semi-structured customer review data, allowing flexibility in schema design.
```

### Example Code Snippet

Here's a hypothetical `Docker Compose` configuration deploying each service with its own database technology:

```yaml
version: '3.8'
services:
  
  authentication-service:
    image: auth-service:latest
    ports:
      - "8081:8080"
    environment:
      REDIS_HOST: redis
    depends_on:
      - redis
  
  redis:
    image: redis:6.2

  catalog-service:
    image: catalog-service:latest
    ports:
      - "8082:8080"
    environment:
      ES_HOST: es
    depends_on:
      - es
  
  es:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.10.0
    environment:
      discovery.type: single-node

  order-service:
    image: order-service:latest
    ports:
      - "8083:8080"
    environment:
      POSTGRES_URL: postgres://postgres:password@postgres:5432/orders
    depends_on:
      - postgres
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: password
  
  feedback-service:
    image: feedback-service:latest
    ports:
      - "8084:8080"
    environment:
      MONGODB_HOST: mongo
    depends_on:
      - mongo

  mongo:
    image: mongo:4.4
```

### Related Patterns

- **Event Sourcing**: Captures all changes as a sequence of events, which can be useful when modeling systems with complex behaviors.
- **CQRS (Command Query Responsibility Segregation)**: Segregates operations that modify data from those that do not, complementing the Database per Service pattern.
- **Saga Pattern**: Handles distributed transactions across microservices through a series of compensating transactions.

### Additional Resources

- [Microservices Architecture: Align Development, Deployment, and Messaging](#)
- [Building Microservices with ASP.NET Core](#)
- [Designing Data-Intensive Applications](#)

### Conclusion

The Database per Service pattern is essential for realizing the full potential of microservices architectures. By allowing each service to own its database, teams can achieve a higher degree of autonomy, facilitating independent development and deployment cycles while addressing specific application needs with the best-suited database technologies. While offering significant advantages, it requires careful consideration of data consistency and transactional boundaries to effectively implement cross-service interactions.

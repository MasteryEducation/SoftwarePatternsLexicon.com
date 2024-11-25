---
linkTitle: "Multi-Tenancy"
title: "Multi-Tenancy: Supporting Multiple Clients or Customers with a Single Instance"
description: "A detailed exploration of the Multi-Tenancy design pattern in machine learning, with examples, related design patterns, additional resources, and a final summary."
categories:
- Deployment Patterns
tags:
- Multi-Tenancy
- Machine Learning
- Deployment
- Scalability
- Isolation
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/deployment-patterns/delivery-methods/multi-tenancy"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In the realm of machine learning deployment, supporting multiple clients or customers efficiently is a key concern. This requirement becomes more complex when we consider the need for isolation, performance, and customization for different tenants. **Multi-Tenancy** is a design pattern that addresses these challenges by enabling multiple tenants (clients/customers) to share a single instance of a machine learning application, while maintaining logical separation and ensuring data privacy and security.

## Detailed Description

Multi-Tenancy is a deployment pattern that allows one instance of an application to serve multiple tenants. Each tenant's data is isolated, and the system is designed to handle specific customizations and scaling requirements for each tenant.

Key attributes:

1. **Isolation**: Data and customization for each tenant are kept separate to ensure privacy and security.
2. **Resource Efficiency**: A single instance handles multiple tenants, which means lower resource usage compared to managing separate instances for each tenant.
3. **Scalability**: The architecture must be able to handle increasing loads from multiple tenants efficiently.
4. **Customization**: Support specific configurations and customizations for different tenants.

## Examples

### Example 1: Multi-Tenancy with Python and Flask

Using Flask, a lightweight WSGI web application framework in Python, we can create a basic multi-tenant application where different tenants access their specific data based on subdomains.

```python
from flask import Flask, request

app = Flask(__name__)

tenants = {
    'tenant1': {'data': 'Data for Tenant 1'},
    'tenant2': {'data': 'Data for Tenant 2'},
}

@app.route('/')
def hello_tenant():
    tenant = request.host.split('.')[0]  # Get subdomain
    if tenant in tenants:
        return f"Hello, {tenant}. Your data is: {tenants[tenant]['data']}"
    else:
        return "Hello, unknown tenant"

if __name__ == '__main__':
    app.run()
```

In this example, the application checks the subdomain to determine the tenant and serves data accordingly.

### Example 2: Multi-Tenancy with Java and Spring Boot

Using Spring Boot, we can create a multi-tenant application where each tenant has a separate schema in a single database.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.datasource.lookup.AbstractRoutingDataSource;
import org.springframework.jdbc.datasource.lookup.MapDataSourceLookup;
import org.springframework.stereotype.Component;

@Component
public class TenantRoutingDataSource extends AbstractRoutingDataSource {

    @Autowired
    private TenantIdentifierResolver tenantIdentifierResolver;

    @Override
    protected Object determineCurrentLookupKey() {
        return tenantIdentifierResolver.resolveCurrentTenantIdentifier();
    }
}

// Configuration class for setting up multi-tenancy
@Configuration
public class DataSourceConfig {

    @Bean
    public DataSource dataSource() {
        DataSourceLookup dataSourceLookup = new MapDataSourceLookup();
        AbstractRoutingDataSource dataSource = new TenantRoutingDataSource();
        dataSource.setDefaultTargetDataSource(dataSourceLookup.getDataSource("default"));
        dataSource.setTargetDataSources(dataSourceLookup.getAllDataSources());
        return dataSource;
    }
}

// Cont4roller to handle requests
@RestController
public class TenantController {

    @GetMapping("/{tenant}/data")
    public String getTenantData(@PathVariable String tenant) {
        return tenant + " data";
    }
}
```

In this Spring Boot example, `TenantRoutingDataSource` dynamically switches between different data sources based on the tenant identifier resolved by `TenantIdentifierResolver`.

## Related Design Patterns

### 1. **Service Abstraction Layer**

This design pattern abstracts the complexities of different service implementations, providing a unified interface to the tenants. It helps in managing multiple tenants by abstracting tenant-specific logic into separate layers.

### 2. **Gateway Aggregation**

This pattern involves aggregating data from multiple services into a single gateway request. It can be particularly useful in a multi-tenant architecture where data from various tenants must be aggregated and managed efficiently.

### 3. **Façade**

Façade design pattern is used to provide a simplified interface to complex subsystems. In a multi-tenant system, a façade can offer a simplified interface to tenant-specific services, reducing the complexity of managing multiple configurations.

## Additional Resources

1. **Books**:
   - "Designing Data-Intensive Applications" by Martin Kleppmann: This book provides insights into designing efficient data architectures, including multi-tenant databases.
   - "Patterns of Enterprise Application Architecture" by Martin Fowler: Covers numerous design patterns, including those relevant for multi-tenancy.

2. **Articles**:
   - ["Multi-Tenancy Models in SaaS Applications"](https://www.infoq.com/articles/saas-multi-tenancy/)
   - ["Scaling Machine Learning Models with Kubernetes and Multi-Tenancy Patterns"](https://towardsdatascience.com/)

3. **Tutorials**:
   - [Spring Boot Multi-Tenancy Example](https://www.baeldung.com/spring-boot-multi-tenancy)
   - [Building Multi-Tenant Applications with Flask](https://flask.palletsprojects.com/en/2.x/patterns/appfactories/)

## Final Summary

Multi-Tenancy is a pivotal design pattern for deploying scalable and efficient machine learning applications serving multiple clients. By sharing a single instance among different tenants, businesses can optimize resources, maintain data isolation, and ensure tenant-specific customizations and scaling. Understanding and implementing multi-tenancy can significantly enhance the efficiency and scalability of your machine learning deployments, making it a vital pattern in modern software architecture.

The examples provided in Python and Java illustrate basic implementations, and the related design patterns like Service Abstraction Layer and Façade further enhance the robustness of multi-tenant systems. Learn more about multi-tenancy through the recommended books, articles, and tutorials to deepen your understanding and application of this crucial pattern.

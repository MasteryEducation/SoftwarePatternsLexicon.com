---
canonical: "https://softwarepatternslexicon.com/patterns-java/30/2"
title: "Strangler Fig Pattern: Incremental Legacy System Migration in Java"
description: "Explore the Strangler Fig pattern for incrementally replacing legacy systems with modern implementations in Java, minimizing disruption and risk."
linkTitle: "30.2 Strangler Fig Pattern"
tags:
- "Java"
- "Design Patterns"
- "Strangler Fig"
- "Legacy Systems"
- "Migration"
- "Software Architecture"
- "Continuous Delivery"
- "Risk Mitigation"
date: 2024-11-25
type: docs
nav_weight: 302000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 30.2 Strangler Fig Pattern

### Introduction

The Strangler Fig pattern is a powerful architectural strategy for modernizing legacy systems. Named after the strangler fig tree, which grows around an existing tree and eventually replaces it, this pattern involves incrementally building a new system around the old one. This approach allows developers to replace outdated components with minimal disruption and risk, facilitating a smoother transition to modern technologies.

### Origin and Analogy

The Strangler Fig pattern draws its name from the natural phenomenon of the strangler fig tree. In nature, the strangler fig starts as a seed deposited on a host tree. Over time, it grows around the host, eventually replacing it entirely. Similarly, in software development, the Strangler Fig pattern involves gradually replacing parts of a legacy system with new implementations, eventually phasing out the old system entirely.

### How the Pattern Works

The Strangler Fig pattern involves several key steps:

1. **Identifying Components to Replace**: Begin by identifying the components of the legacy system that need replacement. Focus on areas that are most critical or problematic.

2. **Routing Traffic Through a Facade or Proxy**: Implement a facade or proxy to route traffic between the old and new systems. This allows you to control the flow of data and functionality, ensuring a seamless transition.

3. **Rewriting Components Incrementally**: Gradually rewrite components of the legacy system, replacing them with new implementations. This can be done in stages, allowing for continuous integration and testing.

4. **Decommissioning Old Components**: Once a component has been successfully replaced and tested, decommission the old component. This reduces maintenance overhead and ensures that the system remains up-to-date.

### Steps for Implementing the Strangler Fig Pattern

#### 1. Identifying Components to Replace

The first step in implementing the Strangler Fig pattern is to identify which components of the legacy system need to be replaced. This involves analyzing the system to determine which parts are outdated, inefficient, or difficult to maintain. Focus on components that are critical to the system's functionality or that pose significant risks if they fail.

#### 2. Routing Traffic Through a Facade or Proxy

Once you have identified the components to replace, the next step is to route traffic through a facade or proxy. This allows you to control the flow of data and functionality between the old and new systems. By using a facade or proxy, you can ensure that users and other systems interact with the new components without being affected by the ongoing migration.

```java
// Example of a Proxy in Java
public interface LegacyService {
    void performOperation();
}

public class LegacyServiceImpl implements LegacyService {
    public void performOperation() {
        System.out.println("Performing operation in legacy system.");
    }
}

public class NewService {
    public void performOperation() {
        System.out.println("Performing operation in new system.");
    }
}

public class ServiceProxy implements LegacyService {
    private LegacyService legacyService;
    private NewService newService;
    private boolean useNewService;

    public ServiceProxy(LegacyService legacyService, NewService newService) {
        this.legacyService = legacyService;
        this.newService = newService;
        this.useNewService = false; // Start with legacy
    }

    public void switchToNewService() {
        this.useNewService = true;
    }

    @Override
    public void performOperation() {
        if (useNewService) {
            newService.performOperation();
        } else {
            legacyService.performOperation();
        }
    }
}
```

#### 3. Rewriting Components Incrementally

With the facade or proxy in place, you can begin rewriting components incrementally. This involves developing new implementations for the identified components and integrating them into the system. By doing this incrementally, you can test each component thoroughly before moving on to the next, reducing the risk of introducing errors or instability.

#### 4. Decommissioning Old Components

After successfully replacing a component and ensuring it functions correctly, you can decommission the old component. This involves removing the old code and any dependencies it may have had. Decommissioning reduces maintenance overhead and ensures that the system remains efficient and up-to-date.

### Benefits of the Strangler Fig Pattern

The Strangler Fig pattern offers several benefits, particularly in terms of risk mitigation and continuous delivery:

- **Risk Mitigation**: By replacing components incrementally, the Strangler Fig pattern minimizes the risk of system failure. Each component is thoroughly tested before being integrated, ensuring that the system remains stable throughout the migration process.

- **Continuous Delivery**: The pattern supports continuous delivery by allowing new components to be integrated and deployed as they are developed. This enables organizations to deliver new features and improvements to users more quickly.

- **Reduced Disruption**: The gradual nature of the Strangler Fig pattern means that users and other systems experience minimal disruption during the migration process. This is particularly important for mission-critical systems where downtime is not an option.

### Challenges of the Strangler Fig Pattern

While the Strangler Fig pattern offers many benefits, it also presents several challenges:

- **Data Synchronization**: Ensuring that data remains consistent between the old and new systems can be challenging, particularly if the systems have different data models or storage mechanisms.

- **Integration of New and Old Components**: Integrating new components with the existing system can be complex, particularly if the legacy system is poorly documented or lacks modularity.

- **Technical Debt**: While the pattern helps reduce technical debt over time, the initial stages of migration may temporarily increase complexity as both old and new components coexist.

### Example of Using the Strangler Fig Pattern in Java Applications

Consider a legacy Java application that handles customer orders. The application is outdated and difficult to maintain, so the decision is made to migrate to a modern architecture using the Strangler Fig pattern.

1. **Identify Components**: The order processing module is identified as a critical component that needs replacement.

2. **Implement a Proxy**: A proxy is implemented to route order processing requests to either the old or new system, depending on the status of the migration.

3. **Rewrite Components**: The order processing module is rewritten using modern Java features such as Streams and Lambdas, improving performance and maintainability.

4. **Decommission Old Components**: Once the new order processing module is fully tested and integrated, the old module is decommissioned.

### Conclusion

The Strangler Fig pattern is a powerful tool for modernizing legacy systems in Java. By allowing developers to replace components incrementally, it minimizes risk and disruption while supporting continuous delivery. However, it also presents challenges, particularly in terms of data synchronization and integration. By carefully planning and executing the migration process, organizations can successfully transition to modern architectures while maintaining system stability and performance.

### Related Patterns

- **[Adapter Pattern]({{< ref "/patterns-java/6/6" >}} "Adapter Pattern")**: Useful for integrating new components with legacy systems.
- **[Facade Pattern]({{< ref "/patterns-java/6/6" >}} "Facade Pattern")**: Can be used to simplify interactions between old and new components.

### Known Uses

- **Netflix**: Utilized the Strangler Fig pattern to transition from a monolithic architecture to a microservices-based architecture.
- **Amazon**: Employed the pattern to incrementally replace legacy systems with modern, scalable solutions.

## Strangler Fig Pattern Quiz: Test Your Knowledge

{{< quizdown >}}

### What is the primary benefit of the Strangler Fig pattern?

- [x] Incremental replacement of legacy systems
- [ ] Immediate replacement of entire systems
- [ ] Simplification of legacy code
- [ ] Reduction of system functionality

> **Explanation:** The Strangler Fig pattern allows for incremental replacement, minimizing risk and disruption.

### Which step involves routing traffic through a facade or proxy?

- [x] Step 2
- [ ] Step 1
- [ ] Step 3
- [ ] Step 4

> **Explanation:** Step 2 involves routing traffic through a facade or proxy to control data flow.

### What is a challenge of the Strangler Fig pattern?

- [x] Data synchronization
- [ ] Immediate system downtime
- [ ] Lack of modularity
- [ ] Reduced system performance

> **Explanation:** Data synchronization between old and new systems can be challenging.

### How does the Strangler Fig pattern support continuous delivery?

- [x] By allowing new components to be integrated and deployed incrementally
- [ ] By requiring complete system rewrites
- [ ] By reducing the need for testing
- [ ] By simplifying deployment processes

> **Explanation:** The pattern supports continuous delivery through incremental integration and deployment.

### What analogy is the Strangler Fig pattern based on?

- [x] A tree growing around and replacing another tree
- [ ] A vine climbing a wall
- [ ] A plant spreading across a field
- [ ] A root system expanding underground

> **Explanation:** The pattern is based on the strangler fig tree growing around and replacing its host.

### What is the first step in implementing the Strangler Fig pattern?

- [x] Identifying components to replace
- [ ] Decommissioning old components
- [ ] Rewriting components incrementally
- [ ] Routing traffic through a proxy

> **Explanation:** The first step is identifying which components need replacement.

### Which Java feature can be used in rewriting components?

- [x] Streams and Lambdas
- [ ] Java Applets
- [ ] JavaBeans
- [ ] JavaFX

> **Explanation:** Streams and Lambdas are modern Java features that can improve performance and maintainability.

### What is a benefit of the Strangler Fig pattern?

- [x] Risk mitigation
- [ ] Increased system complexity
- [ ] Immediate system replacement
- [ ] Reduced testing requirements

> **Explanation:** The pattern mitigates risk by allowing for incremental replacement and testing.

### What is a known use of the Strangler Fig pattern?

- [x] Netflix's transition to microservices
- [ ] Google's development of Android
- [ ] Apple's design of iOS
- [ ] Microsoft's creation of Windows

> **Explanation:** Netflix used the pattern to transition from a monolithic to a microservices architecture.

### True or False: The Strangler Fig pattern requires complete system downtime during migration.

- [x] False
- [ ] True

> **Explanation:** The pattern minimizes disruption by allowing for incremental migration without complete system downtime.

{{< /quizdown >}}

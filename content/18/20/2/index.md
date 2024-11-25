---
linkTitle: "Vertical Scaling"
title: "Vertical Scaling: Increasing the Capacity of Existing Instances"
category: "Scalability and Elasticity in Cloud"
series: "Cloud Computing: Essential Patterns & Practices"
description: "Vertical Scaling refers to the process of increasing the capacity of existing server instances to handle more demand. This pattern is part of the scalability strategies in cloud computing, focusing on enhancing resources within a single server instance."
categories:
- Scalability
- Elasticity
- Cloud Computing
tags:
- vertical-scaling
- scalability
- cloud-computing
- elasticity
- server-capacity
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/18/20/2"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Vertical Scaling, also known as "scaling up," involves enhancing the resources of an existing server to meet increased demand. This can include adding more CPU power, RAM, or storage to a single instance. Vertical scaling is a straightforward approach to scaling because it simply adds more power to existing systems without changing the software architecture significantly.

## Detailed Explanation

### Architectural Approaches

Vertical scaling usually requires minimal changes to applications and databases since they continue to run within the same operational environment, but with more resources. Here, we discuss key factors considered when implementing vertical scaling:

1. **Resource Augmentation**: Increase the RAM, CPU cores, storage capacity, or network bandwidth of a server to improve performance and handle more load.

2. **System Compatibility**: Ensure that the underlying operating system and application stack can utilize the additional resources efficiently and support the required scalability.

3. **Minimal Infrastructure Change**: Unlike horizontal scaling, vertical scaling does not necessitate a change in infrastructure architecture, making it suitable for non-distributed workloads.

### Best Practices

- **Monitoring and Analysis**: Continuously monitor performance metrics to identify bottlenecks that vertical scaling can alleviate.
  
- **Cloud Service Provider Solutions**: Leverage cloud-specific tools and settings (e.g., AWS EC2 Instance Types, Azure VM Sizes) for easy vertical scaling.

- **Automated Scaling**: Use managed services like AWS Auto Scaling to adjust resources dynamically based on pre-set performance metrics.

## Example Code

Consider this Java pseudo-code for simulating vertical scaling by dynamically adjusting resource usage within an application.

```java
public class VerticalScaler {
    private ServerInstance server;

    public VerticalScaler(ServerInstance server) {
        this.server = server;
    }

    public void scaleUp(Capacity capacity) {
        server.adjustResources(capacity.getCpu(), capacity.getRam(), capacity.getStorage());
        System.out.println("Scaled up server to: " + capacity);
    }

    public static void main(String[] args) {
        ServerInstance instance = new ServerInstance("large-instance");
        VerticalScaler verticalScaler = new VerticalScaler(instance);

        Capacity increasedCapacity = new Capacity(16, 64, 200); // CPU cores, RAM GB, Storage GB
        verticalScaler.scaleUp(increasedCapacity);
    }
}
```

## Related Patterns

- **Horizontal Scaling**: Adding more instances instead of boosting the power of a single instance. These scale out strategies are often combined with vertical scaling for maximum availability and flexibility.
  
- **Load Balancing**: Often used alongside both vertical and horizontal scaling to distribute traffic efficiently across multiple server instances.

- **Elastic Resource Allocation**: Ensuring application resources can be allocated and deallocated based on real-time demand.

## Additional Resources

- *AWS Auto Scaling Documentation*: Discover ways to automatically adjust your AWS resources to maintain workload performance.
- *Azure Virtual Machine Scale Sets*: Documentation on Azure VMs and vertical scaling across different instances.
- *Google Cloud Compute Engine*: Best practices for scaling Google Cloud resources vertically.

## Summary

Vertical Scaling is an efficient method to increase the capacity of an existing server to better handle its workload. While simple to implement and manage, it generally remains more suitable for tasks that do not require distributed resources. Balancing with horizontal scaling and load balancing allows businesses to create robust, scalable, and reliable applications in the cloud.

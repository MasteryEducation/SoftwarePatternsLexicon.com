---
linkTitle: "Health Checks and Heartbeats"
title: "Health Checks and Heartbeats: Ensuring System Availability"
category: "Monitoring, Observability, and Logging in Cloud"
series: "Cloud Computing: Essential Patterns & Practices"
description: "A comprehensive overview of Health Checks and Heartbeats patterns for maintaining system availability and reliability in cloud-based architectures."
categories:
- Cloud
- Monitoring
- Reliability
tags:
- health-check
- heartbeat
- monitoring
- system-availability
- cloud-architecture
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/18/10/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In cloud-based architectures, ensuring that services are available and performing optimally is crucial. Two fundamental patterns used to achieve this are **Health Checks** and **Heartbeats**. These patterns allow for continuous monitoring and quick detection of issues, ensuring that systems remain reliable and available.

## Pattern Overview

### Health Checks

Health Checks are mechanisms used to evaluate the health and functionality of a service or component within a system. They provide a way for administrators and automated systems to determine if a service is operating correctly. Health checks are typically performed at regular intervals and can include:

- **Liveness Checks:** Determine if an application is running by returning a simple status.
- **Readiness Checks:** Verify that a service is ready to handle requests, often by checking dependencies like databases or external services.

#### Implementation Example

In a Spring Boot application, health checks can be implemented using Actuator:

```java
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;

@Component
public class CustomHealthCheck implements HealthIndicator {

    @Override
    public Health health() {
        // Replace with actual check logic
        boolean healthy = checkServiceHealth();
        if (healthy) {
            return Health.up().build();
        } else {
            return Health.down().withDetail("Error Code", "Service unavailable").build();
        }
    }

    private boolean checkServiceHealth() {
        // Implement your health check logic here
        return true;
    }
}
```

### Heartbeats

Heartbeats are used to periodically signal that a service is alive and functioning. They act as a "pulse" from a service to indicate its liveness. Heartbeats are typically used in distributed systems to detect failed nodes or services and can trigger failover or recovery processes.

#### Implementation Example

A simple example using a scheduled heartbeat in Kotlin:

```kotlin
import java.util.Timer
import kotlin.concurrent.schedule

class HeartbeatService {

    fun start() {
        Timer().schedule(0, 10000) { sendHeartbeat() }
    }

    private fun sendHeartbeat() {
        println("Heartbeat: Service is alive")
        // Add code to send heartbeat to monitoring system
    }
}

fun main() {
    val service = HeartbeatService()
    service.start()
}
```

## Architectural Considerations

1. **Frequency and Latency:** Balance between frequent checks for timely detection of issues and minimizing unnecessary load on the system.
2. **Scalability:** Ensure health checks and heartbeats scale as the system grows.
3. **Security:** Protect health endpoints to prevent abuse or information leakage.
4. **Integration:** Integrate with logging and alerting systems to notify administrators of any issues.

### Related Patterns

- **Circuit Breaker:** Temporarily stops requests from being sent to a service that consistently fails its health checks.
- **Retry Pattern:** Attempts to resend failed requests, often used with health checks to determine when to resume normal operations.

## Additional Resources

- [Spring Boot Actuator Documentation](https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html)
- [Kubernetes Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [AWS CloudWatch](https://aws.amazon.com/cloudwatch/) for monitoring cloud resources and services.

## Summary

Health Checks and Heartbeats are critical patterns for maintaining the availability and reliability of cloud-based systems. By regularly assessing the health and liveness of services, administrators can proactively manage potential issues, leading to more resilient and robust applications. Integrating these patterns within a comprehensive monitoring and alerting framework ensures that your cloud infrastructure operates smoothly and efficiently.

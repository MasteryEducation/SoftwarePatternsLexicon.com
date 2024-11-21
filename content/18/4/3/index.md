---
linkTitle: "Security Groups and Network ACLs"
title: "Security Groups and Network ACLs: Controlling Inbound and Outbound Traffic to Resources"
category: "Networking Services in Cloud"
series: "Cloud Computing: Essential Patterns & Practices"
description: "Explore the use of Security Groups and Network ACLs in cloud computing to effectively manage inbound and outbound traffic to your cloud resources. Understand how these tools function, their differences, and best practices for implementation."
categories:
- Networking
- Security
- Cloud Architecture
tags:
- Security Groups
- Network ACLs
- Cloud Security
- Traffic Control
- Firewall
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/18/4/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

In cloud computing environments, managing and securing network traffic is crucial for maintaining robust application and data protection. Security Groups and Network Access Control Lists (ACLs) serve as foundational elements in controlling inbound and outbound traffic to cloud resources.

## Security Groups

### Introduction
Security Groups function as virtual firewalls for your cloud resources. They are used to control inbound and outbound traffic at the instance level. A security group acts as an implicit deny-all on both ingress and egress traffic unless explicitly allowed by security group rules.

### Key Features
- **Instance-Level Security:** Security Groups operate at the instance rather than subnet level, allowing for more granular control over individual resource access.
- **Stateful Traffic Management:** Any incoming traffic that matches an allow rule is paired with an automatic allowance for outbound traffic to the same destination, unless otherwise specified.
- **Flexible Rule Configurations:** Users can define varying rules based on IP prefixes, protocols, and port numbers to meet specific security requirements.

### Architectural Approach
Security Groups are typically associated with instances, and modifications to rules are applied in real-time, simplifying management. This dynamic adaptability ensures a seamless integration while adapting to evolving security needs.

### Example Code
```shell
aws ec2 create-security-group --group-name MySecurityGroup --description "My security group for instances"

aws ec2 authorize-security-group-ingress --group-name MySecurityGroup --protocol tcp --port 22 --cidr 0.0.0.0/0
```

## Network ACLs

### Introduction
Network ACLs are an optional layer of security that acts as a firewall for controlling traffic in and out of one or more subnets. Unlike Security Groups, Network ACLs are stateless.

### Key Features
- **Subnet-Level Control:** Network ACLs are associated with one or more subnets and evaluate rule traffic against incoming and outgoing operations.
- **Stateless Traffic Management:** Each inbound and outbound rule is evaluated independently, offering strict control for each direction of traffic.
- **Ordered Rule Evaluation:** Rules in Network ACLs are evaluated in number order, and the first matching rule determines the action.

### Architectural Approach
Network ACLs provide a broader control mechanism at the subnet level. They are best employed in scenarios requiring a more extensive traffic filtering mechanism.

### Example Code
```shell
aws ec2 create-network-acl --vpc-id vpc-123456

aws ec2 create-network-acl-entry --network-acl-id acl-123456 --ingress --rule-number 100 --protocol tcp --port-range From=80,To=80 --cidr-block 0.0.0.0/0 --rule-action allow
```

## Best Practices
- **Layered Security:** Utilize both Security Groups and Network ACLs in tandem for a multi-layered defense strategy.
- **Rule Management:** Regularly review and adjust rules to ensure they comply with compliance standards and evolving security needs.
- **Least Privilege Principle:** Adhere to the principle of least privilege by allowing only necessary traffic through security configurations.

## Related Patterns
- **Zero Trust Security Model:** Implement a zero-trust framework to minimize insider and outsider threats.
- **Network Segmentation:** Use network segmentation to isolate and protect critical infrastructure elements.

## Additional Resources
- [AWS Security Groups Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html)
- [AWS Network ACLs Documentation](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-network-acls.html)

## Summary
Security Groups and Network ACLs are integral components of cloud security that offer flexible and robust mechanisms to manage network traffic. By leveraging both paradigms, cloud architects can ensure that resources are adequately protected against unauthorized access and provide a strong defense against potential security threats. Understanding their differences and capabilities is essential for designing secure cloud architectures.

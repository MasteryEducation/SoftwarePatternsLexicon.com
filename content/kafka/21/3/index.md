---
canonical: "https://softwarepatternslexicon.com/kafka/21/3"
title: "Kafka CLI Tools and Commands: Mastering Apache Kafka Command-Line Interface for Experts"
description: "Explore the comprehensive guide to Kafka CLI tools and commands, including syntax examples, descriptions, and practical applications for expert software engineers and enterprise architects."
linkTitle: "Kafka CLI Tools and Commands"
tags:
- "Apache Kafka"
- "Kafka CLI"
- "Kafka Commands"
- "Kafka Administration"
- "Kafka Monitoring"
- "Kafka Testing"
- "Kafka Tools"
- "Kafka Management"
date: 2024-11-25
type: docs
nav_weight: 213000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## C. Kafka CLI Tools and Commands

Apache Kafka, a distributed event streaming platform, provides a robust command-line interface (CLI) that allows expert software engineers and enterprise architects to manage, monitor, and test Kafka clusters efficiently. This section delves into the essential Kafka CLI tools and commands, offering syntax examples, detailed explanations, and practical applications.

### Overview of Kafka CLI Tools

Kafka's CLI tools are categorized into several groups based on their functionality: administration, monitoring, and testing. Each tool serves a specific purpose, from managing topics and partitions to producing and consuming messages for testing purposes.

### Administration Tools

#### 1. `kafka-topics`

The `kafka-topics` tool is used for managing Kafka topics. It allows you to create, delete, list, and describe topics within a Kafka cluster.

- **Creating a Topic**: Use the `--create` option to create a new topic.

  ```bash
  kafka-topics --bootstrap-server localhost:9092 --create --topic my-topic --partitions 3 --replication-factor 2
  ```

  - **Options**:
    - `--bootstrap-server`: Specifies the Kafka broker address.
    - `--create`: Indicates the creation of a new topic.
    - `--topic`: Names the topic.
    - `--partitions`: Sets the number of partitions.
    - `--replication-factor`: Sets the replication factor.

- **Listing Topics**: Use the `--list` option to display all topics.

  ```bash
  kafka-topics --bootstrap-server localhost:9092 --list
  ```

- **Describing a Topic**: Use the `--describe` option to get details about a topic.

  ```bash
  kafka-topics --bootstrap-server localhost:9092 --describe --topic my-topic
  ```

- **Deleting a Topic**: Use the `--delete` option to remove a topic.

  ```bash
  kafka-topics --bootstrap-server localhost:9092 --delete --topic my-topic
  ```

#### 2. `kafka-configs`

The `kafka-configs` tool is used to manage configurations for brokers, topics, and clients.

- **Describing Configurations**: Use the `--describe` option to view configurations.

  ```bash
  kafka-configs --bootstrap-server localhost:9092 --entity-type topics --entity-name my-topic --describe
  ```

- **Altering Configurations**: Use the `--alter` option to change configurations.

  ```bash
  kafka-configs --bootstrap-server localhost:9092 --entity-type topics --entity-name my-topic --alter --add-config retention.ms=604800000
  ```

#### 3. `kafka-acls`

The `kafka-acls` tool manages access control lists (ACLs) for Kafka resources.

- **Adding ACLs**: Use the `--add` option to add an ACL.

  ```bash
  kafka-acls --bootstrap-server localhost:9092 --add --allow-principal User:Alice --operation Read --topic my-topic
  ```

- **Listing ACLs**: Use the `--list` option to view existing ACLs.

  ```bash
  kafka-acls --bootstrap-server localhost:9092 --list --topic my-topic
  ```

### Monitoring Tools

#### 1. `kafka-consumer-groups`

The `kafka-consumer-groups` tool is used to manage and monitor consumer groups.

- **Listing Consumer Groups**: Use the `--list` option to display all consumer groups.

  ```bash
  kafka-consumer-groups --bootstrap-server localhost:9092 --list
  ```

- **Describing Consumer Groups**: Use the `--describe` option to get details about a consumer group.

  ```bash
  kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group my-group
  ```

- **Resetting Offsets**: Use the `--reset-offsets` option to reset offsets for a consumer group.

  ```bash
  kafka-consumer-groups --bootstrap-server localhost:9092 --group my-group --reset-offsets --to-earliest --execute --topic my-topic
  ```

#### 2. `kafka-run-class`

The `kafka-run-class` tool is a utility for running Kafka classes. It is often used to start Kafka services or execute specific classes.

- **Running a Class**: Specify the class name and any necessary arguments.

  ```bash
  kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic my-topic
  ```

### Testing Tools

#### 1. `kafka-console-producer`

The `kafka-console-producer` tool is used to produce messages to a Kafka topic from the command line.

- **Producing Messages**: Use the following command to start producing messages.

  ```bash
  kafka-console-producer --broker-list localhost:9092 --topic my-topic
  ```

  - **Options**:
    - `--broker-list`: Specifies the Kafka broker address.
    - `--topic`: Names the topic to which messages will be sent.

#### 2. `kafka-console-consumer`

The `kafka-console-consumer` tool is used to consume messages from a Kafka topic and display them on the console.

- **Consuming Messages**: Use the following command to start consuming messages.

  ```bash
  kafka-console-consumer --bootstrap-server localhost:9092 --topic my-topic --from-beginning
  ```

  - **Options**:
    - `--bootstrap-server`: Specifies the Kafka broker address.
    - `--topic`: Names the topic from which messages will be consumed.
    - `--from-beginning`: Consumes messages from the beginning of the topic.

#### 3. `kafka-verifiable-producer` and `kafka-verifiable-consumer`

These tools are used for testing and verifying the performance and reliability of Kafka producers and consumers.

- **Verifiable Producer**: Use the following command to start a verifiable producer.

  ```bash
  kafka-verifiable-producer --broker-list localhost:9092 --topic my-topic
  ```

- **Verifiable Consumer**: Use the following command to start a verifiable consumer.

  ```bash
  kafka-verifiable-consumer --bootstrap-server localhost:9092 --topic my-topic --group my-group
  ```

### Advanced Usage and Best Practices

#### 1. Automating Kafka CLI Commands

Automate repetitive tasks using shell scripts or automation tools like Ansible or Terraform. This approach ensures consistency and reduces the risk of human error.

#### 2. Monitoring and Alerting

Integrate Kafka CLI tools with monitoring solutions like Prometheus and Grafana to visualize metrics and set up alerts for critical events.

#### 3. Security Considerations

Ensure secure communication by using SSL/TLS encryption and SASL authentication when executing Kafka CLI commands. Refer to [12.1 Authentication Mechanisms]({{< ref "/kafka/12/1" >}} "Authentication Mechanisms") for more details.

#### 4. Performance Tuning

Optimize Kafka performance by adjusting configurations such as batch size and compression settings. See [10.1 Producer and Consumer Performance Tuning]({{< ref "/kafka/10/1" >}} "Producer and Consumer Performance Tuning") for further guidance.

### Conclusion

Mastering Kafka CLI tools and commands is essential for managing and optimizing Kafka clusters effectively. By understanding the purpose and usage of each tool, expert software engineers and enterprise architects can ensure efficient operation and maintenance of their Kafka environments.

## Test Your Knowledge: Kafka CLI Tools and Commands Quiz

{{< quizdown >}}

### Which Kafka CLI tool is used to create a new topic?

- [x] kafka-topics
- [ ] kafka-configs
- [ ] kafka-acls
- [ ] kafka-console-producer

> **Explanation:** The `kafka-topics` tool is used to manage topics, including creating new ones.

### What option is used with `kafka-console-consumer` to consume messages from the beginning of a topic?

- [x] --from-beginning
- [ ] --start-from-beginning
- [ ] --beginning
- [ ] --reset

> **Explanation:** The `--from-beginning` option allows consuming messages from the start of the topic.

### How do you list all consumer groups using Kafka CLI?

- [x] kafka-consumer-groups --list
- [ ] kafka-topics --list
- [ ] kafka-configs --list
- [ ] kafka-acls --list

> **Explanation:** The `kafka-consumer-groups --list` command lists all consumer groups.

### Which command is used to alter topic configurations?

- [x] kafka-configs --alter
- [ ] kafka-topics --alter
- [ ] kafka-acls --alter
- [ ] kafka-console-producer --alter

> **Explanation:** The `kafka-configs --alter` command is used to change configurations for topics, brokers, or clients.

### What is the purpose of the `kafka-verifiable-producer` tool?

- [x] To test and verify producer performance
- [ ] To consume messages from a topic
- [ ] To manage topic configurations
- [ ] To list consumer groups

> **Explanation:** The `kafka-verifiable-producer` tool is used for testing and verifying the performance of Kafka producers.

### Which option is used with `kafka-topics` to delete a topic?

- [x] --delete
- [ ] --remove
- [ ] --drop
- [ ] --erase

> **Explanation:** The `--delete` option is used to remove a topic.

### How can you describe a specific consumer group using Kafka CLI?

- [x] kafka-consumer-groups --describe --group <group-name>
- [ ] kafka-topics --describe --group <group-name>
- [ ] kafka-configs --describe --group <group-name>
- [ ] kafka-acls --describe --group <group-name>

> **Explanation:** The `kafka-consumer-groups --describe --group <group-name>` command provides details about a specific consumer group.

### What command lists all topics in a Kafka cluster?

- [x] kafka-topics --list
- [ ] kafka-consumer-groups --list
- [ ] kafka-configs --list
- [ ] kafka-acls --list

> **Explanation:** The `kafka-topics --list` command displays all topics in the cluster.

### Which tool is used to manage ACLs in Kafka?

- [x] kafka-acls
- [ ] kafka-topics
- [ ] kafka-configs
- [ ] kafka-console-producer

> **Explanation:** The `kafka-acls` tool is used to manage access control lists for Kafka resources.

### True or False: The `kafka-console-producer` can be used to consume messages from a topic.

- [ ] True
- [x] False

> **Explanation:** The `kafka-console-producer` is used to produce messages to a topic, not consume them.

{{< /quizdown >}}

By mastering these Kafka CLI tools and commands, you can effectively manage and optimize your Kafka clusters, ensuring robust and efficient data streaming solutions.

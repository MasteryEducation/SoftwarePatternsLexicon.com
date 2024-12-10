---
canonical: "https://softwarepatternslexicon.com/kafka/5/5/4"
title: "Mastering Apache Kafka with .NET and C# Clients"
description: "Explore Kafka integration with .NET applications using C#, enabling developers to build high-performance Kafka clients in the Microsoft ecosystem."
linkTitle: "5.5.4 .NET and C# Clients"
tags:
- "Apache Kafka"
- "CSharp"
- "DotNet"
- "Confluent"
- "Asynchronous Programming"
- "Serialization"
- "Error Handling"
- "Kafka Clients"
date: 2024-11-25
type: docs
nav_weight: 55400
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 5.5.4 .NET and C# Clients

### Introduction

Apache Kafka has become a cornerstone for building real-time data pipelines and streaming applications. With its robust architecture and scalability, Kafka is widely adopted across various industries. For developers in the Microsoft ecosystem, integrating Kafka with .NET applications using C# offers a powerful way to leverage Kafka's capabilities. This section explores the official Confluent .NET client, providing insights into producing and consuming messages in C#, asynchronous programming patterns, serialization, error handling, and compatibility considerations.

### Confluent .NET Client Overview

The Confluent .NET client is a high-performance library designed to integrate Kafka with .NET applications. It provides a simple yet powerful API for producing and consuming messages, supporting both synchronous and asynchronous operations. The client is built on top of `librdkafka`, a widely used C library for Kafka, ensuring efficient and reliable communication with Kafka brokers.

#### Key Features

- **High Performance**: Built on `librdkafka`, the client offers low latency and high throughput.
- **Asynchronous Programming**: Supports async/await patterns for non-blocking operations.
- **Serialization Support**: Integrates with Confluent Schema Registry for Avro, Protobuf, and JSON serialization.
- **Error Handling**: Provides robust error handling mechanisms to ensure reliability.
- **Compatibility**: Supports .NET Core, .NET 5+, and .NET Framework 4.6.1+.

For more details, refer to the [Confluent's .NET Client documentation](https://docs.confluent.io/clients-confluent-kafka-dotnet/current/overview.html).

### Producing Messages in C#

Producing messages to Kafka involves creating a producer instance, configuring it, and sending messages to a specified topic. The Confluent .NET client simplifies this process with its intuitive API.

#### Basic Producer Example

Here's a basic example of producing messages to a Kafka topic using C#:

```csharp
using Confluent.Kafka;
using System;
using System.Threading.Tasks;

class KafkaProducer
{
    public static async Task Main(string[] args)
    {
        var config = new ProducerConfig
        {
            BootstrapServers = "localhost:9092"
        };

        using (var producer = new ProducerBuilder<Null, string>(config).Build())
        {
            try
            {
                var deliveryReport = await producer.ProduceAsync("my-topic", new Message<Null, string> { Value = "Hello, Kafka!" });
                Console.WriteLine($"Delivered '{deliveryReport.Value}' to '{deliveryReport.TopicPartitionOffset}'");
            }
            catch (ProduceException<Null, string> e)
            {
                Console.WriteLine($"Delivery failed: {e.Error.Reason}");
            }
        }
    }
}
```

**Explanation**:
- **ProducerConfig**: Configures the producer with the Kafka broker's address.
- **ProduceAsync**: Sends a message asynchronously to the specified topic.
- **Delivery Report**: Provides feedback on the delivery status of the message.

#### Asynchronous Programming Patterns

The Confluent .NET client supports asynchronous programming, allowing developers to build non-blocking applications. The `async` and `await` keywords are used to perform asynchronous operations, improving application responsiveness and scalability.

### Consuming Messages in C#

Consuming messages from Kafka involves creating a consumer instance, subscribing to topics, and polling for messages. The Confluent .NET client provides a straightforward API for consuming messages.

#### Basic Consumer Example

Here's a basic example of consuming messages from a Kafka topic using C#:

```csharp
using Confluent.Kafka;
using System;
using System.Threading;

class KafkaConsumer
{
    public static void Main(string[] args)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers = "localhost:9092",
            GroupId = "test-consumer-group",
            AutoOffsetReset = AutoOffsetReset.Earliest
        };

        using (var consumer = new ConsumerBuilder<Ignore, string>(config).Build())
        {
            consumer.Subscribe("my-topic");

            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => {
                e.Cancel = true;
                cts.Cancel();
            };

            try
            {
                while (true)
                {
                    try
                    {
                        var cr = consumer.Consume(cts.Token);
                        Console.WriteLine($"Consumed message '{cr.Value}' at: '{cr.TopicPartitionOffset}'.");
                    }
                    catch (ConsumeException e)
                    {
                        Console.WriteLine($"Error occurred: {e.Error.Reason}");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                consumer.Close();
            }
        }
    }
}
```

**Explanation**:
- **ConsumerConfig**: Configures the consumer with the Kafka broker's address, consumer group, and offset reset policy.
- **Subscribe**: Subscribes the consumer to a topic.
- **Consume**: Polls for messages from the topic.

### Serialization and Deserialization

Serialization is crucial for converting data into a format suitable for transmission over the network. The Confluent .NET client supports various serialization formats, including Avro, Protobuf, and JSON, through the Confluent Schema Registry.

#### Configuring Avro Serialization

To use Avro serialization, configure the producer and consumer with Avro serializers:

```csharp
var producerConfig = new ProducerConfig
{
    BootstrapServers = "localhost:9092"
};

var avroProducerConfig = new AvroSerializerConfig
{
    SchemaRegistryUrl = "http://localhost:8081"
};

using (var producer = new ProducerBuilder<Null, MyAvroRecord>(producerConfig)
    .SetValueSerializer(new AvroSerializer<MyAvroRecord>(avroProducerConfig))
    .Build())
{
    // Produce messages
}
```

**Explanation**:
- **AvroSerializerConfig**: Configures the Avro serializer with the Schema Registry URL.
- **SetValueSerializer**: Sets the serializer for the message value.

### Error Handling

Robust error handling is essential for building reliable Kafka applications. The Confluent .NET client provides mechanisms to handle errors during production and consumption.

#### Handling Produce Errors

When producing messages, handle errors using `ProduceException`:

```csharp
try
{
    var deliveryReport = await producer.ProduceAsync("my-topic", new Message<Null, string> { Value = "Hello, Kafka!" });
}
catch (ProduceException<Null, string> e)
{
    Console.WriteLine($"Delivery failed: {e.Error.Reason}");
}
```

#### Handling Consume Errors

When consuming messages, handle errors using `ConsumeException`:

```csharp
try
{
    var cr = consumer.Consume(cts.Token);
}
catch (ConsumeException e)
{
    Console.WriteLine($"Error occurred: {e.Error.Reason}");
}
```

### Compatibility Considerations

The Confluent .NET client supports .NET Core, .NET 5+, and .NET Framework 4.6.1+. However, developers should be aware of potential compatibility issues with different .NET frameworks, especially when dealing with older versions.

#### .NET Core vs. .NET Framework

- **.NET Core**: Offers cross-platform support and is recommended for new applications.
- **.NET Framework**: Limited to Windows and may require additional configuration for compatibility.

### Best Practices

- **Use Asynchronous Operations**: Leverage async/await for non-blocking operations.
- **Configure Serialization**: Use the Confluent Schema Registry for managing schemas and serialization.
- **Handle Errors Gracefully**: Implement robust error handling to ensure application reliability.
- **Monitor Performance**: Use monitoring tools to track Kafka client performance and identify bottlenecks.

### Conclusion

Integrating Kafka with .NET applications using C# provides a powerful way to build high-performance, real-time data processing systems. By leveraging the Confluent .NET client, developers can efficiently produce and consume messages, implement asynchronous programming patterns, and manage serialization and error handling. With the right configurations and best practices, .NET developers can fully harness the capabilities of Kafka in their applications.

## Test Your Knowledge: Mastering Kafka with .NET and C# Clients

{{< quizdown >}}

### What is the primary benefit of using the Confluent .NET client for Kafka?

- [x] High performance and low latency
- [ ] Built-in UI for monitoring
- [ ] Automatic schema generation
- [ ] Native support for all .NET versions

> **Explanation:** The Confluent .NET client is built on `librdkafka`, offering high performance and low latency for Kafka operations.

### Which method is used to send messages asynchronously in the Confluent .NET client?

- [x] ProduceAsync
- [ ] SendAsync
- [ ] PublishAsync
- [ ] DispatchAsync

> **Explanation:** `ProduceAsync` is the method used to send messages asynchronously in the Confluent .NET client.

### What is the purpose of the Schema Registry in Kafka?

- [x] To manage and enforce schemas for serialized data
- [ ] To store Kafka configuration settings
- [ ] To monitor Kafka cluster health
- [ ] To provide a UI for Kafka management

> **Explanation:** The Schema Registry manages and enforces schemas for serialized data, ensuring compatibility and consistency.

### Which exception is used to handle errors during message production?

- [x] ProduceException
- [ ] KafkaException
- [ ] SerializationException
- [ ] ConsumerException

> **Explanation:** `ProduceException` is used to handle errors that occur during message production in Kafka.

### What is the recommended .NET version for new Kafka applications?

- [x] .NET Core
- [ ] .NET Framework 4.5
- [ ] .NET Framework 4.6.1
- [ ] .NET Standard

> **Explanation:** .NET Core is recommended for new applications due to its cross-platform support and modern features.

### How can you ensure non-blocking operations in a .NET Kafka application?

- [x] Use async/await patterns
- [ ] Use synchronous methods
- [ ] Increase thread count
- [ ] Disable logging

> **Explanation:** Using async/await patterns ensures non-blocking operations, improving application responsiveness.

### Which configuration is necessary for Avro serialization in Kafka?

- [x] SchemaRegistryUrl
- [ ] BootstrapServers
- [ ] GroupId
- [ ] AutoOffsetReset

> **Explanation:** `SchemaRegistryUrl` is necessary for configuring Avro serialization with the Schema Registry.

### What is the role of the `ConsumerConfig` class in Kafka?

- [x] To configure consumer settings like broker address and group ID
- [ ] To configure producer settings
- [ ] To manage Kafka topics
- [ ] To monitor consumer performance

> **Explanation:** `ConsumerConfig` is used to configure consumer settings such as broker address and group ID.

### Which method is used to subscribe a consumer to a Kafka topic?

- [x] Subscribe
- [ ] Listen
- [ ] Register
- [ ] Connect

> **Explanation:** The `Subscribe` method is used to subscribe a consumer to a Kafka topic.

### True or False: The Confluent .NET client supports both synchronous and asynchronous operations.

- [x] True
- [ ] False

> **Explanation:** The Confluent .NET client supports both synchronous and asynchronous operations, allowing flexibility in application design.

{{< /quizdown >}}

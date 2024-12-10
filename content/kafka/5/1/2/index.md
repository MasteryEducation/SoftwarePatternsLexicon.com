---
canonical: "https://softwarepatternslexicon.com/kafka/5/1/2"

title: "Asynchronous and Synchronous Sending in Apache Kafka"
description: "Explore the intricacies of asynchronous and synchronous message sending in Apache Kafka producers, including performance implications, error handling, and best practices."
linkTitle: "5.1.2 Asynchronous and Synchronous Sending"
tags:
- "Apache Kafka"
- "Kafka Producer"
- "Asynchronous Sending"
- "Synchronous Sending"
- "Performance Optimization"
- "Error Handling"
- "Java"
- "Scala"
date: 2024-11-25
type: docs
nav_weight: 51200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 5.1.2 Asynchronous and Synchronous Sending

In the realm of Apache Kafka, understanding the nuances between asynchronous and synchronous message sending is crucial for optimizing performance and ensuring reliable data delivery. This section delves into the default asynchronous nature of Kafka producers, explores how to perform synchronous sends, and discusses the implications of each approach on throughput and latency. We will also cover the use of callbacks for handling send results asynchronously, provide code examples in multiple languages, and highlight performance considerations and best practices. Finally, we will suggest strategies for error handling in both modes.

### Understanding Kafka's Default Asynchronous Sending

By default, Kafka producers operate in an asynchronous mode. This means that when a message is sent, the producer does not wait for an acknowledgment from the Kafka broker before proceeding to send the next message. This approach is designed to maximize throughput by allowing the producer to send messages in rapid succession without being blocked by network latency or broker processing time.

#### Benefits of Asynchronous Sending

- **High Throughput**: Asynchronous sending allows producers to achieve high throughput by batching multiple messages together and sending them in a single request.
- **Non-Blocking**: The producer thread is not blocked while waiting for an acknowledgment, allowing it to continue processing other tasks.
- **Efficient Resource Utilization**: By not waiting for acknowledgments, producers can make better use of system resources, leading to improved overall performance.

#### Asynchronous Sending with Callbacks

To handle the results of asynchronous sends, Kafka provides a mechanism for registering callbacks. These callbacks are invoked once the broker acknowledges the receipt of a message, allowing the application to handle success or failure scenarios.

##### Java Example: Asynchronous Sending with Callbacks

```java
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.clients.producer.ProducerConfig;

import java.util.Properties;

public class AsyncProducerExample {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer");
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer");

        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        ProducerRecord<String, String> record = new ProducerRecord<>("my-topic", "key", "value");

        producer.send(record, new Callback() {
            @Override
            public void onCompletion(RecordMetadata metadata, Exception exception) {
                if (exception == null) {
                    System.out.println("Message sent successfully to " + metadata.topic() + " partition " + metadata.partition());
                } else {
                    exception.printStackTrace();
                }
            }
        });

        producer.close();
    }
}
```

In this example, the `send` method is used to send a message asynchronously. The `Callback` interface is implemented to handle the result of the send operation. If the message is successfully sent, the metadata is printed; otherwise, the exception is logged.

##### Scala Example: Asynchronous Sending with Callbacks

```scala
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord, Callback, RecordMetadata, ProducerConfig}
import java.util.Properties

object AsyncProducerExample {
  def main(args: Array[String]): Unit = {
    val props = new Properties()
    props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092")
    props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")
    props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")

    val producer = new KafkaProducer[String, String](props)

    val record = new ProducerRecord[String, String]("my-topic", "key", "value")

    producer.send(record, new Callback {
      override def onCompletion(metadata: RecordMetadata, exception: Exception): Unit = {
        if (exception == null) {
          println(s"Message sent successfully to ${metadata.topic()} partition ${metadata.partition()}")
        } else {
          exception.printStackTrace()
        }
      }
    })

    producer.close()
  }
}
```

### Synchronous Sending in Kafka

While asynchronous sending is the default and most efficient mode for high throughput, there are scenarios where synchronous sending is preferred. In synchronous mode, the producer waits for an acknowledgment from the broker before sending the next message. This ensures that each message is successfully received before proceeding, which can be critical for applications requiring strong delivery guarantees.

#### Benefits of Synchronous Sending

- **Delivery Guarantees**: Ensures that each message is acknowledged by the broker before proceeding, reducing the risk of message loss.
- **Simplified Error Handling**: Errors can be handled immediately after a send operation, simplifying the logic for retry mechanisms.

#### Java Example: Synchronous Sending

```java
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.ProducerConfig;

import java.util.Properties;
import java.util.concurrent.ExecutionException;

public class SyncProducerExample {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer");
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer");

        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        ProducerRecord<String, String> record = new ProducerRecord<>("my-topic", "key", "value");

        try {
            producer.send(record).get(); // Synchronous send
            System.out.println("Message sent successfully");
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        producer.close();
    }
}
```

In this example, the `send` method is followed by a `get` call, which blocks the producer thread until the broker acknowledges the message.

#### Scala Example: Synchronous Sending

```scala
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord, ProducerConfig}
import java.util.Properties
import scala.util.{Failure, Success, Try}

object SyncProducerExample {
  def main(args: Array[String]): Unit = {
    val props = new Properties()
    props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092")
    props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")
    props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")

    val producer = new KafkaProducer[String, String](props)

    val record = new ProducerRecord[String, String]("my-topic", "key", "value")

    Try(producer.send(record).get()) match {
      case Success(_) => println("Message sent successfully")
      case Failure(exception) => exception.printStackTrace()
    }

    producer.close()
  }
}
```

### Performance Considerations

When choosing between asynchronous and synchronous sending, it's important to consider the trade-offs between throughput and latency.

- **Throughput**: Asynchronous sending generally provides higher throughput as it allows for batching and non-blocking operations. This is ideal for applications where high message rates are critical.
- **Latency**: Synchronous sending introduces additional latency as each message must be acknowledged before the next is sent. This can be acceptable for applications where delivery guarantees are more important than speed.

### Best Practices

- **Batching**: Use batching to improve throughput in asynchronous mode. Configure the `linger.ms` and `batch.size` settings to optimize batch sizes and reduce the number of requests sent to the broker.
- **Retries and Idempotence**: Enable retries and idempotence to handle transient errors and ensure message delivery. Configure `retries` and `enable.idempotence` settings appropriately.
- **Error Handling**: Implement robust error handling mechanisms for both asynchronous and synchronous modes. Use callbacks to handle errors in asynchronous mode and exception handling in synchronous mode.
- **Monitoring and Logging**: Monitor producer metrics and log send results to identify and troubleshoot issues. Use tools like Prometheus and Grafana for real-time monitoring.

### Error Handling Strategies

Error handling is a critical aspect of message sending in Kafka. Both asynchronous and synchronous modes require different strategies to handle errors effectively.

#### Asynchronous Error Handling

- **Callbacks**: Use callbacks to handle errors in asynchronous mode. Implement logic to retry or log errors based on the exception type.
- **Retry Mechanisms**: Configure retry settings to handle transient errors automatically. Use exponential backoff to avoid overwhelming the broker.

#### Synchronous Error Handling

- **Exception Handling**: Use try-catch blocks to handle exceptions in synchronous mode. Implement logic to retry or log errors based on the exception type.
- **Transaction Support**: Use transactions to ensure atomicity and consistency in message delivery. This is particularly useful for applications requiring exactly-once semantics.

### Conclusion

Understanding the differences between asynchronous and synchronous sending in Kafka is essential for building efficient and reliable data pipelines. By leveraging the strengths of each approach and implementing best practices for performance optimization and error handling, you can ensure that your Kafka producers operate effectively in any scenario.

## Test Your Knowledge: Asynchronous and Synchronous Sending in Kafka

{{< quizdown >}}

### What is the default sending mode for Kafka producers?

- [x] Asynchronous
- [ ] Synchronous
- [ ] Batch
- [ ] Transactional

> **Explanation:** Kafka producers are designed to send messages asynchronously by default to maximize throughput.

### Which method is used to perform a synchronous send in Kafka?

- [x] `send(record).get()`
- [ ] `send(record)`
- [ ] `sendSync(record)`
- [ ] `sendAsync(record)`

> **Explanation:** The `send(record).get()` method blocks the producer thread until the broker acknowledges the message, making it synchronous.

### What is a key benefit of asynchronous sending in Kafka?

- [x] High throughput
- [ ] Low latency
- [ ] Guaranteed delivery
- [ ] Simplified error handling

> **Explanation:** Asynchronous sending allows for high throughput by enabling non-blocking operations and batching.

### How can you handle errors in asynchronous sending?

- [x] Use callbacks
- [ ] Use try-catch blocks
- [ ] Use transactions
- [ ] Use synchronous sends

> **Explanation:** Callbacks are used in asynchronous sending to handle errors and process send results.

### What is a potential drawback of synchronous sending?

- [x] Increased latency
- [ ] Reduced throughput
- [x] Simplified error handling
- [ ] Complex implementation

> **Explanation:** Synchronous sending increases latency as each message must be acknowledged before the next is sent.

### Which setting can be configured to improve batching in asynchronous mode?

- [x] `linger.ms`
- [ ] `acks`
- [ ] `compression.type`
- [ ] `max.in.flight.requests.per.connection`

> **Explanation:** The `linger.ms` setting controls the time to wait before sending a batch, allowing for larger batches and improved throughput.

### What is a recommended strategy for handling transient errors in Kafka?

- [x] Enable retries
- [ ] Use synchronous sending
- [x] Implement idempotence
- [ ] Disable batching

> **Explanation:** Enabling retries and implementing idempotence are effective strategies for handling transient errors in Kafka.

### Which tool can be used for real-time monitoring of Kafka producers?

- [x] Prometheus
- [ ] Jenkins
- [ ] Terraform
- [ ] Ansible

> **Explanation:** Prometheus is a popular tool for real-time monitoring of Kafka producers and other components.

### What is the purpose of the `Callback` interface in Kafka?

- [x] To handle send results asynchronously
- [ ] To perform synchronous sends
- [ ] To configure producer settings
- [ ] To manage consumer offsets

> **Explanation:** The `Callback` interface is used to handle send results asynchronously, allowing for error handling and logging.

### True or False: Synchronous sending is always preferred for high-throughput applications.

- [ ] True
- [x] False

> **Explanation:** False. Asynchronous sending is generally preferred for high-throughput applications due to its non-blocking nature and ability to batch messages.

{{< /quizdown >}}

By mastering both asynchronous and synchronous sending techniques, you can optimize your Kafka producers for a wide range of applications, ensuring efficient and reliable message delivery.

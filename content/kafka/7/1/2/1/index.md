---
canonical: "https://softwarepatternslexicon.com/kafka/7/1/2/1"
title: "Mastering Apache Kafka Connectors: Commonly Used Source and Sink Connectors"
description: "Explore the most popular Kafka Connectors, including JDBC, FileStream, Elasticsearch, and HDFS, and learn how to integrate them into your data pipelines for efficient data processing."
linkTitle: "7.1.2.1 Commonly Used Connectors"
tags:
- "Apache Kafka"
- "Kafka Connect"
- "JDBC Connector"
- "Elasticsearch Connector"
- "HDFS Connector"
- "Data Integration"
- "Real-Time Data Processing"
- "Enterprise Architecture"
date: 2024-11-25
type: docs
nav_weight: 71210
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 7.1.2.1 Commonly Used Connectors

Apache Kafka Connect is a robust framework for streaming data between Apache Kafka and other systems. It simplifies the integration of various data sources and sinks, allowing for seamless data flow in real-time. This section delves into some of the most commonly used connectors, highlighting their functionalities, typical use cases, and providing links to their documentation for further exploration.

### Introduction to Kafka Connectors

Kafka Connectors are essential components in the Kafka ecosystem, enabling the integration of external systems with Kafka. They are categorized into two types:

- **Source Connectors**: These connectors ingest data from external systems into Kafka topics.
- **Sink Connectors**: These connectors export data from Kafka topics to external systems.

Each connector is designed to handle specific data formats and protocols, making it crucial to choose the right connector for your use case.

### Popular Source and Sink Connectors

#### 1. JDBC Connector

The JDBC Connector is one of the most widely used connectors in the Kafka ecosystem. It allows for the integration of relational databases with Kafka, enabling the streaming of data from and to databases using the JDBC protocol.

- **Functionality**: The JDBC Connector can act as both a source and a sink. As a source, it captures changes from a database and streams them into Kafka topics. As a sink, it writes data from Kafka topics into a database.
- **Use Cases**: Ideal for applications that require real-time data synchronization between databases and Kafka. Commonly used in scenarios such as change data capture (CDC) and data warehousing.
- **Documentation**: [Kafka Connect JDBC](https://www.confluent.io/hub/confluentinc/kafka-connect-jdbc)

**Example Scenario**: A financial institution uses the JDBC Connector to stream transaction data from its SQL database into Kafka for real-time fraud detection.

#### 2. FileStream Connector

The FileStream Connector is a simple yet powerful connector that enables the streaming of data from files into Kafka and vice versa.

- **Functionality**: As a source, it reads data from files and streams it into Kafka topics. As a sink, it writes data from Kafka topics into files.
- **Use Cases**: Suitable for scenarios where data is stored in log files or flat files and needs to be ingested into Kafka for processing. Often used in log aggregation and batch processing applications.

**Example Scenario**: A retail company uses the FileStream Connector to ingest sales data from daily log files into Kafka for real-time analytics.

#### 3. Elasticsearch Connector

The Elasticsearch Connector is designed to integrate Kafka with Elasticsearch, a popular search and analytics engine.

- **Functionality**: Primarily a sink connector, it streams data from Kafka topics into Elasticsearch indices, enabling real-time search and analytics capabilities.
- **Use Cases**: Commonly used in applications that require full-text search and real-time analytics. Ideal for log analysis, monitoring, and alerting systems.
- **Documentation**: [Kafka Connect Elasticsearch](https://www.confluent.io/hub/confluentinc/kafka-connect-elasticsearch)

**Example Scenario**: An e-commerce platform uses the Elasticsearch Connector to index product data from Kafka, allowing users to perform fast searches on the website.

#### 4. HDFS Connector

The HDFS Connector facilitates the integration of Kafka with Hadoop Distributed File System (HDFS), enabling the storage of Kafka data in a distributed file system.

- **Functionality**: As a sink connector, it writes data from Kafka topics into HDFS, supporting various file formats such as Avro, Parquet, and JSON.
- **Use Cases**: Ideal for big data applications that require long-term storage and batch processing of Kafka data. Commonly used in data lake architectures and ETL pipelines.

**Example Scenario**: A media company uses the HDFS Connector to store video metadata from Kafka into HDFS for batch processing and analytics.

### Detailed Exploration of Connectors

#### JDBC Connector

The JDBC Connector is a versatile tool for integrating relational databases with Kafka. It supports a wide range of databases, including MySQL, PostgreSQL, Oracle, and SQL Server. The connector can be configured to perform incremental loads, capturing only the changes made to the database since the last poll.

**Key Features**:

- **Incremental Querying**: Supports incremental queries using timestamp columns or custom query logic.
- **Schema Evolution**: Automatically adapts to changes in the database schema.
- **Batch Processing**: Configurable batch size for efficient data transfer.

**Java Example**:

```java
import org.apache.kafka.connect.jdbc.JdbcSourceConnector;
import java.util.HashMap;
import java.util.Map;

public class JdbcConnectorExample {
    public static void main(String[] args) {
        Map<String, String> config = new HashMap<>();
        config.put("connector.class", "JdbcSourceConnector");
        config.put("tasks.max", "1");
        config.put("connection.url", "jdbc:mysql://localhost:3306/mydb");
        config.put("mode", "incrementing");
        config.put("incrementing.column.name", "id");
        config.put("topic.prefix", "jdbc-");

        JdbcSourceConnector connector = new JdbcSourceConnector();
        connector.start(config);
    }
}
```

**Scala Example**:

```scala
import org.apache.kafka.connect.jdbc.JdbcSourceConnector
import scala.collection.JavaConverters._

object JdbcConnectorExample extends App {
  val config = Map(
    "connector.class" -> "JdbcSourceConnector",
    "tasks.max" -> "1",
    "connection.url" -> "jdbc:mysql://localhost:3306/mydb",
    "mode" -> "incrementing",
    "incrementing.column.name" -> "id",
    "topic.prefix" -> "jdbc-"
  ).asJava

  val connector = new JdbcSourceConnector()
  connector.start(config)
}
```

**Kotlin Example**:

```kotlin
import org.apache.kafka.connect.jdbc.JdbcSourceConnector

fun main() {
    val config = mapOf(
        "connector.class" to "JdbcSourceConnector",
        "tasks.max" to "1",
        "connection.url" to "jdbc:mysql://localhost:3306/mydb",
        "mode" to "incrementing",
        "incrementing.column.name" to "id",
        "topic.prefix" to "jdbc-"
    )

    val connector = JdbcSourceConnector()
    connector.start(config)
}
```

**Clojure Example**:

```clojure
(ns jdbc-connector-example
  (:import [org.apache.kafka.connect.jdbc JdbcSourceConnector]))

(defn start-jdbc-connector []
  (let [config {"connector.class" "JdbcSourceConnector"
                "tasks.max" "1"
                "connection.url" "jdbc:mysql://localhost:3306/mydb"
                "mode" "incrementing"
                "incrementing.column.name" "id"
                "topic.prefix" "jdbc-"}]
    (doto (JdbcSourceConnector.)
      (.start config))))
```

#### FileStream Connector

The FileStream Connector is a straightforward connector that reads from and writes to files. It is particularly useful for scenarios where data is stored in flat files and needs to be processed in real-time.

**Key Features**:

- **Simple Configuration**: Easy to set up with minimal configuration.
- **Flexible File Formats**: Supports various file formats, including text and binary.
- **Real-Time Processing**: Streams data in real-time, making it suitable for log aggregation.

**Java Example**:

```java
import org.apache.kafka.connect.file.FileStreamSourceConnector;
import java.util.HashMap;
import java.util.Map;

public class FileStreamConnectorExample {
    public static void main(String[] args) {
        Map<String, String> config = new HashMap<>();
        config.put("connector.class", "FileStreamSourceConnector");
        config.put("tasks.max", "1");
        config.put("file", "/path/to/input/file");
        config.put("topic", "file-stream-topic");

        FileStreamSourceConnector connector = new FileStreamSourceConnector();
        connector.start(config);
    }
}
```

**Scala Example**:

```scala
import org.apache.kafka.connect.file.FileStreamSourceConnector
import scala.collection.JavaConverters._

object FileStreamConnectorExample extends App {
  val config = Map(
    "connector.class" -> "FileStreamSourceConnector",
    "tasks.max" -> "1",
    "file" -> "/path/to/input/file",
    "topic" -> "file-stream-topic"
  ).asJava

  val connector = new FileStreamSourceConnector()
  connector.start(config)
}
```

**Kotlin Example**:

```kotlin
import org.apache.kafka.connect.file.FileStreamSourceConnector

fun main() {
    val config = mapOf(
        "connector.class" to "FileStreamSourceConnector",
        "tasks.max" to "1",
        "file" to "/path/to/input/file",
        "topic" to "file-stream-topic"
    )

    val connector = FileStreamSourceConnector()
    connector.start(config)
}
```

**Clojure Example**:

```clojure
(ns file-stream-connector-example
  (:import [org.apache.kafka.connect.file FileStreamSourceConnector]))

(defn start-file-stream-connector []
  (let [config {"connector.class" "FileStreamSourceConnector"
                "tasks.max" "1"
                "file" "/path/to/input/file"
                "topic" "file-stream-topic"}]
    (doto (FileStreamSourceConnector.)
      (.start config))))
```

#### Elasticsearch Connector

The Elasticsearch Connector is a powerful tool for integrating Kafka with Elasticsearch, enabling real-time search and analytics capabilities.

**Key Features**:

- **Real-Time Indexing**: Streams data from Kafka topics into Elasticsearch indices in real-time.
- **Schema Mapping**: Automatically maps Kafka data to Elasticsearch fields.
- **Scalable**: Supports high-throughput data ingestion.

**Java Example**:

```java
import org.apache.kafka.connect.elasticsearch.ElasticsearchSinkConnector;
import java.util.HashMap;
import java.util.Map;

public class ElasticsearchConnectorExample {
    public static void main(String[] args) {
        Map<String, String> config = new HashMap<>();
        config.put("connector.class", "ElasticsearchSinkConnector");
        config.put("tasks.max", "1");
        config.put("topics", "elasticsearch-topic");
        config.put("connection.url", "http://localhost:9200");
        config.put("type.name", "kafka-connect");

        ElasticsearchSinkConnector connector = new ElasticsearchSinkConnector();
        connector.start(config);
    }
}
```

**Scala Example**:

```scala
import org.apache.kafka.connect.elasticsearch.ElasticsearchSinkConnector
import scala.collection.JavaConverters._

object ElasticsearchConnectorExample extends App {
  val config = Map(
    "connector.class" -> "ElasticsearchSinkConnector",
    "tasks.max" -> "1",
    "topics" -> "elasticsearch-topic",
    "connection.url" -> "http://localhost:9200",
    "type.name" -> "kafka-connect"
  ).asJava

  val connector = new ElasticsearchSinkConnector()
  connector.start(config)
}
```

**Kotlin Example**:

```kotlin
import org.apache.kafka.connect.elasticsearch.ElasticsearchSinkConnector

fun main() {
    val config = mapOf(
        "connector.class" to "ElasticsearchSinkConnector",
        "tasks.max" to "1",
        "topics" to "elasticsearch-topic",
        "connection.url" to "http://localhost:9200",
        "type.name" to "kafka-connect"
    )

    val connector = ElasticsearchSinkConnector()
    connector.start(config)
}
```

**Clojure Example**:

```clojure
(ns elasticsearch-connector-example
  (:import [org.apache.kafka.connect.elasticsearch ElasticsearchSinkConnector]))

(defn start-elasticsearch-connector []
  (let [config {"connector.class" "ElasticsearchSinkConnector"
                "tasks.max" "1"
                "topics" "elasticsearch-topic"
                "connection.url" "http://localhost:9200"
                "type.name" "kafka-connect"}]
    (doto (ElasticsearchSinkConnector.)
      (.start config))))
```

#### HDFS Connector

The HDFS Connector is essential for integrating Kafka with Hadoop ecosystems, enabling the storage of Kafka data in HDFS for batch processing and long-term storage.

**Key Features**:

- **File Format Support**: Supports Avro, Parquet, and JSON file formats.
- **Partitioning**: Configurable partitioning strategies for efficient data organization.
- **Batch Processing**: Suitable for ETL pipelines and data lake architectures.

**Java Example**:

```java
import org.apache.kafka.connect.hdfs.HdfsSinkConnector;
import java.util.HashMap;
import java.util.Map;

public class HdfsConnectorExample {
    public static void main(String[] args) {
        Map<String, String> config = new HashMap<>();
        config.put("connector.class", "HdfsSinkConnector");
        config.put("tasks.max", "1");
        config.put("topics", "hdfs-topic");
        config.put("hdfs.url", "hdfs://localhost:9000");
        config.put("flush.size", "1000");

        HdfsSinkConnector connector = new HdfsSinkConnector();
        connector.start(config);
    }
}
```

**Scala Example**:

```scala
import org.apache.kafka.connect.hdfs.HdfsSinkConnector
import scala.collection.JavaConverters._

object HdfsConnectorExample extends App {
  val config = Map(
    "connector.class" -> "HdfsSinkConnector",
    "tasks.max" -> "1",
    "topics" -> "hdfs-topic",
    "hdfs.url" -> "hdfs://localhost:9000",
    "flush.size" -> "1000"
  ).asJava

  val connector = new HdfsSinkConnector()
  connector.start(config)
}
```

**Kotlin Example**:

```kotlin
import org.apache.kafka.connect.hdfs.HdfsSinkConnector

fun main() {
    val config = mapOf(
        "connector.class" to "HdfsSinkConnector",
        "tasks.max" to "1",
        "topics" to "hdfs-topic",
        "hdfs.url" to "hdfs://localhost:9000",
        "flush.size" to "1000"
    )

    val connector = HdfsSinkConnector()
    connector.start(config)
}
```

**Clojure Example**:

```clojure
(ns hdfs-connector-example
  (:import [org.apache.kafka.connect.hdfs HdfsSinkConnector]))

(defn start-hdfs-connector []
  (let [config {"connector.class" "HdfsSinkConnector"
                "tasks.max" "1"
                "topics" "hdfs-topic"
                "hdfs.url" "hdfs://localhost:9000"
                "flush.size" "1000"}]
    (doto (HdfsSinkConnector.)
      (.start config))))
```

### Best Practices for Using Kafka Connectors

- **Configuration Management**: Use configuration management tools to manage connector configurations across environments.
- **Monitoring and Alerting**: Implement monitoring and alerting to detect and respond to connector failures promptly.
- **Schema Management**: Leverage schema registries to manage data schemas and ensure compatibility across systems.
- **Performance Tuning**: Optimize connector configurations for performance, considering factors such as batch size and task parallelism.

### Conclusion

Kafka Connectors play a crucial role in integrating Kafka with external systems, enabling seamless data flow in real-time. By understanding the functionalities and use cases of popular connectors such as JDBC, FileStream, Elasticsearch, and HDFS, you can design efficient data pipelines that meet your organization's needs.

For more information on Kafka Connectors and their configurations, refer to the official [Apache Kafka Documentation](https://kafka.apache.org/documentation/).

## Test Your Knowledge: Mastering Kafka Connectors Quiz

{{< quizdown >}}

### Which connector is commonly used for integrating relational databases with Kafka?

- [x] JDBC Connector
- [ ] FileStream Connector
- [ ] Elasticsearch Connector
- [ ] HDFS Connector

> **Explanation:** The JDBC Connector is designed for integrating relational databases with Kafka, supporting both source and sink operations.

### What is the primary function of the FileStream Connector?

- [x] Streaming data from files to Kafka and vice versa
- [ ] Indexing data into Elasticsearch
- [ ] Writing data to HDFS
- [ ] Capturing changes from databases

> **Explanation:** The FileStream Connector is used to stream data from files into Kafka topics and from Kafka topics into files.

### Which connector is best suited for real-time search and analytics?

- [ ] JDBC Connector
- [ ] FileStream Connector
- [x] Elasticsearch Connector
- [ ] HDFS Connector

> **Explanation:** The Elasticsearch Connector is designed for real-time search and analytics, streaming data from Kafka into Elasticsearch indices.

### What file formats does the HDFS Connector support?

- [x] Avro, Parquet, and JSON
- [ ] CSV and XML
- [ ] Text and Binary
- [ ] SQL and NoSQL

> **Explanation:** The HDFS Connector supports Avro, Parquet, and JSON file formats for storing Kafka data in HDFS.

### Which connector would you use for change data capture (CDC) from a database?

- [x] JDBC Connector
- [ ] FileStream Connector
- [ ] Elasticsearch Connector
- [ ] HDFS Connector

> **Explanation:** The JDBC Connector is ideal for change data capture (CDC) scenarios, streaming changes from databases into Kafka.

### What is a common use case for the FileStream Connector?

- [x] Log aggregation
- [ ] Real-time fraud detection
- [ ] Full-text search
- [ ] Data lake storage

> **Explanation:** The FileStream Connector is commonly used for log aggregation, streaming log data from files into Kafka.

### How does the Elasticsearch Connector handle schema mapping?

- [x] Automatically maps Kafka data to Elasticsearch fields
- [ ] Requires manual schema mapping
- [ ] Uses a separate schema registry
- [ ] Does not support schema mapping

> **Explanation:** The Elasticsearch Connector automatically maps Kafka data to Elasticsearch fields, simplifying the integration process.

### Which connector is primarily used for storing Kafka data in a distributed file system?

- [ ] JDBC Connector
- [ ] FileStream Connector
- [ ] Elasticsearch Connector
- [x] HDFS Connector

> **Explanation:** The HDFS Connector is used to store Kafka data in the Hadoop Distributed File System (HDFS).

### What is a key feature of the JDBC Connector?

- [x] Incremental querying
- [ ] Real-time indexing
- [ ] File format support
- [ ] Log aggregation

> **Explanation:** The JDBC Connector supports incremental querying, capturing only the changes made to the database since the last poll.

### True or False: The FileStream Connector can only be used as a source connector.

- [ ] True
- [x] False

> **Explanation:** The FileStream Connector can be used both as a source and a sink connector, streaming data from files to Kafka and vice versa.

{{< /quizdown >}}

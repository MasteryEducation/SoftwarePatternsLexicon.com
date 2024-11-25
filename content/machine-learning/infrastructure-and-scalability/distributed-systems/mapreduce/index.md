---
linkTitle: "MapReduce"
title: "MapReduce: Distributing Data Processing Tasks Across Multiple Nodes"
description: "An overview of the MapReduce design pattern for distributing data processing tasks across multiple nodes, including detailed explanations, examples, related design patterns, and additional resources."
categories:
- Infrastructure and Scalability
tags:
- machine learning
- distributed systems
- data processing
- big data
- parallel computing
date: 2023-10-05
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/infrastructure-and-scalability/distributed-systems/mapreduce"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

The **MapReduce** design pattern is a fundamental paradigm for processing large datasets across multiple nodes in a distributed system. By breaking down a job into a series of `Map` and `Reduce` operations, this pattern allows for scalable and efficient data processing. It is particularly useful for tasks that can be parallelized, offering significant performance benefits on large-scale data.

## Components of MapReduce

### 1. Map Function

The `Map` function processes input data (usually key-value pairs) and produces a set of intermediate key-value pairs. Each input item is processed independently, allowing for parallel execution.

```python
def map_function(input_data):
    intermediate = []
    for record in input_data:
        key = record['category']
        value = record['amount']
        intermediate.append((key, value))
    return intermediate
```

### 2. Reduce Function

The `Reduce` function takes intermediate key-value pairs produced by the `Map` function and merges them together to form a smaller set of output values.

```python
def reduce_function(intermediate_data):
    output = {}
    for key, values in intermediate_data.items():
        output[key] = sum(values)
    return output
```

### Example Workflow

Here’s a step-by-step example to illustrate how MapReduce works:

1. The input dataset is split into chunks that can be processed in parallel.
2. Each chunk of data is processed by a `Map` function, generating intermediate key-value pairs.
3. Intermediate pairs with the same key are shuffled and grouped together.
4. Each group is processed by a `Reduce` function, producing the final output.

## Example in Multiple Languages

### Python with Hadoop Streaming

```python
import sys

for line in sys.stdin:
    data = line.strip().split()
    category = data[0]
    amount = data[1]
    print(f"{category}\t{amount}")
```

```python
import sys
from collections import defaultdict

category_totals = defaultdict(int)

for line in sys.stdin:
    category, amount = line.strip().split('\t')
    category_totals[category] += int(amount)

for category, total in category_totals.items():
    print(f"{category}\t{total}")
```

### Java

```java
// Mapper class
public class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        StringTokenizer itr = new StringTokenizer(value.toString());
        while (itr.hasMoreTokens()) {
            word.set(itr.nextToken());
            context.write(word, one);
        }
    }
}
```

```java
// Reducer class
public class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();
        }
        result.set(sum);
        context.write(key, result);
    }
}
```

## Related Design Patterns

### 1. **Pipeline Design Pattern**

The Pipeline pattern structures data processing in a series of stages (filters) connected by pipes (channels), each performing distinct operations on the data. This design is complementary to MapReduce, where the stages could be distributed across nodes.

### 2. **Batch Processing Pattern**

Batch Processing involves executing a series of non-interactive jobs all at once. MapReduce is a specific kind of batch processing mechanism focused on large datasets and distributed environments, ensuring fault tolerance and scalability.

### 3. **Data Sharding**

Data Sharding divides a dataset into smaller, more manageable pieces (shards) that can be processed in parallel. MapReduce can use data sharding principles for effectively distributing the map tasks.

## Additional Resources

- **Books**: 
  - "Hadoop: The Definitive Guide" by Tom White
  - "Mining of Massive Datasets" by Jure Leskovec, Anand Rajaraman, and Jeffrey Ullman

- **Online Tutorials**:
  - [Hadoop Tutorial](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html)
  - [Google MapReduce Paper](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)

- **Frameworks**:
  - [Apache Hadoop](https://hadoop.apache.org)
  - [Apache Spark](https://spark.apache.org)

## Summary

The **MapReduce** design pattern allows for scalable, parallel processing of large datasets by distributing tasks across multiple nodes. By dividing tasks into `Map` and `Reduce` functions, it provides a simple yet powerful mechanism for processing big data efficiently. The versatility and scalability offered by MapReduce make it a cornerstone in the infrastructure of modern distributed systems. Understanding and implementing this pattern can significantly enhance the performance and capability of your data processing systems.


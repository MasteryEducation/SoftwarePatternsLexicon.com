---
canonical: "https://softwarepatternslexicon.com/patterns-java/12/3/2"

title: "Understanding Flux and Mono in Reactive Java Programming"
description: "Explore the core concepts of Flux and Mono in Project Reactor, essential for mastering reactive programming in Java."
linkTitle: "12.3.2 Understanding Flux and Mono"
tags:
- "Java"
- "Reactive Programming"
- "Project Reactor"
- "Flux"
- "Mono"
- "Concurrency"
- "Asynchronous"
- "Error Handling"
date: 2024-11-25
type: docs
nav_weight: 123200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 12.3.2 Understanding Flux and Mono

Reactive programming is a paradigm that allows developers to build systems that are resilient, responsive, and elastic. In the Java ecosystem, Project Reactor is a popular library that provides support for building reactive applications. At the heart of Project Reactor are two key types: `Flux` and `Mono`. Understanding these types is crucial for leveraging the full power of reactive programming in Java.

### Introduction to Reactive Programming

Reactive programming is a programming paradigm oriented around data flows and the propagation of change. It is particularly useful for applications that require high concurrency and low latency. Reactive systems are designed to be responsive, resilient, elastic, and message-driven, which are the four tenets of the Reactive Manifesto.

### Project Reactor: An Overview

Project Reactor is a reactive library for building non-blocking applications on the JVM. It is part of the larger Spring ecosystem and provides a powerful set of tools for working with asynchronous data streams. Reactor is based on the Reactive Streams specification, which defines a standard for asynchronous stream processing with non-blocking back pressure.

### Understanding Flux and Mono

#### Flux: The Reactive Stream of Many

`Flux` is a reactive type that represents a stream of 0 to N elements. It is analogous to a `List` or `Stream` in Java, but with the added capability of handling asynchronous data flows. A `Flux` can emit three types of signals:

1. **Next**: Represents the emission of an element.
2. **Error**: Indicates that an error has occurred, terminating the stream.
3. **Complete**: Signals that the stream has finished emitting elements.

```java
import reactor.core.publisher.Flux;

public class FluxExample {
    public static void main(String[] args) {
        Flux<String> stringFlux = Flux.just("Hello", "World", "From", "Flux");
        
        stringFlux.subscribe(
            System.out::println, // onNext
            error -> System.err.println("Error: " + error), // onError
            () -> System.out.println("Completed!") // onComplete
        );
    }
}
```

In this example, `Flux.just` creates a `Flux` that emits a sequence of strings. The `subscribe` method is used to consume the elements, handle errors, and react to the completion signal.

#### Mono: The Reactive Stream of One

`Mono` is a reactive type that represents a stream of 0 or 1 element. It is similar to `Optional` in Java, but with asynchronous capabilities. A `Mono` can also emit three types of signals:

1. **Next**: Represents the emission of a single element.
2. **Error**: Indicates that an error has occurred, terminating the stream.
3. **Complete**: Signals that the stream has finished emitting the single element.

```java
import reactor.core.publisher.Mono;

public class MonoExample {
    public static void main(String[] args) {
        Mono<String> stringMono = Mono.just("Hello Mono");
        
        stringMono.subscribe(
            System.out::println, // onNext
            error -> System.err.println("Error: " + error), // onError
            () -> System.out.println("Completed!") // onComplete
        );
    }
}
```

In this example, `Mono.just` creates a `Mono` that emits a single string. The `subscribe` method is used to consume the element, handle errors, and react to the completion signal.

### Transforming Data with Flux and Mono

One of the powerful features of `Flux` and `Mono` is their ability to transform data using a variety of operators. These operators allow developers to manipulate the data stream in a declarative manner.

#### Common Operators

- **map**: Transforms each element emitted by the `Flux` or `Mono`.

```java
Flux<Integer> numbers = Flux.just(1, 2, 3, 4);
Flux<Integer> squares = numbers.map(n -> n * n);
```

- **filter**: Filters elements based on a predicate.

```java
Flux<Integer> evenNumbers = numbers.filter(n -> n % 2 == 0);
```

- **flatMap**: Transforms each element into a `Flux` or `Mono` and flattens the results.

```java
Flux<String> words = Flux.just("hello", "world");
Flux<String> letters = words.flatMap(word -> Flux.fromArray(word.split("")));
```

- **reduce**: Aggregates elements into a single value.

```java
Mono<Integer> sum = numbers.reduce(0, Integer::sum);
```

### Handling Errors in Reactive Streams

Error handling is a critical aspect of reactive programming. `Flux` and `Mono` provide several operators for dealing with errors gracefully.

- **onErrorReturn**: Returns a fallback value when an error occurs.

```java
Flux<Integer> numbersWithFallback = numbers.map(n -> {
    if (n == 3) throw new RuntimeException("Error on 3");
    return n;
}).onErrorReturn(-1);
```

- **onErrorResume**: Switches to a different `Flux` or `Mono` when an error occurs.

```java
Flux<Integer> numbersWithFallback = numbers.map(n -> {
    if (n == 3) throw new RuntimeException("Error on 3");
    return n;
}).onErrorResume(e -> Flux.just(0, 0, 0));
```

- **doOnError**: Executes a side-effect when an error occurs.

```java
numbers.doOnError(e -> System.err.println("Error: " + e.getMessage()));
```

### Completion Signals and Subscribing to Streams

Subscribing to a `Flux` or `Mono` is how you consume the data they emit. The `subscribe` method takes up to three arguments: a consumer for the next signal, a consumer for the error signal, and a runnable for the complete signal.

```java
Flux<String> data = Flux.just("A", "B", "C");

data.subscribe(
    item -> System.out.println("Received: " + item),
    error -> System.err.println("Error: " + error),
    () -> System.out.println("Stream completed")
);
```

### Execution Flow and Backpressure

Reactive streams are designed to handle backpressure, which is the ability to manage the rate of data flow between producers and consumers. This is crucial in systems where the producer can generate data faster than the consumer can process it.

#### Backpressure Strategies

- **Buffer**: Accumulates elements in a buffer until the consumer is ready.
- **Drop**: Drops elements when the consumer is overwhelmed.
- **Latest**: Keeps only the latest element, dropping older ones.
- **Error**: Throws an error when overwhelmed.

### Practical Applications and Real-World Scenarios

Reactive programming with `Flux` and `Mono` is particularly useful in scenarios where you need to handle a large number of concurrent connections, such as web servers, real-time data processing, and microservices communication.

#### Example: Building a Reactive Web Service

Consider a web service that fetches data from multiple sources and aggregates the results. Using `Flux` and `Mono`, you can handle each data source asynchronously and combine the results efficiently.

```java
public Mono<ResponseEntity<String>> fetchData() {
    Mono<String> data1 = webClient.get().uri("/service1").retrieve().bodyToMono(String.class);
    Mono<String> data2 = webClient.get().uri("/service2").retrieve().bodyToMono(String.class);

    return Mono.zip(data1, data2)
               .map(tuple -> "Combined Result: " + tuple.getT1() + ", " + tuple.getT2())
               .map(ResponseEntity::ok);
}
```

### Conclusion

Understanding `Flux` and `Mono` is essential for mastering reactive programming in Java. These types provide a powerful abstraction for working with asynchronous data streams, allowing developers to build responsive and resilient applications. By leveraging the operators and error handling mechanisms provided by Project Reactor, you can create systems that are both efficient and robust.

### Key Takeaways

- `Flux` represents a stream of 0 to N elements, while `Mono` represents a stream of 0 or 1 element.
- Use operators like `map`, `filter`, and `flatMap` to transform data streams.
- Handle errors gracefully using operators like `onErrorReturn` and `onErrorResume`.
- Subscribing to a `Flux` or `Mono` allows you to consume the data they emit.
- Reactive streams handle backpressure, ensuring efficient data flow between producers and consumers.

### Encouragement for Further Exploration

Experiment with different operators and error handling strategies to see how they affect the behavior of your reactive streams. Consider how you can apply these concepts to your own projects to improve responsiveness and resilience.

---

## Test Your Knowledge: Flux and Mono in Reactive Java Quiz

{{< quizdown >}}

### What is the primary difference between `Flux` and `Mono` in Project Reactor?

- [x] `Flux` can emit 0 to N elements, while `Mono` can emit 0 or 1 element.
- [ ] `Flux` can only emit 1 element, while `Mono` can emit 0 to N elements.
- [ ] `Flux` is synchronous, while `Mono` is asynchronous.
- [ ] `Flux` handles errors, while `Mono` does not.

> **Explanation:** `Flux` is designed to handle streams of 0 to N elements, whereas `Mono` is used for streams that emit at most one element.

### Which operator would you use to transform each element in a `Flux`?

- [x] map
- [ ] filter
- [ ] flatMap
- [ ] reduce

> **Explanation:** The `map` operator is used to transform each element emitted by a `Flux` or `Mono`.

### How can you handle errors in a `Flux` stream?

- [x] onErrorReturn
- [x] onErrorResume
- [ ] map
- [ ] filter

> **Explanation:** `onErrorReturn` and `onErrorResume` are operators specifically designed to handle errors in reactive streams.

### What signal types can a `Mono` emit?

- [x] Next, Error, Complete
- [ ] Next, Error
- [ ] Complete, Error
- [ ] Next, Complete

> **Explanation:** A `Mono` can emit a Next signal (for the single element), an Error signal, or a Complete signal.

### Which operator would you use to combine results from two `Mono` streams?

- [x] zip
- [ ] map
- [ ] filter
- [ ] reduce

> **Explanation:** The `zip` operator is used to combine results from multiple reactive streams.

### What is backpressure in reactive programming?

- [x] The ability to manage the rate of data flow between producers and consumers.
- [ ] The process of buffering data until the consumer is ready.
- [ ] The mechanism for handling errors in a stream.
- [ ] The strategy for transforming data in a stream.

> **Explanation:** Backpressure is a key concept in reactive programming that deals with managing the rate of data flow to prevent overwhelming the consumer.

### Which backpressure strategy keeps only the latest element?

- [x] Latest
- [ ] Buffer
- [ ] Drop
- [ ] Error

> **Explanation:** The Latest strategy keeps only the most recent element, discarding older ones when the consumer is overwhelmed.

### How do you subscribe to a `Flux` stream?

- [x] Using the `subscribe` method.
- [ ] Using the `map` method.
- [ ] Using the `filter` method.
- [ ] Using the `reduce` method.

> **Explanation:** The `subscribe` method is used to consume the elements emitted by a `Flux` or `Mono`.

### What does the `flatMap` operator do?

- [x] Transforms each element into a `Flux` or `Mono` and flattens the results.
- [ ] Filters elements based on a predicate.
- [ ] Aggregates elements into a single value.
- [ ] Transforms each element into a different type.

> **Explanation:** The `flatMap` operator is used to transform each element into a `Flux` or `Mono` and flatten the results into a single stream.

### Reactive programming is particularly useful for applications that require high concurrency and low latency.

- [x] True
- [ ] False

> **Explanation:** Reactive programming is designed to handle high concurrency and low latency, making it ideal for responsive and resilient applications.

{{< /quizdown >}}

---

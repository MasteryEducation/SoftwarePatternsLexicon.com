---
canonical: "https://softwarepatternslexicon.com/patterns-scala/8/13"
title: "Structured Concurrency in Scala: Managing Concurrency with Cats Effect 3 and ZIO"
description: "Explore structured concurrency in Scala, leveraging Cats Effect 3 and ZIO for efficient concurrency management, improved error handling, and hierarchical lifecycles."
linkTitle: "8.13 Concurrency with Structured Concurrency"
categories:
- Scala
- Concurrency
- Functional Programming
tags:
- Structured Concurrency
- Cats Effect
- ZIO
- Error Handling
- Hierarchical Lifecycles
date: 2024-11-17
type: docs
nav_weight: 9300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 8.13 Concurrency with Structured Concurrency

Concurrency is a fundamental aspect of modern software development, especially in a language like Scala that supports both functional and object-oriented paradigms. As systems grow in complexity, managing concurrency becomes a critical challenge. Traditional concurrency models often lead to issues such as resource leaks, uncontrolled execution, and complex error handling. Structured concurrency offers a solution to these problems by organizing concurrent tasks into a well-defined structure, making them easier to reason about, manage, and debug.

### Understanding Structured Concurrency

Structured concurrency is a programming paradigm that treats concurrent operations as structured blocks of code, similar to how structured programming treats control flow. The main idea is to ensure that concurrent tasks are started and completed within a well-defined scope, which simplifies resource management and error handling.

#### Key Concepts of Structured Concurrency

1. **Hierarchical Lifecycles**: Tasks are organized in a hierarchy, where the lifecycle of child tasks is bound to the parent task. This ensures that resources are properly managed and cleaned up when tasks complete or fail.

2. **Improved Error Handling**: Errors in concurrent tasks are propagated in a structured manner, allowing for more predictable and manageable error handling.

3. **Cancellation Propagation**: Structured concurrency provides a mechanism for propagating cancellation signals through the task hierarchy, ensuring that resources are not leaked and tasks are not left running indefinitely.

4. **Resource Safety**: By structuring concurrent tasks, resources such as threads, memory, and file handles are managed more effectively, reducing the risk of resource leaks.

### Implementing Structured Concurrency with Cats Effect 3

Cats Effect 3 is a powerful library for functional programming in Scala that provides tools for managing concurrency, resource safety, and effectful computations. It introduces structured concurrency through its `IO` monad, which allows developers to define and manage concurrent tasks in a functional style.

#### Key Features of Cats Effect 3

- **Fiber-based Concurrency**: Cats Effect 3 uses fibers, lightweight threads managed by the runtime, to execute concurrent tasks. Fibers are cheaper to create and manage than native threads, allowing for high concurrency levels.

- **Resource Management**: The `Resource` data type in Cats Effect 3 ensures that resources are acquired and released safely, even in the presence of errors or cancellations.

- **Error Handling**: Cats Effect 3 provides powerful error handling mechanisms, allowing developers to compose and manage errors in a predictable way.

#### Example: Using Cats Effect 3 for Structured Concurrency

Let's explore a simple example of structured concurrency using Cats Effect 3. We'll create a program that performs two concurrent tasks: fetching data from a remote API and processing the data.

```scala
import cats.effect.{IO, IOApp}
import cats.effect.std.Console
import scala.concurrent.duration._

object StructuredConcurrencyExample extends IOApp.Simple {

  // Simulate a remote API call
  def fetchData: IO[String] = IO.sleep(2.seconds) *> IO.pure("Data from API")

  // Simulate data processing
  def processData(data: String): IO[Unit] = IO.sleep(1.second) *> IO(println(s"Processed: $data"))

  val program: IO[Unit] = for {
    fiber1 <- fetchData.start // Start fetching data concurrently
    fiber2 <- fiber1.join.flatMap(processData).start // Process data after fetching
    _ <- fiber2.join // Wait for processing to complete
  } yield ()

  override def run: IO[Unit] = program
}
```

In this example, we use the `start` method to run tasks concurrently as fibers. The `join` method waits for a fiber to complete, ensuring that resources are managed correctly.

### Implementing Structured Concurrency with ZIO

ZIO is another popular library for functional programming in Scala that provides a comprehensive framework for managing concurrency, resources, and effects. ZIO's approach to structured concurrency is similar to Cats Effect 3, but it offers additional features such as environment management and more granular control over concurrency.

#### Key Features of ZIO

- **Effectful Programming**: ZIO provides a powerful effect system that allows developers to model side effects in a pure functional way.

- **Environment Management**: ZIO's environment system allows for dependency injection and configuration management, making it easier to build modular and testable applications.

- **Concurrency Primitives**: ZIO provides a rich set of concurrency primitives, including fibers, queues, and semaphores, for building complex concurrent systems.

#### Example: Using ZIO for Structured Concurrency

Let's look at a similar example using ZIO to perform concurrent tasks.

```scala
import zio._
import zio.console._
import zio.duration._

object ZioStructuredConcurrencyExample extends App {

  // Simulate a remote API call
  def fetchData: ZIO[Any, Nothing, String] = ZIO.sleep(2.seconds) *> ZIO.succeed("Data from API")

  // Simulate data processing
  def processData(data: String): ZIO[Console, Nothing, Unit] = ZIO.sleep(1.second) *> putStrLn(s"Processed: $data")

  val program: ZIO[Console, Nothing, Unit] = for {
    fiber1 <- fetchData.fork // Start fetching data concurrently
    fiber2 <- fiber1.join.flatMap(processData).fork // Process data after fetching
    _ <- fiber2.join // Wait for processing to complete
  } yield ()

  override def run(args: List[String]): URIO[ZEnv, ExitCode] = program.exitCode
}
```

In this ZIO example, we use the `fork` method to run tasks concurrently as fibers. The `join` method is used to wait for a fiber to complete, similar to Cats Effect 3.

### Managing Concurrency with Hierarchical Lifecycles

One of the key benefits of structured concurrency is the ability to manage concurrency with hierarchical lifecycles. This means that the lifecycle of a child task is bound to its parent task, ensuring that resources are properly managed and cleaned up.

#### Hierarchical Lifecycles in Cats Effect 3

In Cats Effect 3, hierarchical lifecycles are managed using the `Resource` data type. Resources are acquired and released in a structured way, ensuring that they are cleaned up even if an error occurs.

```scala
import cats.effect.{IO, Resource}
import scala.concurrent.duration._

def useResource: Resource[IO, Unit] = {
  val acquire = IO(println("Acquiring resource"))
  val release = IO(println("Releasing resource"))

  Resource.make(acquire)(_ => release)
}

val program: IO[Unit] = useResource.use { _ =>
  IO.sleep(1.second) *> IO(println("Using resource"))
}

program.unsafeRunSync()
```

In this example, the `Resource` data type ensures that the resource is acquired and released correctly, even if an error occurs during its use.

#### Hierarchical Lifecycles in ZIO

ZIO provides similar functionality through its `ZManaged` data type, which ensures that resources are managed safely.

```scala
import zio._
import zio.console._

def useResource: ZManaged[Console, Nothing, Unit] = {
  val acquire = putStrLn("Acquiring resource")
  val release = putStrLn("Releasing resource")

  ZManaged.make(acquire)(_ => release)
}

val program: ZIO[Console, Nothing, Unit] = useResource.use { _ =>
  ZIO.sleep(1.second) *> putStrLn("Using resource")
}

program.exitCode
```

In this ZIO example, `ZManaged` ensures that the resource is acquired and released correctly, similar to Cats Effect 3's `Resource`.

### Benefits Over Traditional Concurrency Models

Structured concurrency offers several benefits over traditional concurrency models:

1. **Predictable Resource Management**: By structuring concurrent tasks, resources are managed more predictably, reducing the risk of leaks and ensuring that resources are cleaned up properly.

2. **Simplified Error Handling**: Errors are propagated in a structured manner, making it easier to handle them predictably and consistently.

3. **Improved Cancellation**: Cancellation signals are propagated through the task hierarchy, ensuring that tasks are not left running indefinitely and that resources are not leaked.

4. **Easier Reasoning**: Structured concurrency makes it easier to reason about concurrent tasks, as they are organized in a well-defined structure.

### Improved Cancellation and Error Handling

One of the key advantages of structured concurrency is improved cancellation and error handling. By organizing tasks into a hierarchy, cancellation signals can be propagated through the task hierarchy, ensuring that tasks are not left running indefinitely and that resources are not leaked.

#### Cancellation in Cats Effect 3

Cats Effect 3 provides a mechanism for propagating cancellation signals through the task hierarchy, ensuring that tasks are not left running indefinitely and that resources are not leaked.

```scala
import cats.effect.{IO, IOApp}
import scala.concurrent.duration._

object CancellationExample extends IOApp.Simple {

  def task: IO[Unit] = IO.sleep(2.seconds) *> IO(println("Task completed"))

  val program: IO[Unit] = for {
    fiber <- task.start
    _ <- IO.sleep(1.second) *> fiber.cancel // Cancel the task after 1 second
  } yield ()

  override def run: IO[Unit] = program
}
```

In this example, the task is cancelled after 1 second, demonstrating how cancellation signals can be propagated through the task hierarchy.

#### Cancellation in ZIO

ZIO provides similar functionality for propagating cancellation signals through the task hierarchy.

```scala
import zio._
import zio.console._
import zio.duration._

object ZioCancellationExample extends App {

  def task: ZIO[Console, Nothing, Unit] = ZIO.sleep(2.seconds) *> putStrLn("Task completed")

  val program: ZIO[Console, Nothing, Unit] = for {
    fiber <- task.fork
    _ <- ZIO.sleep(1.second) *> fiber.interrupt // Cancel the task after 1 second
  } yield ()

  override def run(args: List[String]): URIO[ZEnv, ExitCode] = program.exitCode
}
```

In this ZIO example, the task is cancelled after 1 second, similar to the Cats Effect 3 example.

### Case Studies

To illustrate the practical benefits of structured concurrency, let's explore a couple of case studies that demonstrate how structured concurrency can be applied in real-world scenarios.

#### Case Study 1: Web Server with Structured Concurrency

In this case study, we'll explore how structured concurrency can be used to build a web server that handles multiple requests concurrently while ensuring that resources are managed safely and errors are handled predictably.

```scala
import cats.effect.{IO, IOApp}
import cats.effect.std.Console
import scala.concurrent.duration._

object WebServerExample extends IOApp.Simple {

  def handleRequest(request: String): IO[Unit] = IO.sleep(1.second) *> IO(println(s"Handled request: $request"))

  val server: IO[Unit] = for {
    fiber1 <- handleRequest("Request 1").start
    fiber2 <- handleRequest("Request 2").start
    _ <- fiber1.join
    _ <- fiber2.join
  } yield ()

  override def run: IO[Unit] = server
}
```

In this example, the web server handles multiple requests concurrently using structured concurrency, ensuring that resources are managed safely and errors are handled predictably.

#### Case Study 2: Data Processing Pipeline with Structured Concurrency

In this case study, we'll explore how structured concurrency can be used to build a data processing pipeline that processes data concurrently while ensuring that resources are managed safely and errors are handled predictably.

```scala
import zio._
import zio.console._
import zio.duration._

object DataProcessingPipelineExample extends App {

  def fetchData: ZIO[Any, Nothing, String] = ZIO.sleep(1.second) *> ZIO.succeed("Data")

  def processData(data: String): ZIO[Console, Nothing, Unit] = ZIO.sleep(1.second) *> putStrLn(s"Processed: $data")

  val pipeline: ZIO[Console, Nothing, Unit] = for {
    data <- fetchData
    _ <- processData(data).fork
  } yield ()

  override def run(args: List[String]): URIO[ZEnv, ExitCode] = pipeline.exitCode
}
```

In this example, the data processing pipeline processes data concurrently using structured concurrency, ensuring that resources are managed safely and errors are handled predictably.

### Try It Yourself

Now that we've explored the concepts and examples of structured concurrency, it's time to try it yourself. Here are a few suggestions for experimenting with the code examples:

1. **Modify the Task Duration**: Change the duration of the tasks in the examples to see how it affects the execution and cancellation behavior.

2. **Add More Concurrent Tasks**: Add more concurrent tasks to the examples to see how structured concurrency manages them.

3. **Introduce Errors**: Introduce errors into the tasks to see how structured concurrency handles error propagation and resource management.

4. **Experiment with Resource Management**: Use the `Resource` and `ZManaged` data types to manage resources in the examples, ensuring that they are acquired and released safely.

### Knowledge Check

Before we conclude, let's summarize the key takeaways from this section:

- Structured concurrency organizes concurrent tasks into a well-defined structure, making them easier to reason about, manage, and debug.

- Cats Effect 3 and ZIO provide powerful tools for implementing structured concurrency in Scala, offering features such as hierarchical lifecycles, improved error handling, and cancellation propagation.

- Structured concurrency offers several benefits over traditional concurrency models, including predictable resource management, simplified error handling, and improved cancellation.

- By experimenting with the code examples and trying out different scenarios, you can gain a deeper understanding of how structured concurrency works and how it can be applied in real-world applications.

Remember, this is just the beginning. As you progress, you'll build more complex and interactive systems using structured concurrency. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is structured concurrency?

- [x] A programming paradigm that organizes concurrent tasks into a well-defined structure.
- [ ] A method for optimizing database queries.
- [ ] A design pattern for building user interfaces.
- [ ] A technique for improving network latency.

> **Explanation:** Structured concurrency is a programming paradigm that treats concurrent operations as structured blocks of code, ensuring that tasks are started and completed within a well-defined scope.

### Which library provides structured concurrency in Scala?

- [x] Cats Effect 3
- [x] ZIO
- [ ] Akka
- [ ] Play Framework

> **Explanation:** Cats Effect 3 and ZIO are libraries that provide structured concurrency in Scala, offering tools for managing concurrency, resource safety, and effectful computations.

### What is a key benefit of structured concurrency?

- [x] Predictable resource management
- [ ] Faster execution speed
- [ ] Reduced code complexity
- [ ] Improved user interface design

> **Explanation:** Structured concurrency offers predictable resource management by organizing concurrent tasks into a well-defined structure, ensuring that resources are managed safely.

### How does Cats Effect 3 manage concurrency?

- [x] Using fibers
- [ ] Using threads
- [ ] Using processes
- [ ] Using coroutines

> **Explanation:** Cats Effect 3 uses fibers, which are lightweight threads managed by the runtime, to execute concurrent tasks.

### What is the purpose of the `Resource` data type in Cats Effect 3?

- [x] To ensure resources are acquired and released safely
- [ ] To optimize database queries
- [ ] To manage user sessions
- [ ] To improve network latency

> **Explanation:** The `Resource` data type in Cats Effect 3 ensures that resources are acquired and released safely, even in the presence of errors or cancellations.

### How does ZIO handle concurrency?

- [x] Using fibers
- [ ] Using threads
- [ ] Using processes
- [ ] Using coroutines

> **Explanation:** ZIO handles concurrency using fibers, similar to Cats Effect 3, allowing for lightweight and efficient concurrent execution.

### What is a key feature of ZIO?

- [x] Environment management
- [ ] Faster execution speed
- [ ] Reduced code complexity
- [ ] Improved user interface design

> **Explanation:** ZIO provides environment management, which allows for dependency injection and configuration management, making it easier to build modular and testable applications.

### How does structured concurrency improve error handling?

- [x] By propagating errors in a structured manner
- [ ] By ignoring errors
- [ ] By logging errors to a file
- [ ] By displaying error messages to users

> **Explanation:** Structured concurrency improves error handling by propagating errors in a structured manner, making it easier to handle them predictably and consistently.

### What is a common use case for structured concurrency?

- [x] Building web servers
- [ ] Designing user interfaces
- [ ] Optimizing database queries
- [ ] Improving network latency

> **Explanation:** Structured concurrency is commonly used in building web servers, where it helps manage multiple requests concurrently while ensuring that resources are managed safely and errors are handled predictably.

### True or False: Structured concurrency can help prevent resource leaks.

- [x] True
- [ ] False

> **Explanation:** True. Structured concurrency helps prevent resource leaks by organizing concurrent tasks into a well-defined structure, ensuring that resources are managed safely and cleaned up properly.

{{< /quizdown >}}

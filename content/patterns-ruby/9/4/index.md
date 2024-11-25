---
canonical: "https://softwarepatternslexicon.com/patterns-ruby/9/4"
title: "Ractors in Ruby 3: Achieving True Parallelism"
description: "Explore Ractors in Ruby 3, a groundbreaking feature for achieving true parallelism by running code across multiple CPU cores while avoiding thread safety issues."
linkTitle: "9.4 Ractors in Ruby 3 for Parallelism"
categories:
- Ruby
- Concurrency
- Parallelism
tags:
- Ractors
- Ruby 3
- Parallelism
- Concurrency
- Multithreading
date: 2024-11-23
type: docs
nav_weight: 94000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 9.4 Ractors in Ruby 3 for Parallelism

Ruby 3 introduced a powerful new feature called Ractors, designed to enable true parallelism by running code across multiple CPU cores. This section will delve into the concept of Ractors, their purpose, and how they differ from traditional concurrency models like Threads and Fibers. We'll explore how Ractors enable parallel execution by isolating objects, provide practical examples, and discuss the implications of using Ractors in your Ruby applications.

### Understanding Ractors

#### Definition and Purpose

Ractors, short for "Ruby Actors," are a concurrency abstraction introduced in Ruby 3 to achieve parallel execution. Unlike Threads, which share memory space and require careful synchronization to avoid race conditions, Ractors provide a model where each Ractor has its own isolated memory space. This isolation helps avoid thread safety issues, making it easier to write concurrent programs that can run in parallel on multiple CPU cores.

#### How Ractors Enable Parallel Execution

Ractors achieve parallel execution by isolating objects. Each Ractor has its own heap, and objects cannot be shared directly between Ractors. Instead, communication between Ractors is done through message passing, ensuring that each Ractor operates independently without interfering with others.

### Creating and Communicating Between Ractors

Let's explore how to create Ractors and facilitate communication between them.

#### Creating a Ractor

Creating a Ractor is straightforward. You use the `Ractor.new` method, passing a block of code that the Ractor will execute. Here's a simple example:

```ruby
# Create a Ractor that calculates the sum of an array
ractor = Ractor.new([1, 2, 3, 4, 5]) do |numbers|
  numbers.sum
end

# Retrieve the result from the Ractor
result = ractor.take
puts "Sum: #{result}"  # Output: Sum: 15
```

In this example, we create a Ractor that calculates the sum of an array. The `take` method is used to retrieve the result from the Ractor.

#### Communicating Between Ractors

Communication between Ractors is done through message passing. You can send messages to a Ractor using the `send` method and receive messages using the `receive` method. Here's an example:

```ruby
# Create a Ractor that receives a message and prints it
receiver = Ractor.new do
  message = Ractor.receive
  puts "Received: #{message}"
end

# Send a message to the receiver Ractor
receiver.send("Hello, Ractor!")

# Output: Received: Hello, Ractor!
```

In this example, we create a Ractor that waits for a message and prints it. We then send a message to the Ractor using the `send` method.

### Immutability and Object Sharing Rules

Ractors enforce immutability and strict object sharing rules to ensure safe parallel execution. Here's what you need to know:

- **Immutable Objects**: Immutable objects, such as numbers and symbols, can be shared between Ractors without restrictions.
- **Shareable Objects**: Objects that are explicitly marked as shareable can be shared between Ractors. You can mark an object as shareable using the `Ractor.make_shareable` method.
- **Copying Objects**: Non-shareable objects are copied when sent between Ractors, ensuring that each Ractor has its own independent copy.

Here's an example demonstrating these concepts:

```ruby
# Create a Ractor that processes a shareable object
ractor = Ractor.new do
  obj = Ractor.receive
  puts "Processing: #{obj}"
end

# Create a shareable object
shareable_obj = Ractor.make_shareable({ key: "value" })

# Send the shareable object to the Ractor
ractor.send(shareable_obj)

# Output: Processing: {:key=>"value"}
```

In this example, we create a shareable object and send it to a Ractor for processing.

### Ractors vs. Threads and Fibers

Ractors differ from Threads and Fibers in several key ways:

- **Isolation**: Ractors provide isolated memory spaces, reducing the risk of race conditions and making it easier to write safe concurrent code.
- **Parallelism**: Ractors enable true parallelism by running on multiple CPU cores, whereas Threads are limited by the Global Interpreter Lock (GIL) in MRI Ruby.
- **Communication**: Ractors communicate through message passing, while Threads share memory space and require synchronization mechanisms.

### Practical Considerations and Performance Implications

When using Ractors, consider the following:

- **Overhead**: Ractors introduce some overhead due to message passing and object isolation. Use Ractors when the benefits of parallelism outweigh this overhead.
- **Design**: Design your application to leverage Ractors effectively, focusing on tasks that can be parallelized and benefit from isolated execution.
- **Performance**: Measure and profile your application to ensure that Ractors provide the desired performance improvements.

### Limitations and Potential Pitfalls

While Ractors offer significant advantages, they also have limitations:

- **Complexity**: Designing applications with Ractors can be complex, especially when coordinating multiple Ractors.
- **Debugging**: Debugging Ractor-based applications can be challenging due to the isolated nature of Ractors.
- **Compatibility**: Not all Ruby libraries and gems are Ractor-compatible. Ensure that your dependencies support Ractors before adopting them.

### Conclusion

Ractors in Ruby 3 provide a powerful tool for achieving true parallelism, enabling developers to write concurrent applications that run efficiently on multiple CPU cores. By understanding the concepts of isolation, message passing, and immutability, you can leverage Ractors to build scalable and maintainable applications. Remember to consider the practical implications and limitations of Ractors when designing your applications.

For more information, refer to the official [Ruby Ractors documentation](https://docs.ruby-lang.org/en/master/NEWS_ruby-3_0_0.html#label-Ractor+-Class+for+Parallel+Execution).

### Try It Yourself

Experiment with Ractors by modifying the code examples provided. Try creating multiple Ractors that perform different tasks and communicate with each other. Observe how Ractors handle parallel execution and message passing.

## Quiz: Ractors in Ruby 3 for Parallelism

{{< quizdown >}}

### What is the primary purpose of Ractors in Ruby 3?

- [x] To achieve true parallelism by running code across multiple CPU cores
- [ ] To replace Threads and Fibers entirely
- [ ] To simplify the Ruby syntax
- [ ] To improve memory management

> **Explanation:** Ractors are designed to enable true parallelism by allowing code to run on multiple CPU cores.

### How do Ractors communicate with each other?

- [x] Through message passing
- [ ] By sharing memory space
- [ ] Using global variables
- [ ] Through direct method calls

> **Explanation:** Ractors communicate through message passing, ensuring isolation between their memory spaces.

### Which of the following objects can be shared between Ractors without restrictions?

- [x] Immutable objects
- [ ] Mutable objects
- [ ] Arrays
- [ ] Hashes

> **Explanation:** Immutable objects, such as numbers and symbols, can be shared between Ractors without restrictions.

### What method is used to mark an object as shareable between Ractors?

- [x] `Ractor.make_shareable`
- [ ] `Ractor.share`
- [ ] `Ractor.allow`
- [ ] `Ractor.enable`

> **Explanation:** The `Ractor.make_shareable` method is used to mark an object as shareable between Ractors.

### What is a key difference between Ractors and Threads?

- [x] Ractors provide isolated memory spaces
- [ ] Threads are faster than Ractors
- [ ] Ractors use global variables for communication
- [ ] Threads do not require synchronization

> **Explanation:** Ractors provide isolated memory spaces, reducing the risk of race conditions.

### What is a potential drawback of using Ractors?

- [x] Increased complexity in application design
- [ ] Reduced performance compared to Threads
- [ ] Lack of support for parallel execution
- [ ] Incompatibility with all Ruby versions

> **Explanation:** Designing applications with Ractors can be complex due to the need for coordination and message passing.

### Which Ruby feature limits Threads from achieving true parallelism?

- [x] Global Interpreter Lock (GIL)
- [ ] Lack of message passing
- [ ] Absence of isolation
- [ ] Limited CPU core access

> **Explanation:** The Global Interpreter Lock (GIL) in MRI Ruby limits Threads from achieving true parallelism.

### What should you consider when using Ractors in your application?

- [x] The overhead introduced by message passing
- [ ] The need to replace all Threads with Ractors
- [ ] The requirement to use global variables
- [ ] The necessity to disable the GIL

> **Explanation:** Ractors introduce overhead due to message passing, so it's important to consider this when designing applications.

### True or False: Ractors can run on multiple CPU cores simultaneously.

- [x] True
- [ ] False

> **Explanation:** Ractors are designed to run on multiple CPU cores simultaneously, enabling true parallelism.

### What is a common challenge when debugging Ractor-based applications?

- [x] The isolated nature of Ractors
- [ ] The lack of message passing
- [ ] The absence of error messages
- [ ] The need for global variables

> **Explanation:** Debugging Ractor-based applications can be challenging due to the isolated nature of Ractors.

{{< /quizdown >}}

Remember, this is just the beginning. As you progress, you'll build more complex and interactive applications using Ractors. Keep experimenting, stay curious, and enjoy the journey!

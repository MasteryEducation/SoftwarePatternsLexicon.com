---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/8/1"
title: "Understanding Creational Patterns in Erlang: Leveraging Functional Paradigms"
description: "Explore how Erlang's functional and concurrent nature influences the implementation of creational design patterns, transforming traditional object-oriented approaches into functional solutions."
linkTitle: "8.1 Understanding Creational Patterns in Erlang"
categories:
- Erlang
- Functional Programming
- Design Patterns
tags:
- Creational Patterns
- Erlang
- Functional Design
- Software Architecture
- Concurrency
date: 2024-11-23
type: docs
nav_weight: 81000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 8.1 Understanding Creational Patterns in Erlang

In the world of software design, creational patterns play a pivotal role in managing object creation mechanisms. These patterns provide various ways to create objects while hiding the creation logic, making the system more flexible and reusable. However, when we transition from object-oriented languages to a functional language like Erlang, the approach to these patterns changes significantly. Let's delve into how Erlang's unique features influence the implementation of creational patterns and explore the patterns covered in this section.

### The Role of Creational Patterns in Software Design

Creational design patterns abstract the instantiation process, allowing systems to be more independent of how their objects are created, composed, and represented. In traditional object-oriented programming (OOP), these patterns help manage the complexity of creating objects, especially when dealing with complex hierarchies or when the instantiation process involves multiple steps.

#### Key Objectives of Creational Patterns:

- **Encapsulation of Object Creation**: Hide the instantiation logic to reduce dependencies.
- **Flexibility and Reusability**: Allow systems to use different strategies for object creation.
- **Scalability**: Facilitate the creation of complex objects without modifying existing code.

### Erlang's Influence on Creational Patterns

Erlang, being a functional language, does not have objects in the traditional sense. Instead, it focuses on functions and immutable data. This paradigm shift means that creational patterns in Erlang are more about managing the creation and initialization of data structures and processes rather than objects.

#### Erlang's Unique Features:

1. **Immutability**: Once a data structure is created, it cannot be changed. This leads to patterns that focus on creating new data structures rather than modifying existing ones.
2. **Processes as Primary Units**: Erlang uses lightweight processes for concurrency, which can be leveraged in creational patterns to manage state and behavior.
3. **Functional Composition**: Functions are first-class citizens, allowing for higher-order functions and function composition to play a role in pattern implementation.
4. **Pattern Matching**: Provides a powerful mechanism to destructure and match data, influencing how patterns are implemented.

### Overview of Creational Patterns in Erlang

In this section, we will explore several creational patterns adapted for Erlang's functional paradigm. These include:

- **Factory Pattern**: Using functions and modules to encapsulate the creation logic.
- **Builder Pattern**: Leveraging functional approaches to construct complex data structures.
- **Singleton Pattern**: Managing application-wide state using Erlang's process model.
- **Prototype Pattern**: Utilizing process cloning to create new instances.
- **Dependency Injection**: Achieving flexibility through parameters and behaviors.
- **Registry Pattern**: Using Erlang processes to manage global state.
- **Dynamic Module Loading**: Facilitating code updates and polymorphism through behaviors.

### Rethinking Object Creation in Erlang

In Erlang, the concept of object creation is transformed into creating and managing processes and data structures. This requires a shift in mindset from traditional OOP approaches. Instead of focusing on classes and objects, we focus on functions, modules, and processes.

#### Key Considerations:

- **Process-Based State Management**: Use processes to encapsulate state and behavior, akin to objects in OOP.
- **Functional Data Structures**: Embrace immutability and leverage functional data structures for state representation.
- **Higher-Order Functions**: Use functions to encapsulate creation logic and enable flexible composition.

### Factory Pattern in Erlang

The Factory Pattern in Erlang involves using functions and modules to encapsulate the creation logic. This pattern is particularly useful when you need to create instances of complex data structures or processes.

#### Example:

```erlang
-module(factory_example).
-export([create_process/1]).

create_process(Type) ->
    case Type of
        worker -> spawn(fun worker_process/0);
        manager -> spawn(fun manager_process/0)
    end.

worker_process() ->
    receive
        {work, Task} -> io:format("Working on ~p~n", [Task]);
        stop -> exit(normal)
    end.

manager_process() ->
    receive
        {assign, Task} -> io:format("Assigning task ~p~n", [Task]);
        stop -> exit(normal)
    end.
```

In this example, the `create_process/1` function acts as a factory, creating different types of processes based on the input type.

### Builder Pattern Using Functional Approaches

The Builder Pattern is used to construct complex data structures step by step. In Erlang, this can be achieved using functional composition and higher-order functions.

#### Example:

```erlang
-module(builder_example).
-export([build_car/1]).

build_car(Options) ->
    Car = #{},
    Car1 = add_engine(Car, Options),
    Car2 = add_wheels(Car1, Options),
    add_paint(Car2, Options).

add_engine(Car, Options) ->
    Engine = proplists:get_value(engine, Options, default_engine),
    maps:put(engine, Engine, Car).

add_wheels(Car, Options) ->
    Wheels = proplists:get_value(wheels, Options, default_wheels),
    maps:put(wheels, Wheels, Car).

add_paint(Car, Options) ->
    Paint = proplists:get_value(paint, Options, default_paint),
    maps:put(paint, Paint, Car).
```

Here, the `build_car/1` function uses helper functions to add components to a car, demonstrating a step-by-step construction process.

### Singleton Pattern and Application Environment

The Singleton Pattern ensures that a particular resource or configuration is shared across the application. In Erlang, this can be achieved using a process to manage the state.

#### Example:

```erlang
-module(singleton_example).
-export([start/0, get_config/0, set_config/1]).

start() ->
    register(singleton, spawn(fun() -> loop(#{}))).

loop(Config) ->
    receive
        {get, Caller} ->
            Caller ! Config,
            loop(Config);
        {set, NewConfig} ->
            loop(NewConfig)
    end.

get_config() ->
    singleton ! {get, self()},
    receive
        Config -> Config
    end.

set_config(NewConfig) ->
    singleton ! {set, NewConfig}.
```

In this example, a registered process named `singleton` is used to manage configuration, ensuring a single point of access.

### Prototype Pattern through Process Cloning

The Prototype Pattern involves creating new instances by copying existing ones. In Erlang, this can be achieved by cloning processes.

#### Example:

```erlang
-module(prototype_example).
-export([clone_process/1, process_loop/0]).

clone_process(OriginalPid) ->
    spawn(fun() -> process_loop(OriginalPid) end).

process_loop(OriginalPid) ->
    receive
        {clone, Caller} ->
            NewPid = clone_process(self()),
            Caller ! {cloned, NewPid};
        stop -> exit(normal)
    end.
```

Here, the `clone_process/1` function creates a new process by copying the behavior of an existing process.

### Dependency Injection via Parameters and Behaviors

Dependency Injection in Erlang can be achieved by passing dependencies as parameters or using behaviors to define interfaces.

#### Example:

```erlang
-module(di_example).
-export([start/1]).

start(Logger) ->
    Logger:log("Starting application").

-module(console_logger).
-export([log/1]).

log(Message) ->
    io:format("Console: ~s~n", [Message]).

-module(file_logger).
-export([log/1]).

log(Message) ->
    {ok, File} = file:open("log.txt", [write]),
    io:format(File, "File: ~s~n", [Message]),
    file:close(File).
```

In this example, the `start/1` function accepts a logger module, allowing for flexible logging strategies.

### Registry Pattern with Erlang Processes

The Registry Pattern involves managing global state or resources. In Erlang, this can be implemented using processes to register and manage resources.

#### Example:

```erlang
-module(registry_example).
-export([register/2, lookup/1]).

register(Name, Pid) ->
    register(Name, Pid).

lookup(Name) ->
    whereis(Name).
```

This simple example demonstrates how to register and look up processes by name, providing a global registry.

### Dynamic Module Loading and Code Updates

Erlang's hot code swapping feature allows for dynamic module loading and updates, facilitating polymorphism and flexibility.

#### Example:

```erlang
-module(dynamic_example).
-export([load_module/1, call_function/2]).

load_module(Module) ->
    code:load_file(Module).

call_function(Module, Function) ->
    apply(Module, Function, []).
```

This example demonstrates loading a module at runtime and calling a function, showcasing Erlang's dynamic capabilities.

### Erlang Unique Features and Considerations

Erlang's functional and concurrent nature provides unique opportunities and challenges when implementing creational patterns. Key considerations include:

- **Concurrency**: Leverage processes for managing state and behavior.
- **Immutability**: Focus on creating new data structures rather than modifying existing ones.
- **Pattern Matching**: Use pattern matching to simplify and clarify creation logic.

### Differences and Similarities with OOP Patterns

While the core intent of creational patterns remains the same, their implementation in Erlang differs due to the language's functional paradigm. Key differences include:

- **No Classes or Objects**: Focus on functions, modules, and processes.
- **State Management**: Use processes to encapsulate state instead of objects.
- **Function Composition**: Utilize higher-order functions for flexible creation logic.

### Try It Yourself

Experiment with the provided code examples by modifying the creation logic or adding new features. For instance, try adding a new type of process in the Factory Pattern example or implement a new logging strategy in the Dependency Injection example.

### Knowledge Check

- How does Erlang's immutability influence creational patterns?
- What role do processes play in managing state in Erlang?
- How can pattern matching simplify creation logic in Erlang?

### Embrace the Journey

Remember, understanding creational patterns in Erlang is just the beginning. As you continue exploring, you'll discover more ways to leverage Erlang's unique features to build robust and scalable applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Understanding Creational Patterns in Erlang

{{< quizdown >}}

### What is a key objective of creational patterns in software design?

- [x] Encapsulation of object creation
- [ ] Direct modification of objects
- [ ] Increasing code complexity
- [ ] Reducing code readability

> **Explanation:** Creational patterns aim to encapsulate the instantiation process, making systems more flexible and independent of object creation logic.

### How does Erlang's immutability affect creational patterns?

- [x] It focuses on creating new data structures
- [ ] It allows direct modification of existing data
- [ ] It complicates the creation process
- [ ] It has no effect on creational patterns

> **Explanation:** Immutability in Erlang means that once data is created, it cannot be changed, leading to a focus on creating new data structures.

### What is a primary unit of concurrency in Erlang?

- [x] Processes
- [ ] Threads
- [ ] Classes
- [ ] Objects

> **Explanation:** Erlang uses lightweight processes as the primary unit of concurrency, enabling efficient state management and behavior encapsulation.

### Which pattern involves using functions and modules to encapsulate creation logic in Erlang?

- [x] Factory Pattern
- [ ] Singleton Pattern
- [ ] Prototype Pattern
- [ ] Builder Pattern

> **Explanation:** The Factory Pattern in Erlang uses functions and modules to encapsulate the creation logic, allowing for flexible and reusable code.

### How can dependency injection be achieved in Erlang?

- [x] Via parameters and behaviors
- [ ] By modifying global variables
- [ ] Through direct object manipulation
- [ ] Using inheritance

> **Explanation:** Dependency injection in Erlang can be achieved by passing dependencies as parameters or using behaviors to define interfaces.

### What is a key difference between creational patterns in OOP and Erlang?

- [x] Erlang focuses on functions and processes, not objects
- [ ] Erlang uses classes and objects
- [ ] Erlang does not support creational patterns
- [ ] Erlang relies on inheritance for pattern implementation

> **Explanation:** Erlang's functional paradigm focuses on functions, modules, and processes rather than classes and objects, influencing how creational patterns are implemented.

### Which pattern ensures a single point of access to a resource in Erlang?

- [x] Singleton Pattern
- [ ] Factory Pattern
- [ ] Builder Pattern
- [ ] Prototype Pattern

> **Explanation:** The Singleton Pattern in Erlang uses a process to manage a shared resource, ensuring a single point of access.

### What feature allows for dynamic module loading and code updates in Erlang?

- [x] Hot code swapping
- [ ] Static typing
- [ ] Object-oriented inheritance
- [ ] Global variables

> **Explanation:** Erlang's hot code swapping feature allows for dynamic module loading and updates, facilitating flexibility and polymorphism.

### How does the Prototype Pattern work in Erlang?

- [x] By cloning processes
- [ ] By modifying existing objects
- [ ] Through class inheritance
- [ ] Using global variables

> **Explanation:** The Prototype Pattern in Erlang involves creating new instances by cloning existing processes, leveraging Erlang's process model.

### True or False: Erlang's pattern matching can simplify creation logic.

- [x] True
- [ ] False

> **Explanation:** Pattern matching in Erlang provides a powerful mechanism to destructure and match data, simplifying creation logic and enhancing code clarity.

{{< /quizdown >}}

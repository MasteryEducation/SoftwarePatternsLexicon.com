---
canonical: "https://softwarepatternslexicon.com/patterns-java/32/3"
title: "Design Pattern Reference Cheat Sheet"
description: "Comprehensive guide to Java design patterns with intent, structure, and key components for quick reference."
linkTitle: "32.3 Design Pattern Reference Cheat Sheet"
tags:
- "Java"
- "Design Patterns"
- "Creational Patterns"
- "Structural Patterns"
- "Behavioral Patterns"
- "Software Architecture"
- "UML Diagrams"
- "Software Design"
date: 2024-11-25
type: docs
nav_weight: 323000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 32.3 Design Pattern Reference Cheat Sheet

This section provides a comprehensive reference sheet summarizing key design patterns discussed in the guide. It includes their intent, structure, key participants, and example use cases. The patterns are organized by category: Creational, Structural, and Behavioral. This cheat sheet is designed to be a quick reference for experienced Java developers and software architects.

### Creational Patterns

Creational patterns deal with object creation mechanisms, trying to create objects in a manner suitable to the situation. They help make a system independent of how its objects are created, composed, and represented.

#### Singleton Pattern

- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Key Participants**: Singleton class.
- **Structure**:

    ```mermaid
    classDiagram
        class Singleton {
            -Singleton instance
            -Singleton()
            +getInstance(): Singleton
        }
    ```

    - **Caption**: The Singleton class diagram shows a single instance managed internally.

- **Example Use Cases**: Logging, Configuration settings, Thread pools.

#### Factory Method Pattern

- **Intent**: Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- **Key Participants**: Creator, ConcreteCreator, Product, ConcreteProduct.
- **Structure**:

    ```mermaid
    classDiagram
        class Creator {
            +factoryMethod(): Product
        }
        class ConcreteCreator {
            +factoryMethod(): ConcreteProduct
        }
        class Product
        class ConcreteProduct
        Creator <|-- ConcreteCreator
        Product <|-- ConcreteProduct
        Creator --> Product
        ConcreteCreator --> ConcreteProduct
    ```

    - **Caption**: The Factory Method pattern structure showing the relationship between Creator and Product.

- **Example Use Cases**: GUI libraries, Document generation.

#### Abstract Factory Pattern

- **Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- **Key Participants**: AbstractFactory, ConcreteFactory, AbstractProduct, ConcreteProduct.
- **Structure**:

    ```mermaid
    classDiagram
        class AbstractFactory {
            +createProductA(): AbstractProductA
            +createProductB(): AbstractProductB
        }
        class ConcreteFactory1 {
            +createProductA(): ProductA1
            +createProductB(): ProductB1
        }
        class ConcreteFactory2 {
            +createProductA(): ProductA2
            +createProductB(): ProductB2
        }
        class AbstractProductA
        class AbstractProductB
        class ProductA1
        class ProductA2
        class ProductB1
        class ProductB2
        AbstractFactory <|-- ConcreteFactory1
        AbstractFactory <|-- ConcreteFactory2
        AbstractProductA <|-- ProductA1
        AbstractProductA <|-- ProductA2
        AbstractProductB <|-- ProductB1
        AbstractProductB <|-- ProductB2
        ConcreteFactory1 --> ProductA1
        ConcreteFactory1 --> ProductB1
        ConcreteFactory2 --> ProductA2
        ConcreteFactory2 --> ProductB2
    ```

    - **Caption**: The Abstract Factory pattern structure illustrating the creation of product families.

- **Example Use Cases**: UI toolkits, Cross-platform support.

### Structural Patterns

Structural patterns ease the design by identifying a simple way to realize relationships between entities.

#### Adapter Pattern

- **Intent**: Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.
- **Key Participants**: Target, Adapter, Adaptee, Client.
- **Structure**:

    ```mermaid
    classDiagram
        class Target {
            +request()
        }
        class Adapter {
            +request()
        }
        class Adaptee {
            +specificRequest()
        }
        Target <|-- Adapter
        Adapter --> Adaptee
    ```

    - **Caption**: The Adapter pattern structure showing how Adapter translates requests from Target to Adaptee.

- **Example Use Cases**: Legacy code integration, Third-party library adaptation.

#### Composite Pattern

- **Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
- **Key Participants**: Component, Leaf, Composite.
- **Structure**:

    ```mermaid
    classDiagram
        class Component {
            +operation()
        }
        class Leaf {
            +operation()
        }
        class Composite {
            +add(Component)
            +remove(Component)
            +operation()
        }
        Component <|-- Leaf
        Component <|-- Composite
        Composite --> Component
    ```

    - **Caption**: The Composite pattern structure showing the hierarchical composition of objects.

- **Example Use Cases**: File systems, UI components.

#### Proxy Pattern

- **Intent**: Provide a surrogate or placeholder for another object to control access to it.
- **Key Participants**: Proxy, RealSubject, Subject.
- **Structure**:

    ```mermaid
    classDiagram
        class Subject {
            +request()
        }
        class RealSubject {
            +request()
        }
        class Proxy {
            +request()
        }
        Subject <|-- RealSubject
        Subject <|-- Proxy
        Proxy --> RealSubject
    ```

    - **Caption**: The Proxy pattern structure showing the Proxy controlling access to RealSubject.

- **Example Use Cases**: Virtual proxies, Access control, Caching.

### Behavioral Patterns

Behavioral patterns are concerned with algorithms and the assignment of responsibilities between objects.

#### Observer Pattern

- **Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- **Key Participants**: Subject, Observer, ConcreteSubject, ConcreteObserver.
- **Structure**:

    ```mermaid
    classDiagram
        class Subject {
            +attach(Observer)
            +detach(Observer)
            +notify()
        }
        class Observer {
            +update()
        }
        class ConcreteSubject {
            -state
            +getState()
            +setState()
        }
        class ConcreteObserver {
            +update()
        }
        Subject <|-- ConcreteSubject
        Observer <|-- ConcreteObserver
        ConcreteSubject --> Observer
    ```

    - **Caption**: The Observer pattern structure showing the relationship between Subject and Observers.

- **Example Use Cases**: Event handling systems, Model-View-Controller (MVC) architectures.

#### Strategy Pattern

- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
- **Key Participants**: Strategy, ConcreteStrategy, Context.
- **Structure**:

    ```mermaid
    classDiagram
        class Context {
            +setStrategy(Strategy)
            +executeStrategy()
        }
        class Strategy {
            +execute()
        }
        class ConcreteStrategyA {
            +execute()
        }
        class ConcreteStrategyB {
            +execute()
        }
        Strategy <|-- ConcreteStrategyA
        Strategy <|-- ConcreteStrategyB
        Context --> Strategy
    ```

    - **Caption**: The Strategy pattern structure showing how Context uses different strategies.

- **Example Use Cases**: Sorting algorithms, Payment processing systems.

#### Command Pattern

- **Intent**: Encapsulate a request as an object, thereby letting you parameterize clients with queues, requests, and operations.
- **Key Participants**: Command, ConcreteCommand, Invoker, Receiver.
- **Structure**:

    ```mermaid
    classDiagram
        class Command {
            +execute()
        }
        class ConcreteCommand {
            +execute()
        }
        class Invoker {
            +setCommand(Command)
            +executeCommand()
        }
        class Receiver {
            +action()
        }
        Command <|-- ConcreteCommand
        Invoker --> Command
        ConcreteCommand --> Receiver
    ```

    - **Caption**: The Command pattern structure showing how commands are executed by Invoker.

- **Example Use Cases**: Transactional systems, Undo/Redo functionality.

### Conclusion

This cheat sheet provides a quick reference to some of the most commonly used design patterns in Java. Each pattern is presented with its intent, key participants, and a structural diagram to help you understand its application in real-world scenarios. For more detailed explanations and code examples, refer to the respective sections within this guide.

## Test Your Knowledge: Java Design Patterns Quiz

{{< quizdown >}}

### What is the primary intent of the Singleton pattern?

- [x] To ensure a class has only one instance and provide a global point of access to it.
- [ ] To create a family of related objects.
- [ ] To define a one-to-many dependency between objects.
- [ ] To encapsulate a request as an object.

> **Explanation:** The Singleton pattern ensures a class has only one instance and provides a global point of access to it.

### Which pattern is used to convert the interface of a class into another interface clients expect?

- [x] Adapter Pattern
- [ ] Proxy Pattern
- [ ] Composite Pattern
- [ ] Observer Pattern

> **Explanation:** The Adapter pattern is used to convert the interface of a class into another interface clients expect.

### In the Factory Method pattern, what is the role of the Creator?

- [x] To define an interface for creating an object.
- [ ] To compose objects into tree structures.
- [ ] To provide a surrogate for another object.
- [ ] To encapsulate a request as an object.

> **Explanation:** In the Factory Method pattern, the Creator defines an interface for creating an object.

### What is the main benefit of using the Composite pattern?

- [x] To compose objects into tree structures to represent part-whole hierarchies.
- [ ] To ensure a class has only one instance.
- [ ] To encapsulate a request as an object.
- [ ] To define a family of algorithms.

> **Explanation:** The Composite pattern allows you to compose objects into tree structures to represent part-whole hierarchies.

### Which pattern is best suited for implementing undo/redo functionality?

- [x] Command Pattern
- [ ] Strategy Pattern
- [x] Observer Pattern
- [ ] Singleton Pattern

> **Explanation:** The Command pattern is best suited for implementing undo/redo functionality as it encapsulates requests as objects.

### What is the primary purpose of the Abstract Factory pattern?

- [x] To provide an interface for creating families of related or dependent objects.
- [ ] To define a one-to-many dependency between objects.
- [ ] To convert the interface of a class into another interface.
- [ ] To encapsulate a request as an object.

> **Explanation:** The Abstract Factory pattern provides an interface for creating families of related or dependent objects.

### Which pattern allows the algorithm to vary independently from clients that use it?

- [x] Strategy Pattern
- [ ] Command Pattern
- [x] Observer Pattern
- [ ] Singleton Pattern

> **Explanation:** The Strategy pattern allows the algorithm to vary independently from clients that use it.

### What is the role of the Proxy in the Proxy pattern?

- [x] To control access to another object.
- [ ] To define a family of algorithms.
- [ ] To compose objects into tree structures.
- [ ] To encapsulate a request as an object.

> **Explanation:** The Proxy controls access to another object, acting as a surrogate or placeholder.

### Which pattern is used to define a one-to-many dependency between objects?

- [x] Observer Pattern
- [ ] Adapter Pattern
- [ ] Composite Pattern
- [ ] Strategy Pattern

> **Explanation:** The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

### True or False: The Factory Method pattern is used to ensure a class has only one instance.

- [ ] True
- [x] False

> **Explanation:** False. The Singleton pattern is used to ensure a class has only one instance, not the Factory Method pattern.

{{< /quizdown >}}

---
canonical: "https://softwarepatternslexicon.com/patterns-cpp/23/3"
title: "Comprehensive C++ Design Patterns Reference Cheat Sheet"
description: "Explore a detailed reference guide to C++ design patterns, including intent, applicability, and key features for expert developers."
linkTitle: "23.3 Pattern Reference Cheat Sheet"
categories:
- C++ Design Patterns
- Software Architecture
- Programming Best Practices
tags:
- C++
- Design Patterns
- Software Development
- Architecture
- Programming
date: 2024-11-17
type: docs
nav_weight: 23300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 23.3 Pattern Reference Cheat Sheet

Welcome to the ultimate reference guide for mastering C++ design patterns. This cheat sheet provides a concise overview of each pattern, including its intent, applicability, and key features. Whether you're an expert software engineer or an architect, this guide will serve as a valuable resource for building robust, scalable, and maintainable C++ applications.

### Creational Patterns

#### Singleton Pattern
- **Category**: Creational
- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Singleton {
          -static Singleton* instance
          -Singleton()
          +static Singleton* getInstance()
      }
  ```
- **Key Participants**: Singleton
- **Applicability**: Use when exactly one instance of a class is needed to control actions across the system.
- **Design Considerations**: 
  - Ensure thread safety in multithreaded environments.
  - Use Meyers' Singleton for simplicity and thread safety.
  - Example:
    ```cpp
    class Singleton {
    public:
        static Singleton& getInstance() {
            static Singleton instance;
            return instance;
        }
    private:
        Singleton() {}
    };
    ```
- **Differences and Similarities**: Often confused with the Multiton pattern, which allows multiple instances but controls their creation.

#### Factory Method Pattern
- **Category**: Creational
- **Intent**: Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Product {
          <<interface>>
      }
      class ConcreteProduct
      class Creator {
          <<interface>>
          +factoryMethod() Product
      }
      class ConcreteCreator
      Product <|-- ConcreteProduct
      Creator <|-- ConcreteCreator
      ConcreteCreator --> ConcreteProduct
  ```
- **Key Participants**: Creator, ConcreteCreator, Product, ConcreteProduct
- **Applicability**: Use when a class can't anticipate the class of objects it must create.
- **Design Considerations**: 
  - Provides flexibility in terms of object creation.
  - Can lead to a proliferation of classes.
  - Example:
    ```cpp
    class Product {
    public:
        virtual void use() = 0;
    };

    class ConcreteProduct : public Product {
    public:
        void use() override { /* implementation */ }
    };

    class Creator {
    public:
        virtual Product* factoryMethod() = 0;
    };

    class ConcreteCreator : public Creator {
    public:
        Product* factoryMethod() override { return new ConcreteProduct(); }
    };
    ```
- **Differences and Similarities**: Similar to Abstract Factory, but focuses on creating a single product.

#### Builder Pattern
- **Category**: Creational
- **Intent**: Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Builder {
          <<interface>>
          +buildPart()
      }
      class ConcreteBuilder
      class Director {
          +construct()
      }
      class Product
      Builder <|-- ConcreteBuilder
      Director --> Builder
      ConcreteBuilder --> Product
  ```
- **Key Participants**: Builder, ConcreteBuilder, Director, Product
- **Applicability**: Use when the construction process must allow different representations for the object that's constructed.
- **Design Considerations**: 
  - Useful for constructing objects with many optional parts.
  - Can be combined with the Composite pattern for building complex trees.
  - Example:
    ```cpp
    class Product {
    public:
        void addPart(const std::string& part) { /* add part */ }
    };

    class Builder {
    public:
        virtual void buildPart() = 0;
        virtual Product* getResult() = 0;
    };

    class ConcreteBuilder : public Builder {
    private:
        Product* product;
    public:
        ConcreteBuilder() { product = new Product(); }
        void buildPart() override { product->addPart("PartA"); }
        Product* getResult() override { return product; }
    };

    class Director {
    public:
        void construct(Builder* builder) {
            builder->buildPart();
        }
    };
    ```
- **Differences and Similarities**: Often compared with Factory Method, but Builder focuses on step-by-step construction.

#### Prototype Pattern
- **Category**: Creational
- **Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Prototype {
          <<interface>>
          +clone() Prototype
      }
      class ConcretePrototype
      Prototype <|-- ConcretePrototype
  ```
- **Key Participants**: Prototype, ConcretePrototype
- **Applicability**: Use when the cost of creating a new instance of a class is more expensive than copying an existing instance.
- **Design Considerations**: 
  - Useful for creating objects that are costly to create from scratch.
  - Ensure deep copies when necessary.
  - Example:
    ```cpp
    class Prototype {
    public:
        virtual Prototype* clone() const = 0;
    };

    class ConcretePrototype : public Prototype {
    public:
        Prototype* clone() const override { return new ConcretePrototype(*this); }
    };
    ```
- **Differences and Similarities**: Similar to the Factory Method, but Prototype uses cloning instead of instantiation.

### Structural Patterns

#### Adapter Pattern
- **Category**: Structural
- **Intent**: Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Target {
          <<interface>>
          +request()
      }
      class Adapter
      class Adaptee {
          +specificRequest()
      }
      Target <|-- Adapter
      Adapter --> Adaptee
  ```
- **Key Participants**: Target, Adapter, Adaptee
- **Applicability**: Use when you want to use an existing class, and its interface does not match the one you need.
- **Design Considerations**: 
  - Can be implemented using multiple inheritance in C++.
  - Consider using object composition over class inheritance.
  - Example:
    ```cpp
    class Target {
    public:
        virtual void request() = 0;
    };

    class Adaptee {
    public:
        void specificRequest() { /* implementation */ }
    };

    class Adapter : public Target {
    private:
        Adaptee* adaptee;
    public:
        Adapter(Adaptee* a) : adaptee(a) {}
        void request() override { adaptee->specificRequest(); }
    };
    ```
- **Differences and Similarities**: Often confused with the Bridge pattern, but Adapter focuses on interface compatibility.

#### Bridge Pattern
- **Category**: Structural
- **Intent**: Decouple an abstraction from its implementation so that the two can vary independently.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Abstraction {
          +operation()
      }
      class RefinedAbstraction
      class Implementor {
          <<interface>>
          +implementation()
      }
      class ConcreteImplementor
      Abstraction <|-- RefinedAbstraction
      Abstraction --> Implementor
      Implementor <|-- ConcreteImplementor
  ```
- **Key Participants**: Abstraction, RefinedAbstraction, Implementor, ConcreteImplementor
- **Applicability**: Use when you want to separate an abstraction from its implementation.
- **Design Considerations**: 
  - Useful for avoiding a permanent binding between an abstraction and its implementation.
  - Example:
    ```cpp
    class Implementor {
    public:
        virtual void implementation() = 0;
    };

    class ConcreteImplementor : public Implementor {
    public:
        void implementation() override { /* implementation */ }
    };

    class Abstraction {
    protected:
        Implementor* implementor;
    public:
        Abstraction(Implementor* impl) : implementor(impl) {}
        virtual void operation() { implementor->implementation(); }
    };

    class RefinedAbstraction : public Abstraction {
    public:
        RefinedAbstraction(Implementor* impl) : Abstraction(impl) {}
        void operation() override { /* refined operation */ }
    };
    ```
- **Differences and Similarities**: Similar to Adapter, but Bridge focuses on separating abstraction from implementation.

#### Composite Pattern
- **Category**: Structural
- **Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Component {
          <<interface>>
          +operation()
      }
      class Leaf
      class Composite
      Component <|-- Leaf
      Component <|-- Composite
      Composite --> Component
  ```
- **Key Participants**: Component, Leaf, Composite
- **Applicability**: Use when you want to represent part-whole hierarchies of objects.
- **Design Considerations**: 
  - Simplifies client code by treating individual objects and compositions uniformly.
  - Example:
    ```cpp
    class Component {
    public:
        virtual void operation() = 0;
    };

    class Leaf : public Component {
    public:
        void operation() override { /* implementation */ }
    };

    class Composite : public Component {
    private:
        std::vector<Component*> children;
    public:
        void operation() override {
            for (auto child : children) {
                child->operation();
            }
        }
        void add(Component* component) { children.push_back(component); }
    };
    ```
- **Differences and Similarities**: Often used with the Decorator pattern to add responsibilities to objects.

### Behavioral Patterns

#### Observer Pattern
- **Category**: Behavioral
- **Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Subject {
          +attach(Observer)
          +detach(Observer)
          +notify()
      }
      class Observer {
          <<interface>>
          +update()
      }
      class ConcreteObserver
      Subject --> Observer
      Observer <|-- ConcreteObserver
  ```
- **Key Participants**: Subject, Observer, ConcreteObserver
- **Applicability**: Use when a change to one object requires changing others, and you don't know how many objects need to be changed.
- **Design Considerations**: 
  - Can lead to memory leaks if observers are not properly detached.
  - Example:
    ```cpp
    class Observer {
    public:
        virtual void update() = 0;
    };

    class Subject {
    private:
        std::vector<Observer*> observers;
    public:
        void attach(Observer* observer) { observers.push_back(observer); }
        void detach(Observer* observer) { /* remove observer */ }
        void notify() {
            for (auto observer : observers) {
                observer->update();
            }
        }
    };

    class ConcreteObserver : public Observer {
    public:
        void update() override { /* update logic */ }
    };
    ```
- **Differences and Similarities**: Similar to the Mediator pattern, but Observer focuses on state changes.

#### Strategy Pattern
- **Category**: Behavioral
- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Context {
          +setStrategy(Strategy)
          +executeStrategy()
      }
      class Strategy {
          <<interface>>
          +algorithmInterface()
      }
      class ConcreteStrategyA
      class ConcreteStrategyB
      Context --> Strategy
      Strategy <|-- ConcreteStrategyA
      Strategy <|-- ConcreteStrategyB
  ```
- **Key Participants**: Context, Strategy, ConcreteStrategyA, ConcreteStrategyB
- **Applicability**: Use when you want to define a class that has one behavior that's similar to other behaviors in a list.
- **Design Considerations**: 
  - Avoids the use of conditional statements.
  - Example:
    ```cpp
    class Strategy {
    public:
        virtual void algorithmInterface() = 0;
    };

    class ConcreteStrategyA : public Strategy {
    public:
        void algorithmInterface() override { /* implementation A */ }
    };

    class ConcreteStrategyB : public Strategy {
    public:
        void algorithmInterface() override { /* implementation B */ }
    };

    class Context {
    private:
        Strategy* strategy;
    public:
        void setStrategy(Strategy* s) { strategy = s; }
        void executeStrategy() { strategy->algorithmInterface(); }
    };
    ```
- **Differences and Similarities**: Often compared with the State pattern, but Strategy focuses on interchangeable algorithms.

#### Command Pattern
- **Category**: Behavioral
- **Intent**: Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Command {
          <<interface>>
          +execute()
      }
      class ConcreteCommand
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
- **Key Participants**: Command, ConcreteCommand, Invoker, Receiver
- **Applicability**: Use when you want to parameterize objects with operations.
- **Design Considerations**: 
  - Supports undo/redo functionality.
  - Example:
    ```cpp
    class Command {
    public:
        virtual void execute() = 0;
    };

    class Receiver {
    public:
        void action() { /* perform action */ }
    };

    class ConcreteCommand : public Command {
    private:
        Receiver* receiver;
    public:
        ConcreteCommand(Receiver* r) : receiver(r) {}
        void execute() override { receiver->action(); }
    };

    class Invoker {
    private:
        Command* command;
    public:
        void setCommand(Command* c) { command = c; }
        void executeCommand() { command->execute(); }
    };
    ```
- **Differences and Similarities**: Similar to the Strategy pattern, but Command focuses on encapsulating requests.

### Concurrency Patterns

#### Active Object Pattern
- **Category**: Concurrency
- **Intent**: Decouple method execution from method invocation for objects that each reside in their own thread of control.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class ActiveObject {
          +method()
      }
      class Scheduler {
          +enqueue()
          +dispatch()
      }
      class Proxy
      ActiveObject --> Scheduler
      Proxy --> ActiveObject
  ```
- **Key Participants**: ActiveObject, Scheduler, Proxy
- **Applicability**: Use when you need to separate method invocation from execution.
- **Design Considerations**: 
  - Useful for improving performance in concurrent systems.
  - Example:
    ```cpp
    class ActiveObject {
    public:
        void method() { /* asynchronous execution */ }
    };

    class Scheduler {
    public:
        void enqueue(ActiveObject* obj) { /* add to queue */ }
        void dispatch() { /* execute from queue */ }
    };

    class Proxy {
    private:
        ActiveObject* activeObject;
    public:
        Proxy(ActiveObject* ao) : activeObject(ao) {}
        void method() { activeObject->method(); }
    };
    ```
- **Differences and Similarities**: Often compared with the Proxy pattern, but Active Object focuses on concurrency.

#### Monitor Object Pattern
- **Category**: Concurrency
- **Intent**: Synchronize method execution to ensure that only one method at a time runs within an object.
- **Structure Diagram**:
  ```mermaid
  classDiagram
      class Monitor {
          +synchronizedMethod()
      }
  ```
- **Key Participants**: Monitor
- **Applicability**: Use when you need to control access to an object's methods in a multithreaded environment.
- **Design Considerations**: 
  - Encapsulates synchronization within the object.
  - Example:
    ```cpp
    class Monitor {
    private:
        std::mutex mtx;
    public:
        void synchronizedMethod() {
            std::lock_guard<std::mutex> lock(mtx);
            // critical section
        }
    };
    ```
- **Differences and Similarities**: Similar to the Singleton pattern in terms of encapsulating control, but Monitor focuses on synchronization.

### Functional Patterns

#### Lambda Expressions
- **Category**: Functional
- **Intent**: Provide a concise way to represent anonymous functions.
- **Structure Diagram**: Not applicable
- **Key Participants**: Lambda, Capture List
- **Applicability**: Use when you need to define short, inline functions that can be passed as arguments.
- **Design Considerations**: 
  - Captures variables by value or reference.
  - Example:
    ```cpp
    auto add = [](int a, int b) { return a + b; };
    int result = add(5, 3); // result is 8
    ```
- **Differences and Similarities**: Similar to function pointers but more flexible and type-safe.

#### Currying and Partial Application
- **Category**: Functional
- **Intent**: Transform a function that takes multiple arguments into a sequence of functions that each take a single argument.
- **Structure Diagram**: Not applicable
- **Key Participants**: Curried Function
- **Applicability**: Use when you need to fix some arguments of a function and generate a new function.
- **Design Considerations**: 
  - Useful for creating specialized functions.
  - Example:
    ```cpp
    auto add = [](int a, int b) { return a + b; };
    auto addFive = std::bind(add, 5, std::placeholders::_1);
    int result = addFive(3); // result is 8
    ```
- **Differences and Similarities**: Similar to partial application but focuses on transforming functions.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Singleton pattern?

- [x] Ensure a class has only one instance and provide a global point of access to it.
- [ ] Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- [ ] Separate the construction of a complex object from its representation.
- [ ] Specify the kinds of objects to create using a prototypical instance.

> **Explanation:** The Singleton pattern ensures that a class has only one instance and provides a global point of access to it.

### Which pattern is used to convert the interface of a class into another interface clients expect?

- [ ] Bridge
- [x] Adapter
- [ ] Composite
- [ ] Proxy

> **Explanation:** The Adapter pattern is used to convert the interface of a class into another interface that clients expect.

### What is the key difference between the Strategy and State patterns?

- [x] Strategy focuses on interchangeable algorithms, while State focuses on changing behavior based on state.
- [ ] Strategy is used for concurrency, while State is used for synchronization.
- [ ] Strategy uses inheritance, while State uses composition.
- [ ] Strategy is a creational pattern, while State is a structural pattern.

> **Explanation:** The Strategy pattern focuses on interchangeable algorithms, while the State pattern focuses on changing behavior based on state.

### Which pattern is often used to implement undo/redo functionality?

- [ ] Observer
- [ ] Strategy
- [x] Command
- [ ] Visitor

> **Explanation:** The Command pattern is often used to implement undo/redo functionality by encapsulating requests as objects.

### What is the main purpose of the Bridge pattern?

- [x] Decouple an abstraction from its implementation so that the two can vary independently.
- [ ] Compose objects into tree structures to represent part-whole hierarchies.
- [ ] Convert the interface of a class into another interface clients expect.
- [ ] Define a family of algorithms, encapsulate each one, and make them interchangeable.

> **Explanation:** The Bridge pattern is used to decouple an abstraction from its implementation so that the two can vary independently.

### In which pattern do you define a one-to-many dependency between objects?

- [ ] Strategy
- [x] Observer
- [ ] Command
- [ ] State

> **Explanation:** The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

### Which pattern is used to separate method execution from method invocation for objects that each reside in their own thread of control?

- [x] Active Object
- [ ] Proxy
- [ ] Command
- [ ] Strategy

> **Explanation:** The Active Object pattern is used to separate method execution from method invocation for objects that each reside in their own thread of control.

### What is the primary intent of the Prototype pattern?

- [ ] Ensure a class has only one instance and provide a global point of access to it.
- [ ] Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- [ ] Separate the construction of a complex object from its representation.
- [x] Specify the kinds of objects to create using a prototypical instance.

> **Explanation:** The Prototype pattern specifies the kinds of objects to create using a prototypical instance and creates new objects by copying this prototype.

### Which pattern is often confused with the Adapter pattern but focuses on separating abstraction from implementation?

- [x] Bridge
- [ ] Composite
- [ ] Proxy
- [ ] Decorator

> **Explanation:** The Bridge pattern is often confused with the Adapter pattern but focuses on separating abstraction from implementation.

### The Command pattern is primarily used for:

- [x] Encapsulating a request as an object.
- [ ] Defining a family of algorithms.
- [ ] Composing objects into tree structures.
- [ ] Decoupling an abstraction from its implementation.

> **Explanation:** The Command pattern is used for encapsulating a request as an object, allowing for parameterization and queuing of requests.

{{< /quizdown >}}

Remember, mastering design patterns is a journey. As you continue to explore and apply these patterns, you'll find new ways to enhance your software architecture and design skills. Keep experimenting, stay curious, and enjoy the process!

---
canonical: "https://softwarepatternslexicon.com/object-oriented/10/1"

title: "Glossary of Object-Oriented Design Patterns and Concepts"
description: "Comprehensive Glossary of Key Object-Oriented Design Patterns and Concepts with Pseudocode Examples"
linkTitle: "10.1. Glossary of Terms"
categories:
- Object-Oriented Design
- Design Patterns
- Software Development
tags:
- OOP
- Design Patterns
- Glossary
- Software Architecture
- Pseudocode
date: 2024-11-17
type: docs
nav_weight: 10100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 10.1. Glossary of Terms

Welcome to the Glossary of Terms for Object-Oriented Design Patterns. This section serves as a comprehensive reference for key concepts and patterns in object-oriented programming (OOP). Whether you're a seasoned developer or new to design patterns, this glossary will provide clear definitions, illustrative examples, and insightful explanations to enhance your understanding.

### Object-Oriented Programming (OOP)

**Definition:** A programming paradigm based on the concept of "objects," which can contain data and code to manipulate that data. Key principles include encapsulation, inheritance, polymorphism, and abstraction.

**Example:** In OOP, a `Car` object might have properties like `color` and `speed`, and methods like `accelerate()` and `brake()`.

### Encapsulation

**Definition:** The bundling of data with the methods that operate on that data. Encapsulation restricts direct access to some of an object's components, which can prevent the accidental modification of data.

**Example:** A class `BankAccount` might encapsulate the balance and provide methods like `deposit()` and `withdraw()` to modify it.

```pseudocode
class BankAccount
    private balance

    method deposit(amount)
        balance = balance + amount

    method withdraw(amount)
        if amount <= balance
            balance = balance - amount
```

### Inheritance

**Definition:** A mechanism where a new class is derived from an existing class. The new class inherits attributes and behaviors (methods) from the parent class.

**Example:** A `Truck` class might inherit from a `Vehicle` class, gaining its properties and methods while adding additional features.

```pseudocode
class Vehicle
    method startEngine()

class Truck inherits Vehicle
    method loadCargo()
```

### Polymorphism

**Definition:** The ability of different classes to be treated as instances of the same class through a common interface. It allows methods to do different things based on the object it is acting upon.

**Example:** A `Shape` interface with a method `draw()` can be implemented by `Circle` and `Square` classes, each providing a specific implementation of `draw()`.

```pseudocode
interface Shape
    method draw()

class Circle implements Shape
    method draw()
        // Draw circle

class Square implements Shape
    method draw()
        // Draw square
```

### Abstraction

**Definition:** The concept of hiding the complex reality while exposing only the necessary parts. It helps in reducing programming complexity and effort.

**Example:** An abstract class `Animal` might define an abstract method `makeSound()`, which is implemented by subclasses like `Dog` and `Cat`.

```pseudocode
abstract class Animal
    abstract method makeSound()

class Dog extends Animal
    method makeSound()
        // Bark

class Cat extends Animal
    method makeSound()
        // Meow
```

### Design Pattern

**Definition:** A general repeatable solution to a commonly occurring problem in software design. Patterns are templates designed to help write code that is easy to understand and reuse.

**Example:** The Singleton pattern ensures a class has only one instance and provides a global point of access to it.

### Singleton Pattern

**Category:** Creational

**Intent:** Ensure a class has only one instance and provide a global point of access to it.

**Key Participants:** Singleton class

**Applicability:** Use when exactly one instance of a class is needed to coordinate actions across the system.

**Sample Code Snippet:**

```pseudocode
class Singleton
    private static instance

    private Singleton()

    static method getInstance()
        if instance is null
            instance = new Singleton()
        return instance
```

### Factory Method Pattern

**Category:** Creational

**Intent:** Define an interface for creating an object, but let subclasses alter the type of objects that will be created.

**Key Participants:** Creator, ConcreteCreator

**Applicability:** Use when a class can't anticipate the class of objects it must create.

**Sample Code Snippet:**

```pseudocode
abstract class Creator
    abstract method factoryMethod()

class ConcreteCreator extends Creator
    method factoryMethod()
        return new ConcreteProduct()
```

### Abstract Factory Pattern

**Category:** Creational

**Intent:** Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

**Key Participants:** AbstractFactory, ConcreteFactory

**Applicability:** Use when a system should be independent of how its products are created.

**Sample Code Snippet:**

```pseudocode
interface AbstractFactory
    method createProductA()
    method createProductB()

class ConcreteFactory1 implements AbstractFactory
    method createProductA()
        return new ProductA1()

    method createProductB()
        return new ProductB1()
```

### Builder Pattern

**Category:** Creational

**Intent:** Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.

**Key Participants:** Builder, ConcreteBuilder, Director

**Applicability:** Use when the construction process must allow different representations for the object that is constructed.

**Sample Code Snippet:**

```pseudocode
class Director
    method construct(builder)
        builder.buildPartA()
        builder.buildPartB()

interface Builder
    method buildPartA()
    method buildPartB()

class ConcreteBuilder implements Builder
    method buildPartA()
        // Build part A

    method buildPartB()
        // Build part B
```

### Prototype Pattern

**Category:** Creational

**Intent:** Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

**Key Participants:** Prototype, ConcretePrototype

**Applicability:** Use when a system should be independent of how its products are created, composed, and represented.

**Sample Code Snippet:**

```pseudocode
interface Prototype
    method clone()

class ConcretePrototype implements Prototype
    method clone()
        return new ConcretePrototype(this)
```

### Adapter Pattern

**Category:** Structural

**Intent:** Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.

**Key Participants:** Adapter, Adaptee, Target

**Applicability:** Use when you want to use an existing class, and its interface does not match the one you need.

**Sample Code Snippet:**

```pseudocode
class Adapter implements Target
    private adaptee

    method request()
        adaptee.specificRequest()
```

### Bridge Pattern

**Category:** Structural

**Intent:** Decouple an abstraction from its implementation so that the two can vary independently.

**Key Participants:** Abstraction, Implementor

**Applicability:** Use when you want to avoid a permanent binding between an abstraction and its implementation.

**Sample Code Snippet:**

```pseudocode
class Abstraction
    private implementor

    method operation()
        implementor.operationImpl()

interface Implementor
    method operationImpl()
```

### Composite Pattern

**Category:** Structural

**Intent:** Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.

**Key Participants:** Component, Composite, Leaf

**Applicability:** Use when you want to represent part-whole hierarchies of objects.

**Sample Code Snippet:**

```pseudocode
interface Component
    method operation()

class Leaf implements Component
    method operation()
        // Leaf operation

class Composite implements Component
    private children

    method operation()
        for each child in children
            child.operation()
```

### Decorator Pattern

**Category:** Structural

**Intent:** Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**Key Participants:** Component, Decorator

**Applicability:** Use to add responsibilities to individual objects dynamically and transparently, without affecting other objects.

**Sample Code Snippet:**

```pseudocode
interface Component
    method operation()

class ConcreteComponent implements Component
    method operation()
        // Concrete operation

class Decorator implements Component
    private component

    method operation()
        component.operation()
        // Additional behavior
```

### Facade Pattern

**Category:** Structural

**Intent:** Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.

**Key Participants:** Facade

**Applicability:** Use when you want to provide a simple interface to a complex subsystem.

**Sample Code Snippet:**

```pseudocode
class Facade
    private subsystem1
    private subsystem2

    method operation()
        subsystem1.operation1()
        subsystem2.operation2()
```

### Flyweight Pattern

**Category:** Structural

**Intent:** Use sharing to support large numbers of fine-grained objects efficiently.

**Key Participants:** Flyweight, FlyweightFactory

**Applicability:** Use when many objects must be manipulated and storage costs are high.

**Sample Code Snippet:**

```pseudocode
class Flyweight
    method operation(extrinsicState)

class FlyweightFactory
    private flyweights

    method getFlyweight(key)
        if flyweights[key] is null
            flyweights[key] = new ConcreteFlyweight()
        return flyweights[key]
```

### Proxy Pattern

**Category:** Structural

**Intent:** Provide a surrogate or placeholder for another object to control access to it.

**Key Participants:** Proxy, RealSubject

**Applicability:** Use when you need a more versatile or sophisticated reference to an object than a simple pointer.

**Sample Code Snippet:**

```pseudocode
class Proxy implements Subject
    private realSubject

    method request()
        if realSubject is null
            realSubject = new RealSubject()
        realSubject.request()
```

### Chain of Responsibility Pattern

**Category:** Behavioral

**Intent:** Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.

**Key Participants:** Handler

**Applicability:** Use when more than one object may handle a request, and the handler isn't known a priori.

**Sample Code Snippet:**

```pseudocode
class Handler
    private nextHandler

    method handleRequest(request)
        if canHandle(request)
            // Handle request
        else if nextHandler is not null
            nextHandler.handleRequest(request)
```

### Command Pattern

**Category:** Behavioral

**Intent:** Encapsulate a request as an object, thereby letting you parameterize clients with queues, requests, and operations.

**Key Participants:** Command, Invoker, Receiver

**Applicability:** Use when you want to parameterize objects with operations.

**Sample Code Snippet:**

```pseudocode
interface Command
    method execute()

class ConcreteCommand implements Command
    private receiver

    method execute()
        receiver.action()
```

### Interpreter Pattern

**Category:** Behavioral

**Intent:** Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language.

**Key Participants:** AbstractExpression, TerminalExpression, NonTerminalExpression

**Applicability:** Use when you want to interpret sentences in a language.

**Sample Code Snippet:**

```pseudocode
abstract class AbstractExpression
    method interpret(context)

class TerminalExpression extends AbstractExpression
    method interpret(context)
        // Interpret terminal

class NonTerminalExpression extends AbstractExpression
    method interpret(context)
        // Interpret non-terminal
```

### Iterator Pattern

**Category:** Behavioral

**Intent:** Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

**Key Participants:** Iterator, Aggregate

**Applicability:** Use to access an aggregate object's contents without exposing its internal structure.

**Sample Code Snippet:**

```pseudocode
interface Iterator
    method hasNext()
    method next()

interface Aggregate
    method createIterator()

class ConcreteAggregate implements Aggregate
    method createIterator()
        return new ConcreteIterator(this)
```

### Mediator Pattern

**Category:** Behavioral

**Intent:** Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly.

**Key Participants:** Mediator, Colleague

**Applicability:** Use when you want to reduce communication complexity between multiple objects.

**Sample Code Snippet:**

```pseudocode
interface Mediator
    method notify(sender, event)

class ConcreteMediator implements Mediator
    method notify(sender, event)
        // Handle event
```

### Memento Pattern

**Category:** Behavioral

**Intent:** Without violating encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later.

**Key Participants:** Memento, Originator, Caretaker

**Applicability:** Use when you want to save and restore the state of an object.

**Sample Code Snippet:**

```pseudocode
class Memento
    private state

class Originator
    method saveToMemento()
        return new Memento(state)

    method restoreFromMemento(memento)
        state = memento.getState()
```

### Observer Pattern

**Category:** Behavioral

**Intent:** Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**Key Participants:** Subject, Observer

**Applicability:** Use when an abstraction has two aspects, one dependent on the other.

**Sample Code Snippet:**

```pseudocode
interface Observer
    method update()

class Subject
    private observers

    method attach(observer)
        observers.add(observer)

    method notifyObservers()
        for each observer in observers
            observer.update()
```

### State Pattern

**Category:** Behavioral

**Intent:** Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.

**Key Participants:** Context, State

**Applicability:** Use when an object's behavior depends on its state, and it must change its behavior at runtime depending on that state.

**Sample Code Snippet:**

```pseudocode
interface State
    method handle(context)

class Context
    private state

    method setState(state)
        this.state = state

    method request()
        state.handle(this)
```

### Strategy Pattern

**Category:** Behavioral

**Intent:** Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**Key Participants:** Strategy, Context

**Applicability:** Use when you want to use different variants of an algorithm.

**Sample Code Snippet:**

```pseudocode
interface Strategy
    method execute()

class Context
    private strategy

    method setStrategy(strategy)
        this.strategy = strategy

    method executeStrategy()
        strategy.execute()
```

### Template Method Pattern

**Category:** Behavioral

**Intent:** Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.

**Key Participants:** AbstractClass, ConcreteClass

**Applicability:** Use to implement the invariant parts of an algorithm once and leave it up to subclasses to implement the behavior that can vary.

**Sample Code Snippet:**

```pseudocode
abstract class AbstractClass
    method templateMethod()
        step1()
        step2()

    abstract method step1()
    abstract method step2()

class ConcreteClass extends AbstractClass
    method step1()
        // Implementation of step1

    method step2()
        // Implementation of step2
```

### Visitor Pattern

**Category:** Behavioral

**Intent:** Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.

**Key Participants:** Visitor, ConcreteVisitor, Element

**Applicability:** Use when you want to perform operations on the elements of a complex object structure.

**Sample Code Snippet:**

```pseudocode
interface Visitor
    method visitConcreteElementA(elementA)
    method visitConcreteElementB(elementB)

interface Element
    method accept(visitor)

class ConcreteElementA implements Element
    method accept(visitor)
        visitor.visitConcreteElementA(this)
```

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the Singleton pattern?

- [x] To ensure a class has only one instance and provide a global access point to it.
- [ ] To define a family of algorithms and make them interchangeable.
- [ ] To compose objects into tree structures to represent part-whole hierarchies.
- [ ] To provide a way to access elements of an aggregate object sequentially.

> **Explanation:** The Singleton pattern ensures that a class has only one instance and provides a global point of access to it.

### Which pattern is used to decouple an abstraction from its implementation?

- [ ] Adapter Pattern
- [x] Bridge Pattern
- [ ] Composite Pattern
- [ ] Decorator Pattern

> **Explanation:** The Bridge pattern is used to decouple an abstraction from its implementation so that the two can vary independently.

### What is the main advantage of the Decorator pattern?

- [x] It allows additional responsibilities to be attached to an object dynamically.
- [ ] It provides a unified interface to a set of interfaces in a subsystem.
- [ ] It uses sharing to support large numbers of fine-grained objects efficiently.
- [ ] It provides a surrogate or placeholder for another object to control access to it.

> **Explanation:** The Decorator pattern allows additional responsibilities to be attached to an object dynamically, providing a flexible alternative to subclassing for extending functionality.

### Which pattern would you use to avoid coupling the sender of a request to its receiver?

- [ ] Command Pattern
- [x] Chain of Responsibility Pattern
- [ ] Observer Pattern
- [ ] State Pattern

> **Explanation:** The Chain of Responsibility pattern avoids coupling the sender of a request to its receiver by giving more than one object a chance to handle the request.

### What is the role of the Memento pattern?

- [x] To capture and externalize an object's internal state without violating encapsulation.
- [ ] To define a one-to-many dependency between objects.
- [ ] To allow an object to alter its behavior when its internal state changes.
- [ ] To define a family of algorithms and make them interchangeable.

> **Explanation:** The Memento pattern captures and externalizes an object's internal state so that the object can be restored to this state later, without violating encapsulation.

### Which pattern is used to provide a way to access elements of an aggregate object sequentially without exposing its underlying representation?

- [ ] Visitor Pattern
- [ ] Strategy Pattern
- [x] Iterator Pattern
- [ ] Template Method Pattern

> **Explanation:** The Iterator pattern provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

### What is the main intent of the Factory Method pattern?

- [x] To define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- [ ] To provide an interface for creating families of related or dependent objects.
- [ ] To separate the construction of a complex object from its representation.
- [ ] To specify the kinds of objects to create using a prototypical instance.

> **Explanation:** The Factory Method pattern defines an interface for creating an object, but lets subclasses alter the type of objects that will be created.

### Which pattern allows an object to alter its behavior when its internal state changes?

- [ ] Observer Pattern
- [ ] Strategy Pattern
- [x] State Pattern
- [ ] Template Method Pattern

> **Explanation:** The State pattern allows an object to alter its behavior when its internal state changes, making the object appear to change its class.

### What is the primary purpose of the Visitor pattern?

- [x] To represent an operation to be performed on the elements of an object structure.
- [ ] To define a one-to-many dependency between objects.
- [ ] To encapsulate a request as an object.
- [ ] To define a representation for a language and an interpreter for its sentences.

> **Explanation:** The Visitor pattern represents an operation to be performed on the elements of an object structure, allowing new operations to be defined without changing the classes of the elements.

### True or False: The Abstract Factory pattern is used to create a single object.

- [ ] True
- [x] False

> **Explanation:** The Abstract Factory pattern provides an interface for creating families of related or dependent objects without specifying their concrete classes, not just a single object.

{{< /quizdown >}}

Remember, this is just the beginning. As you progress, you'll build more complex and interactive systems using these patterns. Keep experimenting, stay curious, and enjoy the journey!

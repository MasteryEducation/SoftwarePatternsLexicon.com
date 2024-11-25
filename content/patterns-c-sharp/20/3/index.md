---
canonical: "https://softwarepatternslexicon.com/patterns-c-sharp/20/3"

title: "C# Design Patterns Reference Cheat Sheet"
description: "Explore a comprehensive guide to C# design patterns, including Creational, Structural, and Behavioral patterns, with intent, applicability, and key features."
linkTitle: "20.3 Pattern Reference Cheat Sheet"
categories:
- Design Patterns
- CSharp Programming
- Software Architecture
tags:
- CSharp Design Patterns
- Creational Patterns
- Structural Patterns
- Behavioral Patterns
- Software Engineering
date: 2024-11-17
type: docs
nav_weight: 20300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 20.3 Pattern Reference Cheat Sheet

Welcome to the Pattern Reference Cheat Sheet, a quick-reference guide for all design patterns discussed in our comprehensive guide. This section is designed to provide expert software engineers and enterprise architects with a concise overview of each pattern, summarizing key information for easy recall and comparison. Let's dive into the world of design patterns, categorized into Creational, Structural, and Behavioral patterns.

---

### Creational Design Patterns

#### Singleton Pattern
- **Category**: Creational
- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Singleton {
      -Singleton instance
      -Singleton()
      +GetInstance() Singleton
    }
  ```
- **Key Participants**: Singleton
- **Applicability**: Use when exactly one instance of a class is needed to control access to shared resources.
- **Design Considerations**: Ensure thread safety in multi-threaded applications. Use `Lazy<T>` in C# for lazy initialization.
- **Differences and Similarities**: Often confused with the Factory Method, but Singleton focuses on instance control.

#### Factory Method Pattern
- **Category**: Creational
- **Intent**: Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Creator {
      +FactoryMethod() Product
    }
    class ConcreteCreator {
      +FactoryMethod() ConcreteProduct
    }
    class Product
    class ConcreteProduct
    Creator <|-- ConcreteCreator
    Product <|-- ConcreteProduct
  ```
- **Key Participants**: Creator, ConcreteCreator, Product, ConcreteProduct
- **Applicability**: Use when a class cannot anticipate the class of objects it must create.
- **Design Considerations**: Promotes loose coupling by eliminating the need to bind application-specific classes into the code.
- **Differences and Similarities**: Similar to Abstract Factory but focuses on a single product.

#### Abstract Factory Pattern
- **Category**: Creational
- **Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class AbstractFactory {
      +CreateProductA() ProductA
      +CreateProductB() ProductB
    }
    class ConcreteFactory1 {
      +CreateProductA() ProductA1
      +CreateProductB() ProductB1
    }
    class ProductA
    class ProductB
    class ProductA1
    class ProductB1
    AbstractFactory <|-- ConcreteFactory1
    ProductA <|-- ProductA1
    ProductB <|-- ProductB1
  ```
- **Key Participants**: AbstractFactory, ConcreteFactory, ProductA, ProductB
- **Applicability**: Use when the system needs to be independent of how its products are created.
- **Design Considerations**: Ensures consistency among products. Can be complex to implement.
- **Differences and Similarities**: Similar to Factory Method but focuses on families of products.

#### Builder Pattern
- **Category**: Creational
- **Intent**: Separate the construction of a complex object from its representation so that the same construction process can create different representations.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Builder {
      +BuildPart()
    }
    class ConcreteBuilder {
      +BuildPart()
      +GetResult() Product
    }
    class Director {
      +Construct()
    }
    class Product
    Builder <|-- ConcreteBuilder
    Director --> Builder
    ConcreteBuilder --> Product
  ```
- **Key Participants**: Builder, ConcreteBuilder, Director, Product
- **Applicability**: Use when the construction process must allow different representations for the object that's constructed.
- **Design Considerations**: Useful for creating complex objects with numerous parts.
- **Differences and Similarities**: Similar to Abstract Factory but focuses on step-by-step construction.

#### Prototype Pattern
- **Category**: Creational
- **Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Prototype {
      +Clone() Prototype
    }
    class ConcretePrototype1 {
      +Clone() ConcretePrototype1
    }
    class ConcretePrototype2 {
      +Clone() ConcretePrototype2
    }
    Prototype <|-- ConcretePrototype1
    Prototype <|-- ConcretePrototype2
  ```
- **Key Participants**: Prototype, ConcretePrototype
- **Applicability**: Use when a system should be independent of how its products are created, composed, and represented.
- **Design Considerations**: Reduces the need for subclasses. Consider deep vs. shallow copy.
- **Differences and Similarities**: Similar to Factory Method but focuses on cloning.

#### Object Pool Pattern
- **Category**: Creational
- **Intent**: Manage a pool of reusable objects to improve performance and resource utilization.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class ObjectPool {
      +GetObject() PooledObject
      +ReleaseObject(PooledObject)
    }
    class PooledObject
    ObjectPool --> PooledObject
  ```
- **Key Participants**: ObjectPool, PooledObject
- **Applicability**: Use when object instantiation is costly and objects are frequently reused.
- **Design Considerations**: Manage lifecycle and state of pooled objects carefully.
- **Differences and Similarities**: Different from Singleton as it manages multiple instances.

#### Dependency Injection Pattern
- **Category**: Creational
- **Intent**: Allow the removal of hard-coded dependencies and make it possible to change them, whether at runtime or compile time.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Client {
      +SetService(Service)
    }
    class Service
    class Injector {
      +Inject() Client
    }
    Client --> Service
    Injector --> Client
  ```
- **Key Participants**: Client, Service, Injector
- **Applicability**: Use when you want to decouple the creation of a dependency from its usage.
- **Design Considerations**: Promotes loose coupling and easier testing.
- **Differences and Similarities**: Often used with IoC containers.

#### Service Locator Pattern
- **Category**: Creational
- **Intent**: Encapsulate the processes involved in obtaining a service with a strong abstraction layer.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class ServiceLocator {
      +GetService() Service
    }
    class Service
    ServiceLocator --> Service
  ```
- **Key Participants**: ServiceLocator, Service
- **Applicability**: Use when you want to centralize service access.
- **Design Considerations**: Can lead to hidden dependencies and make testing difficult.
- **Differences and Similarities**: Alternative to Dependency Injection but less favored due to potential for hidden dependencies.

#### Multiton Pattern
- **Category**: Creational
- **Intent**: Ensure a class has only named instances and provide a global point of access to them.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Multiton {
      -instances
      +GetInstance(key) Multiton
    }
  ```
- **Key Participants**: Multiton
- **Applicability**: Use when you need to manage a fixed number of instances.
- **Design Considerations**: Similar to Singleton but allows multiple instances.
- **Differences and Similarities**: Similar to Singleton but supports multiple instances.

#### Active Object Pattern
- **Category**: Creational
- **Intent**: Decouple method execution from method invocation for objects that each reside in their own thread of control.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class ActiveObject {
      +Method()
    }
    class Scheduler {
      +Schedule()
    }
    ActiveObject --> Scheduler
  ```
- **Key Participants**: ActiveObject, Scheduler
- **Applicability**: Use when you need to manage concurrency in a more controlled manner.
- **Design Considerations**: Useful for asynchronous method invocation.
- **Differences and Similarities**: Different from Proxy as it involves concurrency.

#### Double-Checked Locking Pattern
- **Category**: Creational
- **Intent**: Reduce the overhead of acquiring a lock by first testing the locking criterion without actually acquiring the lock.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class DoubleCheckedLocking {
      -instance
      +GetInstance() DoubleCheckedLocking
    }
  ```
- **Key Participants**: DoubleCheckedLocking
- **Applicability**: Use when you need to ensure thread safety with minimal locking overhead.
- **Design Considerations**: Ensure proper memory visibility in multi-threaded environments.
- **Differences and Similarities**: Often used in Singleton implementations.

#### Monostate Pattern
- **Category**: Creational
- **Intent**: Ensure all instances of a class share the same state.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Monostate {
      -sharedState
      +Method()
    }
  ```
- **Key Participants**: Monostate
- **Applicability**: Use when you want to share state across all instances of a class.
- **Design Considerations**: Different from Singleton as it allows multiple instances.
- **Differences and Similarities**: Similar to Singleton but focuses on shared state.

#### Service Layer Pattern
- **Category**: Creational
- **Intent**: Define an application's boundary with a layer of services that establishes a set of available operations and coordinates the application's response in each operation.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class ServiceLayer {
      +Operation()
    }
    class Service
    ServiceLayer --> Service
  ```
- **Key Participants**: ServiceLayer, Service
- **Applicability**: Use when you want to encapsulate business logic.
- **Design Considerations**: Promotes separation of concerns.
- **Differences and Similarities**: Different from Facade as it focuses on business logic.

---

### Structural Design Patterns

#### Adapter Pattern
- **Category**: Structural
- **Intent**: Convert the interface of a class into another interface clients expect.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Target {
      +Request()
    }
    class Adapter {
      +Request()
    }
    class Adaptee {
      +SpecificRequest()
    }
    Target <|.. Adapter
    Adapter --> Adaptee
  ```
- **Key Participants**: Target, Adapter, Adaptee
- **Applicability**: Use when you want to use an existing class, and its interface does not match the one you need.
- **Design Considerations**: Promotes reusability of existing classes.
- **Differences and Similarities**: Similar to Facade but focuses on interface conversion.

#### Bridge Pattern
- **Category**: Structural
- **Intent**: Decouple an abstraction from its implementation so that the two can vary independently.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Abstraction {
      +Operation()
    }
    class RefinedAbstraction {
      +Operation()
    }
    class Implementor {
      +OperationImpl()
    }
    class ConcreteImplementor {
      +OperationImpl()
    }
    Abstraction <|-- RefinedAbstraction
    Implementor <|-- ConcreteImplementor
    Abstraction --> Implementor
  ```
- **Key Participants**: Abstraction, Implementor
- **Applicability**: Use when you want to avoid a permanent binding between an abstraction and its implementation.
- **Design Considerations**: Promotes flexibility and extensibility.
- **Differences and Similarities**: Different from Adapter as it focuses on decoupling.

#### Composite Pattern
- **Category**: Structural
- **Intent**: Compose objects into tree structures to represent part-whole hierarchies.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Component {
      +Operation()
    }
    class Leaf {
      +Operation()
    }
    class Composite {
      +Operation()
      +Add(Component)
      +Remove(Component)
    }
    Component <|-- Leaf
    Component <|-- Composite
    Composite --> Component
  ```
- **Key Participants**: Component, Leaf, Composite
- **Applicability**: Use when you want to represent part-whole hierarchies of objects.
- **Design Considerations**: Simplifies client code by treating individual objects and compositions uniformly.
- **Differences and Similarities**: Different from Decorator as it focuses on hierarchy.

#### Decorator Pattern
- **Category**: Structural
- **Intent**: Attach additional responsibilities to an object dynamically.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Component {
      +Operation()
    }
    class ConcreteComponent {
      +Operation()
    }
    class Decorator {
      +Operation()
    }
    class ConcreteDecorator {
      +Operation()
    }
    Component <|-- ConcreteComponent
    Component <|-- Decorator
    Decorator <|-- ConcreteDecorator
    Decorator --> Component
  ```
- **Key Participants**: Component, Decorator
- **Applicability**: Use when you want to add responsibilities to individual objects dynamically.
- **Design Considerations**: Promotes flexibility but can lead to complexity.
- **Differences and Similarities**: Similar to Composite but focuses on adding behavior.

#### Facade Pattern
- **Category**: Structural
- **Intent**: Provide a unified interface to a set of interfaces in a subsystem.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Facade {
      +Operation()
    }
    class Subsystem1 {
      +Operation1()
    }
    class Subsystem2 {
      +Operation2()
    }
    Facade --> Subsystem1
    Facade --> Subsystem2
  ```
- **Key Participants**: Facade, Subsystem
- **Applicability**: Use when you want to provide a simple interface to a complex subsystem.
- **Design Considerations**: Simplifies usage of complex systems.
- **Differences and Similarities**: Different from Adapter as it focuses on simplification.

#### Flyweight Pattern
- **Category**: Structural
- **Intent**: Use sharing to support large numbers of fine-grained objects efficiently.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Flyweight {
      +Operation()
    }
    class ConcreteFlyweight {
      +Operation()
    }
    class FlyweightFactory {
      +GetFlyweight() Flyweight
    }
    Flyweight <|-- ConcreteFlyweight
    FlyweightFactory --> Flyweight
  ```
- **Key Participants**: Flyweight, FlyweightFactory
- **Applicability**: Use when you need to manage a large number of similar objects.
- **Design Considerations**: Reduces memory usage but can increase complexity.
- **Differences and Similarities**: Different from Singleton as it focuses on sharing.

#### Proxy Pattern
- **Category**: Structural
- **Intent**: Provide a surrogate or placeholder for another object to control access to it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Subject {
      +Request()
    }
    class RealSubject {
      +Request()
    }
    class Proxy {
      +Request()
    }
    Subject <|-- RealSubject
    Subject <|-- Proxy
    Proxy --> RealSubject
  ```
- **Key Participants**: Subject, Proxy, RealSubject
- **Applicability**: Use when you need to control access to an object.
- **Design Considerations**: Can introduce additional complexity.
- **Differences and Similarities**: Similar to Decorator but focuses on access control.

#### Data Access Patterns in C#

##### Data Access Object (DAO) Pattern
- **Category**: Structural
- **Intent**: Abstract and encapsulate all access to the data source.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class DAO {
      +GetData()
      +SaveData()
    }
    class DataSource
    DAO --> DataSource
  ```
- **Key Participants**: DAO, DataSource
- **Applicability**: Use when you want to separate data access logic from business logic.
- **Design Considerations**: Promotes separation of concerns.
- **Differences and Similarities**: Similar to Repository but focuses on data access.

##### Repository Pattern
- **Category**: Structural
- **Intent**: Mediate between the domain and data mapping layers using a collection-like interface for accessing domain objects.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Repository {
      +Add()
      +Remove()
      +FindById()
    }
    class DomainObject
    Repository --> DomainObject
  ```
- **Key Participants**: Repository, DomainObject
- **Applicability**: Use when you want to abstract data access and business logic.
- **Design Considerations**: Promotes separation of concerns.
- **Differences and Similarities**: Similar to DAO but focuses on domain logic.

##### Data Transfer Object (DTO) Pattern
- **Category**: Structural
- **Intent**: Transfer data between software application subsystems.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class DTO {
      +GetData()
      +SetData()
    }
  ```
- **Key Participants**: DTO
- **Applicability**: Use when you need to transfer data between layers.
- **Design Considerations**: Simplifies data transfer but can lead to duplication.
- **Differences and Similarities**: Different from DAO as it focuses on data transfer.

##### Unit of Work Pattern
- **Category**: Structural
- **Intent**: Maintain a list of objects affected by a business transaction and coordinate the writing out of changes.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class UnitOfWork {
      +Commit()
      +RegisterNew()
      +RegisterDirty()
      +RegisterDeleted()
    }
    class Entity
    UnitOfWork --> Entity
  ```
- **Key Participants**: UnitOfWork, Entity
- **Applicability**: Use when you need to manage transactions.
- **Design Considerations**: Promotes consistency and integrity.
- **Differences and Similarities**: Different from Repository as it focuses on transaction management.

##### Data Mapper Pattern
- **Category**: Structural
- **Intent**: Separate the in-memory objects from the database.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class DataMapper {
      +MapToDatabase()
      +MapToObject()
    }
    class DomainObject
    class Database
    DataMapper --> DomainObject
    DataMapper --> Database
  ```
- **Key Participants**: DataMapper, DomainObject, Database
- **Applicability**: Use when you need to map objects to database tables.
- **Design Considerations**: Promotes separation of concerns.
- **Differences and Similarities**: Similar to DAO but focuses on mapping.

#### Converter Pattern
- **Category**: Structural
- **Intent**: Convert one type of data to another.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Converter {
      +Convert()
    }
  ```
- **Key Participants**: Converter
- **Applicability**: Use when you need to convert data types.
- **Design Considerations**: Promotes flexibility.
- **Differences and Similarities**: Similar to Adapter but focuses on data conversion.

#### Naked Objects Pattern
- **Category**: Structural
- **Intent**: Expose the domain objects directly to the user interface.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class NakedObject {
      +Operation()
    }
  ```
- **Key Participants**: NakedObject
- **Applicability**: Use when you want to expose domain logic directly.
- **Design Considerations**: Promotes simplicity but can lead to tight coupling.
- **Differences and Similarities**: Different from Facade as it focuses on direct exposure.

#### Extension Object Pattern
- **Category**: Structural
- **Intent**: Add functionality to a class without altering its structure.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class ExtensionObject {
      +ExtensionMethod()
    }
  ```
- **Key Participants**: ExtensionObject
- **Applicability**: Use when you need to add functionality to existing classes.
- **Design Considerations**: Promotes flexibility.
- **Differences and Similarities**: Similar to Decorator but focuses on extension.

#### Converter vs. Adapter
- **Category**: Structural
- **Intent**: Compare and contrast the Converter and Adapter patterns.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Converter {
      +Convert()
    }
    class Adapter {
      +Request()
    }
  ```
- **Key Participants**: Converter, Adapter
- **Applicability**: Use Converter for data type conversion and Adapter for interface conversion.
- **Design Considerations**: Understand the context of use.
- **Differences and Similarities**: Converter focuses on data conversion, Adapter on interface conversion.

---

### Behavioral Design Patterns

#### Strategy Pattern
- **Category**: Behavioral
- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Context {
      +SetStrategy(Strategy)
      +ExecuteStrategy()
    }
    class Strategy {
      +AlgorithmInterface()
    }
    class ConcreteStrategyA {
      +AlgorithmInterface()
    }
    class ConcreteStrategyB {
      +AlgorithmInterface()
    }
    Context --> Strategy
    Strategy <|-- ConcreteStrategyA
    Strategy <|-- ConcreteStrategyB
  ```
- **Key Participants**: Context, Strategy
- **Applicability**: Use when you need to switch between algorithms at runtime.
- **Design Considerations**: Promotes flexibility but can increase complexity.
- **Differences and Similarities**: Similar to State but focuses on algorithms.

#### Observer Pattern
- **Category**: Behavioral
- **Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Subject {
      +Attach(Observer)
      +Detach(Observer)
      +Notify()
    }
    class Observer {
      +Update()
    }
    class ConcreteObserver {
      +Update()
    }
    Subject --> Observer
    Observer <|-- ConcreteObserver
  ```
- **Key Participants**: Subject, Observer
- **Applicability**: Use when an object should notify other objects without knowing who they are.
- **Design Considerations**: Promotes loose coupling.
- **Differences and Similarities**: Similar to Publish/Subscribe but focuses on state changes.

#### Command Pattern
- **Category**: Behavioral
- **Intent**: Encapsulate a request as an object, thereby allowing for parameterization of clients with queues, requests, and operations.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Command {
      +Execute()
    }
    class ConcreteCommand {
      +Execute()
    }
    class Invoker {
      +SetCommand(Command)
      +ExecuteCommand()
    }
    class Receiver {
      +Action()
    }
    Command <|-- ConcreteCommand
    Invoker --> Command
    ConcreteCommand --> Receiver
  ```
- **Key Participants**: Command, Invoker, Receiver
- **Applicability**: Use when you need to parameterize objects with operations.
- **Design Considerations**: Promotes flexibility but can increase complexity.
- **Differences and Similarities**: Similar to Strategy but focuses on requests.

#### Chain of Responsibility Pattern
- **Category**: Behavioral
- **Intent**: Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Handler {
      +SetSuccessor(Handler)
      +HandleRequest()
    }
    class ConcreteHandler1 {
      +HandleRequest()
    }
    class ConcreteHandler2 {
      +HandleRequest()
    }
    Handler <|-- ConcreteHandler1
    Handler <|-- ConcreteHandler2
    Handler --> Handler
  ```
- **Key Participants**: Handler
- **Applicability**: Use when you want to pass requests along a chain of handlers.
- **Design Considerations**: Promotes flexibility but can lead to complexity.
- **Differences and Similarities**: Different from Command as it focuses on request handling.

#### Mediator Pattern
- **Category**: Behavioral
- **Intent**: Define an object that encapsulates how a set of objects interact.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Mediator {
      +Send()
    }
    class ConcreteMediator {
      +Send()
    }
    class Colleague {
      +Receive()
    }
    Mediator <|-- ConcreteMediator
    ConcreteMediator --> Colleague
  ```
- **Key Participants**: Mediator, Colleague
- **Applicability**: Use when you want to reduce the complexity of communication between multiple objects.
- **Design Considerations**: Promotes loose coupling but can lead to complexity.
- **Differences and Similarities**: Different from Observer as it focuses on communication.

#### Memento Pattern
- **Category**: Behavioral
- **Intent**: Capture and externalize an object's internal state so that the object can be restored to this state later.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Originator {
      +CreateMemento() Memento
      +SetMemento(Memento)
    }
    class Memento
    class Caretaker {
      +SaveMemento(Memento)
      +RestoreMemento() Memento
    }
    Originator --> Memento
    Caretaker --> Memento
  ```
- **Key Participants**: Originator, Memento, Caretaker
- **Applicability**: Use when you need to restore an object to a previous state.
- **Design Considerations**: Promotes encapsulation but can lead to increased memory usage.
- **Differences and Similarities**: Different from Command as it focuses on state restoration.

#### State Pattern
- **Category**: Behavioral
- **Intent**: Allow an object to alter its behavior when its internal state changes.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Context {
      +SetState(State)
      +Request()
    }
    class State {
      +Handle()
    }
    class ConcreteStateA {
      +Handle()
    }
    class ConcreteStateB {
      +Handle()
    }
    Context --> State
    State <|-- ConcreteStateA
    State <|-- ConcreteStateB
  ```
- **Key Participants**: Context, State
- **Applicability**: Use when an object's behavior depends on its state.
- **Design Considerations**: Promotes flexibility but can increase complexity.
- **Differences and Similarities**: Similar to Strategy but focuses on state.

#### Template Method Pattern
- **Category**: Behavioral
- **Intent**: Define the skeleton of an algorithm in an operation, deferring some steps to subclasses.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class AbstractClass {
      +TemplateMethod()
      +PrimitiveOperation1()
      +PrimitiveOperation2()
    }
    class ConcreteClass {
      +PrimitiveOperation1()
      +PrimitiveOperation2()
    }
    AbstractClass <|-- ConcreteClass
  ```
- **Key Participants**: AbstractClass, ConcreteClass
- **Applicability**: Use when you want to let subclasses redefine certain steps of an algorithm.
- **Design Considerations**: Promotes code reuse but can lead to tight coupling.
- **Differences and Similarities**: Different from Strategy as it focuses on algorithm structure.

#### Visitor Pattern
- **Category**: Behavioral
- **Intent**: Represent an operation to be performed on the elements of an object structure.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Visitor {
      +VisitConcreteElementA()
      +VisitConcreteElementB()
    }
    class ConcreteVisitor {
      +VisitConcreteElementA()
      +VisitConcreteElementB()
    }
    class Element {
      +Accept(Visitor)
    }
    class ConcreteElementA {
      +Accept(Visitor)
    }
    class ConcreteElementB {
      +Accept(Visitor)
    }
    Visitor <|-- ConcreteVisitor
    Element <|-- ConcreteElementA
    Element <|-- ConcreteElementB
    ConcreteElementA --> Visitor
    ConcreteElementB --> Visitor
  ```
- **Key Participants**: Visitor, Element
- **Applicability**: Use when you need to perform operations on elements of an object structure.
- **Design Considerations**: Promotes flexibility but can lead to complexity.
- **Differences and Similarities**: Different from Command as it focuses on operations.

#### Interpreter Pattern
- **Category**: Behavioral
- **Intent**: Define a representation for a grammar and an interpreter to interpret sentences in the language.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class AbstractExpression {
      +Interpret(Context)
    }
    class TerminalExpression {
      +Interpret(Context)
    }
    class NonterminalExpression {
      +Interpret(Context)
    }
    AbstractExpression <|-- TerminalExpression
    AbstractExpression <|-- NonterminalExpression
  ```
- **Key Participants**: AbstractExpression, TerminalExpression, NonterminalExpression
- **Applicability**: Use when you need to interpret a language.
- **Design Considerations**: Promotes flexibility but can lead to complexity.
- **Differences and Similarities**: Different from Visitor as it focuses on interpretation.

#### Iterator Pattern
- **Category**: Behavioral
- **Intent**: Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Iterator {
      +First()
      +Next()
      +IsDone()
      +CurrentItem()
    }
    class ConcreteIterator {
      +First()
      +Next()
      +IsDone()
      +CurrentItem()
    }
    class Aggregate {
      +CreateIterator() Iterator
    }
    class ConcreteAggregate {
      +CreateIterator() Iterator
    }
    Iterator <|-- ConcreteIterator
    Aggregate <|-- ConcreteAggregate
    ConcreteAggregate --> Iterator
  ```
- **Key Participants**: Iterator, Aggregate
- **Applicability**: Use when you need to traverse a collection.
- **Design Considerations**: Promotes encapsulation.
- **Differences and Similarities**: Different from Composite as it focuses on traversal.

#### Observer vs. Publish/Subscribe Pattern
- **Category**: Behavioral
- **Intent**: Compare and contrast the Observer and Publish/Subscribe patterns.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Observer {
      +Update()
    }
    class Publisher {
      +Publish()
    }
  ```
- **Key Participants**: Observer, Publisher
- **Applicability**: Use Observer for state changes and Publish/Subscribe for event handling.
- **Design Considerations**: Understand the context of use.
- **Differences and Similarities**: Observer focuses on state changes, Publish/Subscribe on events.

#### Specification Pattern
- **Category**: Behavioral
- **Intent**: Encapsulate business rules in a reusable and combinable way.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Specification {
      +IsSatisfiedBy(Candidate)
    }
    class CompositeSpecification {
      +IsSatisfiedBy(Candidate)
    }
    class AndSpecification {
      +IsSatisfiedBy(Candidate)
    }
    class OrSpecification {
      +IsSatisfiedBy(Candidate)
    }
    Specification <|-- CompositeSpecification
    CompositeSpecification <|-- AndSpecification
    CompositeSpecification <|-- OrSpecification
  ```
- **Key Participants**: Specification
- **Applicability**: Use when you need to encapsulate business rules.
- **Design Considerations**: Promotes flexibility.
- **Differences and Similarities**: Different from Strategy as it focuses on rules.

#### Balking Pattern
- **Category**: Behavioral
- **Intent**: Prevent an object from executing an action if it is not in the appropriate state.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Balking {
      +Request()
    }
  ```
- **Key Participants**: Balking
- **Applicability**: Use when you need to prevent actions in inappropriate states.
- **Design Considerations**: Promotes safety.
- **Differences and Similarities**: Different from State as it focuses on prevention.

#### Double Dispatch Pattern
- **Category**: Behavioral
- **Intent**: Resolve a method call to a method in a hierarchy based on the runtime types of two objects.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class DoubleDispatch {
      +Dispatch()
    }
  ```
- **Key Participants**: DoubleDispatch
- **Applicability**: Use when you need to resolve method calls based on two types.
- **Design Considerations**: Promotes flexibility.
- **Differences and Similarities**: Different from Visitor as it focuses on dispatch.

---

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Singleton pattern?

- [x] Ensure a class has only one instance and provide a global point of access to it.
- [ ] Convert the interface of a class into another interface clients expect.
- [ ] Define a family of algorithms, encapsulate each one, and make them interchangeable.
- [ ] Provide a way to access the elements of an aggregate object sequentially.

> **Explanation:** The Singleton pattern ensures a class has only one instance and provides a global point of access to it.

### Which pattern is used to decouple an abstraction from its implementation?

- [ ] Adapter
- [x] Bridge
- [ ] Composite
- [ ] Decorator

> **Explanation:** The Bridge pattern is used to decouple an abstraction from its implementation so that the two can vary independently.

### What is the key difference between the Factory Method and Abstract Factory patterns?

- [x] Factory Method focuses on a single product, while Abstract Factory focuses on families of products.
- [ ] Factory Method is used for cloning objects, while Abstract Factory is used for creating objects.
- [ ] Factory Method is used for creating complex objects, while Abstract Factory is used for simple objects.
- [ ] Factory Method is used for managing object lifecycles, while Abstract Factory is used for managing object states.

> **Explanation:** The Factory Method pattern focuses on creating a single product, while the Abstract Factory pattern focuses on creating families of related or dependent products.

### Which pattern is best suited for adding responsibilities to individual objects dynamically?

- [ ] Composite
- [ ] Facade
- [x] Decorator
- [ ] Proxy

> **Explanation:** The Decorator pattern is best suited for adding responsibilities to individual objects dynamically without altering their structure.

### What is the primary purpose of the Observer pattern?

- [ ] Define a representation for a grammar and an interpreter to interpret sentences in the language.
- [x] Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- [ ] Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
- [ ] Capture and externalize an object's internal state so that the object can be restored to this state later.

> **Explanation:** The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

### Which pattern is used to provide a unified interface to a set of interfaces in a subsystem?

- [ ] Adapter
- [ ] Bridge
- [x] Facade
- [ ] Proxy

> **Explanation:** The Facade pattern is used to provide a unified interface to a set of interfaces in a subsystem, simplifying the usage of complex systems.

### What is the primary intent of the Command pattern?

- [ ] Define a family of algorithms, encapsulate each one, and make them interchangeable.
- [ ] Allow an object to alter its behavior when its internal state changes.
- [x] Encapsulate a request as an object, thereby allowing for parameterization of clients with queues, requests, and operations.
- [ ] Provide a surrogate or placeholder for another object to control access to it.

> **Explanation:** The Command pattern encapsulates a request as an object, allowing for parameterization of clients with queues, requests, and operations.

### Which pattern is used to manage a pool of reusable objects to improve performance and resource utilization?

- [ ] Singleton
- [ ] Prototype
- [ ] Builder
- [x] Object Pool

> **Explanation:** The Object Pool pattern is used to manage a pool of reusable objects to improve performance and resource utilization.

### What is the primary purpose of the Strategy pattern?

- [x] Define a family of algorithms, encapsulate each one, and make them interchangeable.
- [ ] Capture and externalize an object's internal state so that the object can be restored to this state later.
- [ ] Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
- [ ] Define a representation for a grammar and an interpreter to interpret sentences in the language.

> **Explanation:** The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable.

### True or False: The Adapter pattern is used to provide a surrogate or placeholder for another object to control access to it.

- [ ] True
- [x] False

> **Explanation:** False. The Adapter pattern is used to convert the interface of a class into another interface clients expect, not to provide a surrogate or placeholder for another object.

{{< /quizdown >}}

---

Remember, this cheat sheet is just the beginning. As you progress, you'll build more complex and interactive applications using these patterns. Keep experimenting, stay curious, and enjoy the journey!

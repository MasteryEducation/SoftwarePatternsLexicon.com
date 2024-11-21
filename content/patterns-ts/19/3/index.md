---
canonical: "https://softwarepatternslexicon.com/patterns-ts/19/3"
title: "Design Patterns in TypeScript: Comprehensive Pattern Reference Cheat Sheet"
description: "Explore a detailed cheat sheet for design patterns in TypeScript, covering Creational, Structural, Behavioral, and Architectural patterns with examples and diagrams."
linkTitle: "19.3 Pattern Reference Cheat Sheet"
categories:
- Design Patterns
- TypeScript
- Software Engineering
tags:
- Creational Patterns
- Structural Patterns
- Behavioral Patterns
- Architectural Patterns
- TypeScript
date: 2024-11-17
type: docs
nav_weight: 19300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 19.3 Pattern Reference Cheat Sheet

Welcome to the Pattern Reference Cheat Sheet, your quick guide to understanding and applying design patterns in TypeScript. This section is designed to provide expert developers with a concise summary of each pattern covered in the guide. We will explore Creational, Structural, Behavioral, and Architectural patterns, offering insights into their intent, key features, applicability, and more. Let's dive in!

---

### Creational Patterns

**Creational patterns** focus on the process of object creation, providing various ways to create objects while hiding the creation logic.

#### Singleton Pattern

- **Category**: Creational
- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Key Features**:
  - Single instance management.
  - Global access point.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Singleton {
      -instance: Singleton
      +getInstance(): Singleton
    }
  ```

- **Key Participants**: Singleton class.
- **Applicability**: Use when exactly one instance of a class is needed, like configuration settings.
- **Example**: Database connection manager.

#### Factory Method Pattern

- **Category**: Creational
- **Intent**: Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- **Key Features**:
  - Interface for object creation.
  - Subclass responsibility for instantiation.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Creator {
      +factoryMethod(): Product
    }
    class ConcreteCreator {
      +factoryMethod(): ConcreteProduct
    }
    Creator <|-- ConcreteCreator
    class Product
    class ConcreteProduct
    Product <|-- ConcreteProduct
  ```

- **Key Participants**: Creator, ConcreteCreator, Product, ConcreteProduct.
- **Applicability**: Use when a class can't anticipate the class of objects it must create.
- **Example**: GUI frameworks with different button types.

#### Abstract Factory Pattern

- **Category**: Creational
- **Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- **Key Features**:
  - Family of related objects.
  - Abstract interfaces.
- **Structure Diagram**:

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
    AbstractFactory <|-- ConcreteFactory1
    class AbstractProductA
    class ProductA1
    AbstractProductA <|-- ProductA1
    class AbstractProductB
    class ProductB1
    AbstractProductB <|-- ProductB1
  ```

- **Key Participants**: AbstractFactory, ConcreteFactory, AbstractProduct, ConcreteProduct.
- **Applicability**: Use when the system needs to be independent of how its products are created.
- **Example**: UI component libraries with different themes.

#### Builder Pattern

- **Category**: Creational
- **Intent**: Separate the construction of a complex object from its representation, allowing the same construction process to create various representations.
- **Key Features**:
  - Step-by-step construction.
  - Different representations.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Builder {
      +buildPart(): void
    }
    class ConcreteBuilder {
      +buildPart(): void
      +getResult(): Product
    }
    Builder <|-- ConcreteBuilder
    class Director {
      +construct(): void
    }
    class Product
    Director --> Builder
    ConcreteBuilder --> Product
  ```

- **Key Participants**: Builder, ConcreteBuilder, Director, Product.
- **Applicability**: Use when the construction process must allow different representations.
- **Example**: Building a complex document with different formats.

#### Prototype Pattern

- **Category**: Creational
- **Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
- **Key Features**:
  - Cloning.
  - Prototypical instance.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Prototype {
      +clone(): Prototype
    }
    class ConcretePrototype {
      +clone(): ConcretePrototype
    }
    Prototype <|-- ConcretePrototype
  ```

- **Key Participants**: Prototype, ConcretePrototype.
- **Applicability**: Use when the cost of creating a new instance of a class is more expensive than copying an existing instance.
- **Example**: Copying objects in a game with different attributes.

#### Object Pool Pattern

- **Category**: Creational
- **Intent**: Manage a pool of reusable objects to improve performance.
- **Key Features**:
  - Reuse of expensive-to-create objects.
  - Resource management.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class ObjectPool {
      +getObject(): PooledObject
      +releaseObject(obj: PooledObject): void
    }
    class PooledObject
    ObjectPool --> PooledObject
  ```

- **Key Participants**: ObjectPool, PooledObject.
- **Applicability**: Use when object creation is costly and frequent.
- **Example**: Database connection pooling.

#### Dependency Injection Pattern

- **Category**: Creational
- **Intent**: Pass dependencies to a class instead of hard-coding them.
- **Key Features**:
  - Inversion of control.
  - Decoupling.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Client {
      -service: Service
      +setService(service: Service): void
    }
    class Service
    Client --> Service
  ```

- **Key Participants**: Client, Service.
- **Applicability**: Use when you want to decouple classes and improve testability.
- **Example**: Injecting services in a web application.

#### Lazy Initialization Pattern

- **Category**: Creational
- **Intent**: Delay the creation of an object until it is needed.
- **Key Features**:
  - Deferred creation.
  - Resource optimization.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Lazy {
      -instance: Resource
      +getInstance(): Resource
    }
    class Resource
    Lazy --> Resource
  ```

- **Key Participants**: Lazy, Resource.
- **Applicability**: Use when you want to improve performance by avoiding unnecessary computations.
- **Example**: Loading configuration settings only when accessed.

#### Multiton Pattern

- **Category**: Creational
- **Intent**: Allow controlled creation of multiple instances with a key.
- **Key Features**:
  - Instance management by key.
  - Controlled access.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Multiton {
      -instances: Map<Key, Multiton>
      +getInstance(key: Key): Multiton
    }
  ```

- **Key Participants**: Multiton.
- **Applicability**: Use when you need a limited number of instances, each identified by a key.
- **Example**: Managing instances of a logger for different contexts.

---

### Structural Patterns

**Structural patterns** deal with object composition, ensuring that if one part changes, the entire structure doesn't need to.

#### Adapter Pattern

- **Category**: Structural
- **Intent**: Allow incompatible interfaces to work together via a mediator.
- **Key Features**:
  - Interface translation.
  - Compatibility.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Target {
      +request(): void
    }
    class Adapter {
      +request(): void
    }
    class Adaptee {
      +specificRequest(): void
    }
    Target <|-- Adapter
    Adapter --> Adaptee
  ```

- **Key Participants**: Target, Adapter, Adaptee.
- **Applicability**: Use when you want to use an existing class but its interface doesn't match the one you need.
- **Example**: Integrating legacy code with new systems.

#### Bridge Pattern

- **Category**: Structural
- **Intent**: Decouple an abstraction from its implementation so that the two can vary independently.
- **Key Features**:
  - Abstraction-implementation separation.
  - Flexibility.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Abstraction {
      -implementor: Implementor
      +operation(): void
    }
    class RefinedAbstraction {
      +operation(): void
    }
    class Implementor {
      +operationImpl(): void
    }
    class ConcreteImplementor {
      +operationImpl(): void
    }
    Abstraction <|-- RefinedAbstraction
    Abstraction --> Implementor
    Implementor <|-- ConcreteImplementor
  ```

- **Key Participants**: Abstraction, Implementor.
- **Applicability**: Use when you want to avoid a permanent binding between an abstraction and its implementation.
- **Example**: Cross-platform GUI applications.

#### Composite Pattern

- **Category**: Structural
- **Intent**: Compose objects into tree structures to represent part-whole hierarchies.
- **Key Features**:
  - Tree structure.
  - Uniform treatment of individual and composite objects.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Component {
      +operation(): void
    }
    class Leaf {
      +operation(): void
    }
    class Composite {
      +add(Component): void
      +remove(Component): void
      +operation(): void
    }
    Component <|-- Leaf
    Component <|-- Composite
    Composite --> Component
  ```

- **Key Participants**: Component, Leaf, Composite.
- **Applicability**: Use when you want to represent part-whole hierarchies of objects.
- **Example**: File system directories.

#### Decorator Pattern

- **Category**: Structural
- **Intent**: Attach additional responsibilities to an object dynamically.
- **Key Features**:
  - Dynamic behavior addition.
  - Flexible alternative to subclassing.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Component {
      +operation(): void
    }
    class ConcreteComponent {
      +operation(): void
    }
    class Decorator {
      -component: Component
      +operation(): void
    }
    class ConcreteDecorator {
      +operation(): void
    }
    Component <|-- ConcreteComponent
    Component <|-- Decorator
    Decorator <|-- ConcreteDecorator
    Decorator --> Component
  ```

- **Key Participants**: Component, Decorator.
- **Applicability**: Use when you want to add responsibilities to individual objects without affecting other objects.
- **Example**: Adding scrollbars to windows.

#### Facade Pattern

- **Category**: Structural
- **Intent**: Provide a simplified interface to a complex subsystem.
- **Key Features**:
  - Simplified interface.
  - Subsystem encapsulation.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Facade {
      +operation(): void
    }
    class Subsystem1 {
      +operation1(): void
    }
    class Subsystem2 {
      +operation2(): void
    }
    Facade --> Subsystem1
    Facade --> Subsystem2
  ```

- **Key Participants**: Facade, Subsystem classes.
- **Applicability**: Use when you want to provide a simple interface to a complex subsystem.
- **Example**: Simplifying a complex API for client use.

#### Flyweight Pattern

- **Category**: Structural
- **Intent**: Use sharing to support large numbers of fine-grained objects efficiently.
- **Key Features**:
  - Shared state.
  - Memory optimization.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Flyweight {
      +operation(extrinsicState): void
    }
    class ConcreteFlyweight {
      +operation(extrinsicState): void
    }
    class FlyweightFactory {
      +getFlyweight(key): Flyweight
    }
    Flyweight <|-- ConcreteFlyweight
    FlyweightFactory --> Flyweight
  ```

- **Key Participants**: Flyweight, FlyweightFactory.
- **Applicability**: Use when many objects must be created efficiently.
- **Example**: Text editors handling large documents.

#### Proxy Pattern

- **Category**: Structural
- **Intent**: Provide a surrogate or placeholder for another object to control access.
- **Key Features**:
  - Access control.
  - Surrogate object.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Subject {
      +request(): void
    }
    class RealSubject {
      +request(): void
    }
    class Proxy {
      +request(): void
    }
    Subject <|-- RealSubject
    Subject <|-- Proxy
    Proxy --> RealSubject
  ```

- **Key Participants**: Subject, RealSubject, Proxy.
- **Applicability**: Use when you need to control access to an object.
- **Example**: Virtual proxies for loading large images.

#### Module Pattern

- **Category**: Structural
- **Intent**: Encapsulate code within modules for better organization and maintainability.
- **Key Features**:
  - Encapsulation.
  - Public and private access.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Module {
      +publicMethod(): void
      -privateMethod(): void
    }
  ```

- **Key Participants**: Module.
- **Applicability**: Use when you want to encapsulate code and manage dependencies.
- **Example**: JavaScript modules.

#### Extension Object Pattern

- **Category**: Structural
- **Intent**: Add functionality to objects dynamically by attaching new extension objects.
- **Key Features**:
  - Dynamic extension.
  - Flexibility.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class CoreObject {
      +operation(): void
    }
    class Extension {
      +extendedOperation(): void
    }
    CoreObject --> Extension
  ```

- **Key Participants**: CoreObject, Extension.
- **Applicability**: Use when you want to add new functionality to objects without altering their class definitions.
- **Example**: Plugin systems.

---

### Behavioral Patterns

**Behavioral patterns** focus on communication between objects, ensuring that they can interact in a flexible and dynamic way.

#### Chain of Responsibility Pattern

- **Category**: Behavioral
- **Intent**: Pass a request along a chain of handlers until one handles it.
- **Key Features**:
  - Decoupled sender and receiver.
  - Dynamic request handling.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Handler {
      +setNext(handler: Handler): void
      +handleRequest(request: Request): void
    }
    class ConcreteHandler {
      +handleRequest(request: Request): void
    }
    Handler <|-- ConcreteHandler
    Handler --> Handler
  ```

- **Key Participants**: Handler, ConcreteHandler.
- **Applicability**: Use when multiple objects can handle a request, and the handler isn't known in advance.
- **Example**: Event handling systems.

#### Command Pattern

- **Category**: Behavioral
- **Intent**: Encapsulate a request as an object, allowing parameterization and queuing.
- **Key Features**:
  - Encapsulation of requests.
  - Parameterization.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Command {
      +execute(): void
    }
    class ConcreteCommand {
      +execute(): void
    }
    class Invoker {
      +setCommand(command: Command): void
      +executeCommand(): void
    }
    class Receiver {
      +action(): void
    }
    Command <|-- ConcreteCommand
    Invoker --> Command
    ConcreteCommand --> Receiver
  ```

- **Key Participants**: Command, Invoker, Receiver.
- **Applicability**: Use when you want to parameterize objects with operations.
- **Example**: GUI button actions.

#### Interpreter Pattern

- **Category**: Behavioral
- **Intent**: Define a representation of a grammar and an interpreter to work with it.
- **Key Features**:
  - Grammar representation.
  - Interpretation.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class AbstractExpression {
      +interpret(context: Context): void
    }
    class TerminalExpression {
      +interpret(context: Context): void
    }
    class NonTerminalExpression {
      +interpret(context: Context): void
    }
    AbstractExpression <|-- TerminalExpression
    AbstractExpression <|-- NonTerminalExpression
  ```

- **Key Participants**: AbstractExpression, TerminalExpression, NonTerminalExpression.
- **Applicability**: Use when you need to interpret a language.
- **Example**: SQL parsing engines.

#### Iterator Pattern

- **Category**: Behavioral
- **Intent**: Provide a way to access elements of a collection sequentially without exposing the underlying representation.
- **Key Features**:
  - Sequential access.
  - Encapsulation of collection details.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Iterator {
      +next(): Element
      +hasNext(): bool
    }
    class ConcreteIterator {
      +next(): Element
      +hasNext(): bool
    }
    class Aggregate {
      +createIterator(): Iterator
    }
    class ConcreteAggregate {
      +createIterator(): Iterator
    }
    Iterator <|-- ConcreteIterator
    Aggregate <|-- ConcreteAggregate
    ConcreteAggregate --> ConcreteIterator
  ```

- **Key Participants**: Iterator, ConcreteIterator, Aggregate.
- **Applicability**: Use when you need to traverse a collection without exposing its internal structure.
- **Example**: Iterating over a list of files.

#### Mediator Pattern

- **Category**: Behavioral
- **Intent**: Define an object that encapsulates how a set of objects interact.
- **Key Features**:
  - Centralized communication.
  - Reduced dependencies.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Mediator {
      +notify(sender: Colleague, event: string): void
    }
    class ConcreteMediator {
      +notify(sender: Colleague, event: string): void
    }
    class Colleague {
      +setMediator(mediator: Mediator): void
    }
    class ConcreteColleague {
      +setMediator(mediator: Mediator): void
    }
    Mediator <|-- ConcreteMediator
    Colleague <|-- ConcreteColleague
    ConcreteColleague --> Mediator
  ```

- **Key Participants**: Mediator, Colleague.
- **Applicability**: Use when you want to reduce the complexity of communication between multiple objects.
- **Example**: Chat systems.

#### Memento Pattern

- **Category**: Behavioral
- **Intent**: Capture and restore an object's internal state without violating encapsulation.
- **Key Features**:
  - State capture.
  - Encapsulation.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Originator {
      +createMemento(): Memento
      +restore(memento: Memento): void
    }
    class Memento
    class Caretaker {
      +addMemento(memento: Memento): void
      +getMemento(index: int): Memento
    }
    Originator --> Memento
    Caretaker --> Memento
  ```

- **Key Participants**: Originator, Memento, Caretaker.
- **Applicability**: Use when you need to save and restore the state of an object.
- **Example**: Undo functionality in text editors.

#### Observer Pattern

- **Category**: Behavioral
- **Intent**: Define a one-to-many dependency so that when one object changes state, all its dependents are notified.
- **Key Features**:
  - State change notification.
  - Loose coupling.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Subject {
      +attach(observer: Observer): void
      +detach(observer: Observer): void
      +notify(): void
    }
    class ConcreteSubject {
      +getState(): State
      +setState(state: State): void
    }
    class Observer {
      +update(): void
    }
    class ConcreteObserver {
      +update(): void
    }
    Subject <|-- ConcreteSubject
    Observer <|-- ConcreteObserver
    ConcreteSubject --> Observer
  ```

- **Key Participants**: Subject, Observer.
- **Applicability**: Use when an object should be able to notify other objects without making assumptions about who these objects are.
- **Example**: Event listeners in GUI frameworks.

#### State Pattern

- **Category**: Behavioral
- **Intent**: Allow an object to alter its behavior when its internal state changes.
- **Key Features**:
  - State-specific behavior.
  - State transitions.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Context {
      +setState(state: State): void
      +request(): void
    }
    class State {
      +handle(context: Context): void
    }
    class ConcreteState {
      +handle(context: Context): void
    }
    Context --> State
    State <|-- ConcreteState
  ```

- **Key Participants**: Context, State.
- **Applicability**: Use when an object's behavior depends on its state and it must change behavior at runtime.
- **Example**: Finite state machines.

#### Strategy Pattern

- **Category**: Behavioral
- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable.
- **Key Features**:
  - Encapsulated algorithms.
  - Interchangeability.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Context {
      +setStrategy(strategy: Strategy): void
      +executeStrategy(): void
    }
    class Strategy {
      +execute(): void
    }
    class ConcreteStrategy {
      +execute(): void
    }
    Context --> Strategy
    Strategy <|-- ConcreteStrategy
  ```

- **Key Participants**: Context, Strategy.
- **Applicability**: Use when you want to use different variants of an algorithm.
- **Example**: Sorting algorithms.

#### Template Method Pattern

- **Category**: Behavioral
- **Intent**: Define the skeleton of an algorithm, deferring exact steps to subclasses.
- **Key Features**:
  - Algorithm skeleton.
  - Subclass customization.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class AbstractClass {
      +templateMethod(): void
      +primitiveOperation1(): void
      +primitiveOperation2(): void
    }
    class ConcreteClass {
      +primitiveOperation1(): void
      +primitiveOperation2(): void
    }
    AbstractClass <|-- ConcreteClass
  ```

- **Key Participants**: AbstractClass, ConcreteClass.
- **Applicability**: Use when you want to let subclasses redefine certain steps of an algorithm without changing its structure.
- **Example**: Data processing frameworks.

#### Visitor Pattern

- **Category**: Behavioral
- **Intent**: Represent an operation to be performed on elements of an object structure.
- **Key Features**:
  - Operation separation.
  - New operations without modifying classes.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Visitor {
      +visitConcreteElementA(element: ConcreteElementA): void
      +visitConcreteElementB(element: ConcreteElementB): void
    }
    class ConcreteVisitor {
      +visitConcreteElementA(element: ConcreteElementA): void
      +visitConcreteElementB(element: ConcreteElementB): void
    }
    class Element {
      +accept(visitor: Visitor): void
    }
    class ConcreteElementA {
      +accept(visitor: Visitor): void
    }
    class ConcreteElementB {
      +accept(visitor: Visitor): void
    }
    Visitor <|-- ConcreteVisitor
    Element <|-- ConcreteElementA
    Element <|-- ConcreteElementB
    ConcreteElementA --> Visitor
    ConcreteElementB --> Visitor
  ```

- **Key Participants**: Visitor, Element.
- **Applicability**: Use when you need to perform operations on elements of a complex object structure.
- **Example**: Compilers processing syntax trees.

#### Specification Pattern

- **Category**: Behavioral
- **Intent**: Combine business rules with logic to evaluate objects.
- **Key Features**:
  - Business rule encapsulation.
  - Logical combinations.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Specification {
      +isSatisfiedBy(candidate: Candidate): bool
    }
    class CompositeSpecification {
      +isSatisfiedBy(candidate: Candidate): bool
    }
    class AndSpecification {
      +isSatisfiedBy(candidate: Candidate): bool
    }
    class OrSpecification {
      +isSatisfiedBy(candidate: Candidate): bool
    }
    Specification <|-- CompositeSpecification
    CompositeSpecification <|-- AndSpecification
    CompositeSpecification <|-- OrSpecification
  ```

- **Key Participants**: Specification, CompositeSpecification.
- **Applicability**: Use when you need to evaluate objects against a set of criteria.
- **Example**: Filtering products based on multiple criteria.

#### Publish/Subscribe Pattern

- **Category**: Behavioral
- **Intent**: Decouple components by using a message broker.
- **Key Features**:
  - Decoupled communication.
  - Event-driven architecture.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Publisher {
      +publish(message: Message): void
    }
    class Subscriber {
      +receive(message: Message): void
    }
    class MessageBroker {
      +subscribe(subscriber: Subscriber): void
      +publish(message: Message): void
    }
    Publisher --> MessageBroker
    Subscriber --> MessageBroker
  ```

- **Key Participants**: Publisher, Subscriber, MessageBroker.
- **Applicability**: Use when you want to enable asynchronous communication between components.
- **Example**: Notification systems.

---

### Architectural Patterns

**Architectural patterns** provide a blueprint for system organization, focusing on the high-level structure of software systems.

#### Model-View-Controller (MVC) Pattern

- **Category**: Architectural
- **Intent**: Divide an application into three interconnected components to separate internal representations from user interactions.
- **Key Features**:
  - Separation of concerns.
  - Modular design.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Model {
      +getData(): Data
      +setData(data: Data): void
    }
    class View {
      +render(data: Data): void
    }
    class Controller {
      +updateView(): void
    }
    Controller --> Model
    Controller --> View
  ```

- **Key Participants**: Model, View, Controller.
- **Applicability**: Use when you want to separate user interface logic from business logic.
- **Example**: Web applications.

#### Model-View-ViewModel (MVVM) Pattern

- **Category**: Architectural
- **Intent**: Structure code to separate development of user interfaces from the business logic, facilitating data binding.
- **Key Features**:
  - Data binding.
  - Separation of concerns.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Model {
      +getData(): Data
      +setData(data: Data): void
    }
    class View {
      +render(data: Data): void
    }
    class ViewModel {
      +bindModel(model: Model): void
    }
    ViewModel --> Model
    ViewModel --> View
  ```

- **Key Participants**: Model, View, ViewModel.
- **Applicability**: Use when you want to facilitate a clear separation between UI and business logic.
- **Example**: Modern web applications with frameworks like Angular.

#### Flux and Redux Architecture

- **Category**: Architectural
- **Intent**: Manage application state in predictable ways.
- **Key Features**:
  - Unidirectional data flow.
  - Centralized state management.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Store {
      +dispatch(action: Action): void
      +getState(): State
    }
    class Action {
      +type: string
    }
    class Reducer {
      +reduce(state: State, action: Action): State
    }
    Store --> Action
    Store --> Reducer
  ```

- **Key Participants**: Store, Action, Reducer.
- **Applicability**: Use when you need predictable state management.
- **Example**: Large-scale applications requiring complex state management.

#### Microservices Architecture

- **Category**: Architectural
- **Intent**: Design applications as suites of independently deployable services.
- **Key Features**:
  - Service independence.
  - Scalability.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Service {
      +operation(): void
    }
    class API {
      +request(): void
    }
    class Database {
      +query(): void
    }
    Service --> API
    Service --> Database
  ```

- **Key Participants**: Service, API, Database.
- **Applicability**: Use when you need scalable and independently deployable services.
- **Example**: Scalable backend systems.

#### Event-Driven Architecture

- **Category**: Architectural
- **Intent**: Build systems that react to events.
- **Key Features**:
  - Asynchronous communication.
  - Event-driven processing.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Event {
      +type: string
    }
    class EventHandler {
      +handle(event: Event): void
    }
    class EventBus {
      +publish(event: Event): void
      +subscribe(handler: EventHandler): void
    }
    EventBus --> Event
    EventBus --> EventHandler
  ```

- **Key Participants**: Event, EventHandler, EventBus.
- **Applicability**: Use when you want to decouple components through events.
- **Example**: Real-time applications.

#### Service-Oriented Architecture (SOA)

- **Category**: Architectural
- **Intent**: Structure applications around reusable services.
- **Key Features**:
  - Interoperability.
  - Reusability.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Service {
      +operation(): void
    }
    class Consumer {
      +requestService(): void
    }
    class ServiceRegistry {
      +register(service: Service): void
      +getService(name: string): Service
    }
    Consumer --> ServiceRegistry
    ServiceRegistry --> Service
  ```

- **Key Participants**: Service, Consumer, ServiceRegistry.
- **Applicability**: Use when you need interoperable and reusable services.
- **Example**: Enterprise applications.

#### Hexagonal Architecture (Ports and Adapters)

- **Category**: Architectural
- **Intent**: Isolate application core from external factors via ports and adapters.
- **Key Features**:
  - Strong separation of concerns.
  - Flexibility.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class Core {
      +businessLogic(): void
    }
    class Port {
      +request(): void
    }
    class Adapter {
      +adapt(): void
    }
    Core --> Port
    Port --> Adapter
  ```

- **Key Participants**: Core, Port, Adapter.
- **Applicability**: Use when you want to isolate application logic from external dependencies.
- **Example**: Applications requiring strong separation of concerns.

#### Micro Frontends

- **Category**: Architectural
- **Intent**: Architect front-end applications as independent microservices.
- **Key Features**:
  - Independent deployment.
  - Fragmentation.
- **Structure Diagram**:

  ```mermaid
  classDiagram
    class MicroFrontend {
      +render(): void
    }
    class Container {
      +loadMicroFrontend(): void
    }
    MicroFrontend --> Container
  ```

- **Key Participants**: MicroFrontend, Container.
- **Applicability**: Use when you need to manage large-scale front-end applications.
- **Example**: Large-scale front-end applications.

---

## Quiz Time!

{{< quizdown >}}

### Which pattern ensures a class has only one instance?

- [x] Singleton Pattern
- [ ] Factory Method Pattern
- [ ] Prototype Pattern
- [ ] Builder Pattern

> **Explanation:** The Singleton Pattern ensures a class has only one instance and provides a global point of access to it.


### What is the main intent of the Factory Method Pattern?

- [x] Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- [ ] Provide an interface for creating families of related objects.
- [ ] Separate the construction of a complex object from its representation.
- [ ] Specify the kinds of objects to create using a prototypical instance.

> **Explanation:** The Factory Method Pattern defines an interface for creating an object, but lets subclasses alter the type of objects that will be created.


### Which pattern is used to compose objects into tree structures?

- [ ] Adapter Pattern
- [x] Composite Pattern
- [ ] Decorator Pattern
- [ ] Facade Pattern

> **Explanation:** The Composite Pattern is used to compose objects into tree structures to represent part-whole hierarchies.


### What is the primary purpose of the Proxy Pattern?

- [ ] Provide a simplified interface to a complex subsystem.
- [x] Provide a surrogate or placeholder for another object to control access.
- [ ] Allow incompatible interfaces to work together.
- [ ] Use sharing to support large numbers of fine-grained objects efficiently.

> **Explanation:** The Proxy Pattern provides a surrogate or placeholder for another object to control access.


### Which pattern is characterized by encapsulating a request as an object?

- [x] Command Pattern
- [ ] Observer Pattern
- [ ] Strategy Pattern
- [ ] State Pattern

> **Explanation:** The Command Pattern encapsulates a request as an object, allowing parameterization and queuing.


### What is the main benefit of the Observer Pattern?

- [x] Define a one-to-many dependency so that when one object changes state, all its dependents are notified.
- [ ] Allow an object to alter its behavior when its internal state changes.
- [ ] Define a family of algorithms, encapsulate each one, and make them interchangeable.
- [ ] Capture and restore an object's internal state without violating encapsulation.

> **Explanation:** The Observer Pattern defines a one-to-many dependency so that when one object changes state, all its dependents are notified.


### Which pattern is used to manage application state in predictable ways?

- [ ] MVC Pattern
- [ ] MVVM Pattern
- [x] Flux and Redux Architecture
- [ ] Microservices Architecture

> **Explanation:** Flux and Redux Architecture is used to manage application state in predictable ways.


### What is the primary intent of the Builder Pattern?

- [ ] Ensure a class has only one instance.
- [ ] Provide an interface for creating families of related objects.
- [x] Separate the construction of a complex object from its representation.
- [ ] Specify the kinds of objects to create using a prototypical instance.

> **Explanation:** The Builder Pattern separates the construction of a complex object from its representation, allowing the same construction process to create various representations.


### Which pattern involves a centralized communication object?

- [ ] Chain of Responsibility Pattern
- [ ] Command Pattern
- [x] Mediator Pattern
- [ ] Memento Pattern

> **Explanation:** The Mediator Pattern involves a centralized communication object that encapsulates how a set of objects interact.


### True or False: The Strategy Pattern allows for the interchangeability of algorithms.

- [x] True
- [ ] False

> **Explanation:** The Strategy Pattern allows for the interchangeability of algorithms by defining a family of algorithms, encapsulating each one, and making them interchangeable.

{{< /quizdown >}}

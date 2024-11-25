---
canonical: "https://softwarepatternslexicon.com/patterns-java/17/3"
title: "Design Patterns in Java: Comprehensive Reference Cheat Sheet"
description: "A quick-reference guide for all design patterns discussed, summarizing key information for easy recall and comparison."
linkTitle: "17.3 Pattern Reference Cheat Sheet"
categories:
- Java Design Patterns
- Software Engineering
- Programming
tags:
- Design Patterns
- Java
- Software Development
- Creational Patterns
- Structural Patterns
date: 2024-11-17
type: docs
nav_weight: 17300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 17.3 Pattern Reference Cheat Sheet

Welcome to the Pattern Reference Cheat Sheet, your go-to guide for a quick overview of design patterns in Java. This section is designed to provide expert developers with a concise yet comprehensive summary of each pattern, including its intent, applicability, and key features. Whether you're revisiting a familiar pattern or exploring new ones, this cheat sheet will serve as a handy reference.

### Creational Patterns

#### Singleton Pattern
- **Category**: Creational
- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Singleton {
      - Singleton uniqueInstance
      + getInstance() Singleton
    }
  ```
- **Key Participants**: Singleton
- **Applicability**: Use when exactly one instance of a class is needed, such as in logging, configuration settings, or device drivers.
- **Sample Code Snippet**:
  ```java
  public class Singleton {
      private static Singleton instance;

      private Singleton() {}

      public static Singleton getInstance() {
          if (instance == null) {
              instance = new Singleton();
          }
          return instance;
      }
  }
  ```

#### Factory Method Pattern
- **Category**: Creational
- **Intent**: Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Creator {
      + factoryMethod() : Product
    }
    class ConcreteCreator {
      + factoryMethod() : ConcreteProduct
    }
    class Product
    class ConcreteProduct
    Creator <|-- ConcreteCreator
    Product <|-- ConcreteProduct
    Creator --> Product
  ```
- **Key Participants**: Creator, ConcreteCreator, Product, ConcreteProduct
- **Applicability**: Use when a class can't anticipate the class of objects it must create or when a class wants its subclasses to specify the objects it creates.
- **Sample Code Snippet**:
  ```java
  abstract class Product {}
  
  class ConcreteProduct extends Product {}
  
  abstract class Creator {
      public abstract Product factoryMethod();
  }
  
  class ConcreteCreator extends Creator {
      public Product factoryMethod() {
          return new ConcreteProduct();
      }
  }
  ```

#### Abstract Factory Pattern
- **Category**: Creational
- **Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class AbstractFactory {
      + createProductA() : AbstractProductA
      + createProductB() : AbstractProductB
    }
    class ConcreteFactory1 {
      + createProductA() : ProductA1
      + createProductB() : ProductB1
    }
    class AbstractProductA
    class AbstractProductB
    class ProductA1
    class ProductB1
    AbstractFactory <|-- ConcreteFactory1
    AbstractProductA <|-- ProductA1
    AbstractProductB <|-- ProductB1
    AbstractFactory --> AbstractProductA
    AbstractFactory --> AbstractProductB
  ```
- **Key Participants**: AbstractFactory, ConcreteFactory, AbstractProduct, ConcreteProduct
- **Applicability**: Use when a system should be independent of how its products are created, composed, and represented.
- **Sample Code Snippet**:
  ```java
  interface AbstractFactory {
      ProductA createProductA();
      ProductB createProductB();
  }

  class ConcreteFactory1 implements AbstractFactory {
      public ProductA createProductA() {
          return new ProductA1();
      }
      public ProductB createProductB() {
          return new ProductB1();
      }
  }
  ```

#### Builder Pattern
- **Category**: Creational
- **Intent**: Separate the construction of a complex object from its representation so that the same construction process can create different representations.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Builder {
      + buildPart() : void
    }
    class ConcreteBuilder {
      + buildPart() : void
      + getResult() : Product
    }
    class Director {
      + construct() : void
    }
    class Product
    Builder <|-- ConcreteBuilder
    Director --> Builder
    ConcreteBuilder --> Product
  ```
- **Key Participants**: Builder, ConcreteBuilder, Director, Product
- **Applicability**: Use when the algorithm for creating a complex object should be independent of the parts that make up the object and how they're assembled.
- **Sample Code Snippet**:
  ```java
  class Product {
      private String partA;
      private String partB;

      public void setPartA(String partA) { this.partA = partA; }
      public void setPartB(String partB) { this.partB = partB; }
  }

  abstract class Builder {
      protected Product product = new Product();
      public abstract void buildPartA();
      public abstract void buildPartB();
      public Product getResult() { return product; }
  }

  class ConcreteBuilder extends Builder {
      public void buildPartA() { product.setPartA("Part A"); }
      public void buildPartB() { product.setPartB("Part B"); }
  }

  class Director {
      private Builder builder;
      public Director(Builder builder) { this.builder = builder; }
      public void construct() {
          builder.buildPartA();
          builder.buildPartB();
      }
  }
  ```

#### Prototype Pattern
- **Category**: Creational
- **Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Prototype {
      + clone() : Prototype
    }
    class ConcretePrototype1 {
      + clone() : ConcretePrototype1
    }
    class ConcretePrototype2 {
      + clone() : ConcretePrototype2
    }
    Prototype <|-- ConcretePrototype1
    Prototype <|-- ConcretePrototype2
  ```
- **Key Participants**: Prototype, ConcretePrototype
- **Applicability**: Use when a system should be independent of how its products are created, composed, and represented, or when classes to instantiate are specified at runtime.
- **Sample Code Snippet**:
  ```java
  interface Prototype {
      Prototype clone();
  }

  class ConcretePrototype implements Prototype {
      private String field;

      public ConcretePrototype(String field) {
          this.field = field;
      }

      public Prototype clone() {
          return new ConcretePrototype(this.field);
      }
  }
  ```

#### Object Pool Pattern
- **Category**: Creational
- **Intent**: Manage a set of initialized objects ready to be used, rather than creating and destroying them on demand.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class ObjectPool {
      + acquire() : PooledObject
      + release(PooledObject) : void
    }
    class PooledObject
    ObjectPool --> PooledObject
  ```
- **Key Participants**: ObjectPool, PooledObject
- **Applicability**: Use when the cost of initializing a class instance is high, the rate of instantiation is high, or the number of instances in use at any one time is low.
- **Sample Code Snippet**:
  ```java
  class PooledObject {}

  class ObjectPool {
      private List<PooledObject> available = new ArrayList<>();

      public PooledObject acquire() {
          if (available.isEmpty()) {
              return new PooledObject();
          }
          return available.remove(available.size() - 1);
      }

      public void release(PooledObject obj) {
          available.add(obj);
      }
  }
  ```

#### Dependency Injection Pattern
- **Category**: Creational
- **Intent**: A technique whereby one object supplies the dependencies of another object.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Client {
      - Service service
      + setService(Service) : void
    }
    class Service
    Client --> Service
  ```
- **Key Participants**: Client, Service
- **Applicability**: Use when you want to decouple the creation of a client's dependencies from the client's behavior.
- **Sample Code Snippet**:
  ```java
  interface Service {
      void execute();
  }

  class ServiceImpl implements Service {
      public void execute() {
          System.out.println("Service Executed");
      }
  }

  class Client {
      private Service service;

      public void setService(Service service) {
          this.service = service;
      }

      public void doSomething() {
          service.execute();
      }
  }
  ```

### Structural Patterns

#### Adapter Pattern
- **Category**: Structural
- **Intent**: Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Target {
      + request() : void
    }
    class Adapter {
      + request() : void
    }
    class Adaptee {
      + specificRequest() : void
    }
    Target <|-- Adapter
    Adapter --> Adaptee
  ```
- **Key Participants**: Target, Adapter, Adaptee
- **Applicability**: Use when you want to use an existing class, and its interface does not match the one you need.
- **Sample Code Snippet**:
  ```java
  interface Target {
      void request();
  }

  class Adaptee {
      public void specificRequest() {
          System.out.println("Specific Request");
      }
  }

  class Adapter implements Target {
      private Adaptee adaptee;

      public Adapter(Adaptee adaptee) {
          this.adaptee = adaptee;
      }

      public void request() {
          adaptee.specificRequest();
      }
  }
  ```

#### Bridge Pattern
- **Category**: Structural
- **Intent**: Decouple an abstraction from its implementation so that the two can vary independently.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Abstraction {
      + operation() : void
    }
    class RefinedAbstraction {
      + operation() : void
    }
    class Implementor {
      + operationImpl() : void
    }
    class ConcreteImplementorA {
      + operationImpl() : void
    }
    class ConcreteImplementorB {
      + operationImpl() : void
    }
    Abstraction <|-- RefinedAbstraction
    Abstraction --> Implementor
    Implementor <|-- ConcreteImplementorA
    Implementor <|-- ConcreteImplementorB
  ```
- **Key Participants**: Abstraction, Implementor
- **Applicability**: Use when you want to avoid a permanent binding between an abstraction and its implementation.
- **Sample Code Snippet**:
  ```java
  interface Implementor {
      void operationImpl();
  }

  class ConcreteImplementorA implements Implementor {
      public void operationImpl() {
          System.out.println("ConcreteImplementorA");
      }
  }

  abstract class Abstraction {
      protected Implementor implementor;

      protected Abstraction(Implementor implementor) {
          this.implementor = implementor;
      }

      public abstract void operation();
  }

  class RefinedAbstraction extends Abstraction {
      public RefinedAbstraction(Implementor implementor) {
          super(implementor);
      }

      public void operation() {
          implementor.operationImpl();
      }
  }
  ```

#### Composite Pattern
- **Category**: Structural
- **Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Component {
      + operation() : void
    }
    class Leaf {
      + operation() : void
    }
    class Composite {
      + operation() : void
      + add(Component) : void
      + remove(Component) : void
    }
    Component <|-- Leaf
    Component <|-- Composite
    Composite --> Component
  ```
- **Key Participants**: Component, Leaf, Composite
- **Applicability**: Use when you want to represent part-whole hierarchies of objects.
- **Sample Code Snippet**:
  ```java
  interface Component {
      void operation();
  }

  class Leaf implements Component {
      public void operation() {
          System.out.println("Leaf");
      }
  }

  class Composite implements Component {
      private List<Component> children = new ArrayList<>();

      public void operation() {
          for (Component child : children) {
              child.operation();
          }
      }

      public void add(Component component) {
          children.add(component);
      }

      public void remove(Component component) {
          children.remove(component);
      }
  }
  ```

#### Decorator Pattern
- **Category**: Structural
- **Intent**: Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Component {
      + operation() : void
    }
    class ConcreteComponent {
      + operation() : void
    }
    class Decorator {
      + operation() : void
    }
    class ConcreteDecoratorA {
      + operation() : void
    }
    class ConcreteDecoratorB {
      + operation() : void
    }
    Component <|-- ConcreteComponent
    Component <|-- Decorator
    Decorator <|-- ConcreteDecoratorA
    Decorator <|-- ConcreteDecoratorB
    Decorator --> Component
  ```
- **Key Participants**: Component, ConcreteComponent, Decorator
- **Applicability**: Use to add responsibilities to individual objects dynamically and transparently, without affecting other objects.
- **Sample Code Snippet**:
  ```java
  interface Component {
      void operation();
  }

  class ConcreteComponent implements Component {
      public void operation() {
          System.out.println("ConcreteComponent");
      }
  }

  abstract class Decorator implements Component {
      protected Component component;

      public Decorator(Component component) {
          this.component = component;
      }

      public void operation() {
          component.operation();
      }
  }

  class ConcreteDecoratorA extends Decorator {
      public ConcreteDecoratorA(Component component) {
          super(component);
      }

      public void operation() {
          super.operation();
          System.out.println("ConcreteDecoratorA");
      }
  }
  ```

#### Facade Pattern
- **Category**: Structural
- **Intent**: Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Facade {
      + operation() : void
    }
    class SubsystemClass1 {
      + operation1() : void
    }
    class SubsystemClass2 {
      + operation2() : void
    }
    Facade --> SubsystemClass1
    Facade --> SubsystemClass2
  ```
- **Key Participants**: Facade, Subsystem classes
- **Applicability**: Use when you want to provide a simple interface to a complex subsystem.
- **Sample Code Snippet**:
  ```java
  class SubsystemClass1 {
      public void operation1() {
          System.out.println("SubsystemClass1 operation1");
      }
  }

  class SubsystemClass2 {
      public void operation2() {
          System.out.println("SubsystemClass2 operation2");
      }
  }

  class Facade {
      private SubsystemClass1 subsystem1;
      private SubsystemClass2 subsystem2;

      public Facade() {
          subsystem1 = new SubsystemClass1();
          subsystem2 = new SubsystemClass2();
      }

      public void operation() {
          subsystem1.operation1();
          subsystem2.operation2();
      }
  }
  ```

#### Flyweight Pattern
- **Category**: Structural
- **Intent**: Use sharing to support large numbers of fine-grained objects efficiently.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Flyweight {
      + operation(extrinsicState) : void
    }
    class ConcreteFlyweight {
      + operation(extrinsicState) : void
    }
    class FlyweightFactory {
      + getFlyweight(key) : Flyweight
    }
    Flyweight <|-- ConcreteFlyweight
    FlyweightFactory --> Flyweight
  ```
- **Key Participants**: Flyweight, ConcreteFlyweight, FlyweightFactory
- **Applicability**: Use when you need to create a large number of similar objects and want to reduce memory usage.
- **Sample Code Snippet**:
  ```java
  interface Flyweight {
      void operation(String extrinsicState);
  }

  class ConcreteFlyweight implements Flyweight {
      private String intrinsicState;

      public ConcreteFlyweight(String intrinsicState) {
          this.intrinsicState = intrinsicState;
      }

      public void operation(String extrinsicState) {
          System.out.println("Intrinsic: " + intrinsicState + ", Extrinsic: " + extrinsicState);
      }
  }

  class FlyweightFactory {
      private Map<String, Flyweight> flyweights = new HashMap<>();

      public Flyweight getFlyweight(String key) {
          if (!flyweights.containsKey(key)) {
              flyweights.put(key, new ConcreteFlyweight(key));
          }
          return flyweights.get(key);
      }
  }
  ```

#### Proxy Pattern
- **Category**: Structural
- **Intent**: Provide a surrogate or placeholder for another object to control access to it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Subject {
      + request() : void
    }
    class RealSubject {
      + request() : void
    }
    class Proxy {
      + request() : void
    }
    Subject <|-- RealSubject
    Subject <|-- Proxy
    Proxy --> RealSubject
  ```
- **Key Participants**: Subject, RealSubject, Proxy
- **Applicability**: Use when you need a more versatile or sophisticated reference to an object than a simple pointer.
- **Sample Code Snippet**:
  ```java
  interface Subject {
      void request();
  }

  class RealSubject implements Subject {
      public void request() {
          System.out.println("RealSubject request");
      }
  }

  class Proxy implements Subject {
      private RealSubject realSubject;

      public Proxy() {
          this.realSubject = new RealSubject();
      }

      public void request() {
          System.out.println("Proxy request");
          realSubject.request();
      }
  }
  ```

### Behavioral Patterns

#### Chain of Responsibility Pattern
- **Category**: Behavioral
- **Intent**: Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Handler {
      + handleRequest() : void
    }
    class ConcreteHandler1 {
      + handleRequest() : void
    }
    class ConcreteHandler2 {
      + handleRequest() : void
    }
    Handler <|-- ConcreteHandler1
    Handler <|-- ConcreteHandler2
    Handler --> Handler
  ```
- **Key Participants**: Handler, ConcreteHandler
- **Applicability**: Use when more than one object may handle a request, and the handler isn't known a priori.
- **Sample Code Snippet**:
  ```java
  abstract class Handler {
      protected Handler successor;

      public void setSuccessor(Handler successor) {
          this.successor = successor;
      }

      public abstract void handleRequest(String request);
  }

  class ConcreteHandler1 extends Handler {
      public void handleRequest(String request) {
          if (request.equals("ConcreteHandler1")) {
              System.out.println("Handled by ConcreteHandler1");
          } else if (successor != null) {
              successor.handleRequest(request);
          }
      }
  }

  class ConcreteHandler2 extends Handler {
      public void handleRequest(String request) {
          if (request.equals("ConcreteHandler2")) {
              System.out.println("Handled by ConcreteHandler2");
          } else if (successor != null) {
              successor.handleRequest(request);
          }
      }
  }
  ```

#### Command Pattern
- **Category**: Behavioral
- **Intent**: Encapsulate a request as an object, thereby letting you parameterize clients with queues, requests, and operations.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Command {
      + execute() : void
    }
    class ConcreteCommand {
      + execute() : void
    }
    class Invoker {
      + setCommand(Command) : void
      + executeCommand() : void
    }
    class Receiver {
      + action() : void
    }
    Command <|-- ConcreteCommand
    Invoker --> Command
    ConcreteCommand --> Receiver
  ```
- **Key Participants**: Command, ConcreteCommand, Invoker, Receiver
- **Applicability**: Use when you want to parameterize objects with operations, queue operations, or support undoable operations.
- **Sample Code Snippet**:
  ```java
  interface Command {
      void execute();
  }

  class Receiver {
      public void action() {
          System.out.println("Receiver action");
      }
  }

  class ConcreteCommand implements Command {
      private Receiver receiver;

      public ConcreteCommand(Receiver receiver) {
          this.receiver = receiver;
      }

      public void execute() {
          receiver.action();
      }
  }

  class Invoker {
      private Command command;

      public void setCommand(Command command) {
          this.command = command;
      }

      public void executeCommand() {
          command.execute();
      }
  }
  ```

#### Interpreter Pattern
- **Category**: Behavioral
- **Intent**: Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class AbstractExpression {
      + interpret(Context) : void
    }
    class TerminalExpression {
      + interpret(Context) : void
    }
    class NonterminalExpression {
      + interpret(Context) : void
    }
    AbstractExpression <|-- TerminalExpression
    AbstractExpression <|-- NonterminalExpression
  ```
- **Key Participants**: AbstractExpression, TerminalExpression, NonterminalExpression
- **Applicability**: Use when you have a language to interpret, and you can represent statements in the language as abstract syntax trees.
- **Sample Code Snippet**:
  ```java
  interface Expression {
      boolean interpret(String context);
  }

  class TerminalExpression implements Expression {
      private String data;

      public TerminalExpression(String data) {
          this.data = data;
      }

      public boolean interpret(String context) {
          return context.contains(data);
      }
  }

  class OrExpression implements Expression {
      private Expression expr1;
      private Expression expr2;

      public OrExpression(Expression expr1, Expression expr2) {
          this.expr1 = expr1;
          this.expr2 = expr2;
      }

      public boolean interpret(String context) {
          return expr1.interpret(context) || expr2.interpret(context);
      }
  }
  ```

#### Iterator Pattern
- **Category**: Behavioral
- **Intent**: Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Iterator {
      + first() : void
      + next() : void
      + isDone() : boolean
      + currentItem() : Object
    }
    class ConcreteIterator {
      + first() : void
      + next() : void
      + isDone() : boolean
      + currentItem() : Object
    }
    class Aggregate {
      + createIterator() : Iterator
    }
    class ConcreteAggregate {
      + createIterator() : Iterator
    }
    Iterator <|-- ConcreteIterator
    Aggregate <|-- ConcreteAggregate
    ConcreteAggregate --> Iterator
  ```
- **Key Participants**: Iterator, ConcreteIterator, Aggregate, ConcreteAggregate
- **Applicability**: Use when you need to access an aggregate object's contents without exposing its internal representation.
- **Sample Code Snippet**:
  ```java
  interface Iterator {
      boolean hasNext();
      Object next();
  }

  class NameRepository {
      private String[] names = {"John", "Jane", "Doe"};

      public Iterator getIterator() {
          return new NameIterator();
      }

      private class NameIterator implements Iterator {
          int index;

          public boolean hasNext() {
              return index < names.length;
          }

          public Object next() {
              if (this.hasNext()) {
                  return names[index++];
              }
              return null;
          }
      }
  }
  ```

#### Mediator Pattern
- **Category**: Behavioral
- **Intent**: Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Mediator {
      + notify(sender, event) : void
    }
    class ConcreteMediator {
      + notify(sender, event) : void
    }
    class Colleague {
      + send() : void
      + receive() : void
    }
    class ConcreteColleague1 {
      + send() : void
      + receive() : void
    }
    class ConcreteColleague2 {
      + send() : void
      + receive() : void
    }
    Mediator <|-- ConcreteMediator
    Colleague <|-- ConcreteColleague1
    Colleague <|-- ConcreteColleague2
    ConcreteMediator --> Colleague
  ```
- **Key Participants**: Mediator, ConcreteMediator, Colleague
- **Applicability**: Use when a set of objects communicate in well-defined but complex ways.
- **Sample Code Snippet**:
  ```java
  interface Mediator {
      void sendMessage(String message, Colleague colleague);
  }

  abstract class Colleague {
      protected Mediator mediator;

      public Colleague(Mediator mediator) {
          this.mediator = mediator;
      }
  }

  class ConcreteColleague1 extends Colleague {
      public ConcreteColleague1(Mediator mediator) {
          super(mediator);
      }

      public void send(String message) {
          mediator.sendMessage(message, this);
      }

      public void receive(String message) {
          System.out.println("Colleague1 received: " + message);
      }
  }

  class ConcreteMediator implements Mediator {
      private ConcreteColleague1 colleague1;
      private ConcreteColleague2 colleague2;

      public void setColleague1(ConcreteColleague1 colleague1) {
          this.colleague1 = colleague1;
      }

      public void setColleague2(ConcreteColleague2 colleague2) {
          this.colleague2 = colleague2;
      }

      public void sendMessage(String message, Colleague colleague) {
          if (colleague == colleague1) {
              colleague2.receive(message);
          } else {
              colleague1.receive(message);
          }
      }
  }
  ```

#### Memento Pattern
- **Category**: Behavioral
- **Intent**: Without violating encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Memento {
      + getState() : State
    }
    class Originator {
      + setMemento(Memento) : void
      + createMemento() : Memento
    }
    class Caretaker {
      + addMemento(Memento) : void
      + getMemento(index) : Memento
    }
    Originator --> Memento
    Caretaker --> Memento
  ```
- **Key Participants**: Memento, Originator, Caretaker
- **Applicability**: Use when you need to save and restore the state of an object.
- **Sample Code Snippet**:
  ```java
  class Memento {
      private String state;

      public Memento(String state) {
          this.state = state;
      }

      public String getState() {
          return state;
      }
  }

  class Originator {
      private String state;

      public void setState(String state) {
          this.state = state;
      }

      public String getState() {
          return state;
      }

      public Memento saveStateToMemento() {
          return new Memento(state);
      }

      public void getStateFromMemento(Memento memento) {
          state = memento.getState();
      }
  }

  class Caretaker {
      private List<Memento> mementoList = new ArrayList<>();

      public void add(Memento state) {
          mementoList.add(state);
      }

      public Memento get(int index) {
          return mementoList.get(index);
      }
  }
  ```

#### Observer Pattern
- **Category**: Behavioral
- **Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Subject {
      + attach(Observer) : void
      + detach(Observer) : void
      + notify() : void
    }
    class ConcreteSubject {
      + getState() : State
      + setState(State) : void
    }
    class Observer {
      + update() : void
    }
    class ConcreteObserver {
      + update() : void
    }
    Subject <|-- ConcreteSubject
    Observer <|-- ConcreteObserver
    ConcreteSubject --> Observer
  ```
- **Key Participants**: Subject, Observer
- **Applicability**: Use when an abstraction has two aspects, one dependent on the other.
- **Sample Code Snippet**:
  ```java
  interface Observer {
      void update(String state);
  }

  class ConcreteObserver implements Observer {
      private String observerState;

      public void update(String state) {
          observerState = state;
          System.out.println("Observer state updated to: " + observerState);
      }
  }

  class Subject {
      private List<Observer> observers = new ArrayList<>();
      private String state;

      public void attach(Observer observer) {
          observers.add(observer);
      }

      public void detach(Observer observer) {
          observers.remove(observer);
      }

      public void notifyObservers() {
          for (Observer observer : observers) {
              observer.update(state);
          }
      }

      public void setState(String state) {
          this.state = state;
          notifyObservers();
      }
  }
  ```

#### State Pattern
- **Category**: Behavioral
- **Intent**: Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Context {
      + request() : void
    }
    class State {
      + handle(Context) : void
    }
    class ConcreteStateA {
      + handle(Context) : void
    }
    class ConcreteStateB {
      + handle(Context) : void
    }
    Context --> State
    State <|-- ConcreteStateA
    State <|-- ConcreteStateB
  ```
- **Key Participants**: Context, State
- **Applicability**: Use when an object's behavior depends on its state and it must change its behavior at runtime depending on that state.
- **Sample Code Snippet**:
  ```java
  interface State {
      void doAction(Context context);
  }

  class ConcreteStateA implements State {
      public void doAction(Context context) {
          System.out.println("State A");
          context.setState(this);
      }
  }

  class ConcreteStateB implements State {
      public void doAction(Context context) {
          System.out.println("State B");
          context.setState(this);
      }
  }

  class Context {
      private State state;

      public void setState(State state) {
          this.state = state;
      }

      public State getState() {
          return state;
      }
  }
  ```

#### Strategy Pattern
- **Category**: Behavioral
- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Context {
      + setStrategy(Strategy) : void
      + executeStrategy() : void
    }
    class Strategy {
      + algorithmInterface() : void
    }
    class ConcreteStrategyA {
      + algorithmInterface() : void
    }
    class ConcreteStrategyB {
      + algorithmInterface() : void
    }
    Context --> Strategy
    Strategy <|-- ConcreteStrategyA
    Strategy <|-- ConcreteStrategyB
  ```
- **Key Participants**: Context, Strategy
- **Applicability**: Use when you want to define a class that has a behavior that can be changed at runtime.
- **Sample Code Snippet**:
  ```java
  interface Strategy {
      void execute();
  }

  class ConcreteStrategyA implements Strategy {
      public void execute() {
          System.out.println("Strategy A");
      }
  }

  class ConcreteStrategyB implements Strategy {
      public void execute() {
          System.out.println("Strategy B");
      }
  }

  class Context {
      private Strategy strategy;

      public void setStrategy(Strategy strategy) {
          this.strategy = strategy;
      }

      public void executeStrategy() {
          strategy.execute();
      }
  }
  ```

#### Template Method Pattern
- **Category**: Behavioral
- **Intent**: Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class AbstractClass {
      + templateMethod() : void
      + primitiveOperation1() : void
      + primitiveOperation2() : void
    }
    class ConcreteClass {
      + primitiveOperation1() : void
      + primitiveOperation2() : void
    }
    AbstractClass <|-- ConcreteClass
  ```
- **Key Participants**: AbstractClass, ConcreteClass
- **Applicability**: Use to implement the invariant parts of an algorithm once and leave it up to subclasses to implement the behavior that can vary.
- **Sample Code Snippet**:
  ```java
  abstract class AbstractClass {
      public final void templateMethod() {
          primitiveOperation1();
          primitiveOperation2();
      }

      protected abstract void primitiveOperation1();
      protected abstract void primitiveOperation2();
  }

  class ConcreteClass extends AbstractClass {
      protected void primitiveOperation1() {
          System.out.println("ConcreteClass Operation1");
      }

      protected void primitiveOperation2() {
          System.out.println("ConcreteClass Operation2");
      }
  }
  ```

#### Visitor Pattern
- **Category**: Behavioral
- **Intent**: Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.
- **Structure Diagram**:
  ```mermaid
  classDiagram
    class Visitor {
      + visitConcreteElementA(ConcreteElementA) : void
      + visitConcreteElementB(ConcreteElementB) : void
    }
    class ConcreteVisitor1 {
      + visitConcreteElementA(ConcreteElementA) : void
      + visitConcreteElementB(ConcreteElementB) : void
    }
    class Element {
      + accept(Visitor) : void
    }
    class ConcreteElementA {
      + accept(Visitor) : void
    }
    class ConcreteElementB {
      + accept(Visitor) : void
    }
    Visitor <|-- ConcreteVisitor1
    Element <|-- ConcreteElementA
    Element <|-- ConcreteElementB
    ConcreteElementA --> Visitor
    ConcreteElementB --> Visitor
  ```
- **Key Participants**: Visitor, ConcreteVisitor, Element
- **Applicability**: Use when you need to perform operations across a set of objects that have different interfaces.
- **Sample Code Snippet**:
  ```java
  interface Visitor {
      void visit(ConcreteElementA element);
      void visit(ConcreteElementB element);
  }

  class ConcreteVisitor implements Visitor {
      public void visit(ConcreteElementA element) {
          System.out.println("Visited ConcreteElementA");
      }

      public void visit(ConcreteElementB element) {
          System.out.println("Visited ConcreteElementB");
      }
  }

  interface Element {
      void accept(Visitor visitor);
  }

  class ConcreteElementA implements Element {
      public void accept(Visitor visitor) {
          visitor.visit(this);
      }
  }

  class ConcreteElementB implements Element {
      public void accept(Visitor visitor) {
          visitor.visit(this);
      }
  }
  ```

### Design Considerations

- **Singleton vs. Static Class**: Singleton ensures a single instance with controlled access, while a static class provides static methods without instance control.
- **Factory Method vs. Abstract Factory**: Factory Method is for creating a single product, while Abstract Factory is for creating families of related products.
- **Adapter vs. Bridge**: Adapter is used for making unrelated classes work together, while Bridge separates abstraction from implementation.
- **Decorator vs. Proxy**: Decorator adds responsibilities to objects, whereas Proxy controls access to them.
- **Strategy vs. State**: Strategy changes the behavior of a class by switching algorithms, while State changes behavior by switching states.
- **Visitor vs. Iterator**: Visitor is for performing operations on elements, while Iterator is for accessing elements sequentially.

### Printable Format

This cheat sheet is designed to fit on a few pages for easy printing and quick reference. Keep it handy as you work through your Java projects to quickly recall the purpose and structure of each design pattern.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Singleton pattern?

- [x] Ensure a class has only one instance and provide a global point of access to it.
- [ ] Provide a way to access elements of a collection sequentially.
- [ ] Define a family of algorithms and make them interchangeable.
- [ ] Represent an operation to be performed on elements of an object structure.

> **Explanation:** The Singleton pattern ensures a class has only one instance and provides a global point of access to it.

### Which pattern is used to convert the interface of a class into another interface clients expect?

- [ ] Proxy
- [x] Adapter
- [ ] Bridge
- [ ] Decorator

> **Explanation:** The Adapter pattern is used to convert the interface of a class into another interface clients expect.

### What is the key difference between the Factory Method and Abstract Factory patterns?

- [x] Factory Method is for creating a single product, while Abstract Factory is for creating families of related products.
- [ ] Factory Method is for creating families of related products, while Abstract Factory is for creating a single product.
- [ ] Factory Method is used for creating objects, while Abstract Factory is used for creating classes.
- [ ] Factory Method is used for creating classes, while Abstract Factory is used for creating objects.

> **Explanation:** Factory Method is for creating a single product, while Abstract Factory is for creating families of related products.

### Which pattern allows an object to alter its behavior when its internal state changes?

- [ ] Strategy
- [ ] Visitor
- [x] State
- [ ] Command

> **Explanation:** The State pattern allows an object to alter its behavior when its internal state changes.

### In which pattern do you encapsulate a request as an object?

- [ ] Observer
- [x] Command
- [ ] Mediator
- [ ] Memento

> **Explanation:** The Command pattern encapsulates a request as an object.

### What is the primary purpose of the Decorator pattern?

- [ ] Provide a surrogate or placeholder for another object to control access to it.
- [x] Attach additional responsibilities to an object dynamically.
- [ ] Define a one-to-many dependency between objects.
- [ ] Allow an object to alter its behavior when its internal state changes.

> **Explanation:** The Decorator pattern attaches additional responsibilities to an object dynamically.

### Which pattern defines a one-to-many dependency between objects?

- [ ] Strategy
- [ ] State
- [x] Observer
- [ ] Visitor

> **Explanation:** The Observer pattern defines a one-to-many dependency between objects.

### What is the main goal of the Proxy pattern?

- [ ] Convert the interface of a class into another interface clients expect.
- [ ] Compose objects into tree structures to represent part-whole hierarchies.
- [x] Provide a surrogate or placeholder for another object to control access to it.
- [ ] Define a family of algorithms and make them interchangeable.

> **Explanation:** The Proxy pattern provides a surrogate or placeholder for another object to control access to it.

### Which pattern is used to provide a way to access elements of a collection sequentially without exposing its underlying representation?

- [ ] Visitor
- [ ] Command
- [ ] Observer
- [x] Iterator

> **Explanation:** The Iterator pattern provides a way to access elements of a collection sequentially without exposing its underlying representation.

### True or False: The Bridge pattern is used to separate abstraction from implementation.

- [x] True
- [ ] False

> **Explanation:** True. The Bridge pattern is used to decouple an abstraction from its implementation so that the two can vary independently.

{{< /quizdown >}}

Remember, this cheat sheet is just the beginning. As you progress, you'll build more complex and interactive applications. Keep experimenting, stay curious, and enjoy the journey!

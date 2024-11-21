---
canonical: "https://softwarepatternslexicon.com/patterns-python/18/3"
title: "Design Patterns in Python: Comprehensive Pattern Reference Cheat Sheet"
description: "Explore a quick-reference guide summarizing all design patterns in Python, including Creational, Structural, Behavioral, and Concurrency patterns. Understand their intent, structure, key participants, applicability, and see sample code snippets."
linkTitle: "18.3 Pattern Reference Cheat Sheet"
categories:
- Python
- Design Patterns
- Software Development
tags:
- Creational Patterns
- Structural Patterns
- Behavioral Patterns
- Concurrency Patterns
- Python Programming
date: 2024-11-17
type: docs
nav_weight: 18300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
canonical: "https://softwarepatternslexicon.com/patterns-python/18/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 18.3 Pattern Reference Cheat Sheet

Welcome to the comprehensive pattern reference cheat sheet for design patterns in Python. This guide is designed to offer quick insights into various design patterns, providing you with a handy reference to their intent, structure, key participants, applicability, and sample code snippets. Whether you're a seasoned developer or just getting started with design patterns, this cheat sheet will serve as a valuable resource for your software development journey.

### Creational Patterns

#### 1. Abstract Factory Pattern

- **Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- **Structure**:

  ```mermaid
  classDiagram
    AbstractFactory <|-- ConcreteFactory
    AbstractProductA <|-- ProductA1
    AbstractProductB <|-- ProductB1
    ConcreteFactory --> ProductA1
    ConcreteFactory --> ProductB1
  ```

- **Key Participants**:
  - `AbstractFactory`
  - `ConcreteFactory`
  - `AbstractProduct`
  - `ConcreteProduct`

- **Applicability**: Use when a system should be independent of how its products are created, composed, and represented.

- **Sample Code Snippet**:

  ```python
  class AbstractFactory:
      def create_product_a(self):
          pass

      def create_product_b(self):
          pass

  class ConcreteFactory1(AbstractFactory):
      def create_product_a(self):
          return ProductA1()

      def create_product_b(self):
          return ProductB1()

  class ProductA1:
      pass

  class ProductB1:
      pass
  ```

#### 2. Builder Pattern

- **Intent**: Separate the construction of a complex object from its representation so that the same construction process can create different representations.
- **Structure**:

  ```mermaid
  classDiagram
    Director --> Builder
    Builder <|-- ConcreteBuilder
    ConcreteBuilder --> Product
  ```

- **Key Participants**:
  - `Builder`
  - `ConcreteBuilder`
  - `Director`
  - `Product`

- **Applicability**: Use when the algorithm for creating a complex object should be independent of the parts that make up the object and how they're assembled.

- **Sample Code Snippet**:

  ```python
  class Builder:
      def build_part(self):
          pass

  class ConcreteBuilder(Builder):
      def __init__(self):
          self.product = Product()

      def build_part(self):
          self.product.add("Part")

  class Product:
      def __init__(self):
          self.parts = []

      def add(self, part):
          self.parts.append(part)
  ```

#### 3. Factory Method Pattern

- **Intent**: Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- **Structure**:

  ```mermaid
  classDiagram
    Creator <|-- ConcreteCreator
    Creator --> Product
    ConcreteCreator --> ConcreteProduct
  ```

- **Key Participants**:
  - `Creator`
  - `ConcreteCreator`
  - `Product`
  - `ConcreteProduct`

- **Applicability**: Use when a class can't anticipate the class of objects it must create.

- **Sample Code Snippet**:

  ```python
  class Creator:
      def factory_method(self):
          pass

  class ConcreteCreator(Creator):
      def factory_method(self):
          return ConcreteProduct()

  class ConcreteProduct:
      pass
  ```

#### 4. Prototype Pattern

- **Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
- **Structure**:

  ```mermaid
  classDiagram
    Prototype <|-- ConcretePrototype
    Client --> Prototype
  ```

- **Key Participants**:
  - `Prototype`
  - `ConcretePrototype`
  - `Client`

- **Applicability**: Use when a system should be independent of how its products are created, composed, and represented.

- **Sample Code Snippet**:

  ```python
  import copy

  class Prototype:
      def clone(self):
          return copy.deepcopy(self)

  class ConcretePrototype(Prototype):
      def __init__(self, value):
          self.value = value
  ```

#### 5. Singleton Pattern

- **Intent**: Ensure a class has only one instance and provide a global point of access to it.
- **Structure**:

  ```mermaid
  classDiagram
    Singleton --> Singleton
  ```

- **Key Participants**:
  - `Singleton`

- **Applicability**: Use when exactly one instance of a class is needed to control actions.

- **Sample Code Snippet**:

  ```python
  class Singleton:
      _instance = None

      def __new__(cls, *args, **kwargs):
          if not cls._instance:
              cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
          return cls._instance
  ```

### Structural Patterns

#### 1. Adapter Pattern

- **Intent**: Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.
- **Structure**:

  ```mermaid
  classDiagram
    Target <|-- Adapter
    Adapter --> Adaptee
  ```

- **Key Participants**:
  - `Target`
  - `Adapter`
  - `Adaptee`

- **Applicability**: Use when you want to use an existing class, and its interface does not match the one you need.

- **Sample Code Snippet**:

  ```python
  class Target:
      def request(self):
          pass

  class Adaptee:
      def specific_request(self):
          return "Adaptee's specific request"

  class Adapter(Target):
      def __init__(self, adaptee):
          self.adaptee = adaptee

      def request(self):
          return self.adaptee.specific_request()
  ```

#### 2. Bridge Pattern

- **Intent**: Decouple an abstraction from its implementation so that the two can vary independently.
- **Structure**:

  ```mermaid
  classDiagram
    Abstraction <|-- RefinedAbstraction
    Implementor <|-- ConcreteImplementor
    Abstraction --> Implementor
  ```

- **Key Participants**:
  - `Abstraction`
  - `RefinedAbstraction`
  - `Implementor`
  - `ConcreteImplementor`

- **Applicability**: Use when you want to avoid a permanent binding between an abstraction and its implementation.

- **Sample Code Snippet**:

  ```python
  class Implementor:
      def operation_impl(self):
          pass

  class ConcreteImplementorA(Implementor):
      def operation_impl(self):
          return "ConcreteImplementorA operation"

  class Abstraction:
      def __init__(self, implementor):
          self.implementor = implementor

      def operation(self):
          return self.implementor.operation_impl()
  ```

#### 3. Composite Pattern

- **Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
- **Structure**:

  ```mermaid
  classDiagram
    Component <|-- Leaf
    Component <|-- Composite
    Composite --> Component
  ```

- **Key Participants**:
  - `Component`
  - `Leaf`
  - `Composite`

- **Applicability**: Use when you want to represent part-whole hierarchies of objects.

- **Sample Code Snippet**:

  ```python
  class Component:
      def operation(self):
          pass

  class Leaf(Component):
      def operation(self):
          return "Leaf operation"

  class Composite(Component):
      def __init__(self):
          self.children = []

      def add(self, component):
          self.children.append(component)

      def operation(self):
          results = []
          for child in self.children:
              results.append(child.operation())
          return "Composite: " + ", ".join(results)
  ```

#### 4. Decorator Pattern

- **Intent**: Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.
- **Structure**:

  ```mermaid
  classDiagram
    Component <|-- ConcreteComponent
    Decorator <|-- ConcreteDecorator
    Decorator --> Component
  ```

- **Key Participants**:
  - `Component`
  - `ConcreteComponent`
  - `Decorator`
  - `ConcreteDecorator`

- **Applicability**: Use to add responsibilities to individual objects dynamically and transparently, without affecting other objects.

- **Sample Code Snippet**:

  ```python
  class Component:
      def operation(self):
          pass

  class ConcreteComponent(Component):
      def operation(self):
          return "ConcreteComponent operation"

  class Decorator(Component):
      def __init__(self, component):
          self.component = component

      def operation(self):
          return self.component.operation()

  class ConcreteDecorator(Decorator):
      def operation(self):
          return f"ConcreteDecorator({self.component.operation()})"
  ```

#### 5. Facade Pattern

- **Intent**: Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.
- **Structure**:

  ```mermaid
  classDiagram
    Facade --> SubsystemClass
  ```

- **Key Participants**:
  - `Facade`
  - `SubsystemClasses`

- **Applicability**: Use when you want to provide a simple interface to a complex subsystem.

- **Sample Code Snippet**:

  ```python
  class SubsystemA:
      def operation_a(self):
          return "SubsystemA operation"

  class SubsystemB:
      def operation_b(self):
          return "SubsystemB operation"

  class Facade:
      def __init__(self):
          self.subsystem_a = SubsystemA()
          self.subsystem_b = SubsystemB()

      def operation(self):
          return f"{self.subsystem_a.operation_a()} + {self.subsystem_b.operation_b()}"
  ```

### Behavioral Patterns

#### 1. Chain of Responsibility Pattern

- **Intent**: Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.
- **Structure**:

  ```mermaid
  classDiagram
    Handler <|-- ConcreteHandler
    Handler --> Handler
  ```

- **Key Participants**:
  - `Handler`
  - `ConcreteHandler`

- **Applicability**: Use when more than one object can handle a request, and the handler isn't known a priori.

- **Sample Code Snippet**:

  ```python
  class Handler:
      def __init__(self, successor=None):
          self.successor = successor

      def handle_request(self, request):
          if self.successor:
              self.successor.handle_request(request)

  class ConcreteHandler(Handler):
      def handle_request(self, request):
          if request == "handle":
              return "Handled by ConcreteHandler"
          else:
              return super().handle_request(request)
  ```

#### 2. Command Pattern

- **Intent**: Encapsulate a request as an object, thereby letting you parameterize clients with queues, requests, and operations.
- **Structure**:

  ```mermaid
  classDiagram
    Command <|-- ConcreteCommand
    Invoker --> Command
    ConcreteCommand --> Receiver
  ```

- **Key Participants**:
  - `Command`
  - `ConcreteCommand`
  - `Invoker`
  - `Receiver`

- **Applicability**: Use to parameterize objects by an action to perform, queue requests, and support undoable operations.

- **Sample Code Snippet**:

  ```python
  class Command:
      def execute(self):
          pass

  class ConcreteCommand(Command):
      def __init__(self, receiver):
          self.receiver = receiver

      def execute(self):
          return self.receiver.action()

  class Receiver:
      def action(self):
          return "Receiver action executed"

  class Invoker:
      def __init__(self, command):
          self.command = command

      def invoke(self):
          return self.command.execute()
  ```

#### 3. Iterator Pattern

- **Intent**: Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
- **Structure**:

  ```mermaid
  classDiagram
    Iterator <|-- ConcreteIterator
    Aggregate <|-- ConcreteAggregate
    ConcreteAggregate --> Iterator
  ```

- **Key Participants**:
  - `Iterator`
  - `ConcreteIterator`
  - `Aggregate`
  - `ConcreteAggregate`

- **Applicability**: Use to access an aggregate object's contents without exposing its internal representation.

- **Sample Code Snippet**:

  ```python
  class Iterator:
      def __init__(self, collection):
          self._collection = collection
          self._index = 0

      def __next__(self):
          if self._index < len(self._collection):
              result = self._collection[self._index]
              self._index += 1
              return result
          raise StopIteration

  class Aggregate:
      def __init__(self):
          self._items = []

      def __iter__(self):
          return Iterator(self._items)

      def add(self, item):
          self._items.append(item)
  ```

#### 4. Observer Pattern

- **Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- **Structure**:

  ```mermaid
  classDiagram
    Subject <|-- ConcreteSubject
    Observer <|-- ConcreteObserver
    Subject --> Observer
  ```

- **Key Participants**:
  - `Subject`
  - `ConcreteSubject`
  - `Observer`
  - `ConcreteObserver`

- **Applicability**: Use when an abstraction has two aspects, one dependent on the other.

- **Sample Code Snippet**:

  ```python
  class Subject:
      def __init__(self):
          self._observers = []

      def attach(self, observer):
          self._observers.append(observer)

      def notify(self):
          for observer in self._observers:
              observer.update()

  class ConcreteObserver:
      def update(self):
          return "Observer updated"
  ```

#### 5. Strategy Pattern

- **Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
- **Structure**:

  ```mermaid
  classDiagram
    Context --> Strategy
    Strategy <|-- ConcreteStrategy
  ```

- **Key Participants**:
  - `Strategy`
  - `ConcreteStrategy`
  - `Context`

- **Applicability**: Use when you want to use different variants of an algorithm within an object and be able to switch from one algorithm to another at runtime.

- **Sample Code Snippet**:

  ```python
  class Strategy:
      def execute(self):
          pass

  class ConcreteStrategyA(Strategy):
      def execute(self):
          return "Strategy A"

  class Context:
      def __init__(self, strategy):
          self.strategy = strategy

      def context_interface(self):
          return self.strategy.execute()
  ```

### Concurrency Patterns

#### 1. Active Object Pattern

- **Intent**: Decouple method execution from method invocation to enhance concurrency.
- **Structure**:

  ```mermaid
  classDiagram
    ActiveObject --> MethodRequest
    Scheduler --> ActiveObject
  ```

- **Key Participants**:
  - `ActiveObject`
  - `MethodRequest`
  - `Scheduler`

- **Applicability**: Use when you need to decouple method execution from method invocation.

- **Sample Code Snippet**:

  ```python
  import threading
  import queue

  class ActiveObject:
      def __init__(self):
          self._queue = queue.Queue()
          self._thread = threading.Thread(target=self._run)
          self._thread.start()

      def _run(self):
          while True:
              method_request = self._queue.get()
              if method_request is None:
                  break
              method_request()

      def enqueue(self, method_request):
          self._queue.put(method_request)

      def stop(self):
          self._queue.put(None)
          self._thread.join()
  ```

#### 2. Balking Pattern

- **Intent**: Prevent an object from executing an action if it is in an inappropriate state.
- **Structure**:

  ```mermaid
  classDiagram
    Client --> Balking
  ```

- **Key Participants**:
  - `Client`
  - `Balking`

- **Applicability**: Use when an object should not perform an action unless it is in a particular state.

- **Sample Code Snippet**:

  ```python
  class Balking:
      def __init__(self):
          self._is_ready = False

      def set_ready(self):
          self._is_ready = True

      def action(self):
          if not self._is_ready:
              return "Balking: Not ready"
          return "Action performed"
  ```

#### 3. Double-Checked Locking Pattern

- **Intent**: Reduce the overhead of acquiring a lock by first testing the locking criterion without actually acquiring the lock.
- **Structure**:

  ```mermaid
  classDiagram
    Client --> DoubleCheckedLocking
  ```

- **Key Participants**:
  - `Client`
  - `DoubleCheckedLocking`

- **Applicability**: Use when you need to reduce the overhead of acquiring a lock by first testing the locking criterion.

- **Sample Code Snippet**:

  ```python
  import threading

  class Singleton:
      _instance = None
      _lock = threading.Lock()

      @classmethod
      def get_instance(cls):
          if cls._instance is None:
              with cls._lock:
                  if cls._instance is None:
                      cls._instance = cls()
          return cls._instance
  ```

### Conclusion

This cheat sheet provides a concise overview of various design patterns, categorized into Creational, Structural, Behavioral, and Concurrency patterns. Each pattern is summarized with its intent, structure, key participants, applicability, and a sample code snippet. Use this guide as a quick reference to recall the essentials of each pattern and apply them effectively in your Python projects.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Singleton pattern?

- [x] Ensure a class has only one instance and provide a global point of access to it.
- [ ] Define a family of algorithms and make them interchangeable.
- [ ] Provide a way to access elements of a collection sequentially.
- [ ] Compose objects into tree structures to represent part-whole hierarchies.

> **Explanation:** The Singleton pattern ensures that a class has only one instance and provides a global point of access to it.

### Which pattern is used to decouple an abstraction from its implementation?

- [ ] Adapter Pattern
- [x] Bridge Pattern
- [ ] Composite Pattern
- [ ] Decorator Pattern

> **Explanation:** The Bridge pattern is used to decouple an abstraction from its implementation so that the two can vary independently.

### What is the key benefit of the Factory Method pattern?

- [ ] It allows for the creation of complex objects without specifying their concrete classes.
- [x] It defines an interface for creating an object but lets subclasses alter the type of objects that will be created.
- [ ] It provides a way to access elements of a collection sequentially.
- [ ] It composes objects into tree structures to represent part-whole hierarchies.

> **Explanation:** The Factory Method pattern defines an interface for creating an object but lets subclasses alter the type of objects that will be created.

### Which pattern allows you to add responsibilities to individual objects dynamically?

- [ ] Composite Pattern
- [ ] Facade Pattern
- [x] Decorator Pattern
- [ ] Observer Pattern

> **Explanation:** The Decorator pattern allows you to add responsibilities to individual objects dynamically and transparently.

### What is the main purpose of the Observer pattern?

- [x] Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified.
- [ ] Provide a way to access elements of a collection sequentially.
- [ ] Compose objects into tree structures to represent part-whole hierarchies.
- [ ] Define an interface for creating an object but let subclasses alter the type of objects that will be created.

> **Explanation:** The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

### In which pattern does the client request an action, but the object may choose not to perform it?

- [ ] Command Pattern
- [ ] Strategy Pattern
- [ ] Active Object Pattern
- [x] Balking Pattern

> **Explanation:** In the Balking pattern, the client requests an action, but the object may choose not to perform it if it is in an inappropriate state.

### Which pattern is useful for creating a complex object step by step?

- [ ] Singleton Pattern
- [ ] Composite Pattern
- [x] Builder Pattern
- [ ] Observer Pattern

> **Explanation:** The Builder pattern is useful for creating a complex object step by step, separating the construction of a complex object from its representation.

### What is the primary intent of the Adapter pattern?

- [ ] Provide a way to access elements of a collection sequentially.
- [x] Convert the interface of a class into another interface clients expect.
- [ ] Compose objects into tree structures to represent part-whole hierarchies.
- [ ] Define an interface for creating an object but let subclasses alter the type of objects that will be created.

> **Explanation:** The Adapter pattern converts the interface of a class into another interface clients expect, allowing incompatible interfaces to work together.

### Which pattern is used to encapsulate a request as an object?

- [x] Command Pattern
- [ ] Strategy Pattern
- [ ] Observer Pattern
- [ ] Decorator Pattern

> **Explanation:** The Command pattern encapsulates a request as an object, allowing parameterization of clients with queues, requests, and operations.

### True or False: The Facade pattern provides a simple interface to a complex subsystem.

- [x] True
- [ ] False

> **Explanation:** The Facade pattern provides a unified interface to a set of interfaces in a subsystem, making the subsystem easier to use.

{{< /quizdown >}}

Remember, this cheat sheet is just the beginning. As you progress, you'll build more complex and interactive systems. Keep experimenting, stay curious, and enjoy the journey!

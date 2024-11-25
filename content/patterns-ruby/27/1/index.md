---
canonical: "https://softwarepatternslexicon.com/patterns-ruby/27/1"
title: "Ruby Design Patterns Glossary of Terms"
description: "Explore the comprehensive glossary of key terms, concepts, and acronyms used in Ruby design patterns to build scalable and maintainable applications."
linkTitle: "27.1 Glossary of Terms"
categories:
- Ruby
- Design Patterns
- Software Development
tags:
- Ruby
- Design Patterns
- Glossary
- Software Architecture
- Programming Concepts
date: 2024-11-23
type: docs
nav_weight: 271000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 27.1 Glossary of Terms

Welcome to the glossary section of "The Ultimate Guide to Ruby Design Patterns: Build Scalable and Maintainable Applications." This glossary serves as a quick reference for readers, providing clear and concise definitions of key terms, concepts, acronyms, and jargon used throughout the guide. The terms are arranged alphabetically for easy navigation, and cross-references to relevant sections are included to enhance understanding.

### A

**Abstract Factory Pattern**  
A creational design pattern that provides an interface for creating families of related or dependent objects without specifying their concrete classes. See [Section 4.4](#4-4-abstract-factory-pattern).

**Actor Model**  
A conceptual model for handling concurrent computation in which "actors" are the universal primitives of concurrent computation. See [Section 9.5](#9-5-actors-model-with-celluloid-and-concurrent-ruby).

**Adapter Pattern**  
A structural design pattern that allows objects with incompatible interfaces to work together. See [Section 5.1](#5-1-adapter-pattern).

**Agile Development**  
A set of principles for software development under which requirements and solutions evolve through the collaborative effort of cross-functional teams.

**Algorithm**  
A step-by-step procedure or formula for solving a problem.

**API (Application Programming Interface)**  
A set of rules and tools for building software applications, allowing different software programs to communicate with each other.

### B

**Behavioral Design Patterns**  
Patterns that focus on communication between objects, defining how they interact and fulfill their responsibilities. See [Section 6](#6-behavioral-design-patterns-in-ruby).

**Block**  
A chunk of code enclosed between `do...end` or curly braces `{...}` that can be passed to methods in Ruby. See [Section 2.5](#2-5-blocks-procs-and-lambdas).

**Builder Pattern**  
A creational design pattern that separates the construction of a complex object from its representation, allowing the same construction process to create different representations. See [Section 4.5](#4-5-builder-pattern).

**Bundler**  
A dependency manager for Ruby projects, ensuring that the correct versions of gems are used. See [Section 3.6](#3-6-gems-and-dependency-management-with-bundler).

### C

**Callback**  
A function passed as an argument to another function, which is then invoked inside the outer function to complete some kind of routine or action. See [Section 6.12](#6-12-callback-and-hook-patterns).

**Chain of Responsibility Pattern**  
A behavioral design pattern that allows an object to pass a request along a chain of potential handlers until the request is handled. See [Section 6.1](#6-1-chain-of-responsibility-pattern).

**Class**  
A blueprint for creating objects in Ruby, defining the properties and behaviors that the objects created from the class will have.

**Clean Code**  
A philosophy and set of practices for writing code that is easy to understand, maintain, and extend. See [Section 16.7](#16-7-clean-code-practices).

**Command Pattern**  
A behavioral design pattern that turns a request into a stand-alone object that contains all information about the request. See [Section 6.2](#6-2-command-pattern).

**CQRS (Command Query Responsibility Segregation)**  
A pattern that separates read and update operations for a data store. See [Section 6.15](#6-15-command-query-responsibility-segregation-cqrs).

### D

**Decorator Pattern**  
A structural design pattern that allows behavior to be added to individual objects, either statically or dynamically, without affecting the behavior of other objects from the same class. See [Section 5.4](#5-4-decorator-pattern).

**Dependency Injection**  
A technique in which an object receives other objects it depends on, rather than creating them internally. See [Section 4.7](#4-7-dependency-injection-in-ruby).

**Design Pattern**  
A general, reusable solution to a commonly occurring problem within a given context in software design.

**Domain-Driven Design (DDD)**  
An approach to software development that emphasizes collaboration between technical and domain experts to iteratively refine a conceptual model that addresses complex domain logic. See [Section 12.7](#12-7-domain-driven-design-ddd-in-ruby).

**DSL (Domain-Specific Language)**  
A programming language or specification language dedicated to a particular problem domain, a particular problem representation technique, and/or a particular solution technique. See [Section 14](#14-domain-specific-languages-dsls).

### E

**Eigenclass**  
A special hidden class in Ruby that holds methods for a single object. See [Section 2.9](#2-9-singleton-classes-and-eigenclasses).

**Encapsulation**  
The bundling of data with the methods that operate on that data, restricting direct access to some of the object's components.

**Event-Driven Architecture**  
A software architecture paradigm promoting the production, detection, consumption of, and reaction to events. See [Section 12.5](#12-5-event-driven-architecture).

**Exception Handling**  
The process of responding to the occurrence of exceptions – anomalous or exceptional conditions requiring special processing. See [Section 3.3](#3-3-exception-handling-and-raising).

### F

**Facade Pattern**  
A structural design pattern that provides a simplified interface to a complex subsystem. See [Section 5.5](#5-5-facade-pattern).

**Factory Method Pattern**  
A creational design pattern that provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created. See [Section 4.3](#4-3-factory-method-pattern).

**Fiber**  
A lightweight concurrency primitive in Ruby that allows you to pause and resume code blocks. See [Section 9.3](#9-3-fibers-and-cooperative-concurrency).

**Functional Programming**  
A programming paradigm where programs are constructed by applying and composing functions. See [Section 7](#7-functional-programming-in-ruby).

### G

**Garbage Collection**  
The process of automatically freeing memory by deleting objects that are no longer reachable in a program.

**Gem**  
A packaged Ruby application or library. See [Section 3.6](#3-6-gems-and-dependency-management-with-bundler).

**GraphQL**  
A query language for APIs and a runtime for executing those queries by using a type system you define for your data.

### H

**Hexagonal Architecture**  
An architectural pattern used to create loosely coupled application components that can be easily connected to their software environment. See [Section 12.6](#12-6-hexagonal-architecture-ports-and-adapters).

**Hook**  
A place in the code where a programmer can insert custom code to alter or augment functionality. See [Section 6.12](#6-12-callback-and-hook-patterns).

### I

**Immutability**  
An object whose state cannot be modified after it is created. See [Section 2.6](#2-6-immutability-and-frozen-objects).

**Inheritance**  
A mechanism in Ruby where a new class is created from an existing class.

**Interface**  
A shared boundary across which two or more separate components of a computer system exchange information.

**Iterator Pattern**  
A behavioral design pattern that provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation. See [Section 6.4](#6-4-iterator-pattern).

### J

**JRuby**  
An implementation of the Ruby programming language atop the Java Virtual Machine, providing integration with Java libraries.

### K

**Kernel**  
The core module from which all Ruby objects inherit, providing methods that are available in every Ruby object.

### L

**Lambda**  
An anonymous function that can be stored in a variable or passed to a method. See [Section 2.5](#2-5-blocks-procs-and-lambdas).

**Lazy Initialization**  
A technique that delays the creation of an object, the calculation of a value, or some other expensive process until the first time it is needed. See [Section 4.8](#4-8-lazy-initialization).

### M

**Memento Pattern**  
A behavioral design pattern that provides the ability to restore an object to its previous state. See [Section 6.6](#6-6-memento-pattern).

**Metaprogramming**  
A programming technique in which computer programs have the ability to treat other programs as their data. See [Section 8](#8-metaprogramming-in-ruby).

**Microservices**  
An architectural style that structures an application as a collection of loosely coupled services. See [Section 22](#22-microservices-and-distributed-systems).

**Mixin**  
A module included in a class to add additional behavior. See [Section 2.7](#2-7-modules-and-mixins).

**Module**  
A collection of methods and constants that can be included in classes. See [Section 2.7](#2-7-modules-and-mixins).

**Monads**  
A design pattern used to handle program-wide concerns in a functional way. See [Section 7.6](#7-6-monads-in-ruby).

### N

**Namespace**  
A container that provides context for the identifiers (names of types, functions, variables, etc.) it holds and allows disambiguation of items that have the same name but reside in different namespaces. See [Section 3.7](#3-7-namespaces-and-organizing-code-with-modules).

**Null Object Pattern**  
A design pattern that uses an object with defined neutral ("null") behavior. See [Section 5.11](#5-11-null-object-pattern).

### O

**Observer Pattern**  
A behavioral design pattern in which an object, called the subject, maintains a list of its dependents, called observers, and notifies them automatically of any state changes. See [Section 6.7](#6-7-observer-pattern).

**Open Classes**  
A feature in Ruby that allows you to modify existing classes by adding new methods or altering existing ones. See [Section 2.10](#2-10-open-classes-and-refinements).

### P

**Pattern Matching**  
A mechanism for checking a value against a pattern. See [Section 3.13](#3-13-pattern-matching-in-ruby-2-7-and-beyond).

**Polymorphism**  
The ability of different objects to respond to the same message (method call) in different ways.

**Proc**  
An object that holds a block of code that can be stored in a variable and passed to methods. See [Section 2.5](#2-5-blocks-procs-and-lambdas).

**Prototype Pattern**  
A creational design pattern that allows cloning of objects, even complex ones, without coupling to their specific classes. See [Section 4.6](#4-6-prototype-pattern).

**Proxy Pattern**  
A structural design pattern that provides an object representing another object. See [Section 5.7](#5-7-proxy-pattern).

### Q

**Queue**  
A collection of entities that are maintained in a sequence and can be modified by the addition of entities at one end of the sequence and the removal of entities from the other end.

### R

**Ractor**  
A Ruby feature for parallel execution without thread safety issues. See [Section 9.4](#9-4-ractors-in-ruby-3-for-parallelism).

**Reactive Programming**  
A programming paradigm oriented around data flows and the propagation of change. See [Section 11](#11-reactive-programming-patterns).

**Recursion**  
A method of solving a problem where the solution depends on solutions to smaller instances of the same problem. See [Section 7.13](#7-13-recursion-and-tail-call-optimization).

**Reflection**  
The ability of a program to examine and modify its own structure and behavior. See [Section 2.11](#2-11-reflection-and-introspection).

**RubyGems**  
A package manager for the Ruby programming language that provides a standard format for distributing Ruby programs and libraries. See [Section 3.6](#3-6-gems-and-dependency-management-with-bundler).

### S

**Saga Pattern**  
A pattern for managing failures in long-running business transactions. See [Section 6.16](#6-16-saga-pattern).

**Singleton Pattern**  
A creational design pattern that restricts the instantiation of a class to one single instance. See [Section 4.2](#4-2-singleton-pattern).

**SOLID Principles**  
A set of five design principles intended to make software designs more understandable, flexible, and maintainable. See [Section 16.6](#16-6-applying-solid-principles).

**State Pattern**  
A behavioral design pattern that allows an object to alter its behavior when its internal state changes. See [Section 6.8](#6-8-state-pattern).

**Strategy Pattern**  
A behavioral design pattern that enables selecting an algorithm's behavior at runtime. See [Section 6.9](#6-9-strategy-pattern).

**Structural Design Patterns**  
Patterns that ease the design by identifying a simple way to realize relationships between entities. See [Section 5](#5-structural-design-patterns-in-ruby).

### T

**Template Method Pattern**  
A behavioral design pattern that defines the skeleton of an algorithm in the superclass but lets subclasses override specific steps of the algorithm without changing its structure. See [Section 6.10](#6-10-template-method-pattern).

**Thread**  
A sequence of executable instructions that can be managed independently by a scheduler.

**Type Checking**  
The process of verifying and enforcing the constraints of types. See [Section 3.12](#3-12-advanced-type-checking-with-sorbet-and-rbs).

### U

**UML (Unified Modeling Language)**  
A standardized modeling language consisting of an integrated set of diagrams, used to visualize the design of a system.

### V

**Visitor Pattern**  
A behavioral design pattern that lets you separate algorithms from the objects on which they operate. See [Section 6.11](#6-11-visitor-pattern).

### W

**Web API**  
An application programming interface for either a web server or a web browser.

### X

**XML (eXtensible Markup Language)**  
A markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable.

### Y

**YAML (YAML Ain't Markup Language)**  
A human-readable data serialization standard that can be used in conjunction with all programming languages and is often used to write configuration files.

### Z

**Zero Downtime Deployment**  
A deployment method that ensures that the application remains available to users during the deployment process.

---

This glossary is designed to be a living document, evolving as new terms and concepts are introduced in the field of Ruby design patterns. We encourage you to explore the cross-referenced sections for a deeper understanding of each term.

## Quiz: Glossary of Terms

{{< quizdown >}}

### What is the purpose of the Abstract Factory Pattern?

- [x] To provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- [ ] To allow objects with incompatible interfaces to work together.
- [ ] To separate the construction of a complex object from its representation.
- [ ] To provide a simplified interface to a complex subsystem.

> **Explanation:** The Abstract Factory Pattern provides an interface for creating families of related or dependent objects without specifying their concrete classes.

### What is a key feature of the Adapter Pattern?

- [x] It allows objects with incompatible interfaces to work together.
- [ ] It provides a way to access the elements of an aggregate object sequentially.
- [ ] It turns a request into a stand-alone object.
- [ ] It provides a simplified interface to a complex subsystem.

> **Explanation:** The Adapter Pattern allows objects with incompatible interfaces to work together by converting the interface of a class into another interface that clients expect.

### What is the primary focus of Behavioral Design Patterns?

- [x] Communication between objects.
- [ ] Creation of objects.
- [ ] Structure of objects.
- [ ] Performance optimization.

> **Explanation:** Behavioral Design Patterns focus on communication between objects, defining how they interact and fulfill their responsibilities.

### What is a Block in Ruby?

- [x] A chunk of code enclosed between `do...end` or curly braces `{...}` that can be passed to methods.
- [ ] A blueprint for creating objects.
- [ ] A collection of methods and constants.
- [ ] An anonymous function stored in a variable.

> **Explanation:** A Block in Ruby is a chunk of code enclosed between `do...end` or curly braces `{...}` that can be passed to methods.

### What does the Builder Pattern achieve?

- [x] It separates the construction of a complex object from its representation.
- [ ] It provides an interface for creating families of related objects.
- [ ] It allows objects with incompatible interfaces to work together.
- [ ] It provides a simplified interface to a complex subsystem.

> **Explanation:** The Builder Pattern separates the construction of a complex object from its representation, allowing the same construction process to create different representations.

### What is the role of Bundler in Ruby?

- [x] It is a dependency manager for Ruby projects.
- [ ] It is a tool for creating gems.
- [ ] It is a testing framework.
- [ ] It is a web server.

> **Explanation:** Bundler is a dependency manager for Ruby projects, ensuring that the correct versions of gems are used.

### What is the Chain of Responsibility Pattern used for?

- [x] Passing a request along a chain of potential handlers until the request is handled.
- [ ] Separating the construction of a complex object from its representation.
- [ ] Providing a simplified interface to a complex subsystem.
- [ ] Allowing objects with incompatible interfaces to work together.

> **Explanation:** The Chain of Responsibility Pattern allows an object to pass a request along a chain of potential handlers until the request is handled.

### What is a Callback in programming?

- [x] A function passed as an argument to another function, which is then invoked inside the outer function.
- [ ] A method that defines the skeleton of an algorithm.
- [ ] A collection of methods and constants.
- [ ] An object that holds a block of code.

> **Explanation:** A Callback is a function passed as an argument to another function, which is then invoked inside the outer function to complete some kind of routine or action.

### What is the purpose of the Command Pattern?

- [x] To turn a request into a stand-alone object that contains all information about the request.
- [ ] To provide an interface for creating families of related objects.
- [ ] To allow objects with incompatible interfaces to work together.
- [ ] To provide a simplified interface to a complex subsystem.

> **Explanation:** The Command Pattern turns a request into a stand-alone object that contains all information about the request, allowing for parameterization of clients with queues, requests, and operations.

### True or False: The Observer Pattern is a structural design pattern.

- [ ] True
- [x] False

> **Explanation:** The Observer Pattern is a behavioral design pattern, not a structural one. It involves an object, called the subject, maintaining a list of its dependents, called observers, and notifying them automatically of any state changes.

{{< /quizdown >}}

Remember, this glossary is just the beginning of your journey into mastering Ruby design patterns. As you progress, you'll build more complex and interactive applications. Keep experimenting, stay curious, and enjoy the journey!

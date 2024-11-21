---
canonical: "https://softwarepatternslexicon.com/patterns-ts/2/7"
title: "GRASP Principles in TypeScript: Mastering Responsibility Assignment"
description: "Explore the GRASP principles for effective responsibility assignment in TypeScript, enhancing maintainability and scalability in software design."
linkTitle: "2.7 GRASP Principles"
categories:
- Software Design
- Object-Oriented Programming
- TypeScript
tags:
- GRASP Principles
- Responsibility Assignment
- TypeScript Design Patterns
- Object-Oriented Design
- Software Engineering
date: 2024-11-17
type: docs
nav_weight: 2700
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.7 GRASP Principles

In the realm of software engineering, the GRASP (General Responsibility Assignment Software Patterns) principles serve as a foundational guide for assigning responsibilities to classes and objects in object-oriented design. These principles are instrumental in creating systems that are not only maintainable and scalable but also robust and adaptable to change. By understanding and applying GRASP, developers can ensure that their designs are well-structured and that responsibilities are appropriately distributed across the system.

### Overview of GRASP

GRASP is a set of nine principles that provide a framework for responsibility assignment in object-oriented design. These principles help developers make informed decisions about which class should be responsible for what functionality, thereby promoting high cohesion and low coupling. The GRASP principles are:

1. **Information Expert**
2. **Creator**
3. **Controller**
4. **Low Coupling**
5. **High Cohesion**
6. **Polymorphism**
7. **Pure Fabrication**
8. **Indirection**
9. **Protected Variations**

Each principle addresses a specific aspect of responsibility assignment and offers guidance on how to achieve a balanced and efficient design. Let's delve into each principle and explore how they can be implemented in TypeScript.

### 1. Information Expert

The Information Expert principle suggests that responsibility should be assigned to the class that has the necessary information to fulfill it. This principle promotes encapsulation and ensures that data and behavior are co-located, making the system easier to understand and maintain.

**TypeScript Example:**

```typescript
class Order {
    private items: Item[] = [];

    addItem(item: Item): void {
        this.items.push(item);
    }

    calculateTotal(): number {
        return this.items.reduce((total, item) => total + item.price, 0);
    }
}

class Item {
    constructor(public name: string, public price: number) {}
}

// Usage
const order = new Order();
order.addItem(new Item('Laptop', 1200));
order.addItem(new Item('Mouse', 25));
console.log(order.calculateTotal()); // Outputs: 1225
```

In this example, the `Order` class is the Information Expert for calculating the total price because it has access to the list of items and their prices.

### 2. Creator

The Creator principle advises that a class should be responsible for creating instances of another class if it contains, aggregates, or closely uses objects of that class. This principle helps in maintaining a clear and logical relationship between classes.

**TypeScript Example:**

```typescript
class Customer {
    private orders: Order[] = [];

    createOrder(): Order {
        const order = new Order();
        this.orders.push(order);
        return order;
    }
}

// Usage
const customer = new Customer();
const newOrder = customer.createOrder();
```

Here, the `Customer` class is the Creator of `Order` instances because it aggregates orders and manages their lifecycle.

### 3. Controller

The Controller principle suggests assigning the responsibility of handling system events to a non-user interface class that represents the overall system or a use-case scenario. This principle helps in decoupling the user interface from the business logic.

**TypeScript Example:**

```typescript
class OrderController {
    private orderService: OrderService;

    constructor(orderService: OrderService) {
        this.orderService = orderService;
    }

    processOrder(orderData: any): void {
        const order = this.orderService.createOrder(orderData);
        this.orderService.processPayment(order);
    }
}

class OrderService {
    createOrder(orderData: any): Order {
        // Logic to create an order
        return new Order();
    }

    processPayment(order: Order): void {
        // Logic to process payment
    }
}

// Usage
const orderService = new OrderService();
const orderController = new OrderController(orderService);
orderController.processOrder({ /* order data */ });
```

The `OrderController` acts as a Controller by managing the order processing workflow, delegating tasks to the `OrderService`.

### 4. Low Coupling

Low Coupling is a principle that aims to reduce dependencies between classes to increase their reusability and flexibility. A system with low coupling is easier to maintain and adapt to changes.

**TypeScript Example:**

```typescript
class PaymentProcessor {
    process(amount: number): void {
        console.log(`Processing payment of $${amount}`);
    }
}

class Order {
    private paymentProcessor: PaymentProcessor;

    constructor(paymentProcessor: PaymentProcessor) {
        this.paymentProcessor = paymentProcessor;
    }

    completeOrder(): void {
        const total = this.calculateTotal();
        this.paymentProcessor.process(total);
    }

    private calculateTotal(): number {
        // Logic to calculate total
        return 100;
    }
}

// Usage
const paymentProcessor = new PaymentProcessor();
const order = new Order(paymentProcessor);
order.completeOrder();
```

In this example, the `Order` class depends on an abstraction (`PaymentProcessor`) rather than a concrete implementation, promoting low coupling.

### 5. High Cohesion

High Cohesion refers to the degree to which the elements of a class belong together. A class with high cohesion has a single, well-defined purpose, making it easier to understand and maintain.

**TypeScript Example:**

```typescript
class Invoice {
    private items: Item[] = [];

    addItem(item: Item): void {
        this.items.push(item);
    }

    generateInvoice(): string {
        // Logic to generate invoice
        return 'Invoice generated';
    }
}

// Usage
const invoice = new Invoice();
invoice.addItem(new Item('Keyboard', 50));
console.log(invoice.generateInvoice());
```

The `Invoice` class has high cohesion as it focuses solely on managing and generating invoices.

### 6. Polymorphism

Polymorphism allows objects to be treated as instances of their parent class, enabling dynamic method invocation. This principle is crucial for designing systems that can handle variations in behavior.

**TypeScript Example:**

```typescript
interface PaymentMethod {
    processPayment(amount: number): void;
}

class CreditCardPayment implements PaymentMethod {
    processPayment(amount: number): void {
        console.log(`Processing credit card payment of $${amount}`);
    }
}

class PayPalPayment implements PaymentMethod {
    processPayment(amount: number): void {
        console.log(`Processing PayPal payment of $${amount}`);
    }
}

function processOrder(paymentMethod: PaymentMethod, amount: number): void {
    paymentMethod.processPayment(amount);
}

// Usage
const creditCardPayment = new CreditCardPayment();
processOrder(creditCardPayment, 200);

const payPalPayment = new PayPalPayment();
processOrder(payPalPayment, 150);
```

In this example, different payment methods implement the `PaymentMethod` interface, allowing for polymorphic behavior in the `processOrder` function.

### 7. Pure Fabrication

Pure Fabrication involves creating a class that does not represent a concept in the problem domain but is necessary to achieve low coupling and high cohesion. This principle is useful for separating concerns and improving system design.

**TypeScript Example:**

```typescript
class Logger {
    log(message: string): void {
        console.log(`Log: ${message}`);
    }
}

class Order {
    private logger: Logger;

    constructor(logger: Logger) {
        this.logger = logger;
    }

    completeOrder(): void {
        // Order completion logic
        this.logger.log('Order completed');
    }
}

// Usage
const logger = new Logger();
const order = new Order(logger);
order.completeOrder();
```

The `Logger` class is a Pure Fabrication, created to handle logging responsibilities without affecting the `Order` class's primary purpose.

### 8. Indirection

Indirection involves introducing an intermediary to mediate between two components, reducing direct coupling. This principle is often used to decouple classes and improve system flexibility.

**TypeScript Example:**

```typescript
class Authentication {
    authenticate(user: string, password: string): boolean {
        // Authentication logic
        return true;
    }
}

class AuthenticationProxy {
    private auth: Authentication;

    constructor(auth: Authentication) {
        this.auth = auth;
    }

    login(user: string, password: string): boolean {
        console.log('Logging in...');
        return this.auth.authenticate(user, password);
    }
}

// Usage
const auth = new Authentication();
const authProxy = new AuthenticationProxy(auth);
authProxy.login('user', 'password');
```

The `AuthenticationProxy` acts as an intermediary, adding a layer of indirection between the client and the `Authentication` class.

### 9. Protected Variations

Protected Variations is a principle that aims to shield elements from the impact of variations in other elements. By using interfaces and abstract classes, developers can protect parts of the system from changes.

**TypeScript Example:**

```typescript
interface Notification {
    send(message: string): void;
}

class EmailNotification implements Notification {
    send(message: string): void {
        console.log(`Sending email: ${message}`);
    }
}

class SMSNotification implements Notification {
    send(message: string): void {
        console.log(`Sending SMS: ${message}`);
    }
}

function notifyUser(notification: Notification, message: string): void {
    notification.send(message);
}

// Usage
const emailNotification = new EmailNotification();
notifyUser(emailNotification, 'Hello via Email');

const smsNotification = new SMSNotification();
notifyUser(smsNotification, 'Hello via SMS');
```

In this example, the `Notification` interface protects the `notifyUser` function from changes in the notification types, allowing for easy extension and modification.

### Importance of Responsibility Assignment

Effective responsibility assignment is crucial for creating maintainable and scalable systems. By adhering to the GRASP principles, developers can ensure that their designs are robust, flexible, and easy to understand. These principles complement the SOLID principles by providing additional guidance on how to distribute responsibilities across classes and objects.

### Encouraging GRASP in Design

When designing software, it's essential to consider the GRASP principles to achieve a balanced and efficient system. By thoughtfully assigning responsibilities, developers can create systems that are easier to maintain, extend, and adapt to changing requirements.

### Try It Yourself

Experiment with the provided TypeScript examples by modifying them to suit different scenarios. Try adding new features or changing the existing ones to see how the GRASP principles help maintain a clean and organized codebase.

## Quiz Time!

{{< quizdown >}}

### Which GRASP principle suggests assigning responsibility to the class with the necessary information?

- [x] Information Expert
- [ ] Creator
- [ ] Controller
- [ ] Low Coupling

> **Explanation:** The Information Expert principle assigns responsibility to the class that has the necessary information to fulfill it.

### What is the primary goal of the Low Coupling principle?

- [x] Reduce dependencies between classes
- [ ] Increase cohesion within classes
- [ ] Assign responsibilities based on information
- [ ] Create intermediary classes

> **Explanation:** Low Coupling aims to reduce dependencies between classes to increase reusability and flexibility.

### Which principle involves creating a class that does not represent a concept in the problem domain?

- [ ] Information Expert
- [ ] Creator
- [x] Pure Fabrication
- [ ] Indirection

> **Explanation:** Pure Fabrication involves creating a class that does not represent a concept in the problem domain but is necessary for achieving low coupling and high cohesion.

### What does the Controller principle aim to decouple?

- [x] User interface from business logic
- [ ] Data from behavior
- [ ] Classes from dependencies
- [ ] Methods from interfaces

> **Explanation:** The Controller principle aims to decouple the user interface from business logic by handling system events in a non-user interface class.

### Which principle uses interfaces and abstract classes to protect elements from variations?

- [ ] Creator
- [ ] Controller
- [ ] Pure Fabrication
- [x] Protected Variations

> **Explanation:** Protected Variations uses interfaces and abstract classes to shield elements from the impact of variations in other elements.

### Which principle suggests that a class should create instances of another class it closely uses?

- [ ] Information Expert
- [x] Creator
- [ ] Controller
- [ ] Low Coupling

> **Explanation:** The Creator principle advises that a class should be responsible for creating instances of another class if it closely uses or aggregates objects of that class.

### What does the Indirection principle introduce to reduce direct coupling?

- [ ] Information Expert
- [ ] Creator
- [x] Intermediary
- [ ] Pure Fabrication

> **Explanation:** Indirection introduces an intermediary to mediate between two components, reducing direct coupling.

### Which principle focuses on assigning responsibilities to achieve high cohesion?

- [ ] Low Coupling
- [x] High Cohesion
- [ ] Information Expert
- [ ] Controller

> **Explanation:** High Cohesion focuses on ensuring that a class has a single, well-defined purpose, making it easier to understand and maintain.

### How does Polymorphism benefit system design?

- [x] By enabling dynamic method invocation
- [ ] By reducing class dependencies
- [ ] By increasing class cohesion
- [ ] By creating intermediary classes

> **Explanation:** Polymorphism allows objects to be treated as instances of their parent class, enabling dynamic method invocation and handling variations in behavior.

### True or False: GRASP principles are only applicable in TypeScript.

- [ ] True
- [x] False

> **Explanation:** GRASP principles are applicable in any object-oriented programming language, not just TypeScript.

{{< /quizdown >}}

Remember, mastering GRASP principles is a journey. As you continue to design and develop software, keep these principles in mind to create systems that are not only functional but also elegant and sustainable.

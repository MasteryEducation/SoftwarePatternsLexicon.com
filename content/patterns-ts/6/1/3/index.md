---
canonical: "https://softwarepatternslexicon.com/patterns-ts/6/1/3"
title: "Chain of Responsibility Pattern: Use Cases and Examples"
description: "Explore practical applications of the Chain of Responsibility Pattern in TypeScript, including middleware pipelines, event handling systems, and support ticket escalation."
linkTitle: "6.1.3 Use Cases and Examples"
categories:
- Design Patterns
- TypeScript
- Software Engineering
tags:
- Chain of Responsibility
- Middleware
- Event Handling
- TypeScript Patterns
- Software Design
date: 2024-11-17
type: docs
nav_weight: 6130
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 6.1.3 Use Cases and Examples

The Chain of Responsibility Pattern is a behavioral design pattern that allows an object to pass a request along a chain of potential handlers until one of them handles the request. This pattern promotes loose coupling in the system by allowing multiple objects to handle a request without the sender needing to know which object will handle it. Let's delve into some practical scenarios where this pattern can be effectively applied, including middleware pipelines, event handling systems, and support ticket escalation.

### Implementing a Web Server Middleware Pipeline

One of the most common use cases for the Chain of Responsibility Pattern is in building middleware pipelines for web servers. Middleware functions are components that process requests and responses in a web application. They can perform tasks such as authentication, logging, and error handling.

#### Example: Middleware Pipeline in TypeScript

Let's implement a simple middleware pipeline using the Chain of Responsibility Pattern in TypeScript. We'll create a series of middleware functions that process HTTP requests.

```typescript
// Define the Request and Response types
interface Request {
  url: string;
  headers: Record<string, string>;
  body: any;
}

interface Response {
  statusCode: number;
  body: any;
}

// Middleware interface
interface Middleware {
  setNext(middleware: Middleware): Middleware;
  handle(request: Request, response: Response): void;
}

// Abstract Middleware class
abstract class AbstractMiddleware implements Middleware {
  private nextMiddleware: Middleware | null = null;

  setNext(middleware: Middleware): Middleware {
    this.nextMiddleware = middleware;
    return middleware;
  }

  handle(request: Request, response: Response): void {
    if (this.nextMiddleware) {
      this.nextMiddleware.handle(request, response);
    }
  }
}

// Concrete Middleware: Authentication
class AuthenticationMiddleware extends AbstractMiddleware {
  handle(request: Request, response: Response): void {
    if (!request.headers['Authorization']) {
      response.statusCode = 401;
      response.body = 'Unauthorized';
      return;
    }
    console.log('Authentication successful');
    super.handle(request, response);
  }
}

// Concrete Middleware: Logging
class LoggingMiddleware extends AbstractMiddleware {
  handle(request: Request, response: Response): void {
    console.log(`Request URL: ${request.url}`);
    super.handle(request, response);
  }
}

// Concrete Middleware: Error Handling
class ErrorHandlingMiddleware extends AbstractMiddleware {
  handle(request: Request, response: Response): void {
    try {
      super.handle(request, response);
    } catch (error) {
      response.statusCode = 500;
      response.body = 'Internal Server Error';
      console.error('Error:', error);
    }
  }
}

// Client code
const request: Request = {
  url: '/api/data',
  headers: { 'Authorization': 'Bearer token' },
  body: {}
};

const response: Response = {
  statusCode: 200,
  body: {}
};

// Set up the middleware chain
const authMiddleware = new AuthenticationMiddleware();
const logMiddleware = new LoggingMiddleware();
const errorMiddleware = new ErrorHandlingMiddleware();

authMiddleware.setNext(logMiddleware).setNext(errorMiddleware);

// Process the request
authMiddleware.handle(request, response);

console.log('Response:', response);
```

In this example, we define a series of middleware classes that extend an abstract middleware class. Each middleware can handle a request and pass it to the next middleware in the chain. This setup allows us to add, remove, or reorder middleware components without changing the client code.

#### Benefits of Middleware Pipelines

- **Maintainability**: Middleware components are modular and can be developed and tested independently.
- **Extensibility**: New middleware can be added to the pipeline without modifying existing components.
- **Separation of Concerns**: Each middleware has a specific responsibility, promoting clean and organized code.

### Building an Event Handling System

Another effective application of the Chain of Responsibility Pattern is in event handling systems. In such systems, different handlers respond to events based on their type or content.

#### Example: Event Handling in TypeScript

Consider an event handling system where different handlers process events based on their type.

```typescript
// Define the Event type
interface Event {
  type: string;
  payload: any;
}

// Event Handler interface
interface EventHandler {
  setNext(handler: EventHandler): EventHandler;
  handle(event: Event): void;
}

// Abstract Event Handler class
abstract class AbstractEventHandler implements EventHandler {
  private nextHandler: EventHandler | null = null;

  setNext(handler: EventHandler): EventHandler {
    this.nextHandler = handler;
    return handler;
  }

  handle(event: Event): void {
    if (this.nextHandler) {
      this.nextHandler.handle(event);
    }
  }
}

// Concrete Event Handler: Login Event
class LoginEventHandler extends AbstractEventHandler {
  handle(event: Event): void {
    if (event.type === 'login') {
      console.log('Handling login event:', event.payload);
      return;
    }
    super.handle(event);
  }
}

// Concrete Event Handler: Logout Event
class LogoutEventHandler extends AbstractEventHandler {
  handle(event: Event): void {
    if (event.type === 'logout') {
      console.log('Handling logout event:', event.payload);
      return;
    }
    super.handle(event);
  }
}

// Concrete Event Handler: Default Event
class DefaultEventHandler extends AbstractEventHandler {
  handle(event: Event): void {
    console.log('Unhandled event type:', event.type);
  }
}

// Client code
const loginHandler = new LoginEventHandler();
const logoutHandler = new LogoutEventHandler();
const defaultHandler = new DefaultEventHandler();

loginHandler.setNext(logoutHandler).setNext(defaultHandler);

// Process an event
const event: Event = { type: 'login', payload: { user: 'Alice' } };
loginHandler.handle(event);
```

In this example, we create a chain of event handlers, each responsible for handling a specific type of event. If a handler cannot process an event, it passes the event to the next handler in the chain.

#### Benefits of Event Handling Systems

- **Flexibility**: Handlers can be added or removed without affecting other parts of the system.
- **Decoupling**: The system is not tightly coupled to specific event types, allowing for easy modification and extension.
- **Responsibility Distribution**: Each handler has a clear responsibility, making the system easier to understand and maintain.

### Creating a Support Ticket System

The Chain of Responsibility Pattern is also useful in creating systems where requests need to escalate through different levels of processing, such as a support ticket system.

#### Example: Support Ticket Escalation in TypeScript

Let's implement a support ticket system where requests escalate through support tiers until resolved.

```typescript
// Define the Support Ticket type
interface SupportTicket {
  id: number;
  issue: string;
  severity: string;
}

// Support Handler interface
interface SupportHandler {
  setNext(handler: SupportHandler): SupportHandler;
  handle(ticket: SupportTicket): void;
}

// Abstract Support Handler class
abstract class AbstractSupportHandler implements SupportHandler {
  private nextHandler: SupportHandler | null = null;

  setNext(handler: SupportHandler): SupportHandler {
    this.nextHandler = handler;
    return handler;
  }

  handle(ticket: SupportTicket): void {
    if (this.nextHandler) {
      this.nextHandler.handle(ticket);
    }
  }
}

// Concrete Support Handler: Tier 1 Support
class Tier1SupportHandler extends AbstractSupportHandler {
  handle(ticket: SupportTicket): void {
    if (ticket.severity === 'low') {
      console.log('Tier 1 handling ticket:', ticket.id);
      return;
    }
    super.handle(ticket);
  }
}

// Concrete Support Handler: Tier 2 Support
class Tier2SupportHandler extends AbstractSupportHandler {
  handle(ticket: SupportTicket): void {
    if (ticket.severity === 'medium') {
      console.log('Tier 2 handling ticket:', ticket.id);
      return;
    }
    super.handle(ticket);
  }
}

// Concrete Support Handler: Tier 3 Support
class Tier3SupportHandler extends AbstractSupportHandler {
  handle(ticket: SupportTicket): void {
    console.log('Tier 3 handling ticket:', ticket.id);
  }
}

// Client code
const tier1Handler = new Tier1SupportHandler();
const tier2Handler = new Tier2SupportHandler();
const tier3Handler = new Tier3SupportHandler();

tier1Handler.setNext(tier2Handler).setNext(tier3Handler);

// Process a support ticket
const ticket: SupportTicket = { id: 101, issue: 'Network issue', severity: 'medium' };
tier1Handler.handle(ticket);
```

In this example, we create a chain of support handlers, each responsible for handling tickets of a specific severity. If a handler cannot resolve a ticket, it escalates the ticket to the next handler in the chain.

#### Benefits of Support Ticket Systems

- **Scalability**: The system can handle an increasing number of tickets by adding more handlers.
- **Efficiency**: Tickets are processed by the appropriate handler, reducing response time.
- **Clear Escalation Path**: The chain provides a clear path for ticket escalation, ensuring that issues are addressed at the right level.

### Addressing System Needs with the Chain of Responsibility Pattern

The Chain of Responsibility Pattern addresses several key needs in these systems:

- **Decoupling**: By separating the sender of a request from its receivers, the pattern reduces dependencies and enhances flexibility.
- **Dynamic Handling**: Requests can be processed dynamically by different handlers, allowing for adaptable and responsive systems.
- **Responsibility Segmentation**: Each handler has a specific responsibility, promoting clean and maintainable code.

### Potential Drawbacks and Considerations

While the Chain of Responsibility Pattern offers many benefits, there are potential drawbacks to consider:

- **Chain Overhead**: The pattern may introduce overhead if the chain is long or if requests frequently traverse the entire chain without being handled.
- **Debugging Challenges**: Debugging can be more complex, as it may be difficult to trace which handler processed a request.
- **Responsibility Confusion**: If not carefully managed, it can be unclear which handler is responsible for processing a request.

### Encouragement to Use the Pattern

Consider using the Chain of Responsibility Pattern when you need a flexible and decoupled mechanism for processing requests. This pattern is particularly useful in systems where requests need to be handled by multiple components, such as middleware pipelines, event handling systems, and support ticket escalation.

### Try It Yourself

Experiment with the provided code examples by adding new middleware, event handlers, or support tiers. Consider how the system behaves with different configurations and explore the flexibility offered by the Chain of Responsibility Pattern.

## Quiz Time!

{{< quizdown >}}

### What is the primary benefit of the Chain of Responsibility Pattern?

- [x] Decoupling the sender from the receiver
- [ ] Improving performance
- [ ] Reducing code complexity
- [ ] Increasing code redundancy

> **Explanation:** The Chain of Responsibility Pattern decouples the sender of a request from its receivers, allowing for flexible and dynamic request handling.

### In a middleware pipeline, what is the role of each middleware component?

- [x] To process requests and pass them to the next middleware
- [ ] To terminate the request processing
- [ ] To modify the server configuration
- [ ] To handle database transactions

> **Explanation:** Each middleware component processes requests and passes them to the next middleware in the chain, allowing for modular request handling.

### How does the Chain of Responsibility Pattern promote maintainability?

- [x] By allowing handlers to be developed and tested independently
- [ ] By reducing the number of classes
- [ ] By increasing the complexity of the code
- [ ] By centralizing all logic in a single handler

> **Explanation:** The pattern promotes maintainability by allowing handlers to be modular and independently developed and tested.

### What is a potential drawback of using the Chain of Responsibility Pattern?

- [x] Chain overhead
- [ ] Increased coupling
- [ ] Reduced flexibility
- [ ] Simplified debugging

> **Explanation:** The pattern may introduce overhead if the chain is long or if requests frequently traverse the entire chain without being handled.

### In an event handling system, what happens if a handler cannot process an event?

- [x] The event is passed to the next handler in the chain
- [ ] The event is discarded
- [ ] The system throws an error
- [ ] The event is logged and ignored

> **Explanation:** If a handler cannot process an event, it passes the event to the next handler in the chain, allowing for flexible event handling.

### What is the role of the `setNext` method in the Chain of Responsibility Pattern?

- [x] To link handlers together in a chain
- [ ] To terminate the chain
- [ ] To initialize the handler
- [ ] To process the request

> **Explanation:** The `setNext` method links handlers together in a chain, allowing requests to be passed along the chain.

### In a support ticket system, how are tickets escalated?

- [x] By passing them to the next handler in the chain
- [ ] By resolving them immediately
- [ ] By discarding them
- [ ] By logging them for later review

> **Explanation:** Tickets are escalated by passing them to the next handler in the chain, ensuring they are addressed at the appropriate level.

### What is a key consideration when implementing the Chain of Responsibility Pattern?

- [x] Ensuring clear responsibility distribution among handlers
- [ ] Centralizing all logic in a single handler
- [ ] Reducing the number of handlers
- [ ] Increasing code redundancy

> **Explanation:** It's important to ensure clear responsibility distribution among handlers to maintain a clean and organized system.

### How can the Chain of Responsibility Pattern improve system flexibility?

- [x] By allowing handlers to be added or removed without affecting other parts of the system
- [ ] By centralizing all logic in a single handler
- [ ] By reducing the number of handlers
- [ ] By increasing code redundancy

> **Explanation:** The pattern improves flexibility by allowing handlers to be added or removed without affecting other parts of the system.

### True or False: The Chain of Responsibility Pattern is useful for systems that require tightly coupled request processing.

- [ ] True
- [x] False

> **Explanation:** The Chain of Responsibility Pattern is useful for systems that require loosely coupled request processing, allowing for flexible and dynamic handling.

{{< /quizdown >}}

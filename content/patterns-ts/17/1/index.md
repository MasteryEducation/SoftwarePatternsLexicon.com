---
canonical: "https://softwarepatternslexicon.com/patterns-ts/17/1"

title: "Building a Web Framework Using Design Patterns"
description: "Learn how to build a simple web framework using TypeScript and design patterns, focusing on scalability and maintainability."
linkTitle: "17.1 Building a Web Framework Using Design Patterns"
categories:
- Web Development
- Design Patterns
- TypeScript
tags:
- Web Framework
- TypeScript
- Design Patterns
- Software Architecture
- Middleware
date: 2024-11-17
type: docs
nav_weight: 17100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 17.1 Building a Web Framework Using Design Patterns

Building a web framework from scratch is a rewarding endeavor that deepens our understanding of the inner workings of web technologies. It allows us to explore how design patterns can be effectively applied to create a scalable and maintainable architecture. In this chapter, we will embark on a journey to construct a simplified web framework using TypeScript, focusing on key design patterns that form the backbone of many successful frameworks.

### Introduction

The motivation behind building a custom web framework lies in the educational value it offers. By understanding the core principles and design patterns that underpin popular frameworks, we can gain insights into their architecture and design decisions. This knowledge not only enhances our ability to use these frameworks effectively but also empowers us to contribute to their development or even create our own tailored solutions.

Our goal is to build a simplified framework that demonstrates the application of design patterns. This framework is not intended to compete with established solutions like Express or NestJS but rather to serve as a learning tool. We will focus on implementing core features such as HTTP server setup, routing mechanisms, middleware support, and error handling.

### Defining Framework Requirements

Before diving into implementation, let's outline the core features our framework will support:

- **HTTP Server Setup and Request Handling**: Establishing a server to listen for incoming requests and process them efficiently.
- **Routing Mechanisms**: Defining routes to map URLs to specific handlers.
- **Middleware Support**: Allowing the addition of reusable functions to process requests and responses.
- **Templating Engine Integration (Optional)**: Enabling dynamic content rendering.
- **Error Handling**: Providing a robust mechanism to catch and handle errors gracefully.

These features provide a solid foundation for a web framework, enabling developers to build applications with clear structure and extensibility.

### Selecting Design Patterns

To implement these features, we will leverage several design patterns:

- **Factory Pattern**: Used for creating server instances, providing a centralized point for configuration and initialization.
- **Strategy Pattern**: Applied in the routing system to allow flexible routing logic.
- **Middleware Pattern** (a type of Chain of Responsibility): Facilitates the processing of requests through a series of middleware functions.
- **Observer Pattern**: Utilized for event handling, such as error notifications within the framework.
- **Template Method Pattern**: Defines the skeleton of algorithms in request processing, allowing specific steps to be overridden.

Each pattern serves a specific purpose and is chosen for its ability to enhance the framework's modularity, flexibility, and maintainability.

### Implementation Details

#### Setting Up the Project

Let's begin by setting up our TypeScript project. We'll use Node.js as our runtime environment and install necessary dependencies.

```bash
mkdir my-web-framework
cd my-web-framework
npm init -y
npm install typescript @types/node ts-node
npx tsc --init
```

This sets up a basic TypeScript project. We can now create our core components.

#### Developing Core Components

##### Server Initialization

We'll use the Factory Pattern to create server instances. This pattern allows us to encapsulate the creation logic and provide a consistent interface for server configuration.

```typescript
// serverFactory.ts
import { createServer, IncomingMessage, ServerResponse } from 'http';

export class ServerFactory {
  static createServer(requestHandler: (req: IncomingMessage, res: ServerResponse) => void) {
    return createServer(requestHandler);
  }
}

// index.ts
import { ServerFactory } from './serverFactory';

const server = ServerFactory.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('Hello, world!');
});

server.listen(3000, () => {
  console.log('Server running at http://localhost:3000/');
});
```

In this example, `ServerFactory` provides a method to create a server with a specified request handler. This encapsulation simplifies server creation and configuration.

##### Routing System

For the routing mechanism, we'll apply the Strategy Pattern. This pattern allows us to define different routing strategies that can be swapped or extended.

```typescript
// router.ts
interface RouteStrategy {
  handleRequest(url: string, method: string): void;
}

class GetRouteStrategy implements RouteStrategy {
  handleRequest(url: string, method: string) {
    if (method === 'GET') {
      console.log(`Handling GET request for ${url}`);
    }
  }
}

class PostRouteStrategy implements RouteStrategy {
  handleRequest(url: string, method: string) {
    if (method === 'POST') {
      console.log(`Handling POST request for ${url}`);
    }
  }
}

class Router {
  private strategy: RouteStrategy;

  constructor(strategy: RouteStrategy) {
    this.strategy = strategy;
  }

  setStrategy(strategy: RouteStrategy) {
    this.strategy = strategy;
  }

  handleRequest(url: string, method: string) {
    this.strategy.handleRequest(url, method);
  }
}

// index.ts
import { Router, GetRouteStrategy, PostRouteStrategy } from './router';

const router = new Router(new GetRouteStrategy());
router.handleRequest('/home', 'GET');

router.setStrategy(new PostRouteStrategy());
router.handleRequest('/submit', 'POST');
```

Here, `Router` uses a strategy to handle requests, allowing us to easily switch between different routing strategies.

##### Middleware Support

Middleware functions are essential in web frameworks for processing requests and responses. We'll use the Chain of Responsibility Pattern to implement middleware support.

```typescript
// middleware.ts
type Middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => void;

class MiddlewareManager {
  private middlewares: Middleware[] = [];

  use(middleware: Middleware) {
    this.middlewares.push(middleware);
  }

  execute(req: IncomingMessage, res: ServerResponse) {
    const executeMiddleware = (index: number) => {
      if (index < this.middlewares.length) {
        this.middlewares[index](req, res, () => executeMiddleware(index + 1));
      }
    };
    executeMiddleware(0);
  }
}

// index.ts
import { MiddlewareManager } from './middleware';

const middlewareManager = new MiddlewareManager();

middlewareManager.use((req, res, next) => {
  console.log('Middleware 1');
  next();
});

middlewareManager.use((req, res, next) => {
  console.log('Middleware 2');
  next();
});

middlewareManager.execute({}, {});
```

The `MiddlewareManager` processes requests through a series of middleware functions, each calling the next in line.

##### Request and Response Handling

To handle requests and generate responses, we'll use the Template Method Pattern. This pattern defines the skeleton of an algorithm, allowing specific steps to be customized.

```typescript
// requestHandler.ts
abstract class RequestHandler {
  handleRequest(req: IncomingMessage, res: ServerResponse) {
    this.parseRequest(req);
    this.generateResponse(res);
  }

  protected abstract parseRequest(req: IncomingMessage): void;
  protected abstract generateResponse(res: ServerResponse): void;
}

class JsonRequestHandler extends RequestHandler {
  protected parseRequest(req: IncomingMessage) {
    console.log('Parsing JSON request');
  }

  protected generateResponse(res: ServerResponse) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'Hello, JSON!' }));
  }
}

// index.ts
import { JsonRequestHandler } from './requestHandler';

const jsonHandler = new JsonRequestHandler();
jsonHandler.handleRequest({}, {});
```

The `RequestHandler` class provides a template for handling requests, with specific parsing and response generation logic defined in subclasses.

##### Error Handling

For error handling, we'll implement a global error handler using the Observer Pattern. This pattern allows different parts of the framework to be notified of errors.

```typescript
// errorHandler.ts
class ErrorHandler {
  private observers: ((error: Error) => void)[] = [];

  addObserver(observer: (error: Error) => void) {
    this.observers.push(observer);
  }

  notifyObservers(error: Error) {
    this.observers.forEach(observer => observer(error));
  }

  handleError(error: Error) {
    console.error('Handling error:', error);
    this.notifyObservers(error);
  }
}

// index.ts
import { ErrorHandler } from './errorHandler';

const errorHandler = new ErrorHandler();

errorHandler.addObserver((error) => {
  console.log('Observer 1 notified of error:', error);
});

errorHandler.addObserver((error) => {
  console.log('Observer 2 notified of error:', error);
});

errorHandler.handleError(new Error('Something went wrong'));
```

The `ErrorHandler` class manages error notifications, allowing observers to respond to errors as needed.

#### Extensibility and Customization

Our framework is designed with extensibility in mind. Developers can add custom middleware, routing strategies, or request handlers by adhering to the defined interfaces and patterns. This approach aligns with the Open/Closed Principle, ensuring the framework is open for extension but closed for modification.

### Integrating TypeScript Features

TypeScript's features, such as interfaces, generics, and advanced typing, play a crucial role in our framework's design. Interfaces define contracts for components, ensuring consistency and reliability. Generics enable flexible and reusable code, while strong typing enhances the developer experience by catching errors early.

### Testing the Framework

Testing is vital to ensure the reliability of our framework. We can use testing frameworks like Jest or Mocha to write unit tests for our components.

```typescript
// middleware.test.ts
import { MiddlewareManager } from './middleware';

test('Middleware execution order', () => {
  const middlewareManager = new MiddlewareManager();
  const executionOrder: string[] = [];

  middlewareManager.use((req, res, next) => {
    executionOrder.push('Middleware 1');
    next();
  });

  middlewareManager.use((req, res, next) => {
    executionOrder.push('Middleware 2');
    next();
  });

  middlewareManager.execute({}, {});

  expect(executionOrder).toEqual(['Middleware 1', 'Middleware 2']);
});
```

This test verifies that middleware functions execute in the correct order, ensuring our Chain of Responsibility implementation works as expected.

### Real-World Application

Let's build a simple application using our framework. We'll handle GET and POST requests, serve static files, and render templates.

```typescript
// app.ts
import { ServerFactory } from './serverFactory';
import { Router, GetRouteStrategy, PostRouteStrategy } from './router';
import { MiddlewareManager } from './middleware';
import { JsonRequestHandler } from './requestHandler';
import { ErrorHandler } from './errorHandler';

const server = ServerFactory.createServer((req, res) => {
  const router = new Router(new GetRouteStrategy());
  const middlewareManager = new MiddlewareManager();
  const errorHandler = new ErrorHandler();

  middlewareManager.use((req, res, next) => {
    console.log('Logging request:', req.url);
    next();
  });

  router.handleRequest(req.url || '', req.method || '');

  const handler = new JsonRequestHandler();
  handler.handleRequest(req, res);

  errorHandler.handleError(new Error('Test error'));
});

server.listen(3000, () => {
  console.log('Server running at http://localhost:3000/');
});
```

This application demonstrates handling requests, applying middleware, and managing errors using our framework.

### Challenges and Lessons Learned

During development, we encountered challenges such as balancing simplicity with functionality and ensuring extensibility without overcomplicating the design. Design patterns helped us overcome these challenges by providing proven solutions for common problems. We learned that while patterns offer structure, they should be applied judiciously to avoid unnecessary complexity.

### Conclusion

Building a web framework using design patterns has reinforced the importance of these patterns in creating scalable and maintainable software. By understanding and applying patterns like Factory, Strategy, and Chain of Responsibility, we can design systems that are modular, flexible, and easy to extend. We encourage readers to experiment further, explore additional patterns, and consider contributing to open-source projects.

### Additional Resources

For further reading on web framework development and design patterns, consider exploring the following resources:

- [MDN Web Docs: Design Patterns](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Design_Patterns)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [Node.js Documentation](https://nodejs.org/en/docs/)

## Quiz Time!

{{< quizdown >}}

### What is the primary motivation for building a custom web framework?

- [x] To understand the inner workings of frameworks and learn how design patterns are applied.
- [ ] To create a framework that competes with established solutions.
- [ ] To avoid using existing frameworks.
- [ ] To build a framework without any design patterns.

> **Explanation:** The primary motivation is educational, focusing on understanding frameworks and design patterns.

### Which design pattern is used for creating server instances in our framework?

- [x] Factory Pattern
- [ ] Strategy Pattern
- [ ] Observer Pattern
- [ ] Template Method Pattern

> **Explanation:** The Factory Pattern is used to encapsulate server creation logic.

### What is the purpose of the Strategy Pattern in the routing system?

- [x] To allow flexible routing logic that can be swapped or extended.
- [ ] To handle errors within the framework.
- [ ] To manage middleware execution order.
- [ ] To create server instances.

> **Explanation:** The Strategy Pattern enables flexible and interchangeable routing logic.

### How does the Chain of Responsibility Pattern benefit middleware support?

- [x] It allows requests to be processed through a series of middleware functions.
- [ ] It defines the skeleton of algorithms in request processing.
- [ ] It encapsulates server creation logic.
- [ ] It manages error notifications.

> **Explanation:** The Chain of Responsibility Pattern facilitates sequential middleware processing.

### Which pattern is used for error handling in the framework?

- [x] Observer Pattern
- [ ] Factory Pattern
- [ ] Strategy Pattern
- [ ] Template Method Pattern

> **Explanation:** The Observer Pattern is used to notify different parts of the framework about errors.

### How does TypeScript enhance the framework's design?

- [x] By providing strong typing, interfaces, and generics for reliability and flexibility.
- [ ] By allowing the use of any type everywhere.
- [ ] By making the code more complex.
- [ ] By removing the need for testing.

> **Explanation:** TypeScript's strong typing and advanced features enhance reliability and flexibility.

### What testing framework can be used to test the framework components?

- [x] Jest
- [ ] TypeScript
- [ ] Node.js
- [ ] Express

> **Explanation:** Jest is a testing framework that can be used to write unit tests for the framework components.

### What is the role of the Template Method Pattern in request handling?

- [x] It defines the skeleton of algorithms, allowing specific steps to be customized.
- [ ] It manages middleware execution order.
- [ ] It encapsulates server creation logic.
- [ ] It handles error notifications.

> **Explanation:** The Template Method Pattern provides a template for handling requests, with customizable steps.

### How can developers extend the framework?

- [x] By adding custom middleware, routing strategies, or request handlers.
- [ ] By modifying the core framework code.
- [ ] By using only predefined components.
- [ ] By avoiding TypeScript features.

> **Explanation:** Developers can extend the framework by adding custom components, adhering to the Open/Closed Principle.

### True or False: The framework built in this chapter is intended to compete with established solutions like Express.

- [ ] True
- [x] False

> **Explanation:** The framework is intended as a learning tool, not to compete with established solutions.

{{< /quizdown >}}

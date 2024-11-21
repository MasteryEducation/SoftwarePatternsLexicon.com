---

linkTitle: "4.1.4 Currying and Partial Application"
title: "Currying and Partial Application in JavaScript and TypeScript"
description: "Explore the concepts of Currying and Partial Application in JavaScript and TypeScript, their implementation, use cases, and best practices."
categories:
- Functional Programming
- JavaScript
- TypeScript
tags:
- Currying
- Partial Application
- Functional Programming
- JavaScript
- TypeScript
date: 2024-10-25
type: docs
nav_weight: 414000
canonical: "https://softwarepatternslexicon.com/patterns-js/4/1/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 4.1.4 Currying and Partial Application

In the realm of functional programming, currying and partial application are powerful techniques that enable developers to write more modular, reusable, and expressive code. These patterns are particularly useful in JavaScript and TypeScript, where functions are first-class citizens.

### Understand the Concepts

#### Currying

Currying is the process of transforming a function that takes multiple arguments into a sequence of functions, each taking a single argument. This allows for more flexible function composition and reuse.

**Example:**

A function `add` that takes two arguments can be curried to take one argument at a time.

```javascript
function add(a) {
  return function(b) {
    return a + b;
  };
}

const addFive = add(5);
console.log(addFive(10)); // Output: 15
```

#### Partial Application

Partial application involves fixing a few arguments of a function, producing another function with a smaller arity (fewer arguments). This is particularly useful for setting default configurations or parameters.

**Example:**

```javascript
function multiply(a, b) {
  return a * b;
}

const double = multiply.bind(null, 2);
console.log(double(5)); // Output: 10
```

### Implementation Steps

#### Currying Implementation

Currying can be implemented manually or with the help of libraries like Lodash, which provides utility functions for currying.

**Manual Implementation:**

```javascript
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    } else {
      return function(...nextArgs) {
        return curried.apply(this, args.concat(nextArgs));
      };
    }
  };
}

function sum(a, b, c) {
  return a + b + c;
}

const curriedSum = curry(sum);
console.log(curriedSum(1)(2)(3)); // Output: 6
```

**Using Lodash:**

```javascript
const _ = require('lodash');

const curriedSum = _.curry(sum);
console.log(curriedSum(1)(2)(3)); // Output: 6
```

#### Partial Application Implementation

Partial application can be achieved using closures to retain fixed arguments.

**Example:**

```javascript
function partial(fn, ...fixedArgs) {
  return function(...remainingArgs) {
    return fn.apply(this, fixedArgs.concat(remainingArgs));
  };
}

function greet(greeting, name) {
  return `${greeting}, ${name}!`;
}

const sayHelloTo = partial(greet, 'Hello');
console.log(sayHelloTo('Alice')); // Output: Hello, Alice!
```

### Code Examples

#### Curried Function for Mathematical Operations

Let's create a curried function for a basic mathematical operation like addition.

```typescript
function add(a: number): (b: number) => number {
  return (b: number) => a + b;
}

const addTen = add(10);
console.log(addTen(5)); // Output: 15
```

#### Partially Apply Functions to Preset Configuration Options

Consider a logging function where you want to preset the log level.

```typescript
function log(level: string, message: string) {
  console.log(`[${level}] ${message}`);
}

const infoLog = partial(log, 'INFO');
infoLog('This is an informational message.'); // Output: [INFO] This is an informational message.
```

### Use Cases

- **Reusable Functions:** Currying and partial application are ideal for creating reusable functions with preset arguments, such as configuration settings or default parameters.
- **Function Composition:** These techniques facilitate function composition, allowing developers to build complex operations from simpler, smaller functions.

### Practice

Write a curried function that processes data with configurable steps. For instance, a function that filters, maps, and reduces an array.

```typescript
function processArray(filterFn: (item: number) => boolean) {
  return function(mapFn: (item: number) => number) {
    return function(reduceFn: (acc: number, item: number) => number, initialValue: number) {
      return function(arr: number[]) {
        return arr.filter(filterFn).map(mapFn).reduce(reduceFn, initialValue);
      };
    };
  };
}

const process = processArray((x) => x > 0)((x) => x * 2)((acc, x) => acc + x, 0);
console.log(process([-1, 2, 3, -4, 5])); // Output: 20
```

### Considerations

- **Flexibility vs. Complexity:** While currying and partial application increase flexibility, they can also introduce complexity. It's crucial to balance these aspects to maintain code readability.
- **Readability:** Currying can improve readability by breaking down functions into smaller, more manageable parts. However, overuse can lead to convoluted code.

### Best Practices

- **Use Libraries:** Utilize libraries like Lodash for reliable and efficient currying and partial application.
- **Keep It Simple:** Avoid over-currying or over-partializing functions, which can lead to difficult-to-understand code.
- **Document Intent:** Clearly document the intent and usage of curried and partially applied functions to aid maintainability.

### Conclusion

Currying and partial application are essential techniques in functional programming, offering significant benefits in terms of code reuse and modularity. By understanding and applying these patterns, developers can write more flexible and maintainable JavaScript and TypeScript code.

## Quiz Time!

{{< quizdown >}}

### What is currying?

- [x] Transforming a function with multiple parameters into a series of functions that each take a single parameter.
- [ ] Fixing a few arguments of a function, producing another function of smaller arity.
- [ ] A method to optimize function performance.
- [ ] A way to handle asynchronous operations.

> **Explanation:** Currying involves transforming a function with multiple parameters into a series of functions that each take a single parameter.

### What is partial application?

- [ ] Transforming a function with multiple parameters into a series of functions that each take a single parameter.
- [x] Fixing a few arguments of a function, producing another function of smaller arity.
- [ ] A method to optimize function performance.
- [ ] A way to handle asynchronous operations.

> **Explanation:** Partial application involves fixing a few arguments of a function, producing another function with a smaller arity.

### Which library provides utility functions for currying in JavaScript?

- [x] Lodash
- [ ] React
- [ ] Angular
- [ ] Vue.js

> **Explanation:** Lodash provides utility functions for currying in JavaScript.

### What is a potential drawback of currying?

- [ ] It always improves performance.
- [x] It can introduce complexity.
- [ ] It reduces code reusability.
- [ ] It makes functions synchronous.

> **Explanation:** Currying can introduce complexity, especially if overused.

### How can partial application be achieved in JavaScript?

- [x] Using closures to retain fixed arguments.
- [ ] By using async/await.
- [ ] Through inheritance.
- [ ] By using promises.

> **Explanation:** Partial application can be achieved using closures to retain fixed arguments.

### What is the output of the following code snippet?
```javascript
const add = (a) => (b) => a + b;
const addFive = add(5);
console.log(addFive(10));
```

- [x] 15
- [ ] 10
- [ ] 5
- [ ] 0

> **Explanation:** The function `add` is curried, and `addFive` is a function that adds 5 to its argument. Thus, `addFive(10)` returns 15.

### Which of the following is a use case for currying?

- [x] Creating reusable functions with preset arguments.
- [ ] Handling asynchronous operations.
- [ ] Optimizing loop performance.
- [ ] Managing state in applications.

> **Explanation:** Currying is useful for creating reusable functions with preset arguments.

### What is the main benefit of using currying in functional programming?

- [x] It allows for more flexible function composition and reuse.
- [ ] It enhances the speed of function execution.
- [ ] It simplifies asynchronous code.
- [ ] It reduces memory usage.

> **Explanation:** Currying allows for more flexible function composition and reuse.

### Can currying be implemented manually without libraries?

- [x] True
- [ ] False

> **Explanation:** Currying can be implemented manually without libraries, although libraries like Lodash can simplify the process.

### Is it possible to use currying and partial application together?

- [x] True
- [ ] False

> **Explanation:** Currying and partial application can be used together to create highly flexible and reusable functions.

{{< /quizdown >}}

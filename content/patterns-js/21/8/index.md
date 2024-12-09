---
canonical: "https://softwarepatternslexicon.com/patterns-js/21/8"
title: "JavaScript Decorators and Annotations (ESNext): Enhancing Code with Modern Techniques"
description: "Explore JavaScript decorators, a proposed ESNext feature for annotating and modifying classes and properties, enabling cleaner and more expressive code. Learn about their current status, implementation using Babel or TypeScript, and practical use cases."
linkTitle: "21.8 Decorators and Annotations (ESNext)"
tags:
- "JavaScript"
- "Decorators"
- "Annotations"
- "ESNext"
- "TypeScript"
- "Babel"
- "Metaprogramming"
- "Advanced Techniques"
date: 2024-11-25
type: docs
nav_weight: 218000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 21.8 Decorators and Annotations (ESNext)

In the ever-evolving landscape of JavaScript, decorators stand out as a powerful feature that promises to bring a new level of expressiveness and modularity to the language. As part of the ECMAScript proposal process, decorators are designed to provide a way to annotate and modify classes and their members, offering a cleaner and more declarative approach to common programming patterns such as logging, validation, and memoization.

### Understanding Decorators

Decorators are a form of syntactic sugar that allows developers to wrap a class, method, or property with additional functionality. They are inspired by similar features in other languages like Python and Java, where decorators and annotations are used extensively to enhance code readability and maintainability.

#### What Are Decorators?

Decorators are functions that take a target (such as a class or method) and return a modified version of that target. They can be used to add metadata, modify behavior, or even replace the target entirely. In JavaScript, decorators are proposed to be used primarily with classes and their members.

```javascript
function log(target, key, descriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function(...args) {
        console.log(`Calling ${key} with`, args);
        return originalMethod.apply(this, args);
    };
    return descriptor;
}

class Example {
    @log
    method(arg) {
        return `Argument received: ${arg}`;
    }
}

const example = new Example();
example.method('test'); // Logs: Calling method with ['test']
```

In this example, the `@log` decorator is applied to the `method` of the `Example` class. It wraps the original method, adding logging functionality without modifying the method's core logic.

### Current Status of Decorators in ECMAScript

Decorators are currently at Stage 3 in the ECMAScript proposal process, which means they are nearing completion but are not yet part of the official JavaScript standard. This stage indicates that the proposal is complete and ready for feedback from the community and implementers.

#### The Proposal Process

The ECMAScript proposal process consists of several stages:

1. **Stage 0 (Strawman):** An initial idea that is being explored.
2. **Stage 1 (Proposal):** A formal proposal with a detailed description and examples.
3. **Stage 2 (Draft):** The proposal is being actively developed and refined.
4. **Stage 3 (Candidate):** The proposal is complete and ready for implementation feedback.
5. **Stage 4 (Finished):** The proposal is ready to be included in the ECMAScript specification.

Decorators have been in development for several years, with various iterations and refinements. As of now, they are implemented in some transpilers like Babel and TypeScript, allowing developers to experiment with them before they become part of the official standard.

### Implementing Decorators with Babel and TypeScript

While decorators are not yet natively supported in JavaScript, you can use transpilers like Babel and TypeScript to implement them in your projects. These tools allow you to write modern JavaScript code and compile it into a version that is compatible with current JavaScript engines.

#### Using Babel

Babel is a popular JavaScript compiler that supports a wide range of ECMAScript proposals, including decorators. To use decorators with Babel, you need to install the necessary plugins and configure your Babel setup.

```bash
npm install --save-dev @babel/core @babel/cli @babel/preset-env @babel/plugin-proposal-decorators
```

In your Babel configuration file (e.g., `.babelrc`), enable the decorators plugin:

```json
{
  "presets": ["@babel/preset-env"],
  "plugins": [["@babel/plugin-proposal-decorators", { "legacy": true }]]
}
```

#### Using TypeScript

TypeScript, a superset of JavaScript, has built-in support for decorators. To enable decorators in TypeScript, you need to set the `experimentalDecorators` option to `true` in your `tsconfig.json` file.

```json
{
  "compilerOptions": {
    "target": "ES5",
    "experimentalDecorators": true
  }
}
```

With this configuration, you can start using decorators in your TypeScript projects without any additional setup.

### Practical Use Cases for Decorators

Decorators can be used in a variety of scenarios to simplify and enhance your code. Here are some common use cases:

#### Logging

Decorators can be used to add logging functionality to methods, allowing you to track method calls and arguments without cluttering your code with logging statements.

```javascript
function log(target, key, descriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function(...args) {
        console.log(`Calling ${key} with`, args);
        return originalMethod.apply(this, args);
    };
    return descriptor;
}
```

#### Validation

You can use decorators to enforce validation rules on method arguments or class properties, ensuring that your code adheres to certain constraints.

```javascript
function validate(target, key, descriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function(...args) {
        if (args.some(arg => arg == null)) {
            throw new Error('Invalid arguments');
        }
        return originalMethod.apply(this, args);
    };
    return descriptor;
}

class Validator {
    @validate
    process(data) {
        console.log('Processing:', data);
    }
}

const validator = new Validator();
validator.process('valid data'); // Works fine
// validator.process(null); // Throws an error
```

#### Memoization

Decorators can be used to implement memoization, a technique that caches the results of expensive function calls and returns the cached result when the same inputs occur again.

```javascript
function memoize(target, key, descriptor) {
    const originalMethod = descriptor.value;
    const cache = new Map();
    descriptor.value = function(...args) {
        const key = JSON.stringify(args);
        if (!cache.has(key)) {
            cache.set(key, originalMethod.apply(this, args));
        }
        return cache.get(key);
    };
    return descriptor;
}

class Calculator {
    @memoize
    fibonacci(n) {
        if (n <= 1) return n;
        return this.fibonacci(n - 1) + this.fibonacci(n - 2);
    }
}

const calculator = new Calculator();
console.log(calculator.fibonacci(10)); // Efficiently calculates Fibonacci numbers
```

### Potential Syntax and Implementation Details

The syntax for decorators in JavaScript is still being finalized, but the current proposal suggests using the `@` symbol followed by the decorator function name. Decorators can be applied to classes, methods, and properties, and they can be stacked to apply multiple decorators to a single target.

#### Class Decorators

Class decorators are applied to the entire class, allowing you to modify the class constructor or add static properties.

```javascript
function sealed(constructor) {
    Object.seal(constructor);
    Object.seal(constructor.prototype);
}

@sealed
class SealedClass {
    // Class implementation
}
```

#### Method Decorators

Method decorators are applied to individual methods, allowing you to modify the method's behavior or add metadata.

```javascript
function readonly(target, key, descriptor) {
    descriptor.writable = false;
    return descriptor;
}

class ReadOnlyExample {
    @readonly
    method() {
        console.log('This method cannot be overridden');
    }
}
```

#### Property Decorators

Property decorators are applied to class properties, allowing you to modify property descriptors or add metadata.

```javascript
function nonEnumerable(target, key) {
    Object.defineProperty(target, key, {
        enumerable: false,
        configurable: true,
        writable: true
    });
}

class HiddenProperty {
    @nonEnumerable
    hidden = 'This property is not enumerable';
}
```

### Staying Updated with the Latest Proposals

As decorators are still a proposal, it's important to stay informed about the latest developments and changes. The ECMAScript proposal process is dynamic, and features can evolve significantly before they are finalized.

#### Resources for Staying Informed

- **TC39 GitHub Repository:** The official repository for ECMAScript proposals, where you can track the progress of decorators and other features. [TC39 Proposals](https://github.com/tc39/proposals)
- **MDN Web Docs:** A comprehensive resource for JavaScript documentation, including information on upcoming features. [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
- **JavaScript Weekly:** A newsletter that provides updates on the latest JavaScript news and developments. [JavaScript Weekly](https://javascriptweekly.com/)

### Conclusion

Decorators offer a powerful way to enhance and modularize your JavaScript code, providing a declarative approach to common programming patterns. While they are not yet part of the official JavaScript standard, tools like Babel and TypeScript allow you to experiment with decorators today. As the proposal progresses, staying informed about the latest developments will ensure that you are ready to leverage decorators when they become a standard feature of JavaScript.

### Try It Yourself

To get hands-on experience with decorators, try modifying the code examples provided in this article. Experiment with creating your own decorators for different use cases, such as caching, access control, or performance monitoring. By exploring the possibilities of decorators, you'll gain a deeper understanding of their potential and how they can improve your JavaScript code.

### Knowledge Check

## Test Your Knowledge on JavaScript Decorators and Annotations

{{< quizdown >}}

### What is the primary purpose of decorators in JavaScript?

- [x] To annotate and modify classes and properties
- [ ] To create new JavaScript data types
- [ ] To replace JavaScript functions with new syntax
- [ ] To manage memory allocation in JavaScript

> **Explanation:** Decorators are used to annotate and modify classes and properties, enhancing code readability and functionality.

### At what stage is the decorators proposal in the ECMAScript process?

- [ ] Stage 1
- [ ] Stage 2
- [x] Stage 3
- [ ] Stage 4

> **Explanation:** Decorators are currently at Stage 3, meaning they are a candidate for inclusion in the ECMAScript standard.

### Which tool can be used to implement decorators in JavaScript today?

- [ ] Node.js
- [x] Babel
- [ ] Webpack
- [ ] ESLint

> **Explanation:** Babel is a JavaScript compiler that supports decorators through plugins, allowing developers to use them before they are officially part of the language.

### How can decorators be enabled in TypeScript?

- [ ] By installing a special TypeScript plugin
- [x] By setting `experimentalDecorators` to `true` in `tsconfig.json`
- [ ] By using a custom TypeScript compiler
- [ ] By writing decorators in a separate file

> **Explanation:** Decorators can be enabled in TypeScript by setting the `experimentalDecorators` option to `true` in the `tsconfig.json` file.

### Which of the following is a common use case for decorators?

- [x] Logging
- [ ] Creating new HTML elements
- [ ] Managing CSS styles
- [ ] Compiling JavaScript code

> **Explanation:** Decorators are commonly used for logging, among other use cases like validation and memoization.

### What symbol is used to denote a decorator in JavaScript?

- [ ] #
- [ ] $
- [x] @
- [ ] %

> **Explanation:** The `@` symbol is used to denote a decorator in JavaScript.

### Can decorators be applied to class properties?

- [x] Yes
- [ ] No

> **Explanation:** Decorators can be applied to class properties, methods, and entire classes.

### What is the role of a method decorator?

- [x] To modify the behavior or add metadata to a method
- [ ] To create a new method in a class
- [ ] To delete a method from a class
- [ ] To change the return type of a method

> **Explanation:** Method decorators are used to modify the behavior or add metadata to a method.

### Which of the following is NOT a benefit of using decorators?

- [ ] Cleaner code
- [ ] Enhanced modularity
- [ ] Improved readability
- [x] Faster execution speed

> **Explanation:** While decorators offer cleaner code, enhanced modularity, and improved readability, they do not inherently improve execution speed.

### True or False: Decorators are already part of the official JavaScript standard.

- [ ] True
- [x] False

> **Explanation:** Decorators are not yet part of the official JavaScript standard; they are still in the proposal stage.

{{< /quizdown >}}

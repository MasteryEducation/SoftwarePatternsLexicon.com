---
linkTitle: "3.3.2 Class Decorators (TypeScript)"
title: "Class Decorators in TypeScript: Enhance and Modify Classes with Ease"
description: "Explore the power of class decorators in TypeScript to modify and enhance classes, methods, and properties. Learn implementation steps, use cases, and best practices."
categories:
- Design Patterns
- JavaScript
- TypeScript
tags:
- Class Decorators
- TypeScript
- Design Patterns
- JavaScript
- Programming
date: 2024-10-25
type: docs
nav_weight: 332000
canonical: "https://softwarepatternslexicon.com/patterns-js/3/3/2"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 3.3.2 Class Decorators (TypeScript)

### Introduction

Class decorators in TypeScript provide a powerful mechanism to modify or enhance classes and their members without altering the original code. They enable developers to add metadata, enforce access control, log method calls, and much more. This article delves into the intent, implementation, and practical applications of class decorators, offering a comprehensive guide to leveraging this feature effectively.

### Understand the Intent

The primary intent of class decorators is to allow developers to modify or enhance classes and their members in a declarative manner. By using decorators, you can:

- Add metadata to classes for various purposes, such as dependency injection.
- Modify the behavior of methods or properties without altering the original class code.
- Implement cross-cutting concerns like logging, validation, or access control.

### Implementation Steps

To implement class decorators in TypeScript, follow these steps:

1. **Enable Experimental Decorators:**
   - Ensure that the `experimentalDecorators` option is enabled in your TypeScript configuration (`tsconfig.json`).

   ```json
   {
     "compilerOptions": {
       "experimentalDecorators": true
     }
   }
   ```

2. **Define Decorator Functions:**
   - Create decorator functions for classes, methods, accessors, properties, or parameters. A class decorator is a function that takes a class constructor as its argument.

   ```typescript
   function LogClass(target: Function) {
     console.log(`Class Decorator Applied to: ${target.name}`);
   }
   ```

3. **Apply Decorators:**
   - Use the `@` symbol to apply decorators above the class or class member.

   ```typescript
   @LogClass
   class ExampleClass {
     constructor() {
       console.log('ExampleClass instance created');
     }
   }
   ```

### Code Examples

#### Class Decorator for Dependency Injection

A common use case for class decorators is registering classes for dependency injection. Here's how you can implement a simple class decorator for this purpose:

```typescript
function Injectable(target: Function) {
  // Register the class in a dependency injection container
  DependencyContainer.register(target);
}

@Injectable
class Service {
  constructor() {
    console.log('Service instance created');
  }
}
```

#### Method Decorator for Logging

Method decorators can be used to log method calls, providing insights into application behavior:

```typescript
function LogMethod(target: Object, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Method ${propertyKey} called with arguments: ${JSON.stringify(args)}`);
    return originalMethod.apply(this, args);
  };
}

class ExampleService {
  @LogMethod
  executeTask(taskName: string) {
    console.log(`Executing task: ${taskName}`);
  }
}

const service = new ExampleService();
service.executeTask('Task1');
```

### Use Cases

Class decorators are versatile and can be applied in various scenarios:

- **Metadata Addition:** Add metadata to classes for frameworks that rely on reflection, such as Angular or NestJS.
- **Behavior Modification:** Modify the behavior of class methods or properties without altering the original code.
- **Cross-Cutting Concerns:** Implement logging, validation, or access control across multiple classes.

### Practice

Try creating a method decorator that validates input parameters before executing the method. This can be useful for ensuring data integrity and preventing errors.

```typescript
function Validate(target: Object, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    if (args.some(arg => arg == null)) {
      throw new Error('Invalid arguments');
    }
    return originalMethod.apply(this, args);
  };
}

class DataService {
  @Validate
  fetchData(id: number) {
    console.log(`Fetching data for ID: ${id}`);
  }
}

const dataService = new DataService();
try {
  dataService.fetchData(null); // This will throw an error
} catch (error) {
  console.error(error.message);
}
```

### Considerations

- **Order of Execution:** Decorators are applied in the order they are declared, but executed in reverse order. Understanding this order is crucial for complex decorator chains.
- **Experimental Feature:** Decorators are an experimental feature in TypeScript and may change in future versions. Keep this in mind when using them in production code.

### Best Practices

- **Use with Caution:** Since decorators are experimental, use them judiciously and be prepared for potential changes in future TypeScript releases.
- **Combine with SOLID Principles:** Ensure that decorators enhance the code's adherence to SOLID principles, such as single responsibility and open/closed principles.
- **Documentation:** Clearly document the purpose and behavior of decorators to aid in code maintainability and readability.

### Conclusion

Class decorators in TypeScript offer a powerful way to enhance and modify classes and their members. By understanding their intent, implementation, and use cases, you can leverage decorators to write cleaner, more maintainable code. However, given their experimental nature, it's important to use them judiciously and stay informed about potential changes in future TypeScript versions.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of class decorators in TypeScript?

- [x] To modify or enhance classes and their members without altering the original code.
- [ ] To replace the need for interfaces in TypeScript.
- [ ] To provide a way to compile TypeScript to JavaScript.
- [ ] To enforce strict type checking in TypeScript.

> **Explanation:** Class decorators allow developers to modify or enhance classes and their members without changing the original code, enabling the addition of metadata and behavior modification.

### Which TypeScript configuration option must be enabled to use decorators?

- [x] `experimentalDecorators`
- [ ] `strict`
- [ ] `allowJs`
- [ ] `noImplicitAny`

> **Explanation:** The `experimentalDecorators` option must be enabled in the TypeScript configuration to use decorators.

### How do you apply a decorator to a class in TypeScript?

- [x] By using the `@` symbol above the class definition.
- [ ] By using the `#` symbol above the class definition.
- [ ] By calling the decorator function inside the class constructor.
- [ ] By defining the decorator inside the class.

> **Explanation:** Decorators are applied using the `@` symbol above the class or class member.

### What is a common use case for class decorators?

- [x] Registering classes for dependency injection.
- [ ] Compiling TypeScript to JavaScript.
- [ ] Enforcing variable immutability.
- [ ] Creating private class members.

> **Explanation:** Class decorators are commonly used to register classes for dependency injection, among other use cases.

### In which order are decorators applied and executed?

- [x] Applied in the order they are declared, executed in reverse order.
- [ ] Applied and executed in the order they are declared.
- [ ] Applied in reverse order, executed in the order they are declared.
- [ ] Applied and executed in reverse order.

> **Explanation:** Decorators are applied in the order they are declared but executed in reverse order.

### What is a potential drawback of using decorators in TypeScript?

- [x] They are an experimental feature and may change in future versions.
- [ ] They increase the size of the compiled JavaScript.
- [ ] They make TypeScript code incompatible with JavaScript.
- [ ] They require additional libraries to function.

> **Explanation:** Decorators are an experimental feature in TypeScript and may change in future versions, which is a potential drawback.

### Which of the following is a method decorator used for?

- [x] Logging method calls.
- [ ] Compiling TypeScript to JavaScript.
- [ ] Creating private class members.
- [ ] Enforcing variable immutability.

> **Explanation:** Method decorators can be used to log method calls, among other things.

### What should you do to ensure decorators enhance code maintainability?

- [x] Clearly document the purpose and behavior of decorators.
- [ ] Use decorators to replace all class methods.
- [ ] Avoid using decorators in production code.
- [ ] Use decorators only for private class members.

> **Explanation:** Clearly documenting the purpose and behavior of decorators helps ensure code maintainability.

### What is a practice exercise mentioned in the article?

- [x] Create a method decorator that validates input parameters before executing the method.
- [ ] Implement a class decorator that compiles TypeScript to JavaScript.
- [ ] Create a property decorator that makes all properties immutable.
- [ ] Implement a decorator that converts classes to interfaces.

> **Explanation:** The article suggests creating a method decorator that validates input parameters before executing the method as a practice exercise.

### True or False: Decorators can only be applied to classes in TypeScript.

- [ ] True
- [x] False

> **Explanation:** Decorators can be applied to classes, methods, accessors, properties, and parameters in TypeScript.

{{< /quizdown >}}

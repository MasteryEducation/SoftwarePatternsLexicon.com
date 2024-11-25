---
canonical: "https://softwarepatternslexicon.com/patterns-haskell/6/1"
title: "Chain of Responsibility with Function Composition in Haskell"
description: "Explore the Chain of Responsibility pattern using function composition in Haskell to create flexible and reusable pipelines for handling requests."
linkTitle: "6.1 Chain of Responsibility with Function Composition"
categories:
- Haskell
- Design Patterns
- Functional Programming
tags:
- Chain of Responsibility
- Function Composition
- Haskell
- Behavioral Patterns
- Software Design
date: 2024-11-23
type: docs
nav_weight: 61000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 6.1 Chain of Responsibility with Function Composition

In this section, we delve into the Chain of Responsibility pattern, a behavioral design pattern that allows a request to be passed along a chain of handlers until one of them handles it. In Haskell, we can implement this pattern using function composition, creating a pipeline of functions that process requests in sequence. This approach leverages Haskell's functional programming capabilities, such as higher-order functions and immutability, to build flexible and reusable systems.

### Understanding the Chain of Responsibility Pattern

**Intent**: The Chain of Responsibility pattern decouples the sender of a request from its receiver by allowing multiple objects to handle the request. The request is passed along a chain of potential handlers until one of them handles it.

**Key Participants**:
- **Handler**: Defines an interface for handling requests.
- **ConcreteHandler**: Handles requests it is responsible for and forwards requests it does not handle to the next handler.
- **Client**: Initiates the request to a handler in the chain.

**Applicability**:
- Use this pattern when multiple objects can handle a request and the handler is not known a priori.
- When you want to issue a request to one of several objects without specifying the receiver explicitly.
- When the set of handlers and their order can change dynamically.

### Implementing Chain of Responsibility in Haskell

In Haskell, we can implement the Chain of Responsibility pattern using function composition. This involves creating a series of functions that each take an input, perform some processing, and pass the result to the next function in the chain.

#### Function Composition in Haskell

Function composition in Haskell is achieved using the `(.)` operator, which allows us to combine two functions into a single function. For example, if we have two functions `f` and `g`, we can compose them as `f . g`, which is equivalent to `\x -> f (g x)`.

```haskell
-- Function composition example
f :: Int -> Int
f x = x + 1

g :: Int -> Int
g x = x * 2

-- Composed function
h :: Int -> Int
h = f . g

-- Usage
main :: IO ()
main = print (h 3) -- Output: 7
```

In this example, `h` is a composed function that first applies `g` to its input and then applies `f` to the result.

#### Building a Validation Chain

Let's consider a practical example where we build a validation chain for user input. Each function in the chain checks a condition and passes the result to the next function.

```haskell
-- Validation function type
type Validator a = a -> Either String a

-- Validator that checks if a number is positive
positiveValidator :: Validator Int
positiveValidator x
  | x > 0     = Right x
  | otherwise = Left "Number must be positive"

-- Validator that checks if a number is even
evenValidator :: Validator Int
evenValidator x
  | even x    = Right x
  | otherwise = Left "Number must be even"

-- Composing validators
validateNumber :: Validator Int
validateNumber = positiveValidator >=> evenValidator

-- Usage
main :: IO ()
main = do
  print $ validateNumber 4  -- Output: Right 4
  print $ validateNumber (-2) -- Output: Left "Number must be positive"
  print $ validateNumber 3  -- Output: Left "Number must be even"
```

In this example, we define two validators: `positiveValidator` and `evenValidator`. We then compose these validators using the `>=>` operator, which is the Kleisli composition operator for monads. This allows us to chain functions that return `Either` values, passing the result of one function as the input to the next.

### Visualizing the Chain of Responsibility

To better understand the flow of requests through the chain, let's visualize the process using a Mermaid.js diagram.

```mermaid
graph TD;
    A[Input] --> B{positiveValidator}
    B -- Right x --> C{evenValidator}
    B -- Left "Number must be positive" --> D[Error]
    C -- Right x --> E[Success]
    C -- Left "Number must be even" --> D[Error]
```

**Diagram Description**: The diagram illustrates the flow of a number through the validation chain. The input is first processed by `positiveValidator`. If the number is positive, it proceeds to `evenValidator`. If any validator fails, an error is returned.

### Design Considerations

- **Flexibility**: The Chain of Responsibility pattern allows you to add, remove, or reorder handlers dynamically, making it highly flexible.
- **Decoupling**: By decoupling the sender and receiver, you can change the chain without affecting the client code.
- **Responsibility**: Ensure that each handler in the chain has a clear responsibility to avoid confusion and maintainability issues.

### Haskell Unique Features

Haskell's strong type system and functional nature make it particularly well-suited for implementing the Chain of Responsibility pattern. The use of higher-order functions and monads allows for elegant composition and error handling.

### Differences and Similarities

The Chain of Responsibility pattern is similar to the Decorator pattern in that both involve a series of operations applied to an object. However, the Chain of Responsibility pattern focuses on passing a request along a chain, while the Decorator pattern focuses on adding behavior to an object.

### Try It Yourself

Experiment with the validation chain by adding new validators or modifying existing ones. For example, try adding a validator that checks if a number is less than 100.

```haskell
-- Validator that checks if a number is less than 100
lessThanHundredValidator :: Validator Int
lessThanHundredValidator x
  | x < 100   = Right x
  | otherwise = Left "Number must be less than 100"

-- Composing validators with the new validator
validateNumberWithLimit :: Validator Int
validateNumberWithLimit = positiveValidator >=> evenValidator >=> lessThanHundredValidator

-- Usage
main :: IO ()
main = do
  print $ validateNumberWithLimit 50  -- Output: Right 50
  print $ validateNumberWithLimit 150 -- Output: Left "Number must be less than 100"
```

### Knowledge Check

- What is the primary purpose of the Chain of Responsibility pattern?
- How does function composition help in implementing this pattern in Haskell?
- What are the benefits of using the `>=>` operator for composing validators?

### Summary

In this section, we explored the Chain of Responsibility pattern and its implementation in Haskell using function composition. We learned how to create a validation chain for user input and visualized the flow of requests through the chain. By leveraging Haskell's functional programming features, we can build flexible and reusable systems that decouple the sender and receiver of requests.

### Embrace the Journey

Remember, this is just the beginning. As you progress, you'll build more complex and interactive systems using Haskell's powerful design patterns. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Chain of Responsibility with Function Composition

{{< quizdown >}}

### What is the primary purpose of the Chain of Responsibility pattern?

- [x] To decouple the sender of a request from its receiver by allowing multiple objects to handle the request.
- [ ] To add behavior to an object dynamically.
- [ ] To ensure that a class has only one instance.
- [ ] To provide a way to access the elements of an aggregate object sequentially.

> **Explanation:** The Chain of Responsibility pattern allows a request to be passed along a chain of handlers until one of them handles it, decoupling the sender from the receiver.

### How does function composition help in implementing the Chain of Responsibility pattern in Haskell?

- [x] By creating a pipeline of functions that process requests in sequence.
- [ ] By adding behavior to an object dynamically.
- [ ] By ensuring that a class has only one instance.
- [ ] By providing a way to access the elements of an aggregate object sequentially.

> **Explanation:** Function composition allows us to combine multiple functions into a single function, creating a pipeline that processes requests in sequence.

### What operator is used for Kleisli composition in Haskell?

- [x] >=> 
- [ ] <=<
- [ ] >>
- [ ] ++

> **Explanation:** The `>=>` operator is used for Kleisli composition in Haskell, allowing us to chain functions that return monadic values.

### Which of the following is a benefit of using the Chain of Responsibility pattern?

- [x] Flexibility in adding, removing, or reordering handlers.
- [ ] Ensuring that a class has only one instance.
- [ ] Providing a way to access the elements of an aggregate object sequentially.
- [ ] Adding behavior to an object dynamically.

> **Explanation:** The Chain of Responsibility pattern allows for flexibility in adding, removing, or reordering handlers without affecting the client code.

### What is the role of the `Either` type in the validation chain example?

- [x] To represent the result of a validation as either a success or an error.
- [ ] To ensure that a class has only one instance.
- [ ] To provide a way to access the elements of an aggregate object sequentially.
- [ ] To add behavior to an object dynamically.

> **Explanation:** The `Either` type is used to represent the result of a validation as either a success (`Right`) or an error (`Left`).

### What is the difference between the Chain of Responsibility and Decorator patterns?

- [x] Chain of Responsibility focuses on passing a request along a chain, while Decorator focuses on adding behavior to an object.
- [ ] Both patterns focus on adding behavior to an object.
- [ ] Both patterns focus on passing a request along a chain.
- [ ] Chain of Responsibility focuses on adding behavior to an object, while Decorator focuses on passing a request along a chain.

> **Explanation:** The Chain of Responsibility pattern focuses on passing a request along a chain of handlers, while the Decorator pattern focuses on adding behavior to an object.

### Which of the following is a key participant in the Chain of Responsibility pattern?

- [x] Handler
- [ ] Singleton
- [ ] Iterator
- [ ] Decorator

> **Explanation:** The Handler is a key participant in the Chain of Responsibility pattern, defining an interface for handling requests.

### What is the purpose of the `positiveValidator` function in the validation chain example?

- [x] To check if a number is positive and return a success or error.
- [ ] To ensure that a class has only one instance.
- [ ] To provide a way to access the elements of an aggregate object sequentially.
- [ ] To add behavior to an object dynamically.

> **Explanation:** The `positiveValidator` function checks if a number is positive and returns a success (`Right`) or an error (`Left`).

### What is the output of the `validateNumber` function when applied to the number 3?

- [x] Left "Number must be even"
- [ ] Right 3
- [ ] Left "Number must be positive"
- [ ] Right 4

> **Explanation:** The `validateNumber` function returns `Left "Number must be even"` when applied to the number 3, as 3 is not even.

### True or False: The Chain of Responsibility pattern allows for dynamic changes to the set of handlers and their order.

- [x] True
- [ ] False

> **Explanation:** True. The Chain of Responsibility pattern allows for dynamic changes to the set of handlers and their order, providing flexibility in handling requests.

{{< /quizdown >}}

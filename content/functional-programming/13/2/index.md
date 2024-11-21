---
canonical: "https://softwarepatternslexicon.com/functional-programming/13/2"

title: "Glossary of Functional Programming Terms: Key Concepts and Definitions"
description: "Explore the essential glossary of terms in functional programming, including definitions of key concepts, acronyms, and abbreviations to enhance your understanding of FP-specific language."
linkTitle: "Glossary of Functional Programming Terms"
categories:
- Functional Programming
- Software Development
- Design Patterns
tags:
- Functional Programming
- Glossary
- Key Concepts
- Acronyms
- Definitions
date: 2024-11-17
type: docs
nav_weight: 13200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## A.2. Glossary of Terms

Welcome to the glossary of terms for functional programming patterns. This section is designed to provide clear and concise definitions of key concepts, acronyms, and abbreviations commonly used in the realm of functional programming (FP). Whether you're a seasoned developer or new to FP, this glossary will serve as a valuable resource to enhance your understanding and fluency in FP-specific language.

### Pure Function

**Definition:** A pure function is a function where the output value is determined only by its input values, without observable side effects. This means that calling a pure function with the same arguments will always produce the same result.

**Example:**

```pseudocode
function add(x, y):
    return x + y
```

**Explanation:** The `add` function is pure because it consistently returns the sum of `x` and `y` without modifying any external state or relying on external variables.

### Immutability

**Definition:** Immutability refers to the concept of data that cannot be changed after it is created. In functional programming, immutable data structures are preferred because they prevent unintended side effects and make programs easier to reason about.

**Example:**

```pseudocode
let numbers = [1, 2, 3]
let newNumbers = numbers.append(4)  // numbers remains unchanged
```

**Explanation:** The `numbers` array remains unchanged when `newNumbers` is created by appending `4`. This demonstrates immutability.

### Higher-Order Function

**Definition:** A higher-order function is a function that takes one or more functions as arguments or returns a function as its result. This allows for greater abstraction and code reuse.

**Example:**

```pseudocode
function applyFunction(f, x):
    return f(x)

function square(n):
    return n * n

applyFunction(square, 5)  // Returns 25
```

**Explanation:** `applyFunction` is a higher-order function because it takes another function `f` as an argument.

### Function Composition

**Definition:** Function composition is the process of combining two or more functions to produce a new function. The output of one function becomes the input of the next.

**Example:**

```pseudocode
function compose(f, g):
    return function(x):
        return f(g(x))

function addOne(n):
    return n + 1

function double(n):
    return n * 2

let addOneThenDouble = compose(double, addOne)
addOneThenDouble(3)  // Returns 8
```

**Explanation:** The `compose` function creates a new function `addOneThenDouble` that first adds one to its input and then doubles the result.

### Recursion

**Definition:** Recursion is a technique where a function calls itself in order to solve a problem. It is often used in place of iterative loops in functional programming.

**Example:**

```pseudocode
function factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

factorial(5)  // Returns 120
```

**Explanation:** The `factorial` function calls itself to compute the factorial of `n`, demonstrating recursion.

### Lazy Evaluation

**Definition:** Lazy evaluation is a strategy that delays the evaluation of an expression until its value is actually needed. This can improve performance by avoiding unnecessary calculations.

**Example:**

```pseudocode
function lazyAdd(x, y):
    return function():
        return x + y

let addLater = lazyAdd(3, 4)
// The addition is not performed until addLater() is called
addLater()  // Returns 7
```

**Explanation:** The `lazyAdd` function returns a function that performs the addition only when invoked.

### Closure

**Definition:** A closure is a function that captures the lexical scope in which it was defined, allowing it to access variables from that scope even when executed outside of it.

**Example:**

```pseudocode
function makeCounter():
    let count = 0
    return function():
        count = count + 1
        return count

let counter = makeCounter()
counter()  // Returns 1
counter()  // Returns 2
```

**Explanation:** The `makeCounter` function returns a closure that has access to the `count` variable, maintaining its state across calls.

### Partial Application

**Definition:** Partial application is the process of fixing a number of arguments to a function, producing another function of smaller arity.

**Example:**

```pseudocode
function add(x, y):
    return x + y

function addFive = partial(add, 5)
addFive(10)  // Returns 15
```

**Explanation:** The `partial` function creates a new function `addFive` by fixing the first argument of `add` to `5`.

### Currying

**Definition:** Currying is the process of transforming a function with multiple arguments into a sequence of functions, each taking a single argument.

**Example:**

```pseudocode
function curryAdd(x):
    return function(y):
        return x + y

let addThree = curryAdd(3)
addThree(4)  // Returns 7
```

**Explanation:** The `curryAdd` function returns a new function that takes the second argument, demonstrating currying.

### Functor

**Definition:** A functor is a type that implements a `map` function, which applies a function to each element within the functor's context.

**Example:**

```pseudocode
class List:
    function map(f):
        // Apply f to each element and return a new List
```

**Explanation:** The `List` class is a functor because it provides a `map` method to apply a function to its elements.

### Monad

**Definition:** A monad is a design pattern used to handle program-wide concerns in a functional way, such as state or I/O. It provides two operations: `bind` (or `flatMap`) and `unit` (or `return`).

**Example:**

```pseudocode
class Maybe:
    function bind(f):
        // Apply f if value is present, otherwise return Nothing
```

**Explanation:** The `Maybe` class is a monad that encapsulates optional values, providing a `bind` method to chain operations.

### Applicative Functor

**Definition:** An applicative functor is a type that allows for function application lifted over a computational context. It provides an `apply` function to apply functions within the context.

**Example:**

```pseudocode
class Maybe:
    function apply(maybeFunc):
        // Apply the function inside maybeFunc if both are Just
```

**Explanation:** The `Maybe` class can be an applicative functor by implementing an `apply` method.

### Observer Pattern

**Definition:** The observer pattern in functional programming, often used in functional reactive programming (FRP), involves managing streams of data over time.

**Example:**

```pseudocode
class Observable:
    function subscribe(observer):
        // Notify observer of changes
```

**Explanation:** The `Observable` class allows observers to subscribe and receive updates, demonstrating the observer pattern.

### Strategy Pattern

**Definition:** The strategy pattern involves defining a family of algorithms, encapsulating each one, and making them interchangeable. In FP, this is achieved using higher-order functions.

**Example:**

```pseudocode
function sort(strategy, data):
    return strategy(data)

function bubbleSort(data):
    // Implement bubble sort

function quickSort(data):
    // Implement quick sort

sort(bubbleSort, [3, 1, 2])  // Uses bubble sort
```

**Explanation:** The `sort` function takes a sorting strategy as an argument, allowing for different sorting algorithms to be used interchangeably.

### Memoization

**Definition:** Memoization is an optimization technique that involves caching the results of expensive function calls and returning the cached result when the same inputs occur again.

**Example:**

```pseudocode
function memoize(f):
    let cache = {}
    return function(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]

function slowFunction(x):
    // Expensive computation

let fastFunction = memoize(slowFunction)
```

**Explanation:** The `memoize` function wraps `slowFunction`, caching its results for faster subsequent calls.

### Transducer

**Definition:** A transducer is a composable algorithmic transformation that is independent of the context in which it is used. It allows for efficient data processing by avoiding intermediate collections.

**Example:**

```pseudocode
function mapTransducer(f):
    return function(reducer):
        return function(acc, value):
            return reducer(acc, f(value))

function filterTransducer(predicate):
    return function(reducer):
        return function(acc, value):
            if predicate(value):
                return reducer(acc, value)
            else:
                return acc
```

**Explanation:** `mapTransducer` and `filterTransducer` are transducers that can be composed to transform data efficiently.

### Monoid

**Definition:** A monoid is an algebraic structure with a single associative binary operation and an identity element.

**Example:**

```pseudocode
class SumMonoid:
    function concat(a, b):
        return a + b

    function empty():
        return 0
```

**Explanation:** The `SumMonoid` class defines a monoid with addition as the operation and `0` as the identity element.

### Algebraic Data Type (ADT)

**Definition:** An algebraic data type is a composite type used in functional programming, created by combining other types. Common examples include sum types and product types.

**Example:**

```pseudocode
type Option = Some(value) | None
```

**Explanation:** The `Option` type is an ADT representing a value that may or may not be present.

### Lens

**Definition:** A lens is a functional programming construct used to access and update nested data structures in an immutable way.

**Example:**

```pseudocode
function lens(getter, setter):
    return { get: getter, set: setter }

let nameLens = lens(
    (person) => person.name,
    (person, newName) => { ...person, name: newName }
)
```

**Explanation:** The `nameLens` allows for accessing and updating the `name` property of a `person` object immutably.

### Interpreter Pattern

**Definition:** The interpreter pattern involves using functions to define language constructs, allowing for the implementation of domain-specific languages (DSLs).

**Example:**

```pseudocode
function interpret(expression):
    // Evaluate the expression based on its type
```

**Explanation:** The `interpret` function evaluates expressions, demonstrating the interpreter pattern.

### Dependency Injection

**Definition:** Dependency injection is a technique where dependencies are passed explicitly to a function, enhancing modularity and testability.

**Example:**

```pseudocode
function createService(database):
    return function():
        // Use the database to perform operations

let service = createService(myDatabase)
```

**Explanation:** The `createService` function takes a `database` dependency, allowing for easy replacement and testing.

### Zipper

**Definition:** A zipper is a data structure that allows for efficient navigation and updates of immutable data structures by providing a focus within the structure.

**Example:**

```pseudocode
class Zipper:
    function moveLeft():
        // Move focus to the left

    function moveRight():
        // Move focus to the right
```

**Explanation:** The `Zipper` class provides methods to navigate and update a data structure efficiently.

### Option/Maybe Monad

**Definition:** The Option or Maybe monad is used to represent nullable values safely, avoiding null reference errors.

**Example:**

```pseudocode
class Maybe:
    function map(f):
        // Apply f if value is present

    function flatMap(f):
        // Chain operations
```

**Explanation:** The `Maybe` class encapsulates optional values, providing methods to safely operate on them.

### Either Monad

**Definition:** The Either monad is used to handle computations with two possible outcomes, typically representing success and failure.

**Example:**

```pseudocode
class Either:
    function map(f):
        // Apply f if Right

    function flatMap(f):
        // Chain operations
```

**Explanation:** The `Either` class represents computations that may succeed or fail, providing methods to handle both cases.

### Try Monad

**Definition:** The Try monad encapsulates exceptions, allowing for safe handling of operations that might throw exceptions.

**Example:**

```pseudocode
class Try:
    function map(f):
        // Apply f if Success

    function flatMap(f):
        // Chain operations
```

**Explanation:** The `Try` class provides methods to safely handle exceptions as values.

### Validated Data

**Definition:** Validated data involves accumulating errors during validation, allowing for multiple validation errors to be collected and reported.

**Example:**

```pseudocode
class Validated:
    function combine(other):
        // Combine validations
```

**Explanation:** The `Validated` class allows for combining multiple validations, collecting errors along the way.

### Immutable Collections

**Definition:** Immutable collections are data structures that cannot be changed after they are created, providing benefits such as thread safety and easier reasoning.

**Example:**

```pseudocode
let immutableList = [1, 2, 3]
// Operations return new collections
```

**Explanation:** The `immutableList` remains unchanged, demonstrating immutability.

### Streams and Infinite Sequences

**Definition:** Streams and infinite sequences represent potentially infinite data, processed lazily to avoid unnecessary computations.

**Example:**

```pseudocode
function stream(start):
    return function():
        // Generate next value
```

**Explanation:** The `stream` function generates values lazily, allowing for infinite sequences.

### Futures and Promises

**Definition:** Futures and promises are abstractions for managing asynchronous computations, representing values that will be available at some point in the future.

**Example:**

```pseudocode
class Future:
    function then(f):
        // Chain asynchronous operations
```

**Explanation:** The `Future` class provides methods to handle asynchronous computations.

### Actors Model

**Definition:** The actors model is a concurrency model that isolates state within actors, using message passing for communication.

**Example:**

```pseudocode
class Actor:
    function send(message):
        // Process message
```

**Explanation:** The `Actor` class encapsulates state and behavior, communicating via messages.

### Data Parallelism

**Definition:** Data parallelism involves parallelizing operations over data, often using patterns like map-reduce to process data concurrently.

**Example:**

```pseudocode
function parallelMap(f, data):
    // Apply f to each element in parallel
```

**Explanation:** The `parallelMap` function demonstrates data parallelism by applying a function to data concurrently.

### Property-Based Testing

**Definition:** Property-based testing involves testing code with generative data, ensuring correctness over a wide range of inputs.

**Example:**

```pseudocode
function propertyTest(property, generator):
    // Test property with generated data
```

**Explanation:** The `propertyTest` function uses generated data to test properties, ensuring robustness.

### Continuation-Passing Style (CPS)

**Definition:** Continuation-passing style is a style of programming where control is passed explicitly in the form of a continuation.

**Example:**

```pseudocode
function cpsAdd(x, y, cont):
    cont(x + y)
```

**Explanation:** The `cpsAdd` function demonstrates CPS by passing the result to a continuation.

### Free Monads

**Definition:** Free monads abstract over effects, allowing for the construction of interpreters for programs.

**Example:**

```pseudocode
class Free:
    function interpret(interpreter):
        // Execute program with interpreter
```

**Explanation:** The `Free` class allows for defining and interpreting programs abstractly.

### Effect Systems

**Definition:** Effect systems use type systems to manage side effects safely, enforcing constraints at compile time.

**Example:**

```pseudocode
function safeFunction() -> Effect[IO]:
    // Perform IO safely
```

**Explanation:** The `safeFunction` demonstrates an effect system by specifying effects in its type signature.

### Category Theory

**Definition:** Category theory is a branch of mathematics that provides a theoretical framework for understanding functional programming concepts like functors and monads.

**Example:**

```pseudocode
class Functor:
    function map(f):
        // Apply f to context
```

**Explanation:** The `Functor` class is rooted in category theory, providing a mathematical basis for its operations.

### Quiz Time!

{{< quizdown >}}

### What is a pure function?

- [x] A function that returns the same result given the same inputs and has no side effects.
- [ ] A function that modifies global state.
- [ ] A function that relies on external variables.
- [ ] A function that can return different results for the same inputs.

> **Explanation:** A pure function consistently returns the same result for the same inputs and does not cause side effects.

### What is immutability?

- [x] The concept of data that cannot be changed after it is created.
- [ ] The ability to modify data at any time.
- [ ] A function that changes its input data.
- [ ] A variable that can be reassigned.

> **Explanation:** Immutability refers to data that remains unchanged once created, preventing unintended side effects.

### What is a higher-order function?

- [x] A function that takes other functions as arguments or returns a function.
- [ ] A function that only performs arithmetic operations.
- [ ] A function that modifies global variables.
- [ ] A function that cannot be nested.

> **Explanation:** Higher-order functions can accept functions as arguments or return them, allowing for greater abstraction.

### What is function composition?

- [x] Combining two or more functions to produce a new function.
- [ ] A function that calls itself.
- [ ] A function that modifies its input.
- [ ] A function that only returns numbers.

> **Explanation:** Function composition involves creating a new function by combining existing functions, where the output of one becomes the input of another.

### What is recursion?

- [x] A technique where a function calls itself to solve a problem.
- [ ] A function that modifies global state.
- [x] A method of looping through data.
- [ ] A function that only returns strings.

> **Explanation:** Recursion involves a function calling itself to solve problems, often used instead of loops in FP.

### What is lazy evaluation?

- [x] Delaying the evaluation of an expression until its value is needed.
- [ ] Evaluating all expressions immediately.
- [ ] A function that modifies its input.
- [ ] A variable that can be reassigned.

> **Explanation:** Lazy evaluation defers computation until necessary, improving performance by avoiding unnecessary calculations.

### What is a closure?

- [x] A function that captures the lexical scope in which it was defined.
- [ ] A function that modifies global variables.
- [x] A function that cannot be nested.
- [ ] A function that only returns numbers.

> **Explanation:** Closures retain access to their defining environment, allowing them to access variables from that scope.

### What is partial application?

- [x] Fixing a number of arguments to a function, producing another function.
- [ ] A function that modifies its input.
- [ ] A function that only returns strings.
- [ ] A function that cannot be nested.

> **Explanation:** Partial application involves creating a new function by fixing some arguments of an existing function.

### What is currying?

- [x] Transforming a function with multiple arguments into a sequence of functions.
- [ ] A function that modifies global variables.
- [ ] A function that only returns numbers.
- [ ] A function that cannot be nested.

> **Explanation:** Currying transforms a function into a series of functions, each taking a single argument.

### True or False: A monad is a design pattern used to handle program-wide concerns in a functional way.

- [x] True
- [ ] False

> **Explanation:** Monads encapsulate computations within a context, providing a functional way to handle concerns like state or I/O.

{{< /quizdown >}}

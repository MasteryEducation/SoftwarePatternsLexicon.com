---
canonical: "https://softwarepatternslexicon.com/patterns-dart/3/4"
title: "Mastering Generics in Dart: A Comprehensive Guide"
description: "Explore the power of generics in Dart to create flexible, reusable, and type-safe code. Learn how to implement generic types and methods, ensuring robust applications with practical examples and best practices."
linkTitle: "3.4 Generics in Dart"
categories:
- Dart Programming
- Flutter Development
- Software Design Patterns
tags:
- Generics
- Type Safety
- Dart Language
- Flutter
- Programming Best Practices
date: 2024-11-17
type: docs
nav_weight: 3400
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 3.4 Generics in Dart

Generics are a powerful feature in Dart that allow developers to create flexible, reusable, and type-safe code. By using generics, you can define classes, methods, and interfaces that work with any data type, while maintaining type safety. This section will delve into the intricacies of generics in Dart, providing you with the knowledge and tools to leverage them effectively in your Flutter applications.

### Understanding Generics

Generics enable you to write code that can handle different data types without sacrificing type safety. They allow you to parameterize types, making your code more flexible and reusable. In Dart, generics are used extensively in collections, such as `List`, `Set`, and `Map`, as well as in custom classes and methods.

#### Why Use Generics?

1. **Type Safety**: Generics help catch type errors at compile time, reducing runtime errors and improving code reliability.
2. **Code Reusability**: By writing generic code, you can create functions and classes that work with any data type, reducing code duplication.
3. **Flexibility**: Generics allow you to create flexible APIs that can work with a wide range of data types.

### Implementing Generic Types

In Dart, you can create generic classes and interfaces by using type parameters. A type parameter is a placeholder for a specific type that is specified when the class or interface is instantiated.

#### Creating a Generic Class

Let's start by creating a simple generic class called `Box` that can hold a value of any type:

```dart
class Box<T> {
  T value;

  Box(this.value);

  void display() {
    print('The value is: $value');
  }
}

void main() {
  var intBox = Box<int>(123);
  intBox.display(); // Output: The value is: 123

  var stringBox = Box<String>('Hello');
  stringBox.display(); // Output: The value is: Hello
}
```

In this example, `Box` is a generic class with a type parameter `T`. When creating an instance of `Box`, you specify the type you want to use, such as `int` or `String`.

#### Generic Interfaces

You can also create generic interfaces in Dart. Here's an example of a generic interface called `Comparable`:

```dart
abstract class Comparable<T> {
  int compareTo(T other);
}

class Number implements Comparable<Number> {
  final int value;

  Number(this.value);

  @override
  int compareTo(Number other) {
    return value - other.value;
  }
}

void main() {
  var num1 = Number(10);
  var num2 = Number(20);

  print(num1.compareTo(num2)); // Output: -10
}
```

In this example, `Comparable` is a generic interface with a type parameter `T`. The `Number` class implements the `Comparable` interface, specifying `Number` as the type parameter.

### Generic Methods

In addition to generic classes and interfaces, Dart allows you to define generic methods. A generic method is a method that has its own type parameters, separate from any class-level type parameters.

#### Creating a Generic Method

Here's an example of a generic method that swaps two elements in a list:

```dart
void swap<T>(List<T> list, int index1, int index2) {
  T temp = list[index1];
  list[index1] = list[index2];
  list[index2] = temp;
}

void main() {
  var numbers = [1, 2, 3, 4];
  swap<int>(numbers, 0, 2);
  print(numbers); // Output: [3, 2, 1, 4]

  var words = ['apple', 'banana', 'cherry'];
  swap<String>(words, 1, 2);
  print(words); // Output: ['apple', 'cherry', 'banana']
}
```

In this example, the `swap` method is a generic method with a type parameter `T`. The method can swap elements of any type in a list.

### Type Constraints

Sometimes, you may want to restrict the types that can be used with a generic class or method. Dart allows you to specify type constraints using the `extends` keyword.

#### Using Type Constraints

Here's an example of a generic class with a type constraint:

```dart
class Pair<T extends num> {
  final T first;
  final T second;

  Pair(this.first, this.second);

  T sum() => first + second;
}

void main() {
  var intPair = Pair<int>(3, 4);
  print(intPair.sum()); // Output: 7

  var doublePair = Pair<double>(2.5, 3.5);
  print(doublePair.sum()); // Output: 6.0

  // var stringPair = Pair<String>('a', 'b'); // Error: 'String' doesn't extend 'num'
}
```

In this example, the `Pair` class has a type constraint `T extends num`, meaning `T` must be a subtype of `num`. This allows the `sum` method to use the `+` operator, which is defined for `num` types.

### Generics in Collections

Dart's collection classes, such as `List`, `Set`, and `Map`, are implemented using generics. This allows them to store elements of any type while maintaining type safety.

#### Generic Lists

A `List` in Dart can hold elements of any type, but you can specify a type parameter to enforce type safety:

```dart
void main() {
  List<int> numbers = [1, 2, 3];
  numbers.add(4);
  print(numbers); // Output: [1, 2, 3, 4]

  // numbers.add('five'); // Error: The argument type 'String' can't be assigned to the parameter type 'int'
}
```

In this example, `numbers` is a `List` of `int`, ensuring that only integers can be added to the list.

#### Generic Maps

A `Map` in Dart can also use generics to specify the types of keys and values:

```dart
void main() {
  Map<String, int> ages = {
    'Alice': 25,
    'Bob': 30,
  };

  ages['Charlie'] = 35;
  print(ages); // Output: {Alice: 25, Bob: 30, Charlie: 35}

  // ages[42] = 'forty-two'; // Error: The argument type 'int' can't be assigned to the parameter type 'String'
}
```

In this example, `ages` is a `Map` with `String` keys and `int` values, ensuring type safety for both keys and values.

### Advanced Generics

Dart provides several advanced features for working with generics, including generic typedefs and covariant/contravariant type parameters.

#### Generic Typedefs

A typedef in Dart can be generic, allowing you to define function signatures with type parameters:

```dart
typedef Comparator<T> = int Function(T a, T b);

int compareInts(int a, int b) => a - b;

void main() {
  Comparator<int> intComparator = compareInts;
  print(intComparator(3, 2)); // Output: 1
}
```

In this example, `Comparator` is a generic typedef that defines a function signature with a type parameter `T`.

#### Covariant and Contravariant Type Parameters

Dart supports covariance and contravariance for type parameters, allowing you to specify how type parameters relate to subtyping.

- **Covariant**: A covariant type parameter can be replaced with a subtype.
- **Contravariant**: A contravariant type parameter can be replaced with a supertype.

Here's an example of covariance in Dart:

```dart
class Animal {}

class Dog extends Animal {}

class Cat extends Animal {}

void main() {
  List<Animal> animals = [Dog(), Cat()];
  List<Dog> dogs = [Dog()];

  animals = dogs; // Covariance: List<Dog> is a subtype of List<Animal>
}
```

In this example, `List<Dog>` is a subtype of `List<Animal>` due to covariance.

### Best Practices for Using Generics

1. **Use Generics for Type Safety**: Always use generics to enforce type safety in your code, especially when working with collections.
2. **Avoid Overusing Generics**: While generics are powerful, overusing them can make your code harder to read and maintain. Use them judiciously.
3. **Leverage Type Constraints**: Use type constraints to restrict the types that can be used with your generic classes and methods, ensuring they work as intended.
4. **Document Your Code**: When using generics, provide clear documentation to explain the purpose and usage of type parameters.

### Visualizing Generics in Dart

To better understand how generics work in Dart, let's visualize the relationship between generic classes and their instances using a class diagram.

```mermaid
classDiagram
    class Box<T> {
        T value
        +Box(T value)
        +display() void
    }

    class Box~int~ {
        +value : int
        +Box(int value)
        +display() void
    }

    class Box~String~ {
        +value : String
        +Box(String value)
        +display() void
    }

    Box<T> <|-- Box~int~
    Box<T> <|-- Box~String~
```

**Diagram Description**: This class diagram illustrates the `Box` class with a type parameter `T`, and two specific instances of `Box` with `int` and `String` types. The diagram shows how the generic class `Box<T>` can be instantiated with different types, resulting in `Box<int>` and `Box<String>`.

### Try It Yourself

Experiment with the code examples provided in this section to deepen your understanding of generics in Dart. Try modifying the `Box` class to include additional methods or properties, or create your own generic classes and methods to solve specific problems.

### References and Links

- [Dart Language Tour: Generics](https://dart.dev/guides/language/language-tour#generics)
- [Effective Dart: Usage](https://dart.dev/guides/language/effective-dart/usage)
- [Dart API Documentation](https://api.dart.dev/)

### Knowledge Check

- What are the benefits of using generics in Dart?
- How do you create a generic class in Dart?
- What is the purpose of type constraints in generics?
- How do covariance and contravariance work in Dart?

### Embrace the Journey

Remember, mastering generics in Dart is a journey that will enhance your ability to write flexible, reusable, and type-safe code. As you continue to explore and experiment with generics, you'll discover new ways to leverage their power in your Flutter applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is the primary benefit of using generics in Dart?

- [x] Type safety
- [ ] Faster execution
- [ ] Reduced memory usage
- [ ] Simplified syntax

> **Explanation:** Generics provide type safety by catching type errors at compile time, ensuring more robust code.

### How do you define a generic class in Dart?

- [x] By using type parameters
- [ ] By using abstract classes
- [ ] By using interfaces
- [ ] By using mixins

> **Explanation:** A generic class is defined using type parameters, allowing it to work with any data type.

### What keyword is used to specify type constraints in Dart generics?

- [x] extends
- [ ] implements
- [ ] with
- [ ] super

> **Explanation:** The `extends` keyword is used to specify type constraints, restricting the types that can be used with a generic class or method.

### Which of the following is an example of a generic collection in Dart?

- [x] List<T>
- [ ] Set
- [ ] Map
- [ ] Queue

> **Explanation:** `List<T>` is a generic collection in Dart, allowing it to store elements of any type.

### What is covariance in Dart generics?

- [x] Allowing a subtype to replace a type parameter
- [ ] Allowing a supertype to replace a type parameter
- [ ] Allowing any type to replace a type parameter
- [ ] Allowing only primitive types to replace a type parameter

> **Explanation:** Covariance allows a subtype to replace a type parameter, enabling more flexible type relationships.

### How can you enforce type safety when using collections in Dart?

- [x] By specifying type parameters
- [ ] By using dynamic types
- [ ] By using abstract classes
- [ ] By using interfaces

> **Explanation:** Specifying type parameters enforces type safety, ensuring that only elements of the specified type can be added to the collection.

### What is a generic method in Dart?

- [x] A method with its own type parameters
- [ ] A method that returns a generic type
- [ ] A method that uses abstract classes
- [ ] A method that uses interfaces

> **Explanation:** A generic method has its own type parameters, allowing it to work with any data type independently of class-level type parameters.

### What is the purpose of a generic typedef in Dart?

- [x] To define function signatures with type parameters
- [ ] To create abstract classes
- [ ] To implement interfaces
- [ ] To define mixins

> **Explanation:** A generic typedef defines function signatures with type parameters, enabling flexible and reusable function definitions.

### What is contravariance in Dart generics?

- [x] Allowing a supertype to replace a type parameter
- [ ] Allowing a subtype to replace a type parameter
- [ ] Allowing any type to replace a type parameter
- [ ] Allowing only primitive types to replace a type parameter

> **Explanation:** Contravariance allows a supertype to replace a type parameter, enabling more flexible type relationships.

### True or False: Generics in Dart can only be used with collections.

- [ ] True
- [x] False

> **Explanation:** False. Generics can be used with classes, methods, interfaces, and typedefs, not just collections.

{{< /quizdown >}}

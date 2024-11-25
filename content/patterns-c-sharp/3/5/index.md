---
canonical: "https://softwarepatternslexicon.com/patterns-c-sharp/3/5"

title: "Mastering C# Generics: A Comprehensive Guide for Expert Developers"
description: "Explore the power of generics in C# to write reusable, type-safe code. Learn about generic types, methods, constraints, and variance, with detailed examples and best practices."
linkTitle: "3.5 Generics"
categories:
- CSharp Programming
- Software Development
- Design Patterns
tags:
- CSharp Generics
- Type Safety
- Reusability
- Constraints
- Variance
date: 2024-11-17
type: docs
nav_weight: 3500
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 3.5 Generics

Generics in C# are a powerful feature that allows developers to define classes, interfaces, and methods with a placeholder for the type of data they store or use. This enables the creation of reusable and type-safe code components, which can work with any data type while maintaining the benefits of compile-time type checking. In this section, we will delve into the intricacies of generics, exploring their syntax, use cases, constraints, and the concepts of covariance and contravariance.

### Introduction to Generics

Generics were introduced in C# 2.0 to address the limitations of non-generic collections and methods, which often required boxing and unboxing operations or type casting, leading to runtime errors and performance issues. By using generics, you can create data structures and algorithms that work with any data type, while avoiding these pitfalls.

#### Key Benefits of Generics

- **Type Safety**: Generics provide compile-time type checking, reducing runtime errors.
- **Code Reusability**: Write code once and use it with different data types.
- **Performance**: Avoid boxing and unboxing, leading to more efficient code execution.

### Generic Types and Methods

Generics can be applied to classes, interfaces, and methods. Let's explore each of these in detail.

#### Generic Classes

A generic class is defined with a type parameter, which acts as a placeholder for the actual data type. Here's a simple example of a generic class:

```csharp
public class GenericList<T>
{
    private T[] items;
    private int count;

    public GenericList(int capacity)
    {
        items = new T[capacity];
        count = 0;
    }

    public void Add(T item)
    {
        if (count < items.Length)
        {
            items[count++] = item;
        }
    }

    public T Get(int index)
    {
        if (index >= 0 && index < count)
        {
            return items[index];
        }
        throw new IndexOutOfRangeException();
    }
}
```

In this example, `GenericList<T>` is a generic class where `T` is the type parameter. You can create instances of `GenericList` with any data type:

```csharp
var intList = new GenericList<int>(10);
intList.Add(1);

var stringList = new GenericList<string>(10);
stringList.Add("Hello");
```

#### Generic Methods

Generic methods allow you to define a method with a type parameter. This is useful when you want a method to operate on different data types without duplicating code. Here's an example:

```csharp
public class Utilities
{
    public static void Swap<T>(ref T a, ref T b)
    {
        T temp = a;
        a = b;
        b = temp;
    }
}
```

You can use the `Swap` method with any data type:

```csharp
int x = 5, y = 10;
Utilities.Swap(ref x, ref y);

string first = "first", second = "second";
Utilities.Swap(ref first, ref second);
```

### Constraints in Generics

Constraints allow you to specify the requirements that a type argument must satisfy. This can include implementing an interface, inheriting from a base class, or having a parameterless constructor.

#### Types of Constraints

1. **Where T : struct**: The type argument must be a value type.
2. **Where T : class**: The type argument must be a reference type.
3. **Where T : new()**: The type argument must have a parameterless constructor.
4. **Where T : BaseClass**: The type argument must inherit from `BaseClass`.
5. **Where T : InterfaceName**: The type argument must implement `InterfaceName`.

Here's an example of a generic class with constraints:

```csharp
public class Repository<T> where T : IEntity, new()
{
    public T Create()
    {
        return new T();
    }
}
```

In this example, `T` must implement the `IEntity` interface and have a parameterless constructor.

### Covariance and Contravariance

Covariance and contravariance are advanced concepts that allow for more flexible use of generics, particularly with interfaces and delegates.

#### Covariance

Covariance allows you to use a more derived type than originally specified. It is applicable to generic interfaces and delegates with out parameters. Here's an example with interfaces:

```csharp
public interface ICovariant<out T>
{
    T Get();
}

public class Sample : ICovariant<string>
{
    public string Get() => "Hello";
}

ICovariant<object> covariant = new Sample();
```

In this example, `ICovariant<string>` can be assigned to `ICovariant<object>` because of covariance.

#### Contravariance

Contravariance allows you to use a less derived type than originally specified. It is applicable to generic interfaces and delegates with in parameters. Here's an example:

```csharp
public interface IContravariant<in T>
{
    void Set(T value);
}

public class Sample : IContravariant<object>
{
    public void Set(object value) { }
}

IContravariant<string> contravariant = new Sample();
```

In this example, `IContravariant<object>` can be assigned to `IContravariant<string>` because of contravariance.

### Practical Applications of Generics

Generics are widely used in the .NET framework, particularly in collections such as `List<T>`, `Dictionary<TKey, TValue>`, and `Queue<T>`. They are also essential in defining reusable algorithms and data structures.

#### Example: Generic Stack

Let's implement a simple generic stack:

```csharp
public class Stack<T>
{
    private T[] elements;
    private int position;

    public Stack(int size)
    {
        elements = new T[size];
        position = 0;
    }

    public void Push(T item)
    {
        if (position < elements.Length)
        {
            elements[position++] = item;
        }
        else
        {
            throw new InvalidOperationException("Stack is full");
        }
    }

    public T Pop()
    {
        if (position > 0)
        {
            return elements[--position];
        }
        throw new InvalidOperationException("Stack is empty");
    }
}
```

You can use this stack with any data type:

```csharp
var intStack = new Stack<int>(5);
intStack.Push(1);
intStack.Push(2);
Console.WriteLine(intStack.Pop());

var stringStack = new Stack<string>(5);
stringStack.Push("A");
stringStack.Push("B");
Console.WriteLine(stringStack.Pop());
```

### Visualizing Generics

To better understand how generics work, let's visualize the relationship between generic types and their constraints using a class diagram.

```mermaid
classDiagram
    class GenericList<T> {
        - T[] items
        - int count
        + Add(T item)
        + Get(int index) T
    }
    class Repository<T> {
        + Create() T
    }
    class IEntity
    GenericList<T> ..|> T
    Repository<T> ..|> T
    T <|-- IEntity
```

**Diagram Description**: This class diagram illustrates the `GenericList<T>` and `Repository<T>` classes, showing how they relate to their type parameters and constraints.

### Best Practices for Using Generics

1. **Use Descriptive Type Parameters**: Use meaningful names for type parameters, such as `TKey` and `TValue` in a dictionary.
2. **Leverage Constraints**: Use constraints to enforce type requirements and ensure type safety.
3. **Avoid Overusing Generics**: Use generics judiciously to avoid unnecessary complexity.
4. **Consider Performance**: Be mindful of performance implications, especially with large data structures.

### Try It Yourself

Experiment with the generic stack example by adding methods such as `Peek` to view the top element without removing it, or `Clear` to empty the stack. Try creating a generic queue or a linked list to deepen your understanding of generics.

### Knowledge Check

- What are the benefits of using generics in C#?
- How do constraints enhance the functionality of generics?
- Explain the difference between covariance and contravariance.
- How can you apply generics to improve code reusability?

### Conclusion

Generics are a cornerstone of modern C# programming, enabling developers to write flexible, reusable, and type-safe code. By understanding and applying generics effectively, you can create robust applications that are easier to maintain and extend. Remember, mastering generics is a journey—keep experimenting and exploring new ways to leverage this powerful feature in your projects.

## Quiz Time!

{{< quizdown >}}

### What is a primary benefit of using generics in C#?

- [x] Type safety
- [ ] Increased runtime errors
- [ ] Slower performance
- [ ] Less code reusability

> **Explanation:** Generics provide compile-time type safety, reducing runtime errors.

### Which constraint ensures a type argument must be a reference type?

- [ ] Where T : struct
- [x] Where T : class
- [ ] Where T : new()
- [ ] Where T : BaseClass

> **Explanation:** The `where T : class` constraint ensures the type argument is a reference type.

### What is covariance?

- [x] Allows using a more derived type than originally specified
- [ ] Allows using a less derived type than originally specified
- [ ] Restricts type usage to value types
- [ ] Restricts type usage to reference types

> **Explanation:** Covariance allows using a more derived type than originally specified, applicable to out parameters.

### What is contravariance?

- [ ] Allows using a more derived type than originally specified
- [x] Allows using a less derived type than originally specified
- [ ] Restricts type usage to value types
- [ ] Restricts type usage to reference types

> **Explanation:** Contravariance allows using a less derived type than originally specified, applicable to in parameters.

### Which of the following is a correct use of a generic method?

- [x] public static void Swap<T>(ref T a, ref T b)
- [ ] public static void Swap(ref T a, ref T b)
- [ ] public static void Swap<T>(T a, T b)
- [ ] public static void Swap(ref a, ref b)

> **Explanation:** The correct syntax for a generic method includes the type parameter `<T>`.

### What does the `new()` constraint ensure?

- [ ] The type argument must be a reference type
- [ ] The type argument must be a value type
- [x] The type argument must have a parameterless constructor
- [ ] The type argument must implement an interface

> **Explanation:** The `new()` constraint ensures the type argument has a parameterless constructor.

### How can generics improve performance?

- [x] By avoiding boxing and unboxing
- [ ] By increasing runtime errors
- [ ] By reducing code reusability
- [ ] By increasing memory usage

> **Explanation:** Generics improve performance by avoiding boxing and unboxing operations.

### What is a common use case for generics in C#?

- [x] Collections like List<T> and Dictionary<TKey, TValue>
- [ ] Static methods
- [ ] Non-generic classes
- [ ] Value types only

> **Explanation:** Generics are commonly used in collections like `List<T>` and `Dictionary<TKey, TValue>`.

### Which of the following is NOT a type of constraint in generics?

- [ ] Where T : struct
- [ ] Where T : class
- [ ] Where T : new()
- [x] Where T : static

> **Explanation:** There is no `where T : static` constraint in C# generics.

### True or False: Generics can only be used with classes.

- [ ] True
- [x] False

> **Explanation:** Generics can be used with classes, interfaces, and methods.

{{< /quizdown >}}

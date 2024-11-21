---
linkTitle: "2.1.4 Prototype"
title: "Prototype Design Pattern in Go: Efficient Object Creation through Cloning"
description: "Explore the Prototype design pattern in Go, focusing on efficient object creation through cloning. Learn implementation steps, use cases, and best practices with code examples."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Prototype Pattern
- Creational Patterns
- Go Design Patterns
- Object Cloning
- Software Development
date: 2024-10-25
type: docs
nav_weight: 214000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/1/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.1.4 Prototype

The Prototype design pattern is a creational pattern that allows you to create new objects by copying an existing object, known as the prototype. This pattern is particularly useful when the cost of creating an object is expensive or when you want to avoid subclasses of an object creator in the client application. In this article, we will explore the Prototype pattern in the context of Go, including its implementation, use cases, and best practices.

### Purpose of the Prototype Pattern

- **Specify the kinds of objects to create using a prototypical instance.**
- **Create new objects by copying an existing object, known as the prototype.**

The Prototype pattern is beneficial in scenarios where creating a new instance of a class is more resource-intensive than copying an existing instance. It also helps in situations where the system should be independent of how its objects are created, composed, and represented.

### Implementation Steps

To implement the Prototype pattern in Go, follow these steps:

1. **Define an Interface with a Cloning Method:**
   Create an interface that includes a method for cloning objects. This method is typically named `Clone()`.

2. **Implement the Interface in Concrete Types:**
   Provide cloning functionality in concrete types by implementing the `Clone()` method.

3. **Use the Prototype to Create New Objects:**
   Use the `Clone()` method to create new instances based on the prototype.

### When to Use

- **When the Cost of Creating an Object is Expensive:**
  If creating an object involves a significant amount of resources, such as network calls or complex computations, the Prototype pattern can help by cloning an existing instance.

- **To Avoid Subclasses of an Object Creator in the Client Application:**
  The Prototype pattern allows you to create objects without specifying their concrete classes, thus avoiding the need for subclasses in the client code.

### Go-Specific Tips

- **Use Go's Built-in Copy Mechanisms for Shallow Copies:**
  Go provides built-in mechanisms for copying data structures, which can be leveraged for implementing shallow copies in the Prototype pattern.

- **Be Cautious with Pointers and Reference Types:**
  When cloning objects that contain pointers or reference types, ensure that you perform deep copies if necessary to avoid unintended sharing of data between instances.

### Example: Cloning Shapes with Different Properties

Let's consider an example where we have a set of shapes, and we want to clone them with different properties.

```go
package main

import (
    "fmt"
)

// Shape is the prototype interface with a Clone method.
type Shape interface {
    Clone() Shape
    GetInfo() string
}

// Circle is a concrete type that implements the Shape interface.
type Circle struct {
    Radius int
    Color  string
}

// Clone creates a copy of the Circle.
func (c *Circle) Clone() Shape {
    return &Circle{
        Radius: c.Radius,
        Color:  c.Color,
    }
}

// GetInfo returns the details of the Circle.
func (c *Circle) GetInfo() string {
    return fmt.Sprintf("Circle: Radius=%d, Color=%s", c.Radius, c.Color)
}

// Rectangle is another concrete type that implements the Shape interface.
type Rectangle struct {
    Width  int
    Height int
    Color  string
}

// Clone creates a copy of the Rectangle.
func (r *Rectangle) Clone() Shape {
    return &Rectangle{
        Width:  r.Width,
        Height: r.Height,
        Color:  r.Color,
    }
}

// GetInfo returns the details of the Rectangle.
func (r *Rectangle) GetInfo() string {
    return fmt.Sprintf("Rectangle: Width=%d, Height=%d, Color=%s", r.Width, r.Height, r.Color)
}

func main() {
    // Create a prototype Circle and Rectangle.
    circle := &Circle{Radius: 5, Color: "Red"}
    rectangle := &Rectangle{Width: 10, Height: 20, Color: "Blue"}

    // Clone the shapes.
    clonedCircle := circle.Clone()
    clonedRectangle := rectangle.Clone()

    // Display the cloned shapes.
    fmt.Println(clonedCircle.GetInfo())
    fmt.Println(clonedRectangle.GetInfo())
}
```

### Deep Copying When Necessary

In cases where your objects contain pointers or reference types, you may need to implement deep copying to ensure that the cloned objects do not share references with the original objects. This can be done by manually copying each field that requires deep copying.

### Advantages and Disadvantages

**Advantages:**

- **Reduces the Cost of Creating Objects:** By cloning existing objects, you can avoid the overhead of creating new instances from scratch.
- **Simplifies Object Creation:** The Prototype pattern abstracts the process of object creation, making it easier to manage and extend.

**Disadvantages:**

- **Complexity in Deep Copying:** Implementing deep copies can be complex, especially for objects with nested references.
- **Potential for Unintended Sharing:** Without careful implementation, cloned objects may unintentionally share data with their prototypes.

### Best Practices

- **Ensure Proper Cloning:** Always verify that your `Clone()` method correctly copies all necessary fields, especially when dealing with pointers and reference types.
- **Use Prototypes Wisely:** Consider the trade-offs between cloning and creating new instances, and choose the approach that best suits your application's needs.

### Comparisons with Other Patterns

The Prototype pattern is often compared with the Factory Method and Abstract Factory patterns. While the Factory patterns focus on creating objects through interfaces, the Prototype pattern emphasizes cloning existing instances. Choose the Prototype pattern when cloning is more efficient or when you need to avoid subclassing.

### Conclusion

The Prototype design pattern is a powerful tool for creating objects efficiently in Go. By leveraging cloning, you can reduce the cost of object creation and simplify your codebase. However, it's essential to implement cloning carefully to avoid unintended data sharing. With the right approach, the Prototype pattern can significantly enhance your application's performance and maintainability.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the Prototype design pattern?

- [x] To create new objects by copying an existing object.
- [ ] To define a family of algorithms and make them interchangeable.
- [ ] To provide a simplified interface to a complex subsystem.
- [ ] To ensure a class has only one instance.

> **Explanation:** The Prototype pattern is used to create new objects by copying an existing object, known as the prototype.

### When is the Prototype pattern particularly useful?

- [x] When the cost of creating an object is expensive.
- [ ] When you need to ensure a class has only one instance.
- [ ] When you want to provide a simplified interface to a complex subsystem.
- [ ] When you need to define a family of algorithms.

> **Explanation:** The Prototype pattern is useful when the cost of creating an object is expensive, allowing you to clone existing instances instead.

### What method is typically included in the Prototype interface?

- [x] Clone()
- [ ] Execute()
- [ ] Initialize()
- [ ] Build()

> **Explanation:** The Prototype interface typically includes a `Clone()` method for creating copies of objects.

### What should you be cautious about when implementing the Prototype pattern in Go?

- [x] Pointers and reference types to avoid unintended sharing.
- [ ] Using too many interfaces.
- [ ] Overloading functions.
- [ ] Using global variables.

> **Explanation:** When implementing the Prototype pattern in Go, be cautious with pointers and reference types to avoid unintended sharing of data.

### Which of the following is a disadvantage of the Prototype pattern?

- [x] Complexity in deep copying.
- [ ] Increased memory usage.
- [ ] Difficulty in understanding the code.
- [ ] Lack of flexibility.

> **Explanation:** One disadvantage of the Prototype pattern is the complexity involved in implementing deep copies, especially for objects with nested references.

### What is a key advantage of using the Prototype pattern?

- [x] Reduces the cost of creating objects.
- [ ] Ensures a class has only one instance.
- [ ] Provides a simplified interface to a complex subsystem.
- [ ] Defines a family of algorithms.

> **Explanation:** A key advantage of the Prototype pattern is that it reduces the cost of creating objects by cloning existing instances.

### How does the Prototype pattern differ from the Factory Method pattern?

- [x] The Prototype pattern clones existing instances, while the Factory Method pattern creates new instances.
- [ ] The Prototype pattern provides a simplified interface, while the Factory Method pattern ensures a single instance.
- [ ] The Prototype pattern defines a family of algorithms, while the Factory Method pattern makes them interchangeable.
- [ ] The Prototype pattern is used for object pooling, while the Factory Method pattern is used for object creation.

> **Explanation:** The Prototype pattern focuses on cloning existing instances, whereas the Factory Method pattern creates new instances through interfaces.

### What is a common method name used in the Prototype pattern?

- [x] Clone
- [ ] Execute
- [ ] Initialize
- [ ] Build

> **Explanation:** The common method name used in the Prototype pattern is `Clone`, which is responsible for creating copies of objects.

### In the provided example, what types of shapes are cloned?

- [x] Circle and Rectangle
- [ ] Triangle and Square
- [ ] Hexagon and Pentagon
- [ ] Ellipse and Parallelogram

> **Explanation:** In the provided example, the types of shapes that are cloned are Circle and Rectangle.

### True or False: The Prototype pattern can help avoid subclasses of an object creator in the client application.

- [x] True
- [ ] False

> **Explanation:** True. The Prototype pattern can help avoid subclasses of an object creator in the client application by allowing objects to be created through cloning.

{{< /quizdown >}}

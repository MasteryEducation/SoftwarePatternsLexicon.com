---
canonical: "https://softwarepatternslexicon.com/patterns-java/3/5"

title: "Java Design Patterns: Composition Over Inheritance"
description: "Explore the principle of Composition Over Inheritance in Java, emphasizing flexibility and reusability in software design. Learn through examples and design patterns like Strategy, Decorator, and Composite."
linkTitle: "3.5 Composition Over Inheritance"
tags:
- "Java"
- "Design Patterns"
- "Composition"
- "Inheritance"
- "Strategy Pattern"
- "Decorator Pattern"
- "Composite Pattern"
- "Object-Oriented Design"
date: 2024-11-25
type: docs
nav_weight: 35000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 3.5 Composition Over Inheritance

In the realm of object-oriented programming, the principle of "Composition Over Inheritance" is a cornerstone for creating flexible and maintainable software. This principle suggests favoring object composition over class inheritance to achieve greater flexibility and reusability in code. Understanding this principle is crucial for experienced Java developers and software architects aiming to design robust systems.

### Understanding Inheritance and Composition

#### Inheritance

Inheritance is a mechanism in object-oriented programming that allows a new class, known as a subclass, to inherit properties and behaviors (methods) from an existing class, referred to as a superclass. This relationship is often described as an "is-a" relationship. For example, a `Dog` class might inherit from an `Animal` class, indicating that a dog is a type of animal.

```java
class Animal {
    void eat() {
        System.out.println("This animal eats.");
    }
}

class Dog extends Animal {
    void bark() {
        System.out.println("The dog barks.");
    }
}
```

While inheritance promotes code reuse and establishes a clear hierarchical relationship, it comes with limitations that can lead to tightly coupled and fragile code.

#### Composition

Composition, on the other hand, is a design principle where a class is composed of one or more objects from other classes, allowing it to delegate responsibilities to these objects. This is often described as a "has-a" relationship. For example, a `Car` class might have an `Engine` object.

```java
class Engine {
    void start() {
        System.out.println("Engine starts.");
    }
}

class Car {
    private Engine engine;

    Car() {
        this.engine = new Engine();
    }

    void startCar() {
        engine.start();
        System.out.println("Car starts.");
    }
}
```

Composition provides greater flexibility by allowing the behavior of a class to be changed at runtime through object composition, rather than being fixed at compile time through inheritance.

### Limitations of Inheritance

1. **Tight Coupling**: Inheritance creates a strong coupling between the superclass and subclass. Changes in the superclass can inadvertently affect all subclasses, leading to fragile code.

2. **Lack of Flexibility**: Inheritance is static and defined at compile time. It does not allow for changing the behavior of a class at runtime.

3. **Inheritance Hierarchy**: Deep inheritance hierarchies can become complex and difficult to manage, leading to code that is hard to understand and maintain.

4. **Fragility**: Subclasses are dependent on the implementation details of their superclasses. Any change in the superclass can break the functionality of subclasses.

### Composition as a Solution

Composition addresses these limitations by promoting loose coupling and enhancing flexibility. It allows for the dynamic composition of behaviors and responsibilities, making it easier to adapt and extend functionality without modifying existing code.

#### Example: Replacing Inheritance with Composition

Consider a scenario where you have different types of birds, each with unique flying behaviors. Using inheritance, you might create a hierarchy like this:

```java
class Bird {
    void fly() {
        System.out.println("This bird flies.");
    }
}

class Sparrow extends Bird {
    void chirp() {
        System.out.println("The sparrow chirps.");
    }
}

class Penguin extends Bird {
    @Override
    void fly() {
        System.out.println("Penguins can't fly.");
    }
}
```

This approach has limitations, as the `Penguin` class must override the `fly` method to reflect its inability to fly. Instead, using composition, you can separate the flying behavior into its own class:

```java
interface FlyBehavior {
    void fly();
}

class CanFly implements FlyBehavior {
    public void fly() {
        System.out.println("This bird flies.");
    }
}

class CannotFly implements FlyBehavior {
    public void fly() {
        System.out.println("This bird can't fly.");
    }
}

class Bird {
    private FlyBehavior flyBehavior;

    Bird(FlyBehavior flyBehavior) {
        this.flyBehavior = flyBehavior;
    }

    void performFly() {
        flyBehavior.fly();
    }
}

class Sparrow extends Bird {
    Sparrow() {
        super(new CanFly());
    }

    void chirp() {
        System.out.println("The sparrow chirps.");
    }
}

class Penguin extends Bird {
    Penguin() {
        super(new CannotFly());
    }
}
```

This design allows for greater flexibility, as the flying behavior can be changed at runtime by simply assigning a different `FlyBehavior` object.

### Design Patterns Emphasizing Composition

Several design patterns leverage the principle of composition over inheritance to achieve flexibility and reusability.

#### Strategy Pattern

The Strategy Pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. It allows the algorithm to vary independently from clients that use it. This pattern is a prime example of using composition to replace inheritance.

- **Example**: In the bird example above, the `FlyBehavior` interface and its implementations (`CanFly` and `CannotFly`) demonstrate the Strategy Pattern.

#### Decorator Pattern

The Decorator Pattern attaches additional responsibilities to an object dynamically. It provides a flexible alternative to subclassing for extending functionality.

- **Example**: Consider a `Coffee` class that can be decorated with various add-ons like milk or sugar. Each add-on is a separate class that implements a common interface, allowing for dynamic composition.

```java
interface Coffee {
    String getDescription();
    double cost();
}

class SimpleCoffee implements Coffee {
    public String getDescription() {
        return "Simple coffee";
    }

    public double cost() {
        return 2.0;
    }
}

class MilkDecorator implements Coffee {
    private Coffee coffee;

    MilkDecorator(Coffee coffee) {
        this.coffee = coffee;
    }

    public String getDescription() {
        return coffee.getDescription() + ", milk";
    }

    public double cost() {
        return coffee.cost() + 0.5;
    }
}
```

#### Composite Pattern

The Composite Pattern allows you to compose objects into tree structures to represent part-whole hierarchies. It lets clients treat individual objects and compositions of objects uniformly.

- **Example**: A graphical user interface might use the Composite Pattern to treat individual components (like buttons and text fields) and composite components (like panels) uniformly.

```java
interface Graphic {
    void draw();
}

class Line implements Graphic {
    public void draw() {
        System.out.println("Drawing a line.");
    }
}

class Picture implements Graphic {
    private List<Graphic> graphics = new ArrayList<>();

    void add(Graphic graphic) {
        graphics.add(graphic);
    }

    public void draw() {
        for (Graphic graphic : graphics) {
            graphic.draw();
        }
    }
}
```

### When to Use Inheritance vs. Composition

#### When to Use Inheritance

- **Is-a Relationship**: Use inheritance when there is a clear hierarchical relationship, and the subclass truly is a type of the superclass.
- **Shared Behavior**: When multiple subclasses share common behavior that is unlikely to change, inheritance can be appropriate.

#### When to Use Composition

- **Has-a Relationship**: Use composition when a class should contain another class, and the relationship is not hierarchical.
- **Dynamic Behavior**: When behavior needs to be changed or extended at runtime, composition is preferable.
- **Avoiding Fragility**: To avoid the fragility and tight coupling of inheritance, composition offers a more robust solution.

### Conclusion

The principle of Composition Over Inheritance is a powerful tool in the software architect's toolkit. By favoring composition, developers can create systems that are more flexible, maintainable, and adaptable to change. This principle is embodied in several design patterns, including Strategy, Decorator, and Composite, each offering unique ways to leverage composition for better software design.

### Exercises

1. **Refactor an Inheritance Hierarchy**: Take an existing class hierarchy in your project and refactor it to use composition. Note the changes in flexibility and maintainability.

2. **Implement a Strategy Pattern**: Create a simple application that uses the Strategy Pattern to change behavior at runtime. Experiment with different strategies and observe the effects.

3. **Design a Decorator**: Implement a Decorator Pattern for a simple class, such as a beverage or a text editor. Add multiple decorators and test their interactions.

4. **Build a Composite Structure**: Design a composite structure using the Composite Pattern. Create a tree-like hierarchy and implement operations that treat individual and composite objects uniformly.

### Key Takeaways

- Composition provides greater flexibility and reusability than inheritance.
- Design patterns like Strategy, Decorator, and Composite leverage composition to solve common design problems.
- Understanding when to use inheritance versus composition is crucial for effective software design.

### Reflection

Consider how you might apply the principle of Composition Over Inheritance in your current projects. Reflect on the benefits and challenges you might encounter and how this principle can lead to more robust and adaptable software.

---

## Test Your Knowledge: Composition Over Inheritance in Java

{{< quizdown >}}

### What is a primary advantage of using composition over inheritance?

- [x] Greater flexibility and reusability
- [ ] Simpler code structure
- [ ] Faster execution time
- [ ] Easier debugging

> **Explanation:** Composition allows for greater flexibility and reusability by enabling dynamic behavior changes and reducing tight coupling.

### Which design pattern is a classic example of composition?

- [x] Strategy Pattern
- [ ] Singleton Pattern
- [ ] Factory Pattern
- [ ] Observer Pattern

> **Explanation:** The Strategy Pattern uses composition to encapsulate algorithms and make them interchangeable.

### What is a common problem associated with deep inheritance hierarchies?

- [x] Complexity and difficulty in maintenance
- [ ] Lack of code reuse
- [ ] Poor performance
- [ ] Limited functionality

> **Explanation:** Deep inheritance hierarchies can become complex and difficult to maintain, leading to fragile code.

### In which scenario is inheritance more appropriate than composition?

- [x] When there is a clear "is-a" relationship
- [ ] When behavior needs to change at runtime
- [ ] When avoiding tight coupling
- [ ] When implementing design patterns

> **Explanation:** Inheritance is appropriate when there is a clear hierarchical "is-a" relationship between classes.

### Which pattern allows objects to be treated uniformly in a tree structure?

- [x] Composite Pattern
- [ ] Strategy Pattern
- [ ] Decorator Pattern
- [ ] Factory Pattern

> **Explanation:** The Composite Pattern allows objects to be composed into tree structures and treated uniformly.

### What is a key benefit of using the Decorator Pattern?

- [x] Dynamic addition of responsibilities to objects
- [ ] Simplified class hierarchy
- [ ] Improved performance
- [ ] Reduced code duplication

> **Explanation:** The Decorator Pattern allows for the dynamic addition of responsibilities to objects without modifying their structure.

### How does the Strategy Pattern enhance flexibility?

- [x] By encapsulating algorithms and making them interchangeable
- [ ] By reducing the number of classes
- [ ] By improving performance
- [ ] By simplifying code

> **Explanation:** The Strategy Pattern enhances flexibility by encapsulating algorithms and allowing them to be swapped at runtime.

### Which of the following is a limitation of inheritance?

- [x] Tight coupling between superclass and subclass
- [ ] Lack of code reuse
- [ ] Poor readability
- [ ] Limited functionality

> **Explanation:** Inheritance creates tight coupling between superclass and subclass, making changes in the superclass affect all subclasses.

### What does the principle of Composition Over Inheritance emphasize?

- [x] Favoring object composition to achieve flexibility
- [ ] Using inheritance for all relationships
- [ ] Avoiding design patterns
- [ ] Simplifying code structure

> **Explanation:** The principle emphasizes using object composition to achieve greater flexibility and adaptability in software design.

### True or False: Composition allows for behavior changes at runtime.

- [x] True
- [ ] False

> **Explanation:** Composition allows for behavior changes at runtime by composing objects with different behaviors.

{{< /quizdown >}}

---

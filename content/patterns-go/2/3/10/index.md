---
linkTitle: "2.3.10 Template Method"
title: "Template Method Design Pattern in Go: Understanding, Implementation, and Best Practices"
description: "Explore the Template Method design pattern in Go, its intent, implementation, and real-world applications. Learn how to define algorithm skeletons with flexible steps using Go's unique features."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Template Method
- Behavioral Patterns
- GoF Patterns
- Go Language
- Software Design
date: 2024-10-25
type: docs
nav_weight: 240000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/3/10"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.3.10 Template Method

The Template Method pattern is a behavioral design pattern that defines the skeleton of an algorithm in a method, allowing subclasses to redefine certain steps of the algorithm without changing its structure. This pattern is particularly useful when you have multiple classes that share similar algorithms with some differing steps, and you want to prevent code duplication.

### Understand the Intent

The primary intent of the Template Method pattern is to:

- Define the skeleton of an algorithm in a method, deferring some steps to subclasses.
- Allow subclasses to redefine certain steps without changing the algorithm's structure.

This pattern is beneficial in scenarios where you need a consistent algorithm structure but require flexibility in specific steps. It promotes code reuse and adherence to the DRY (Don't Repeat Yourself) principle.

### Implementation Steps

Implementing the Template Method pattern in Go involves the following steps:

#### 1. Abstract Class or Struct

- **Define the Template Method:** Create a method that outlines the algorithm's steps. This method should call other methods representing the steps of the algorithm.
- **Implement Invariant Steps:** Implement the steps that remain constant across subclasses within the template method.

#### 2. Primitive Operations

- **Declare Abstract or Placeholder Methods:** Define methods for the steps that can vary. In Go, you can use interfaces or function arguments to achieve this flexibility.

#### 3. Concrete Implementations

- **Create Concrete Types:** Implement the variable steps in concrete types that satisfy the interface or provide specific implementations for the placeholder methods.

### When to Use

Consider using the Template Method pattern in the following scenarios:

- When you have multiple classes that share similar algorithms with some differing steps.
- To prevent code duplication by centralizing the algorithm structure in a single place.
- When you want to enforce a consistent algorithm structure while allowing flexibility in specific steps.

### Go-Specific Tips

In Go, you can leverage interfaces and composition to emulate inheritance, which is typically used in object-oriented languages to implement the Template Method pattern. Additionally, you can pass function arguments to methods to introduce variable behavior if appropriate.

### Example: Data Parser

Let's explore a practical example of the Template Method pattern in Go by implementing a data parser that reads, processes, and writes data. We'll create different parsers for XML and JSON, each implementing the processing step differently.

```go
package main

import (
	"fmt"
)

// DataParser defines the interface for our template method
type DataParser interface {
	ReadData() string
	ProcessData(data string) string
	WriteData(data string)
	Parse() // Template method
}

// BaseParser provides the template method implementation
type BaseParser struct {
	DataParser
}

func (b *BaseParser) Parse() {
	data := b.ReadData()
	processedData := b.ProcessData(data)
	b.WriteData(processedData)
}

// XMLParser implements the DataParser interface for XML
type XMLParser struct {
	BaseParser
}

func (x *XMLParser) ReadData() string {
	fmt.Println("Reading XML data...")
	return "<data>XML Data</data>"
}

func (x *XMLParser) ProcessData(data string) string {
	fmt.Println("Processing XML data...")
	return "Processed " + data
}

func (x *XMLParser) WriteData(data string) {
	fmt.Println("Writing XML data:", data)
}

// JSONParser implements the DataParser interface for JSON
type JSONParser struct {
	BaseParser
}

func (j *JSONParser) ReadData() string {
	fmt.Println("Reading JSON data...")
	return `{"data": "JSON Data"}`
}

func (j *JSONParser) ProcessData(data string) string {
	fmt.Println("Processing JSON data...")
	return "Processed " + data
}

func (j *JSONParser) WriteData(data string) {
	fmt.Println("Writing JSON data:", data)
}

func main() {
	xmlParser := &XMLParser{}
	xmlParser.BaseParser.DataParser = xmlParser
	xmlParser.Parse()

	jsonParser := &JSONParser{}
	jsonParser.BaseParser.DataParser = jsonParser
	jsonParser.Parse()
}
```

### Explanation of the Example

In this example, we have defined a `DataParser` interface that outlines the methods required for parsing data. The `BaseParser` struct implements the `Parse` method, which serves as the template method. It calls `ReadData`, `ProcessData`, and `WriteData` in sequence.

The `XMLParser` and `JSONParser` structs implement the `DataParser` interface, providing specific implementations for reading, processing, and writing XML and JSON data, respectively. By embedding `BaseParser` and setting the `DataParser` field, we ensure that the correct methods are called during parsing.

### Advantages and Disadvantages

#### Advantages

- **Code Reuse:** Centralizes the algorithm structure, promoting code reuse and reducing duplication.
- **Flexibility:** Allows subclasses to implement specific steps, providing flexibility while maintaining a consistent algorithm structure.
- **Maintainability:** Changes to the algorithm structure are made in one place, simplifying maintenance.

#### Disadvantages

- **Complexity:** Can introduce complexity if overused or applied to simple algorithms.
- **Limited Flexibility:** The algorithm structure is fixed, which may not be suitable for highly dynamic scenarios.

### Best Practices

- **Use Sparingly:** Apply the Template Method pattern when you have a clear need for a consistent algorithm structure with variable steps.
- **Leverage Interfaces:** Use interfaces to define the variable steps, promoting flexibility and adherence to Go's idiomatic practices.
- **Avoid Overengineering:** Ensure that the pattern is necessary and not overcomplicating the design.

### Comparisons

The Template Method pattern is often compared to the Strategy pattern. While both patterns allow for flexibility in behavior, the Template Method pattern defines the algorithm structure, whereas the Strategy pattern allows for complete algorithm interchangeability.

### Conclusion

The Template Method pattern is a powerful tool for defining consistent algorithm structures with flexible steps. By leveraging Go's interfaces and composition, you can implement this pattern effectively, promoting code reuse and maintainability. As with any design pattern, it's essential to apply it judiciously, ensuring it aligns with your project's needs and complexity.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Template Method pattern?

- [x] To define the skeleton of an algorithm in a method, deferring some steps to subclasses.
- [ ] To allow complete algorithm interchangeability.
- [ ] To encapsulate a request as an object.
- [ ] To provide a unified interface to a set of interfaces.

> **Explanation:** The Template Method pattern defines the skeleton of an algorithm, allowing subclasses to redefine certain steps without changing the algorithm's structure.

### Which Go feature is commonly used to implement the Template Method pattern?

- [x] Interfaces
- [ ] Goroutines
- [ ] Channels
- [ ] Reflection

> **Explanation:** Interfaces in Go are used to define the variable steps in the Template Method pattern, promoting flexibility and adherence to Go's idiomatic practices.

### What is a disadvantage of the Template Method pattern?

- [x] It can introduce complexity if overused.
- [ ] It allows for complete algorithm interchangeability.
- [ ] It centralizes the algorithm structure.
- [ ] It promotes code reuse.

> **Explanation:** The Template Method pattern can introduce complexity if overused or applied to simple algorithms.

### When should you consider using the Template Method pattern?

- [x] When you have multiple classes that share similar algorithms with some differing steps.
- [ ] When you need complete algorithm interchangeability.
- [ ] When you want to encapsulate a request as an object.
- [ ] When you need to provide a unified interface to a set of interfaces.

> **Explanation:** The Template Method pattern is suitable when you have multiple classes that share similar algorithms with some differing steps, preventing code duplication.

### How does the Template Method pattern promote code reuse?

- [x] By centralizing the algorithm structure in a single place.
- [ ] By allowing complete algorithm interchangeability.
- [ ] By encapsulating a request as an object.
- [ ] By providing a unified interface to a set of interfaces.

> **Explanation:** The Template Method pattern centralizes the algorithm structure, promoting code reuse and reducing duplication.

### What is the role of the template method in the Template Method pattern?

- [x] It outlines the algorithm's steps.
- [ ] It provides complete algorithm interchangeability.
- [ ] It encapsulates a request as an object.
- [ ] It provides a unified interface to a set of interfaces.

> **Explanation:** The template method outlines the algorithm's steps, calling other methods representing the steps of the algorithm.

### Which pattern is often compared to the Template Method pattern?

- [x] Strategy pattern
- [ ] Observer pattern
- [ ] Command pattern
- [ ] Singleton pattern

> **Explanation:** The Template Method pattern is often compared to the Strategy pattern, as both allow for flexibility in behavior.

### What is a best practice when implementing the Template Method pattern in Go?

- [x] Use interfaces to define the variable steps.
- [ ] Use goroutines to define the variable steps.
- [ ] Use channels to define the variable steps.
- [ ] Use reflection to define the variable steps.

> **Explanation:** Using interfaces to define the variable steps promotes flexibility and adherence to Go's idiomatic practices.

### What is a key difference between the Template Method and Strategy patterns?

- [x] The Template Method pattern defines the algorithm structure, while the Strategy pattern allows for complete algorithm interchangeability.
- [ ] The Strategy pattern defines the algorithm structure, while the Template Method pattern allows for complete algorithm interchangeability.
- [ ] Both patterns allow for complete algorithm interchangeability.
- [ ] Both patterns define the algorithm structure.

> **Explanation:** The Template Method pattern defines the algorithm structure, while the Strategy pattern allows for complete algorithm interchangeability.

### True or False: The Template Method pattern is suitable for highly dynamic scenarios.

- [ ] True
- [x] False

> **Explanation:** The Template Method pattern is not suitable for highly dynamic scenarios, as the algorithm structure is fixed.

{{< /quizdown >}}

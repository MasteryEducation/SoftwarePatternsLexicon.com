---
linkTitle: "2.2.7 Proxy"
title: "Proxy Design Pattern in Go: Control Access with Proxies"
description: "Explore the Proxy Design Pattern in Go, its types, implementation, and practical examples. Learn how to control access and enhance functionality using proxies."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Proxy Pattern
- GoF Patterns
- Structural Patterns
- Go Language
- Software Design
date: 2024-10-25
type: docs
nav_weight: 227000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/2/7"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.2.7 Proxy

The Proxy design pattern is a structural pattern that provides a surrogate or placeholder for another object to control access to it. This pattern is particularly useful when you need to add a level of indirection to access an object, allowing you to add functionality such as lazy initialization, access control, or logging without modifying the original object's code.

### Understand the Intent

- **Control Access:** The primary intent of the Proxy pattern is to control access to an object. This can be for reasons such as security, performance, or additional functionality.
- **Indirection Layer:** It introduces an additional layer of indirection when accessing an object, which can be used to manage the object's lifecycle or interactions.

### Implementation Steps

1. **Define an Interface:** Create an interface that both the real subject and the proxy will implement. This ensures that the proxy can be used interchangeably with the real subject.
   
2. **Implement the Real Subject:** Develop the real subject class that performs the actual work.

3. **Create the Proxy Struct:** Implement the proxy struct that holds a reference to the real subject. The proxy will implement the same interface as the real subject.

4. **Control Access:** The proxy will control access to the real subject and may add additional behavior such as logging, caching, or access control.

### Types of Proxies

- **Remote Proxy:** Provides a local representative for an object that exists in a different address space. This is commonly used in distributed systems to manage network communication.

- **Virtual Proxy:** Delays the creation and initialization of expensive objects until they are actually needed. This can significantly improve performance and resource utilization.

- **Protection Proxy:** Controls access to the original object based on access rights. This is useful for implementing security policies.

### When to Use

- When you need to control access to an object, especially in cases where the object is resource-intensive to create or manage.
- To add functionality to an object without changing its code, such as logging, caching, or access control.

### Go-Specific Tips

- **Interfaces:** Use Go interfaces to define the subject's methods. This allows the proxy to be used interchangeably with the real subject.
- **Transparency:** Ensure that the proxy adheres strictly to the subject's interface to maintain transparency and usability.

### Example: Virtual Proxy for Lazy Loading

Let's consider an example where we use a virtual proxy to manage access to a large image object. The proxy will handle the lazy loading of the image, meaning the image will only be loaded when it is actually needed.

```go
package main

import (
	"fmt"
	"sync"
)

// Image is the interface that both RealImage and ProxyImage will implement.
type Image interface {
	Display()
}

// RealImage is the actual object that is expensive to create.
type RealImage struct {
	filename string
}

// Display loads and displays the image.
func (img *RealImage) Display() {
	fmt.Println("Displaying", img.filename)
}

// loadImage simulates loading an image from disk.
func loadImage(filename string) *RealImage {
	fmt.Println("Loading image from disk:", filename)
	return &RealImage{filename: filename}
}

// ProxyImage is the proxy that controls access to the RealImage.
type ProxyImage struct {
	filename string
	realImage *RealImage
	once      sync.Once
}

// Display ensures the real image is loaded before displaying it.
func (proxy *ProxyImage) Display() {
	proxy.once.Do(func() {
		proxy.realImage = loadImage(proxy.filename)
	})
	proxy.realImage.Display()
}

func main() {
	image := &ProxyImage{filename: "large_image.jpg"}
	
	// The image is loaded from disk only when Display is called for the first time.
	image.Display()
	
	// Subsequent calls to Display do not load the image again.
	image.Display()
}
```

#### Explanation

- **Interface Definition:** The `Image` interface defines a `Display` method that both `RealImage` and `ProxyImage` implement.
- **Real Image:** The `RealImage` struct represents the actual image object that is expensive to load.
- **Proxy Image:** The `ProxyImage` struct acts as a proxy to the `RealImage`. It uses a `sync.Once` to ensure that the image is loaded only once, demonstrating lazy loading.
- **Lazy Loading:** The `Display` method in `ProxyImage` uses `sync.Once` to load the image only when it is first accessed, optimizing resource usage.

### Advantages and Disadvantages

#### Advantages

- **Controlled Access:** Provides controlled access to the real subject, which can be useful for security or resource management.
- **Lazy Initialization:** Delays the creation of resource-intensive objects, improving performance.
- **Additional Functionality:** Allows for additional functionality such as logging or caching without modifying the original object.

#### Disadvantages

- **Complexity:** Introduces additional complexity due to the extra layer of indirection.
- **Performance Overhead:** May introduce performance overhead due to the additional method calls.

### Best Practices

- **Interface Adherence:** Ensure that the proxy strictly adheres to the interface of the real subject to maintain transparency.
- **Use Cases:** Use proxies judiciously in scenarios where their benefits outweigh the added complexity.
- **Concurrency:** Consider concurrency implications, especially when using proxies in a multi-threaded environment.

### Comparisons

- **Decorator vs. Proxy:** While both patterns add functionality, a decorator adds responsibilities to an object dynamically, whereas a proxy controls access and may add functionality.
- **Adapter vs. Proxy:** An adapter changes the interface of an object, while a proxy provides the same interface with controlled access.

### Conclusion

The Proxy design pattern is a powerful tool for controlling access to objects, adding functionality, and optimizing resource usage. By understanding its types and implementation strategies, you can effectively leverage proxies in your Go applications to enhance performance and maintainability.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Proxy design pattern?

- [x] To control access to an object
- [ ] To change the interface of an object
- [ ] To add dynamic behavior to an object
- [ ] To encapsulate a request as an object

> **Explanation:** The Proxy design pattern's primary intent is to control access to an object, often adding a level of indirection.

### Which type of proxy delays the creation of expensive objects until they are needed?

- [x] Virtual Proxy
- [ ] Remote Proxy
- [ ] Protection Proxy
- [ ] Adapter Proxy

> **Explanation:** A Virtual Proxy delays the creation and initialization of expensive objects until they are actually needed.

### In Go, what is a common practice when implementing the Proxy pattern?

- [x] Use interfaces to define the subject's methods
- [ ] Use inheritance to extend the real subject
- [ ] Use global variables to manage state
- [ ] Use reflection to dynamically create methods

> **Explanation:** In Go, using interfaces to define the subject's methods allows the proxy to be used interchangeably with the real subject.

### What is a disadvantage of using the Proxy pattern?

- [x] It introduces additional complexity
- [ ] It eliminates the need for interfaces
- [ ] It simplifies the codebase
- [ ] It always improves performance

> **Explanation:** The Proxy pattern introduces additional complexity due to the extra layer of indirection.

### Which proxy type is commonly used in distributed systems?

- [x] Remote Proxy
- [ ] Virtual Proxy
- [ ] Protection Proxy
- [ ] Composite Proxy

> **Explanation:** A Remote Proxy is used to provide a local representative for an object in a different address space, common in distributed systems.

### What is a key benefit of using a Protection Proxy?

- [x] It controls access based on access rights
- [ ] It changes the object's interface
- [ ] It always improves performance
- [ ] It simplifies object creation

> **Explanation:** A Protection Proxy controls access to the original object based on access rights, enhancing security.

### How does a Proxy pattern differ from a Decorator pattern?

- [x] A Proxy controls access, while a Decorator adds responsibilities
- [ ] A Proxy changes the interface, while a Decorator does not
- [ ] A Proxy is used for logging, while a Decorator is not
- [ ] A Proxy is always faster than a Decorator

> **Explanation:** A Proxy controls access to an object, while a Decorator adds responsibilities to an object dynamically.

### What Go feature is often used in a Proxy to ensure lazy loading?

- [x] sync.Once
- [ ] reflect.Type
- [ ] context.Context
- [ ] fmt.Println

> **Explanation:** `sync.Once` is used in Go to ensure that a particular action, such as loading an object, is performed only once.

### Which pattern is used to provide a simplified interface to a complex subsystem?

- [ ] Proxy
- [ ] Adapter
- [ ] Decorator
- [x] Facade

> **Explanation:** The Facade pattern provides a simplified interface to a complex subsystem, not the Proxy pattern.

### True or False: A Proxy pattern can be used to add logging functionality to an object.

- [x] True
- [ ] False

> **Explanation:** True. A Proxy can add additional functionality such as logging without modifying the original object's code.

{{< /quizdown >}}

---
canonical: "https://softwarepatternslexicon.com/patterns-java/7/8/5"
title: "Proxy Pattern Use Cases and Examples"
description: "Explore practical applications of the Proxy Pattern in Java, including lazy loading, security systems, and remote method invocation."
linkTitle: "7.8.5 Use Cases and Examples"
tags:
- "Java"
- "Design Patterns"
- "Proxy Pattern"
- "Lazy Loading"
- "Security"
- "Remote Method Invocation"
- "Performance"
- "Software Architecture"
date: 2024-11-25
type: docs
nav_weight: 78500
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 7.8.5 Use Cases and Examples

The Proxy Pattern is a structural design pattern that provides a surrogate or placeholder for another object to control access to it. This pattern is particularly useful in scenarios where direct access to an object is either undesirable or impractical. The Proxy Pattern can be employed in various contexts, such as implementing lazy loading, enhancing security, and facilitating remote method invocation. This section delves into these use cases, providing detailed examples and explanations to illustrate the practical applications of the Proxy Pattern in Java.

### Lazy Loading with Virtual Proxies

Lazy loading is a design pattern commonly used to defer the initialization of an object until it is needed. This can be particularly beneficial in scenarios where the object is resource-intensive to create or may not be used at all during the lifecycle of an application. Virtual proxies are an ideal solution for implementing lazy loading, as they act as a stand-in for the real object and only instantiate it when necessary.

#### Example: Image Viewer Application

Consider an image viewer application that displays thumbnails of images stored on a remote server. Loading all images at once can be resource-intensive and slow, especially if the images are large. By using a virtual proxy, the application can load only the images that the user chooses to view in full size.

```java
// Interface representing an image
interface Image {
    void display();
}

// RealImage class that loads and displays an image
class RealImage implements Image {
    private String filename;

    public RealImage(String filename) {
        this.filename = filename;
        loadImageFromDisk();
    }

    private void loadImageFromDisk() {
        System.out.println("Loading " + filename);
    }

    @Override
    public void display() {
        System.out.println("Displaying " + filename);
    }
}

// ProxyImage class that acts as a virtual proxy
class ProxyImage implements Image {
    private RealImage realImage;
    private String filename;

    public ProxyImage(String filename) {
        this.filename = filename;
    }

    @Override
    public void display() {
        if (realImage == null) {
            realImage = new RealImage(filename);
        }
        realImage.display();
    }
}

// Client code
public class ProxyPatternDemo {
    public static void main(String[] args) {
        Image image1 = new ProxyImage("photo1.jpg");
        Image image2 = new ProxyImage("photo2.jpg");

        // Image will be loaded from disk
        image1.display();
        // Image will not be loaded from disk
        image1.display();

        // Image will be loaded from disk
        image2.display();
        // Image will not be loaded from disk
        image2.display();
    }
}
```

In this example, the `ProxyImage` class acts as a virtual proxy for the `RealImage` class. The `display()` method of `ProxyImage` checks if the `RealImage` has been instantiated. If not, it creates the `RealImage` object, thereby deferring the loading of the image until it is actually needed.

### Security Systems with Protection Proxies

Protection proxies are used to control access to an object based on access rights. This is particularly useful in security systems where different users have different levels of access to resources.

#### Example: Access Control in a Document Management System

Consider a document management system where documents can be accessed by users with different roles, such as admin, editor, and viewer. A protection proxy can be used to enforce access control based on the user's role.

```java
// Interface representing a document
interface Document {
    void displayContent();
}

// RealDocument class that displays the content of a document
class RealDocument implements Document {
    private String content;

    public RealDocument(String content) {
        this.content = content;
    }

    @Override
    public void displayContent() {
        System.out.println("Document Content: " + content);
    }
}

// ProxyDocument class that acts as a protection proxy
class ProxyDocument implements Document {
    private RealDocument realDocument;
    private String content;
    private String userRole;

    public ProxyDocument(String content, String userRole) {
        this.content = content;
        this.userRole = userRole;
    }

    @Override
    public void displayContent() {
        if ("admin".equals(userRole) || "editor".equals(userRole)) {
            if (realDocument == null) {
                realDocument = new RealDocument(content);
            }
            realDocument.displayContent();
        } else {
            System.out.println("Access Denied: Insufficient permissions");
        }
    }
}

// Client code
public class ProtectionProxyDemo {
    public static void main(String[] args) {
        Document adminDocument = new ProxyDocument("Confidential Document", "admin");
        Document viewerDocument = new ProxyDocument("Confidential Document", "viewer");

        // Admin has access
        adminDocument.displayContent();

        // Viewer does not have access
        viewerDocument.displayContent();
    }
}
```

In this example, the `ProxyDocument` class checks the user's role before granting access to the `RealDocument`. Only users with the "admin" or "editor" roles are allowed to view the document content, while others receive an "Access Denied" message.

### Remote Proxies in RMI (Remote Method Invocation)

Remote proxies are used to represent objects that reside in different address spaces, such as on a remote server. In Java, Remote Method Invocation (RMI) is a mechanism that allows an object residing in one Java Virtual Machine (JVM) to invoke methods on an object in another JVM. Remote proxies play a crucial role in RMI by acting as local representatives for remote objects.

#### Example: Remote Calculator Service

Consider a remote calculator service that performs arithmetic operations. The client application uses a remote proxy to interact with the calculator service.

```java
import java.rmi.Remote;
import java.rmi.RemoteException;

// Remote interface for the calculator service
interface Calculator extends Remote {
    int add(int a, int b) throws RemoteException;
    int subtract(int a, int b) throws RemoteException;
}

// Implementation of the remote calculator service
import java.rmi.server.UnicastRemoteObject;
import java.rmi.RemoteException;

class CalculatorImpl extends UnicastRemoteObject implements Calculator {
    protected CalculatorImpl() throws RemoteException {
        super();
    }

    @Override
    public int add(int a, int b) throws RemoteException {
        return a + b;
    }

    @Override
    public int subtract(int a, int b) throws RemoteException {
        return a - b;
    }
}

// Client code
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class RMIClient {
    public static void main(String[] args) {
        try {
            Registry registry = LocateRegistry.getRegistry("localhost");
            Calculator calculator = (Calculator) registry.lookup("CalculatorService");

            System.out.println("Addition: " + calculator.add(5, 3));
            System.out.println("Subtraction: " + calculator.subtract(5, 3));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

In this example, the `Calculator` interface defines the remote methods, and the `CalculatorImpl` class provides the implementation. The client code uses the `LocateRegistry` to obtain a reference to the remote `Calculator` object, which acts as a remote proxy. The client can then invoke methods on the remote calculator as if it were a local object.

### Performance Improvement with Smart Proxies

Smart proxies can perform additional actions when an object is accessed. These actions can include logging, caching, or other optimizations to improve performance or monitor usage.

#### Example: Caching Proxy for Database Queries

Consider a database application where certain queries are expensive to execute. A caching proxy can be used to cache the results of these queries, reducing the load on the database and improving performance.

```java
import java.util.HashMap;
import java.util.Map;

// Interface representing a database query
interface DatabaseQuery {
    String execute(String query);
}

// RealDatabaseQuery class that executes a query on the database
class RealDatabaseQuery implements DatabaseQuery {
    @Override
    public String execute(String query) {
        // Simulate a database query execution
        System.out.println("Executing query: " + query);
        return "Result of " + query;
    }
}

// CachingProxy class that acts as a smart proxy
class CachingProxy implements DatabaseQuery {
    private RealDatabaseQuery realDatabaseQuery;
    private Map<String, String> cache;

    public CachingProxy() {
        this.realDatabaseQuery = new RealDatabaseQuery();
        this.cache = new HashMap<>();
    }

    @Override
    public String execute(String query) {
        if (cache.containsKey(query)) {
            System.out.println("Returning cached result for query: " + query);
            return cache.get(query);
        } else {
            String result = realDatabaseQuery.execute(query);
            cache.put(query, result);
            return result;
        }
    }
}

// Client code
public class CachingProxyDemo {
    public static void main(String[] args) {
        DatabaseQuery query = new CachingProxy();

        // First execution will query the database
        System.out.println(query.execute("SELECT * FROM users"));

        // Second execution will return cached result
        System.out.println(query.execute("SELECT * FROM users"));
    }
}
```

In this example, the `CachingProxy` class caches the results of database queries. When a query is executed, the proxy checks if the result is already in the cache. If so, it returns the cached result; otherwise, it executes the query on the `RealDatabaseQuery` and stores the result in the cache.

### Conclusion

The Proxy Pattern is a versatile design pattern that can be applied in various scenarios to enhance performance, security, and flexibility. By understanding and implementing the Proxy Pattern, developers can create more efficient and secure applications. Whether it's through lazy loading with virtual proxies, access control with protection proxies, remote method invocation with remote proxies, or performance optimization with smart proxies, the Proxy Pattern offers a robust solution for managing object access and behavior.

### Related Patterns

- **Decorator Pattern**: Similar to the Proxy Pattern, the Decorator Pattern adds additional responsibilities to an object dynamically. However, the Decorator Pattern focuses on enhancing functionality, while the Proxy Pattern focuses on controlling access.
- **Adapter Pattern**: The Adapter Pattern allows incompatible interfaces to work together, whereas the Proxy Pattern provides a surrogate for another object.

### Known Uses

- **Java RMI**: The Java RMI framework uses remote proxies to enable communication between distributed objects.
- **Spring AOP**: The Spring Framework uses proxies to implement aspect-oriented programming (AOP), allowing cross-cutting concerns like logging and transaction management to be applied declaratively.

By leveraging the Proxy Pattern, developers can create applications that are not only efficient and secure but also maintainable and scalable. This pattern is a powerful tool in the software architect's toolkit, offering solutions to common challenges in software design.

## Test Your Knowledge: Proxy Pattern Use Cases Quiz

{{< quizdown >}}

### Which type of proxy is used to defer the creation of an object until it is needed?

- [x] Virtual Proxy
- [ ] Protection Proxy
- [ ] Remote Proxy
- [ ] Smart Proxy

> **Explanation:** A Virtual Proxy is used to implement lazy loading by deferring the creation of an object until it is needed.


### What is the primary purpose of a protection proxy?

- [x] To control access to an object based on access rights
- [ ] To cache results of expensive operations
- [ ] To enhance the functionality of an object
- [ ] To facilitate communication between distributed objects

> **Explanation:** A Protection Proxy controls access to an object based on access rights, ensuring that only authorized users can access certain functionalities.


### In Java RMI, what role does a remote proxy play?

- [x] Acts as a local representative for a remote object
- [ ] Enhances the functionality of a local object
- [ ] Caches results of remote method calls
- [ ] Controls access to a local object

> **Explanation:** In Java RMI, a Remote Proxy acts as a local representative for a remote object, allowing local clients to interact with remote services.


### Which proxy type is used to improve performance by caching results?

- [x] Smart Proxy
- [ ] Virtual Proxy
- [ ] Protection Proxy
- [ ] Remote Proxy

> **Explanation:** A Smart Proxy can perform additional actions such as caching results to improve performance.


### What is a common use case for a virtual proxy?

- [x] Lazy loading of resource-intensive objects
- [ ] Controlling access to sensitive data
- [ ] Facilitating remote method invocation
- [ ] Logging method calls

> **Explanation:** A Virtual Proxy is commonly used for lazy loading, where the creation of resource-intensive objects is deferred until they are needed.


### How does a protection proxy enhance security?

- [x] By restricting access based on user roles
- [ ] By encrypting data
- [ ] By logging access attempts
- [ ] By caching sensitive data

> **Explanation:** A Protection Proxy enhances security by restricting access to an object based on user roles or permissions.


### What is the main difference between a proxy and a decorator pattern?

- [x] Proxy controls access, while Decorator adds functionality
- [ ] Proxy adds functionality, while Decorator controls access
- [ ] Both patterns serve the same purpose
- [ ] Proxy and Decorator are unrelated

> **Explanation:** The Proxy Pattern controls access to an object, while the Decorator Pattern adds additional functionality to an object.


### Which pattern is often used in conjunction with the Proxy Pattern for remote method invocation?

- [x] Remote Proxy
- [ ] Adapter Pattern
- [ ] Decorator Pattern
- [ ] Singleton Pattern

> **Explanation:** The Remote Proxy is often used in conjunction with the Proxy Pattern for remote method invocation, allowing local clients to interact with remote services.


### What is a potential drawback of using proxies?

- [x] Increased complexity and potential performance overhead
- [ ] Reduced security
- [ ] Limited functionality
- [ ] Incompatibility with other patterns

> **Explanation:** Using proxies can increase complexity and introduce potential performance overhead due to the additional layer of indirection.


### True or False: The Proxy Pattern can only be used for security purposes.

- [ ] True
- [x] False

> **Explanation:** False. The Proxy Pattern can be used for various purposes, including lazy loading, performance optimization, and remote method invocation, in addition to security.

{{< /quizdown >}}

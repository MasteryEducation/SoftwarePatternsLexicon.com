---
canonical: "https://softwarepatternslexicon.com/patterns-java/7/8/2"

title: "Java Proxy Pattern: Types of Proxies (Virtual, Protection, Remote)"
description: "Explore the various types of proxies in Java, including Virtual, Protection, and Remote proxies, and learn when and how to use them effectively."
linkTitle: "7.8.2 Types of Proxies (Virtual, Protection, Remote)"
tags:
- "Java"
- "Design Patterns"
- "Proxy Pattern"
- "Virtual Proxy"
- "Protection Proxy"
- "Remote Proxy"
- "Software Architecture"
- "Advanced Java"
date: 2024-11-25
type: docs
nav_weight: 78200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 7.8.2 Types of Proxies (Virtual, Protection, Remote)

The Proxy Pattern is a structural design pattern that provides a surrogate or placeholder for another object to control access to it. This pattern is particularly useful in scenarios where direct access to an object is either costly or undesirable. In Java, proxies can be implemented in various forms, each serving a distinct purpose. This section delves into three primary types of proxies: Virtual, Protection, and Remote, exploring their characteristics, use cases, and implementation strategies.

### Virtual Proxies

#### Definition

Virtual proxies are used to delay the creation and initialization of expensive objects until they are actually needed. This type of proxy is beneficial in scenarios where the cost of creating an object is high, but it might not be used immediately or at all.

#### Motivation

Consider an application that displays high-resolution images. Loading all images at once could be resource-intensive and slow down the application. A virtual proxy can be used to load images on demand, improving performance and resource management.

#### Implementation

Virtual proxies typically implement the same interface as the real object and contain a reference to the real object. They intercept calls to the real object and instantiate it only when necessary.

```java
interface Image {
    void display();
}

class RealImage implements Image {
    private String filename;

    public RealImage(String filename) {
        this.filename = filename;
        loadFromDisk();
    }

    private void loadFromDisk() {
        System.out.println("Loading " + filename);
    }

    @Override
    public void display() {
        System.out.println("Displaying " + filename);
    }
}

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

public class ProxyPatternDemo {
    public static void main(String[] args) {
        Image image = new ProxyImage("test.jpg");
        // Image will be loaded from disk
        image.display();
        // Image will not be loaded from disk
        image.display();
    }
}
```

#### Real-World Example

In web browsers, images are often loaded lazily to improve page load times. Virtual proxies can be used to implement such lazy loading mechanisms.

### Protection Proxies

#### Definition

Protection proxies control access to an object based on access rights. They are used to enforce access control and ensure that only authorized users can perform certain operations on an object.

#### Motivation

In a multi-user application, different users might have different permissions. A protection proxy can ensure that users can only access the resources they are permitted to.

#### Implementation

Protection proxies typically check the credentials of the user before forwarding the request to the real object.

```java
interface Document {
    void displayContent();
}

class SecureDocument implements Document {
    private String content;

    public SecureDocument(String content) {
        this.content = content;
    }

    @Override
    public void displayContent() {
        System.out.println("Document Content: " + content);
    }
}

class ProtectionProxyDocument implements Document {
    private SecureDocument secureDocument;
    private String userRole;

    public ProtectionProxyDocument(String content, String userRole) {
        this.secureDocument = new SecureDocument(content);
        this.userRole = userRole;
    }

    @Override
    public void displayContent() {
        if ("ADMIN".equals(userRole)) {
            secureDocument.displayContent();
        } else {
            System.out.println("Access Denied: Insufficient permissions");
        }
    }
}

public class ProtectionProxyDemo {
    public static void main(String[] args) {
        Document document = new ProtectionProxyDocument("Sensitive Data", "USER");
        document.displayContent(); // Access Denied

        Document adminDocument = new ProtectionProxyDocument("Sensitive Data", "ADMIN");
        adminDocument.displayContent(); // Document Content
    }
}
```

#### Real-World Example

In enterprise applications, protection proxies can be used to enforce role-based access control (RBAC), ensuring that users can only access data and perform actions that their role permits.

### Remote Proxies

#### Definition

Remote proxies represent objects that reside in different address spaces, such as on a different machine or in a different process. They are used to facilitate communication between distributed systems.

#### Motivation

In distributed systems, objects might need to interact with objects located on different servers. Remote proxies provide a way to interact with these remote objects as if they were local.

#### Implementation

Remote proxies typically involve network communication and serialization/deserialization of method calls and responses.

```java
// Remote interface
interface RemoteService {
    String fetchData();
}

// Remote object implementation
class RemoteServiceImpl implements RemoteService {
    @Override
    public String fetchData() {
        return "Data from remote service";
    }
}

// Remote proxy
class RemoteServiceProxy implements RemoteService {
    private RemoteService remoteService;

    public RemoteServiceProxy(RemoteService remoteService) {
        this.remoteService = remoteService;
    }

    @Override
    public String fetchData() {
        // Simulate network communication
        System.out.println("Fetching data from remote service...");
        return remoteService.fetchData();
    }
}

public class RemoteProxyDemo {
    public static void main(String[] args) {
        RemoteService remoteService = new RemoteServiceImpl();
        RemoteService proxy = new RemoteServiceProxy(remoteService);
        System.out.println(proxy.fetchData());
    }
}
```

#### Real-World Example

Remote proxies are commonly used in Java RMI (Remote Method Invocation) and web services, where they allow local code to invoke methods on remote objects seamlessly.

### Differences in Implementation

- **Virtual Proxies**: Focus on lazy initialization and resource management. They are implemented by deferring the creation of the real object until it is needed.
- **Protection Proxies**: Emphasize access control. They are implemented by checking user permissions before allowing access to the real object.
- **Remote Proxies**: Deal with network communication and serialization. They are implemented by handling the complexities of remote method invocation and data transfer.

### Conclusion

Proxies are a powerful tool in software design, providing flexibility and control over object interactions. By understanding the different types of proxies and their use cases, developers can design systems that are efficient, secure, and scalable. Whether it's managing resources with virtual proxies, enforcing security with protection proxies, or enabling distributed computing with remote proxies, the Proxy Pattern offers a versatile solution to many common challenges in software architecture.

### Further Reading

- [Java Documentation](https://docs.oracle.com/en/java/)
- [Cloud Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/)

## Test Your Knowledge: Java Proxy Pattern Quiz

{{< quizdown >}}

### What is the primary purpose of a virtual proxy?

- [x] To delay the creation of an expensive object until it is needed.
- [ ] To control access to an object based on permissions.
- [ ] To represent an object in a different address space.
- [ ] To enhance the performance of a system by caching objects.

> **Explanation:** Virtual proxies are used to delay the creation and initialization of expensive objects until they are actually needed, improving performance and resource management.

### Which type of proxy is used to enforce access control based on user permissions?

- [ ] Virtual Proxy
- [x] Protection Proxy
- [ ] Remote Proxy
- [ ] Caching Proxy

> **Explanation:** Protection proxies control access to an object based on access rights, ensuring that only authorized users can perform certain operations.

### In which scenario would a remote proxy be most appropriate?

- [ ] When objects are expensive to create.
- [ ] When access control is required.
- [x] When objects reside in different address spaces.
- [ ] When objects need to be cached locally.

> **Explanation:** Remote proxies are used to represent objects that reside in different address spaces, facilitating communication between distributed systems.

### What is a key characteristic of a protection proxy?

- [ ] It delays object creation.
- [x] It manages access based on permissions.
- [ ] It handles network communication.
- [ ] It caches objects for performance.

> **Explanation:** Protection proxies manage access to an object based on user permissions, ensuring that only authorized users can access certain functionalities.

### Which of the following is a common use case for virtual proxies?

- [x] Lazy loading of images in a web application.
- [ ] Role-based access control in an enterprise system.
- [ ] Remote method invocation in a distributed system.
- [ ] Caching frequently accessed data.

> **Explanation:** Virtual proxies are often used for lazy loading, such as loading images on demand in a web application to improve performance.

### What is the main advantage of using a remote proxy?

- [x] It allows local code to interact with remote objects seamlessly.
- [ ] It improves the security of an application.
- [ ] It reduces the memory footprint of an application.
- [ ] It simplifies the user interface of an application.

> **Explanation:** Remote proxies enable local code to invoke methods on remote objects as if they were local, facilitating distributed computing.

### How does a virtual proxy improve performance?

- [x] By delaying the creation of expensive objects until they are needed.
- [ ] By caching objects locally.
- [ ] By reducing network latency.
- [ ] By compressing data before transmission.

> **Explanation:** Virtual proxies improve performance by deferring the creation and initialization of expensive objects until they are actually needed.

### What is a common implementation detail of a remote proxy?

- [ ] It uses lazy initialization.
- [ ] It checks user permissions.
- [x] It handles network communication and serialization.
- [ ] It caches frequently accessed data.

> **Explanation:** Remote proxies involve network communication and serialization/deserialization of method calls and responses to facilitate interaction with remote objects.

### Which type of proxy would be used in a system that requires role-based access control?

- [ ] Virtual Proxy
- [x] Protection Proxy
- [ ] Remote Proxy
- [ ] Caching Proxy

> **Explanation:** Protection proxies are used to enforce role-based access control, ensuring that users can only access resources they are permitted to.

### True or False: A virtual proxy can be used to represent objects in different address spaces.

- [ ] True
- [x] False

> **Explanation:** Virtual proxies are used to delay object creation, not to represent objects in different address spaces. Remote proxies are used for that purpose.

{{< /quizdown >}}

By understanding and implementing these types of proxies, developers can create more efficient, secure, and scalable applications. Whether dealing with resource-intensive objects, enforcing security policies, or enabling distributed computing, the Proxy Pattern offers versatile solutions to common challenges in software design.

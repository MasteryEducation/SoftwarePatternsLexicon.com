---
canonical: "https://softwarepatternslexicon.com/patterns-clojure/13/13"
title: "Server-Side Rendering and Templating with Selmer for Clojure Web Development"
description: "Explore server-side rendering and templating in Clojure using Selmer, a powerful templating engine. Learn how to create dynamic HTML templates, pass data from handlers, and ensure performance and security."
linkTitle: "13.13. Server-Side Rendering and Templating with Selmer"
tags:
- "Clojure"
- "Web Development"
- "Selmer"
- "Server-Side Rendering"
- "Templating"
- "HTML"
- "Security"
- "Performance"
date: 2024-11-25
type: docs
nav_weight: 143000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 13.13. Server-Side Rendering and Templating with Selmer

Server-side rendering (SSR) is a technique used to generate HTML content on the server rather than in the browser. This approach can improve the performance of web applications by reducing the time it takes for a page to become interactive. In Clojure, one of the most popular templating engines for SSR is Selmer. In this section, we will explore how to use Selmer for server-side rendering and templating in Clojure web applications.

### Introduction to Selmer

Selmer is a templating engine for Clojure inspired by Django's template language. It allows developers to create dynamic HTML content by embedding Clojure expressions within templates. Selmer is known for its simplicity and flexibility, making it a popular choice for Clojure web developers.

#### Key Features of Selmer

- **Simple Syntax**: Selmer uses a syntax similar to Django templates, which is easy to learn and use.
- **Template Inheritance**: Allows templates to extend other templates, promoting code reuse and maintainability.
- **Macros**: Provides the ability to define reusable template snippets.
- **Security**: Includes features to prevent common security vulnerabilities such as Cross-Site Scripting (XSS).

### Creating HTML Templates with Selmer

To get started with Selmer, you need to create HTML templates that contain placeholders for dynamic content. These placeholders are replaced with actual data when the template is rendered.

#### Basic Template Example

Here's a simple example of a Selmer template:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{title}}</title>
</head>
<body>
    <h1>Welcome, {{name}}!</h1>
    <p>This is a simple Selmer template example.</p>
</body>
</html>
```

In this template, `{{title}}` and `{{name}}` are placeholders that will be replaced with actual values when the template is rendered.

### Rendering Templates with Data

To render a Selmer template, you need to pass a map of data to the `selmer.parser/render` function. This map contains the values that will replace the placeholders in the template.

#### Example: Rendering a Template

```clojure
(require '[selmer.parser :as parser])

(defn render-welcome-page [name]
  (let [template "<!DOCTYPE html>
                  <html lang=\"en\">
                  <head>
                      <meta charset=\"UTF-8\">
                      <title>{{title}}</title>
                  </head>
                  <body>
                      <h1>Welcome, {{name}}!</h1>
                      <p>This is a simple Selmer template example.</p>
                  </body>
                  </html>"
        data {:title "Welcome Page" :name name}]
    (parser/render template data)))

;; Usage
(render-welcome-page "Alice")
```

In this example, the `render-welcome-page` function takes a `name` parameter and uses it to render the template with the provided data.

### Template Inheritance

Selmer supports template inheritance, which allows you to create a base template that can be extended by other templates. This feature is useful for maintaining a consistent layout across multiple pages.

#### Example: Base Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}Default Title{% endblock %}</title>
</head>
<body>
    <header>
        <h1>My Website</h1>
    </header>
    <main>
        {% block content %}
        <!-- Default content goes here -->
        {% endblock %}
    </main>
    <footer>
        <p>© 2024 My Website</p>
    </footer>
</body>
</html>
```

#### Example: Child Template

```html
{% extends "base.html" %}

{% block title %}Home Page{% endblock %}

{% block content %}
<p>Welcome to the home page!</p>
{% endblock %}
```

In this example, the child template extends the base template and overrides the `title` and `content` blocks.

### Using Macros in Selmer

Macros in Selmer allow you to define reusable template snippets. This feature is particularly useful for repeating elements such as navigation bars or footers.

#### Example: Defining and Using a Macro

```html
{% macro nav-bar %}
<nav>
    <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/about">About</a></li>
        <li><a href="/contact">Contact</a></li>
    </ul>
</nav>
{% endmacro %}

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{title}}</title>
</head>
<body>
    {{ nav-bar() }}
    <h1>{{title}}</h1>
    <p>{{content}}</p>
</body>
</html>
```

In this example, the `nav-bar` macro is defined and then used within the template to render a navigation bar.

### Performance Considerations

When using server-side rendering with Selmer, it's important to consider performance. Here are some tips to optimize performance:

- **Template Caching**: Cache rendered templates to avoid re-parsing them on every request.
- **Minimize Template Logic**: Keep logic in templates to a minimum to reduce processing time.
- **Use Efficient Data Structures**: Ensure that the data passed to templates is in an efficient format for quick access.

### Security Considerations

Security is a critical aspect of web development. Selmer provides features to help prevent common vulnerabilities such as XSS.

#### Preventing XSS Attacks

Selmer automatically escapes HTML entities to prevent XSS attacks. However, it's important to be aware of the following:

- **Avoid Unescaped Output**: Do not use unescaped output unless absolutely necessary.
- **Validate Input**: Always validate and sanitize user input before rendering it in templates.

### Conclusion

Selmer is a powerful and flexible templating engine for Clojure that simplifies server-side rendering. By using Selmer, you can create dynamic and secure web applications with ease. Remember to consider performance and security best practices when working with Selmer templates.

### External Links

- [Selmer GitHub Repository](https://github.com/yogthos/Selmer)

## **Ready to Test Your Knowledge?**

{{< quizdown >}}

### What is Selmer in the context of Clojure?

- [x] A templating engine for server-side rendering
- [ ] A database library
- [ ] A concurrency model
- [ ] A testing framework

> **Explanation:** Selmer is a templating engine used for server-side rendering in Clojure web applications.

### How do you pass data to a Selmer template?

- [x] By using a map with keys corresponding to template placeholders
- [ ] By embedding data directly in the HTML
- [ ] By using global variables
- [ ] By using environment variables

> **Explanation:** Data is passed to Selmer templates using a map where keys match the placeholders in the template.

### What feature of Selmer allows templates to extend other templates?

- [x] Template inheritance
- [ ] Macros
- [ ] Data binding
- [ ] Event handling

> **Explanation:** Template inheritance in Selmer allows one template to extend another, promoting code reuse.

### What is a macro in Selmer?

- [x] A reusable template snippet
- [ ] A function for data manipulation
- [ ] A concurrency primitive
- [ ] A database query

> **Explanation:** Macros in Selmer are reusable template snippets that can be defined and used within templates.

### How does Selmer help prevent XSS attacks?

- [x] By automatically escaping HTML entities
- [ ] By using encryption
- [ ] By validating user input
- [ ] By using secure cookies

> **Explanation:** Selmer automatically escapes HTML entities to prevent XSS attacks, ensuring that user input is not executed as code.

### What should you avoid to maintain security in Selmer templates?

- [x] Unescaped output
- [ ] Using macros
- [ ] Template inheritance
- [ ] Using placeholders

> **Explanation:** Unescaped output should be avoided to prevent security vulnerabilities like XSS.

### Which of the following is a performance optimization tip for Selmer?

- [x] Cache rendered templates
- [ ] Use more template logic
- [ ] Increase server load
- [ ] Use larger data structures

> **Explanation:** Caching rendered templates can improve performance by reducing the need to re-parse templates on each request.

### What is the purpose of using a map in Selmer?

- [x] To provide data for template placeholders
- [ ] To define routes
- [ ] To manage state
- [ ] To handle concurrency

> **Explanation:** A map is used to provide data for template placeholders, allowing dynamic content rendering.

### What is the main advantage of using template inheritance in Selmer?

- [x] Code reuse and maintainability
- [ ] Faster rendering
- [ ] Increased security
- [ ] Simplified data handling

> **Explanation:** Template inheritance promotes code reuse and maintainability by allowing templates to extend others.

### True or False: Selmer templates can include logic for data processing.

- [x] True
- [ ] False

> **Explanation:** While Selmer templates can include some logic, it's recommended to keep it minimal for performance reasons.

{{< /quizdown >}}

Remember, mastering server-side rendering with Selmer is just the beginning. As you continue to explore Clojure web development, you'll discover more advanced techniques and patterns. Keep experimenting, stay curious, and enjoy the journey!
